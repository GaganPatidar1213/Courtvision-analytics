import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

PALETTE = ["#F07837", "#8EC0CC", "#C2A8DD", "#BFC989", "#E2B28C"]
PLOTS = ["Histogram", "Box plot", "Violin plot", "Bar chart", "Pie chart", "Line chart",
         "Scatter plot", "Correlation heatmap", "Pair plot", "Density plot", "Area chart", "ECDF"]
MEASURES = ["x", "y", "z", "magnitude"]
LABELS = {key: f"{key} (m/s²)" for key in MEASURES}
LABELS.update(time_s="Elapsed time (s)", activity="Activity", count="Samples")
OPTIONS = dict(template="plotly_dark", color_discrete_sequence=PALETTE, labels=LABELS)
ACTIVITY_COLORS = dict(zip(("dribble", "hold", "pass", "pickup", "shoot"), PALETTE))
UNLABELED = "Unlabeled"


def style_figure(figure):
    figure.update_layout(paper_bgcolor="#101114", plot_bgcolor="#1C1D20",
                         font=dict(family="Inter, Segoe UI, sans-serif", color="#F5F0E8"),
                         title_font=dict(family="Space Grotesk, Segoe UI, sans-serif", size=19),
                         hoverlabel=dict(bgcolor="#1C1D20", font_color="#F5F0E8"))
    figure.update_xaxes(gridcolor="#333335", linecolor="#555451", automargin=True)
    figure.update_yaxes(gridcolor="#333335", linecolor="#555451", automargin=True)
    return figure


def make_plot(kind, data, x="magnitude", y="z", recording=None):
    figure, purpose, observations, _ = chart_analysis(kind, data, x, y, recording)
    return figure, purpose, observations


def activity_labels(data):
    return data.activity.astype("string").str.strip().replace("", pd.NA).fillna(UNLABELED)


def chart_analysis(kind, data, x="magnitude", y="z", recording=None):
    if kind not in PLOTS:
        raise ValueError(f"Unknown chart type: {kind}")
    if data.empty:
        raise ValueError("No samples match the current filters.")
    data = data.assign(activity=activity_labels(data))
    values = data[x]
    n = len(data)
    q1, median, q3 = values.quantile([.25, .5, .75])
    counts = data.activity.value_counts().rename_axis("activity").reset_index(name="count")
    if kind == "Histogram":
        freq, edges = np.histogram(values, bins=30)
        peak = freq.argmax()
        figure = go.Figure(go.Histogram(
            x=values, marker_color=PALETTE[0],
            xbins=dict(start=float(edges[0]), end=float(np.nextafter(edges[-1], np.inf)),
                       size=float(edges[1]-edges[0])),
        ))
        figure.update_layout(template="plotly_dark", xaxis_title=LABELS[x], yaxis_title="Samples")
        description = "Shows the sample count in 30 equal-width acceleration bins."
        findings = [f"The busiest bin spans {edges[peak]:.3f} to {edges[peak+1]:.3f} with {freq[peak]:,} samples.",
                    f"The median is {median:.3f}.", f"The sample mean is {values.mean():.3f}."]
        interpretation = (f"About {freq[peak] / n:.1%} of samples fall in the busiest bin. "
                          f"The middle half spans {q1:.3f} to {q3:.3f} m/s².")
    elif kind in ("Box plot", "Violin plot"):
        function = px.box if kind == "Box plot" else px.violin
        figure = function(data, x="activity", y=x, color="activity", **OPTIONS)
        groups = data.groupby("activity")[x]
        medians = groups.median()
        spread = groups.quantile(.75) - groups.quantile(.25)
        description = ("Compares medians, quartiles and extreme observations across activities."
                       if kind == "Box plot" else "Compares the estimated density shape across activities.")
        findings = [f"{medians.idxmax()} has the highest median ({medians.max():.3f}).",
                    f"{medians.idxmin()} has the lowest median ({medians.min():.3f}).",
                    f"{spread.idxmax()} has the widest IQR ({spread.max():.3f})."]
        interpretation = (f"The middle half of {spread.idxmax()} spans {spread.max():.3f} m/s². "
                          f"The gap between the highest and lowest group medians is "
                          f"{medians.max() - medians.min():.3f} m/s².")
    elif kind in ("Bar chart", "Pie chart"):
        if kind == "Pie chart":
            figure = go.Figure(go.Pie(
                labels=counts.activity.tolist(), values=counts["count"].tolist(),
                sort=False, textinfo="label+percent", textposition="auto",
                insidetextfont=dict(color="#101114", size=13),
                outsidetextfont=dict(color="#F5F0E8", size=13),
                marker=dict(colors=[ACTIVITY_COLORS.get(name, "#9C9A96") for name in counts.activity],
                            line=dict(color="#1C1D20", width=2)),
                hovertemplate="<b>%{label}</b><br>%{value:,} samples<br>%{percent:.1%} of selected samples<extra></extra>",
                showlegend=True,
            ))
            figure.update_layout(template="plotly_dark", legend=dict(orientation="h", y=-.06, x=0))
            description = "Shows each activity’s percentage of the selected sensor samples. Unlabeled samples, if present, appear as their own slice."
        else:
            figure = px.bar(counts, x="activity", y="count", color="activity",
                            color_discrete_map=ACTIVITY_COLORS, **OPTIONS)
            description = "Compares the number of sensor samples recorded for each selected activity."
        high, low = counts["count"].max(), counts["count"].min()
        top = ", ".join(counts.loc[counts["count"].eq(high), "activity"])
        bottom = ", ".join(counts.loc[counts["count"].eq(low), "activity"])
        findings = [f"Most represented: {top} ({high:,} samples per activity, {high/n:.1%} each).",
                    f"Least represented: {bottom} ({low:,} samples per activity, {low/n:.1%} each).",
                    f"The view includes {n:,} samples across {data.recording.nunique()} recordings."]
        interpretation = (f"The largest and smallest groups differ by {high-low:,} samples "
                          f"({(high-low)/n:.1%} of the selection). These shares measure recorded samples, "
                          "so longer recordings contribute more.")
    elif kind in ("Line chart", "Area chart"):
        trial = data.loc[data.recording.eq(recording or data.recording.iloc[0])].sort_values("time_s")
        peak = trial.loc[trial[x].idxmax()]
        if kind == "Line chart":
            shown = trial.assign(filtered=trial[f"{x}_filtered"])
            figure = px.line(shown, x="time_s", y=[x, "filtered"], **OPTIONS)
            figure.update_yaxes(title_text=LABELS[x])
            description = f"{peak.recording} · compares recorded and median-filtered acceleration."
            findings = [f"The raw signal peaks at {peak[x]:.3f} after {peak.time_s:.3f} s.",
                    f"The filtered peak is {shown.filtered.max():.3f}.",
                        f"The mean absolute raw-to-filtered difference is {(shown[x]-shown.filtered).abs().mean():.3f}."]
            interpretation = (f"Median filtering changes the peak by "
                              f"{abs(peak[x] - shown.filtered.max()):.3f} m/s² in this recording.")
        else:
            trial = trial.assign(squared=trial[x].pow(2))
            figure = px.area(trial, x="time_s", y="squared", **OPTIONS)
            figure.update_yaxes(title_text=f"{x}² (m²/s⁴)")
            description = f"{peak.recording} · shows instantaneous squared acceleration."
            findings = [f"The maximum squared value is {trial.squared.max():.3f}.",
                        f"The window-independent RMS across this trial is {np.sqrt(trial.squared.mean()):.3f}.",
                        f"{(trial.squared > trial.squared.mean()).mean():.1%} of samples exceed the mean squared value."]
            interpretation = (f"The peak exceeds the recording’s mean squared acceleration of "
                              f"{trial.squared.mean():.3f} m²/s⁴ by "
                              f"{trial.squared.max()-trial.squared.mean():.3f} m²/s⁴. "
                              "Squaring emphasizes large movements in either direction.")
    elif kind == "Scatter plot":
        sample = data.sample(min(n, 1800), random_state=587)
        correlation = values.corr(data[y])
        figure = px.scatter(sample, x=x, y=y, color="activity", opacity=.55, **OPTIONS)
        description = "Shows paired sensor values using a reproducible sample of up to 1,800 rows."
        findings = [f"Pearson correlation across all selected rows is {correlation:.3f}.",
                    f"{(values.ge(q3) & data[y].ge(data[y].quantile(.75))).mean():.1%} of rows reach both upper quartiles.",
                    f"The plotted sample contains {len(sample):,} rows."]
        interpretation = (f"The absolute correlation is {abs(correlation):.3f} on a 0–1 scale. "
                          "The coefficient measures linear association across the selected samples.")
    elif kind == "Correlation heatmap":
        matrix = data[MEASURES].corr()
        figure = px.imshow(matrix, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                           text_auto=".2f", template="plotly_dark", labels={"color": "Pearson r"})
        pairs = matrix.where(np.triu(np.ones(matrix.shape), 1).astype(bool)).stack()
        if pairs.empty:
            raise ValueError("The selected signals have insufficient variation for correlation.")
        strongest = pairs.abs().idxmax()
        weakest = pairs.abs().idxmin()
        description = "Shows Pearson correlations between the Cartesian axes and derived magnitude."
        findings = [f"The strongest absolute pair is {' / '.join(strongest)} at r = {pairs[strongest]:.3f}.",
                    f"The weakest absolute pair is {' / '.join(weakest)} at r = {pairs[weakest]:.3f}.",
                    f"{(pairs.abs() >= .5).sum()} of {len(pairs)} pairs have |r| ≥ 0.5."]
        interpretation = (f"The strongest pair exceeds the weakest by {pairs.abs().max()-pairs.abs().min():.3f} "
                          "in absolute correlation. Magnitude is derived from the three axes, so it shares their inputs.")
    elif kind == "Pair plot":
        sample = data.sample(min(n, 1800), random_state=587)
        figure = px.scatter_matrix(sample, dimensions=MEASURES, color="activity", opacity=.45, **OPTIONS)
        figure.update_traces(diagonal_visible=False, showupperhalf=False)
        description = "Shows pairwise sensor relationships on a reproducible sample of up to 1,800 rows."
        findings = [f"Sample x/y correlation is {sample.x.corr(sample.y):.3f}.",
                    f"Sample x/z correlation is {sample.x.corr(sample.z):.3f}.",
                    f"Sample y/z correlation is {sample.y.corr(sample.z):.3f}."]
        axis_pairs = sample[["x", "y", "z"]].corr().abs()
        pair = max((("x", "y"), ("x", "z"), ("y", "z")), key=lambda names: axis_pairs.loc[names])
        interpretation = (f"{' / '.join(pair)} has the highest absolute correlation among the three "
                          f"Cartesian axis pairs in the plotted sample ({axis_pairs.loc[pair]:.3f}).")
    elif kind == "Density plot":
        curves = []
        peaks = {}
        for activity, batch in data.groupby("activity"):
            signal = batch[x]
            if signal.nunique() < 2:
                continue
            grid = np.linspace(signal.min(), signal.max(), 180)
            density = gaussian_kde(signal)(grid)
            peaks[activity] = grid[density.argmax()]
            curves.append(pd.DataFrame({x: grid, "Density": density, "activity": activity}))
        if not curves:
            raise ValueError("Density needs at least one activity with two distinct measurements.")
        smooth = pd.concat(curves, ignore_index=True)
        figure = px.line(smooth, x=x, y="Density", color="activity", **OPTIONS)
        description = "Estimates a Gaussian kernel density for each nonconstant activity distribution."
        findings = [f"{activity} has an estimated mode near {value:.3f}."
                    for activity, value in peaks.items()]
        findings += [f"The pooled median is {median:.3f}.", f"The pooled IQR is {q3-q1:.3f}.",
                     f"{len(curves)} activity distributions support KDE."]
        interpretation = (f"The estimated peaks span {min(peaks.values()):.3f} to {max(peaks.values()):.3f} m/s². "
                          "Each curve is smoothed independently, so its peak describes concentration within that activity.")
    else:
        figure = px.ecdf(data, x=x, color="activity", **OPTIONS)
        figure.update_yaxes(title_text="Cumulative proportion")
        description = "Shows the fraction of samples at or below each value, by activity."
        findings = [f"The pooled 25th percentile is {q1:.3f}.",
                    f"The pooled 50th percentile is {median:.3f}.",
                    f"The pooled 75th percentile is {q3:.3f}."]
        interpretation = (f"The middle 50% of selected measurements spans {q3-q1:.3f} m/s². "
                          "The individual curves show what fraction of each activity falls below a given value.")
    title = "ECDF plot" if kind == "ECDF" else kind
    figure.update_layout(title=f"{title} · Basketball sensor data",
                         height=485 if kind != "Pair plot" else 600,
                         margin=dict(l=44, r=20, t=64, b=76 if kind == "Pie chart" else 48),
                         legend_title_text="Activity" if kind == "Pie chart" else "")
    return style_figure(figure), description, findings, interpretation
