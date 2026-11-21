from html import escape
from io import BytesIO

import numpy as np
import pandas as pd

from utils.apriori_analysis import label
from utils.statistics import describe
from utils.visualization import MEASURES, PLOTS, make_plot


def report_sections(raw, data, windows, audit, missing, model, apriori, aclose):
    frequent, rules, runtime = apriori
    closed, mapping, approximate, exact, closed_runtime = aclose
    counts = data.activity.value_counts()
    summary = describe(data[MEASURES])
    macro = model["report"]["macro avg"]
    closed_count = int(closed.itemsets.map(bool).sum())
    confusion = pd.DataFrame(model["matrix"], index=model["classes"], columns=model["classes"])
    confusion.index.name = "Actual / predicted"
    classification = pd.DataFrame({key: value for key, value in model["report"].items()
                                   if isinstance(value, dict)}).T
    importance = pd.DataFrame({"Feature": model["names"],
                               "Importance": model["model"][-1].feature_importances_})
    common = frequent.sort_values("support", ascending=False).head(5)
    common_closed = closed.loc[closed.itemsets.map(bool)].head(5)
    sections = [
        ("Abstract", [
            f"This laboratory study analyzes {len(raw):,} sensor samples and {len(windows):,} labeled windows. "
            f"The default tree scored {model['accuracy']:.1%} held-out accuracy and "
            f"{macro['f1-score']:.1%} macro F1. Apriori recovered {len(frequent)} frequent itemsets "
            f"and A-CLOSE recovered {closed_count} nonempty closed itemsets.",
        ]),
        ("Introduction", ["Wearable sensor signals capture changes in arm motion during basketball drills. "
                          "Trial-level metadata makes supervised activity recognition feasible."]),
        ("Problem statement", ["Prepare the original recordings for descriptive analytics, discover "
                               "window-level item associations, and classify five basketball activities."]),
        ("Objectives", ["Audit data quality, visualize distributions, compare complete and closed "
                        "frequent itemsets, and measure performance on held-out recordings."]),
        ("Dataset description", [
            f"UCI dataset 587 contains {raw.recording.nunique()} recordings, "
            f"including {data.recording.nunique()} labeled trials and "
            f"{raw.loc[raw.activity.eq('mixed')].recording.nunique()} unlabeled mixed recordings. "
            "Each supplies elapsed time, three Cartesian accelerations, magnitude R, and two angles. "
            "X participant values use g and are converted to m/s². The mixed files retain unverified raw units "
            "and share identical sensor payloads. One copy is retained in the cleaned sample dataset.",
            counts.rename("Samples").rename_axis("Activity").reset_index(),
        ]),
        ("Technologies used", ["Python, Streamlit, pandas, NumPy, SciPy, scikit-learn, MLxtend, "
                               "Plotly, Matplotlib, seaborn, ReportLab, and pytest."]),
        ("System architecture", ["UCI ZIP → parsed samples → per-trial cleaning and smoothing → "
                                 "feature windows → exploratory, mining, and classification modules → report export."]),
        ("Data preprocessing methodology", [
            "Required timestamps and sensor channels are coerced to numeric. Invalid required rows, "
            "exact duplicate recordings and packets, and conflicting timestamps are removed. IQR and Z-score checks flag "
            "extremes for review. A centered five-sample median reduces impulsive changes within each recording. "
            "Potential noise flags use a six-scaled-MAD threshold on the residual norm.",
            "Non-overlapping one-second windows require at least 75 samples and 0.75 seconds coverage. "
            "Mean, population standard deviation, min, max, RMS, and sum-of-squares energy are computed "
            "for each filtered axis and filtered magnitude. Scaling and activity one-hot encoding are available for export.",
            pd.DataFrame(audit.items(), columns=["Step", "Count"]),
            (missing / len(raw) * 100).rename("Missing %").rename_axis("Raw field").reset_index(),
        ]),
        ("Descriptive statistics", [
            summary[["count", "mean", "median", "mode", "std", "variance"]].reset_index(names="Sensor"),
            summary[["min", "25%", "50%", "75%", "max", "range", "IQR"]].reset_index(names="Sensor"),
            summary[["skewness", "kurtosis", "missing %"]].reset_index(names="Sensor"),
            data[MEASURES].corr().reset_index(names="Sensor"),
        ]),
        ("Visualizations and inferences", []),
        ("Apriori methodology and results", [
            "Each transaction contains an activity and one empirical tertile bin for each of five features. "
            "MLxtend Apriori enumerates frequent itemsets from subset support and evaluates all "
            "eligible rule partitions. Defaults use support ≥ 0.15, confidence ≥ 0.55 and lift ≥ 1.",
            f"The analysis returned {len(frequent)} itemsets and {len(rules)} rules in {runtime:.3f} s.",
            common.assign(itemsets=common.itemsets.map(label)),
            *rule_excerpt(rules),
        ]),
        ("Decision tree methodology and results", [
            f"The seeded stratified group split held out {model['test_recordings']} recordings "
            f"({len(model['test'])} windows) and trained on {model['train_recordings']} recordings. "
            "The training pipeline chooses up to 12 features using ANOVA F scores. "
            "The balanced tree has maximum depth 5 and requires four windows per leaf.",
            f"The trained decision tree yields {len(model['rules'])} leaf rules. Examples follow.",
            *model["rules"][:5],
        ]),
        ("A-CLOSE methodology and results", [
            "Levelwise search retains frequent minimal generators after rejecting any candidate "
            "with a same-support immediate subset. Closure intersects the matching transactions. "
            "Each closed item's one-item supersets are checked for strictly smaller support.",
            f"The mining pass found {closed_count} nonempty closed itemsets and {len(mapping)} generators "
            f"in {closed_runtime:.3f} s, with {len(approximate)} approximate closed rules "
            f"and {len(exact)} exact implications at the configured thresholds.",
            "Approximate rules relate nested closed sets. Exact implications connect a minimal generator "
            "to the other items in its closure and have confidence one.",
            common_closed.assign(itemsets=common_closed.itemsets.map(label)),
            *rule_excerpt(approximate),
            *rule_excerpt(exact),
        ]),
        ("Algorithm comparison", [
            pd.DataFrame([
                ["Apriori", len(frequent), len(rules), runtime],
                ["A-CLOSE", closed_count, len(approximate) + len(exact), closed_runtime],
            ], columns=["Algorithm", "Nonempty itemsets", "Rules", "Runtime (s)"]),
            "Apriori counts all frequent itemsets. A-CLOSE stores closure classes and represents "
            "different rule families. Timings depend on the machine and run.",
        ]),
        ("Performance evaluation", [
            f"Accuracy {model['accuracy']:.1%}. Macro precision {macro['precision']:.1%}. "
            f"Macro recall {macro['recall']:.1%}. Macro F1 {macro['f1-score']:.1%}.",
            f"Predicting the training majority class ({model['majority_class']}) for every held-out window "
            f"yields {model['baseline_accuracy']:.1%} accuracy.",
            classification.reset_index(names="Group"), confusion.reset_index(),
            importance.sort_values("Importance", ascending=False).head(8),
        ]),
        ("Key findings", [
            f"{counts.idxmax().capitalize()} contributes the most labeled samples ({counts.max():,}). "
            f"Pooled acceleration magnitude has a median of {summary.loc['magnitude', 'median']:.3f} m/s² "
            f"and IQR of {summary.loc['magnitude', 'IQR']:.3f} m/s².",
            "The association tables quantify co-occurrence within trial windows. The classification "
            "report shows how performance varies by activity.",
        ]),
        ("Limitations", [
            "Only four participant identifiers are represented in the labeled archive. "
            "Recording-disjoint evaluation allows participant overlap and does not measure performance on unseen users. "
            "Trial labels may include preparation before the motion begins. Adjacent windows can be correlated. "
            "The archived angles are not validated angular velocity channels. "
            "Whole-dataset rule statistics are descriptive and depend on chosen bins and thresholds.",
        ]),
        ("Conclusion", [
            "The preprocessing pipeline preserves recording provenance while enabling sample-level exploration "
            "and window-level mining. Held-out results provide a baseline for recording-level generalization.",
        ]),
        ("References", [
            "Basketball dataset. UCI Machine Learning Repository (2019). https://doi.org/10.24432/C56G77. CC BY 4.0.",
            "Agrawal and Srikant. Fast Algorithms for Mining Association Rules in Large Databases. "
            "VLDB 1994, pp. 487-499.",
            "Pasquier, Bastide, Taouil and Lakhal. Discovering Frequent Closed Itemsets for Association Rules. "
            "ICDT 1999, pp. 398-416.",
        ]),
    ]
    return sections


def rule_excerpt(rules):
    if rules.empty:
        return ["No rules in this family meet the defaults."]
    return [f"{row.rule} (support {row.support:.3f}, confidence {row.confidence:.3f}, lift {row.lift:.3f})"
            for row in rules.head(3).itertuples()]


def create_report(format_name, raw, data, windows, audit, missing, model, apriori, aclose):
    sections = report_sections(raw, data, windows, audit, missing, model, apriori, aclose)
    charts = [(kind, *make_plot(kind, data)) for kind in PLOTS]
    if format_name == "PDF":
        return build_pdf(sections, charts, data)
    chunks = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
        '<title>COURTSIDE / LAB — Basketball Analytics and Data Mining</title><style>',
        'body{max-width:1100px;margin:3rem auto;padding:0 1.25rem;background:#101114;color:#f5f0e8;font:16px/1.65 Inter,Segoe UI,sans-serif;}',
        'h1,h2{font-family:Space Grotesk,Segoe UI,sans-serif;letter-spacing:-.04em}',
        'h1{font-size:clamp(2rem,5vw,4rem);line-height:1.1} h2{border-top:1px solid #333335;padding-top:1rem}',
        'table{border-collapse:collapse;display:block;overflow:auto} td,th{border:1px solid #333335;padding:.4rem .7rem}',
        'figure{margin:2rem 0} header{border-bottom:2px solid #f07837} a{color:#f07837}',
        'a:focus-visible{outline:2px solid #f07837;outline-offset:3px} nav ol{columns:2;padding-left:1.6rem}',
        'nav li{break-inside:avoid} @media(max-width:600px){nav ol{columns:1}}</style>',
        '<header><p>COURTSIDE / LAB</p><h1>Basketball Analytics &amp; Data Mining</h1>',
        '<p>Laboratory report · UCI Basketball Dataset</p></header>',
    ]
    chunks.append('<nav aria-label="Report contents"><h2>Contents</h2><ol>')
    chunks.extend(f'<li><a href="#report-{index}">{escape(heading)}</a></li>'
                  for index, (heading, _) in enumerate(sections, 1))
    chunks.append('</ol></nav>')
    for index, (heading, contents) in enumerate(sections, 1):
        chunks.append(f'<section><h2 id="report-{index}">{index}. {escape(heading)}</h2>')
        for item in contents:
            chunks.append(item.to_html(index=False, float_format=lambda x: f"{x:.4f}")
                          if isinstance(item, pd.DataFrame) else f"<p>{escape(item)}</p>")
        if not contents:
            for index, (kind, figure, description, findings) in enumerate(charts):
                chunks.append("<figure>" + figure.to_html(full_html=False, include_plotlyjs=index == 0))
                chunks.append(f"<figcaption>Figure {index + 1}. {escape(kind)}. {escape(description)}</figcaption><ul>")
                chunks.extend(f"<li>{escape(finding)}</li>" for finding in findings)
                chunks.append("</ul></figure>")
        chunks.append("</section>")
    return "".join([*chunks, "</html>"]).encode("utf-8")


def render_static(kind, data):
    from matplotlib.figure import Figure
    import seaborn as sns

    size = (7.4, 3.75) if kind == "Pair plot" else (7.4, 3.1)
    fig = Figure(figsize=size, layout="constrained")
    magnitude = "magnitude"
    grouped = list(data.groupby("activity"))
    sample = data.sample(min(len(data), 1800), random_state=587)
    trial = data.loc[data.recording.eq(data.recording.iloc[0])]
    colors = ["#13668A", "#C34D17", "#7549A8", "#147B56", "#A13D70"]
    if kind == "Pair plot":
        axes = fig.subplots(4, 4)
        for row, y in enumerate(MEASURES):
            for col, x in enumerate(MEASURES):
                ax = axes[row, col]
                if row <= col:
                    ax.set_axis_off()
                    continue
                ax.scatter(sample[x], sample[y], s=1, alpha=.25, c=colors[0])
                ax.set(xlabel=x, ylabel=y)
        fig.suptitle("Pair plot (m/s²)", color="#142235")
    else:
        ax = fig.subplots()
        ax.set_title(kind, loc="left", color="#142235", fontweight="bold")
        counts = data.activity.value_counts()
        x_label = "Magnitude (m/s²)"
        if kind == "Histogram":
            ax.hist(data[magnitude], bins=30, color=colors[0])
            ax.set(xlabel=x_label, ylabel="Samples")
        elif kind in {"Box plot", "Violin plot"}:
            groups = [batch[magnitude] for _, batch in grouped]
            ticks = range(1, len(groups) + 1)
            if kind == "Box plot":
                ax.boxplot(groups, flierprops={"markersize": 1})
            else:
                ax.violinplot(groups, showmedians=True)
            ax.set_xticks(list(ticks), [name for name, _ in grouped])
            ax.set(xlabel="Activity", ylabel=x_label)
        elif kind == "Bar chart":
            ax.bar(counts.index, counts.values, color=colors)
            ax.set(xlabel="Activity", ylabel="Samples")
        elif kind == "Pie chart":
            ax.pie(counts.values, labels=counts.index, colors=colors, autopct="%1.1f%%",
                   textprops={"fontsize": 8}, wedgeprops={"edgecolor": "white"})
        elif kind == "Correlation heatmap":
            sns.heatmap(data[MEASURES].corr(), annot=True, fmt=".2f", vmin=-1, vmax=1,
                        cmap="RdBu_r", ax=ax, cbar_kws={"label": "Pearson r"})
        elif kind == "Line chart":
            ax.plot(trial.time_s, trial[magnitude], label="Raw", alpha=.65)
            ax.plot(trial.time_s, trial.magnitude_filtered, label="Filtered")
            ax.set(xlabel="Elapsed time (s)", ylabel=x_label)
            ax.legend(fontsize=8)
        elif kind == "Area chart":
            ax.fill_between(trial.time_s, trial[magnitude].pow(2), color=colors[0], alpha=.7)
            ax.set(xlabel="Elapsed time (s)", ylabel="Magnitude² (m²/s⁴)")
        elif kind == "Scatter plot":
            for (name, batch), color in zip(sample.groupby("activity"), colors):
                ax.scatter(batch[magnitude], batch.z, label=name, c=color, s=5, alpha=.45)
            ax.set(xlabel=x_label, ylabel="z (m/s²)")
            ax.legend(fontsize=7, ncols=3)
        elif kind == "Density plot":
            sns.kdeplot(data=data, x=magnitude, hue="activity", palette=colors, cut=0,
                        common_norm=False, ax=ax)
            ax.set(xlabel=x_label, ylabel="Estimated density")
        elif kind == "ECDF":
            for (name, batch), color in zip(grouped, colors):
                values = np.sort(batch[magnitude])
                ax.step(values, np.arange(1, len(values)+1) / len(values), label=name, color=color)
            ax.set(xlabel=x_label, ylabel="Cumulative proportion")
            ax.legend(fontsize=7, ncols=3)
    image = BytesIO()
    fig.savefig(image, dpi=140, format="png")
    image.seek(0)
    return image, size[1] / size[0]


def build_pdf(sections, charts, data):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    output = BytesIO()
    doc = SimpleDocTemplate(output, pagesize=A4, leftMargin=45, rightMargin=45,
                            topMargin=44, bottomMargin=39, title="Basketball Analytics & Data Mining")
    styles = getSampleStyleSheet()
    styles["Title"].textColor = colors.HexColor("#142235")
    styles["Heading2"].textColor = colors.HexColor("#13668A")
    styles["BodyText"].leading = 13
    styles["Heading2"].keepWithNext = True
    cell_style = ParagraphStyle("Cell", fontName="Helvetica", fontSize=8, leading=10)

    def paragraph(content, style="BodyText"):
        text = str(content).replace("→", " to ").replace("≥", ">=").replace("–", "-")
        return Paragraph(escape(text), styles[style])

    story = [
        paragraph("ASBD LABORATORY REPORT", "Heading2"),
        paragraph("Basketball Analytics & Data Mining", "Title"),
        paragraph("UCI Basketball Dataset · Descriptive analytics, classification, and association mining"),
        Spacer(1, 12),
    ]
    for number, (heading, contents) in enumerate(sections, 1):
        section_heading = paragraph(f"{number}. {heading}", "Heading2")
        story.append(section_heading)
        if heading == "References":
            story[-1] = KeepTogether([section_heading, *[paragraph(item) for item in contents]])
            continue
        for item in contents:
            if isinstance(item, pd.DataFrame):
                rows = [list(item.columns), *item.itertuples(index=False, name=None)]
                cells = [[Paragraph(escape(f"{v:.4g}" if isinstance(v, float) else str(v)), cell_style)
                          for v in row] for row in rows]
                table = Table(cells, colWidths=[doc.width / len(item.columns)] * len(item.columns), repeatRows=1)
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#BEE3F1")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]))
                story.extend([table, Spacer(1, 7)])
            else:
                story.extend([paragraph(item), Spacer(1, 4)])
        if not contents:
            for number, (kind, _, description, findings) in enumerate(charts, 1):
                file, ratio = render_static(kind, data)
                story.append(KeepTogether([
                    Image(file, width=doc.width, height=doc.width * ratio),
                    paragraph(f"Figure {number}. {kind}. {description}"),
                    *[paragraph(f"• {finding}") for finding in findings],
                    Spacer(1, 9),
                ]))

    def footer(canvas, doc):
        canvas.setFillColor(colors.HexColor("#496379"))
        canvas.setFont("Helvetica", 8)
        canvas.drawString(doc.leftMargin, 25, "ASBD   /   BASKETBALL SENSOR ANALYTICS")
        canvas.drawRightString(A4[0] - doc.rightMargin, 25, f"{doc.page}")

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output.getvalue()
