from pathlib import Path

from io import BytesIO
from zipfile import BadZipFile

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.aclose import mine_closed
from utils.apriori_analysis import label, mine, transactions
from utils.data_loader import RAW_COLUMNS, load_archive
from utils.decision_tree import evaluate
from utils.editorial import annotations, chrome, court, facts, micro, relationship, section
from utils.feature_engineering import build_windows, feature_names, transform_for_export
from utils.preprocessing import clean, outliers
from utils.statistics import describe
from utils.visualization import MEASURES, PLOTS, OPTIONS, activity_labels, chart_analysis, make_plot, style_figure


@st.cache_data(show_spinner="Reading UCI recordings and extracting features…")
def dataset():
    raw = load_archive()
    processed, audit, missing = clean(raw)
    windows = build_windows(processed)
    return raw, processed, audit, missing, windows


@st.cache_data(show_spinner="Training on recording-disjoint folds…")
def trained(windows, features, depth=5, leaf=4, selected=12):
    return evaluate(windows, features, depth, leaf, selected)


@st.cache_data(show_spinner="Mining activity-window transactions…", max_entries=16)
def mined(table, support, confidence, lift, closed=False):
    method = mine_closed if closed else mine
    return method(table, support, confidence, lift)


def get_data():
    try:
        return dataset()
    except (OSError, ValueError, RuntimeError, BadZipFile) as error:
        st.error(f"Could not load the UCI archive. Check the connection or data/raw/basketball.zip. {error}")
        st.stop()


def title(heading, detail):
    chrome(heading)
    st.title(heading)
    st.caption(detail)


def csv(frame, name, caption="Download CSV"):
    st.download_button(caption, frame.to_csv(index=False).encode("utf-8"), name + ".csv",
                       "text/csv", key=name, on_click="ignore")


def frame_table(frame):
    st.dataframe(frame, width="stretch", hide_index=True)


def chart(figure):
    st.plotly_chart(style_figure(figure), width="stretch", theme=None,
                    config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}})


def feature_chart(figure, description, findings):
    plot, notes = st.columns([3, 1.18], gap="large")
    with plot:
        chart(figure)
        st.caption(description)
    with notes:
        annotations(findings)


def labeled(data):
    return data.loc[data.activity.ne("mixed")]


def home():
    chrome("Home")
    intro, map_column = st.columns([1, 1.38], gap="large", vertical_alignment="center")
    with intro, st.container(key="lab_home_title"):
        micro("UCI 587 / ASBD RESEARCH WORKSTATION")
        st.title("COURTSIDE / LAB")
        st.html('<p class="lab-hero-copy"><strong>Basketball in motion.</strong> Explore how five activities '
                'register in wearable sensor data, from raw trials to patterns and predictions.</p>')
        st.page_link("pages/1_Dataset_Explorer.py", label="Open the research archive", icon=":material/arrow_forward:")
    with map_column:
        court()
    raw, data, _, _, windows = get_data()
    base = labeled(data)
    facts([("Raw samples", f"{len(raw):,}"), ("Labeled trials", base.recording.nunique()),
           ("Activity windows", f"{len(windows):,}")])
    section("01 / FEATURED ANALYSIS", "How the archive is distributed")
    feature_chart(*make_plot("Bar chart", base))
    section("02 / STUDY DESIGN", "From recording to research finding")
    st.write("The lab compares five labeled basketball activities. Use the court map to inspect trials, "
             "audit the one-second window pipeline, explore association rules, or evaluate the decision tree "
             "on held-out recordings.")


def explorer():
    title("Dataset Explorer", "Inspect the original files, recording metadata, and sample measurements.")
    raw, _, _, _, _ = get_data()
    catalog = raw.groupby(["recording", "activity", "source"], as_index=False).agg(
        samples=("time_s", "size"), duration_s=("time_s", "max"),
        participant=("participant", "first"), trial=("trial", "first"),
        recorded_at=("recorded_at", "first"), duplicate_of=("duplicate_of", "first"),
    )
    facts([("Archive recordings", len(catalog)), ("Raw rows", f"{len(raw):,}"),
           ("Labeled participants", raw.participant.nunique()), ("Sensor / time fields", len(RAW_COLUMNS))])
    section("01 / CATALOG", "Recording inventory")
    category = st.selectbox("Activity filter", ["All", *sorted(raw.activity.unique())])
    chosen = catalog if category == "All" else catalog.loc[catalog.activity.eq(category)]
    frame_table(chosen)
    section("02 / SOURCE VIEW", "Inspect a recording")
    file = st.selectbox("Recording", chosen.recording)
    preview = raw.loc[raw.recording.eq(file)]
    frame_table(preview.head(250))
    with st.expander("Field types and missing values"):
        frame_table(pd.DataFrame({"column": raw.columns, "dtype": raw.dtypes.astype(str).values,
                                  "missing": raw.isna().sum().values}))
    st.caption("Values here retain the original file units. X-prefixed acceleration is recorded in g per UCI. "
               "Mixed files have no sample-level activity labels. Duplicate payloads are marked in the catalog.")
    csv(preview, "recording")


def preprocessing():
    title("Data Preprocessing", "Audit timestamp cleaning, harmonized units, filtering, and feature extraction.")
    raw, data, audit, missing, windows = get_data()
    section("01 / PIPELINE", "From raw packets to activity windows")
    st.html('<ol class="lab-pipeline"><li>Audit packets</li><li>Harmonize units</li>'
            '<li>Smooth within trials</li><li>Extract window features</li></ol>')
    facts([(key.capitalize(), f"{audit[key]:,}") for key in
           ["input rows", "clean rows", "duplicate recording rows", "noise flags"]])
    col1, col2 = st.columns([1.6, 1])
    with col1:
        micro("Cleaning log")
        frame_table(pd.DataFrame(audit.items(), columns=["Step", "Count"]))
    with col2:
        micro("Missing values before / after")
        frame_table(pd.DataFrame({"Field": missing.index, "Before": missing.values,
                                  "After": data[missing.index].isna().sum().values}))
    base = labeled(data)
    with st.expander("Acceleration outlier flags"):
        st.caption("Pooled labeled trials · IQR fences at 1.5 IQR and absolute Z-score > 3. Flags preserve motion peaks.")
        frame_table(outliers(base).rename_axis("Sensor").reset_index())
    section("02 / SIGNAL BENCH", "Compare raw and filtered signals")
    record = st.selectbox("Filter comparison recording", sorted(base.recording.unique()))
    feature_chart(*make_plot("Line chart", base.loc[base.recording.eq(record)]))
    st.write("A centered five-sample median filter works within each recording. "
             "A robust residual threshold flags potential noise. One-second windows require at least "
             "75 samples and 0.75 seconds of coverage. Features summarize three axes and vector magnitude.")
    section("03 / OUTPUT", "Window features and cleaned samples")
    transform = st.radio("Download scaling", ["Original features", "Standardized", "Min-max"], horizontal=True)
    output = transform_for_export(windows, transform)
    st.caption("Export scalers describe the entire export. The classifier uses unscaled windows and fits feature selection on training only.")
    frame_table(output.head(150))
    csv(output, "activity_windows", "Download window features")
    csv(data, "cleaned_samples", "Download cleaned sensor rows")


def descriptive():
    title("Descriptive Analytics", "Summarize distributions and relationships in labeled trials.")
    raw, data, _, missing, windows = get_data()
    base = labeled(data)
    activity = st.multiselect("Activities", sorted(base.activity.unique()), default=sorted(base.activity.unique()))
    base = base.loc[base.activity.isin(activity)]
    if base.empty:
        st.info("Select at least one activity to view summaries.")
        return
    summary = describe(base[MEASURES])
    counts = base.groupby("activity").agg(samples=("time_s", "size"), trials=("recording", "nunique"),
                                          median_magnitude=("magnitude", "median")).reset_index()
    counts["sample %"] = counts.samples / len(base) * 100
    facts([("Samples", f"{len(base):,}"), ("Median magnitude", f"{summary.loc['magnitude', 'median']:.2f} m/s²"),
           ("Magnitude IQR", f"{summary.loc['magnitude', 'IQR']:.2f} m/s²")])
    section("01 / FIELD NOTES", "The shape of acceleration")
    figure, description, findings = make_plot("Histogram", base)
    leader = counts.loc[counts.samples.idxmax()]
    feature_chart(figure, description, [
        *findings, f"{leader.activity.capitalize()} contributes {leader['sample %']:.1f}% of selected samples.",
        f"Magnitude skewness is {summary.loc['magnitude', 'skewness']:.2f}.",
    ])
    section("02 / STATISTICAL RECORD", "Summaries and checks")
    frame_table(summary.rename_axis("Sensor").reset_index())
    frame_table(counts)
    frame_table(base[MEASURES].corr().rename_axis("Sensor").reset_index())
    frame_table((missing / len(raw) * 100).rename("Raw missing %").rename_axis("Field").reset_index())
    st.caption("Trial lengths vary, so sample shares also reflect recording duration.")
    csv(summary.rename_axis("Sensor").reset_index(), "descriptive_statistics")
    workbook = BytesIO()
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index_label="Sensor")
        counts.to_excel(writer, sheet_name="Activity counts", index=False)
        base[MEASURES].corr().to_excel(writer, sheet_name="Correlation", index_label="Sensor")
    st.download_button("Download statistics workbook", workbook.getvalue(), "descriptive_statistics.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", on_click="ignore")


def visualizations():
    title("Data Visualizations", "Explore 12 chart types and download the filtered view.")
    _, data, _, _, _ = get_data()
    base = data.loc[data.activity.ne("mixed").fillna(True)].copy()
    base["activity"] = activity_labels(base)
    with st.container(key="lab_gallery_index"):
        kind = st.radio("Chart index", PLOTS, horizontal=True,
                        format_func=lambda name: f"{PLOTS.index(name)+1:02d} / "
                                                 f"{'ECDF plot' if name == 'ECDF' else name}")
    activity_column, measure_column = st.columns([1.65, 1])
    options = sorted(base.activity.unique())
    activities = activity_column.multiselect("Activities", options, default=options)
    base = base.loc[base.activity.isin(activities)]
    if base.empty:
        st.info("Select at least one activity to display the chart.")
        return
    x = "magnitude"
    y = "z"
    trial = None
    if kind in {"Histogram", "Box plot", "Violin plot", "Line chart", "Scatter plot", "Density plot", "Area chart", "ECDF"}:
        x = measure_column.selectbox("Measure", list(reversed(MEASURES)),
                                     format_func=lambda measure: f"{measure} (m/s²)")
    if kind == "Scatter plot":
        y = measure_column.selectbox("Vertical measure", [name for name in MEASURES if name != x],
                                     format_func=lambda measure: f"{measure} (m/s²)")
    if kind in {"Line chart", "Area chart"}:
        trial = measure_column.selectbox("Recording", sorted(base.recording.unique()))
    section(f"{PLOTS.index(kind)+1:02d} / {len(PLOTS):02d}", "ECDF plot" if kind == "ECDF" else kind)
    try:
        figure, purpose, findings, interpretation = chart_analysis(kind, base, x, y, trial)
    except ValueError as error:
        st.info(str(error))
        return
    st.markdown(f"**Purpose.** {purpose}")
    chart(figure)
    observations, reading = st.columns([2.3, 1], gap="large")
    with observations:
        annotations(findings, "Key observations")
    with reading:
        micro("Interpretation")
        st.write(interpretation)
    st.download_button("Download interactive chart (HTML)", figure.to_html(full_html=True, include_plotlyjs=True),
                       f"{kind.lower().replace(' ', '_')}.html", "text/html", on_click="ignore")


def itemsets_view(frame):
    return frame.assign(itemsets=frame.itemsets.map(label), size=frame.itemsets.map(len))


def rule_view(frame, key):
    if frame.empty:
        st.info("No rules meet these thresholds. Try a lower minimum support or lift.")
        return
    choices = range(len(frame))
    selection = st.selectbox("Inspect an association", choices, key=f"{key}_selection",
                             format_func=lambda index: frame.iloc[index].rule)
    rule = frame.iloc[selection]
    relationship("Antecedent", rule.antecedents, "Consequent", rule.consequents,
                 f"Support {rule.support:.1%}  ·  Confidence {rule.confidence:.1%}  ·  Lift {rule.lift:.2f}")
    figure = px.scatter(frame, x="support", y="confidence", size="lift", color="lift",
                        hover_name="rule", title="Rule support, confidence, and lift",
                        color_continuous_scale=["#745747", "#F07837", "#FFDDBF"], **OPTIONS)
    chart(figure)
    frame_table(frame)
    csv(frame, key, "Download rules")


def mining(closed=False):
    heading = "A-CLOSE" if closed else "Apriori"
    title(heading, "Mine discretized acceleration features and activity labels at the one-second window level.")
    _, _, _, _, windows = get_data()
    table, boundaries = transactions(windows)
    section("01 / THRESHOLDS", "Set the mining conditions")
    a, b, c = st.columns(3)
    support = a.slider("Minimum support", .05, .95, .15, .01)
    confidence = b.slider("Minimum confidence", .10, 1., .55, .05)
    lift = c.slider("Minimum lift", .50, 5., 1., .05)
    with st.expander("Transaction encoding"):
        st.write("Each window contributes one activity item and one value bin per selected feature. "
                 "Bins use pooled empirical tertiles. Associations summarize co-occurrence.")
        frame_table(boundaries)
    frequent, rules, elapsed = mined(table, support, confidence, lift)
    if not closed:
        section("02 / ASSOCIATIONS", "Explore the rules")
        facts([("Association rules", len(rules)), ("Frequent itemsets", len(frequent)),
               ("Transactions", len(table)), ("Runtime", f"{elapsed:.3f} s")])
        rule_tab, item_tab = st.tabs(["Association rules", "Frequent itemsets"])
        with rule_tab:
            rule_view(rules, "apriori_rules")
        with item_tab:
            show = itemsets_view(frequent)
            frame_table(show)
            csv(show, "apriori_itemsets", "Download itemsets")
        return
    closed_sets, mapping, approximate, exact, duration = mined(table, support, confidence, lift, True)
    nonempty = closed_sets.loc[closed_sets.itemsets.map(bool)]
    st.write("A minimal generator has no proper subset with equal support. Its closure contains every "
             "item shared by all matching transactions. A closed itemset has no proper superset with equal support.")
    section("02 / CLOSURE INDEX", "Inspect generators and their closures")
    facts([("Closed itemsets", len(nonempty)), ("Minimal generators", len(mapping)),
           ("Closed rules", len(approximate)), ("Exact implications", len(exact)), ("Runtime", f"{duration:.3f} s")])
    mapping_tab, item_tab, approx_tab, exact_tab = st.tabs(
        ["Minimal generators", "Closed itemsets", "Approximate closed rules", "Exact implications"])
    with item_tab:
        shown = itemsets_view(closed_sets)
        frame_table(shown)
        csv(shown, "closed_itemsets", "Download closed itemsets")
    with mapping_tab:
        shown = mapping.assign(generator=mapping.generator.map(label), closure=mapping.closure.map(label))
        shown = shown.sort_values("support", ascending=False)
        if not shown.empty:
            choices = [i for i in range(len(shown)) if shown.iloc[i].generator != "{}"] or list(range(len(shown)))
            selected_mapping = st.selectbox("Inspect a minimal generator", choices,
                                             format_func=lambda index: shown.iloc[index].generator)
            pair = shown.iloc[selected_mapping]
            relationship("Minimal generator", pair.generator, "Closure", pair.closure,
                         f"Shared support {pair.support:.1%}")
        frame_table(shown)
        csv(shown, "minimal_generators", "Download generator mapping")
    with approx_tab:
        st.caption("A closed antecedent implies the remaining items of a larger closed union at confidence below 1.")
        rule_view(approximate, "closed_rules")
    with exact_tab:
        st.caption("A minimal generator implies the remaining items in its closure with confidence 1.")
        rule_view(exact, "exact_implications")
    section("03 / COMPARISON", "Apriori and A-CLOSE")
    frame_table(pd.DataFrame([
        ["Apriori", len(frequent), len(rules), elapsed],
        ["A-CLOSE", len(nonempty), len(approximate)+len(exact), duration],
    ], columns=["Method", "Reported nonempty itemsets", "Rules", "Seconds"]))


def tree_page():
    from matplotlib import pyplot as plt
    from sklearn.tree import plot_tree

    title("Decision Tree", "Predict five activities using held-out recordings.")
    _, _, _, _, windows = get_data()
    candidates = feature_names(windows)
    st.write("The first seeded four-fold group split with all five labels is used for evaluation. "
             "Trials stay intact across the split. Some participants contribute to both partitions.")
    section("01 / MODEL CONTROLS", "Configure the classifier")
    a, b, c = st.columns(3)
    depth = a.slider("Maximum depth", 2, 8, 5)
    leaf = b.slider("Minimum leaf samples", 1, 20, 4)
    selected = c.slider("Features retained", 1, len(candidates), 12)
    features = st.multiselect("Candidate features", candidates, default=candidates)
    if not features:
        st.info("Select at least one candidate feature.")
        return
    result = trained(windows, features, depth, leaf, selected)
    macro = result["report"]["macro avg"]
    section("02 / HELD-OUT EVALUATION", "Read the model results")
    facts([("Accuracy", f"{result['accuracy']:.1%}"), ("Macro F1", f"{macro['f1-score']:.1%}"),
           ("Macro precision", f"{macro['precision']:.1%}"), ("Macro recall", f"{macro['recall']:.1%}")])
    evaluation, importance_tab, diagram, rules = st.tabs(
        ["Evaluation", "Feature importance", "Tree diagram", "IF–THEN rules"])
    with evaluation:
        figure = px.imshow(result["matrix"], x=result["classes"], y=result["classes"],
                           labels={"x": "Predicted", "y": "Actual", "color": "Windows"},
                           color_continuous_scale=["#242225", "#9D4829", "#FFDDBF"],
                           text_auto=True, template="plotly_dark", title="Held-out confusion matrix")
        feature_chart(figure, "Each row represents the recorded activity and each column represents the prediction.", [
            f"{result['train_recordings']} training trials and {result['test_recordings']} held-out trials.",
            f"The test set contains {len(result['test'])} activity windows.",
            f"The majority baseline ({result['majority_class']}) scores {result['baseline_accuracy']:.1%}.",
        ])
        report = pd.DataFrame({key: val for key, val in result["report"].items() if isinstance(val, dict)}).T
        frame_table(report.rename_axis("Activity").reset_index())
        examples = windows.iloc[result["test"]][["recording", "window", "activity"]].assign(prediction=result["predictions"])
        frame_table(examples)
        csv(examples, "heldout_predictions", "Download predictions")
        row = st.selectbox("Inspect a held-out prediction", examples.index,
                           format_func=lambda index: f"{examples.loc[index, 'recording']} · window {examples.loc[index, 'window']}")
        st.write(f"Predicted **{examples.loc[row, 'prediction']}** · trial label **{examples.loc[row, 'activity']}**")
    with importance_tab:
        importance = pd.DataFrame({"Feature": result["names"], "Importance": result["model"][-1].feature_importances_})
        chart(px.bar(importance.sort_values("Importance"), x="Importance", y="Feature",
                     orientation="h", title="Impurity-based feature importance", **OPTIONS))
    with diagram:
        st.caption("Each split shows a learned threshold. Leaves show the resulting activity prediction.")
        fig, ax = plt.subplots(figsize=(19, 8))
        plot_tree(result["model"][-1], feature_names=result["names"],
                  class_names=result["classes"], filled=True, rounded=True, ax=ax, fontsize=7)
        st.pyplot(fig)
        plt.close(fig)
    with rules:
        micro("Root-to-leaf conditions")
        st.code("\n".join(result["rules"]), language=None)
        st.download_button("Download classification rules", "\n".join(result["rules"]),
                           "classification_rules.txt", on_click="ignore")


def report_page():
    from utils.reporting import create_report

    title("Project Report", "Export the complete laboratory report with reproducible default settings.")
    main, index = st.columns([2.6, 1], gap="large")
    with main:
        section("01 / OVERVIEW", "Basketball sensor analytics", anchor="report-overview")
        st.write("The report follows the UCI recordings from preparation and descriptive analysis through "
                 "association mining, held-out classification, and limitations.")
        section("02 / CONTENTS", "What the report covers", anchor="report-contents")
        annotations([
            "Data provenance, preprocessing methodology, and descriptive statistics.",
            "Twelve visualizations with findings calculated from the recordings.",
            "Apriori and A-CLOSE rules, decision tree evaluation, and methods comparison.",
        ], "Report outline")
    with index:
        st.html('<nav class="lab-report-index" aria-label="Report page contents">'
                '<a href="#report-overview" target="_self">01 / Overview</a>'
                '<a href="#report-contents" target="_self">02 / Contents</a>'
                '<a href="#report-downloads" target="_self">03 / Downloads</a></nav>')
    section("03 / DOWNLOADS", "Generate the report", anchor="report-downloads")
    show = st.radio("Report format", ["HTML", "PDF"], horizontal=True)
    if st.button("Generate report", type="primary"):
        raw, clean_data, audit, missing, windows = get_data()
        table, _ = transactions(windows)
        model = trained(windows, feature_names(windows))
        frequent, rules, elapsed = mined(table, .15, .55, 1.)
        closed = mined(table, .15, .55, 1., True)
        with st.spinner(f"Building the {show} report…"):
            payload = create_report(show, raw, labeled(clean_data), windows, audit, missing,
                                    model, (frequent, rules, elapsed), closed)
        st.session_state["exported_report"] = show, payload
    exported = st.session_state.get("exported_report")
    if exported and exported[0] == show:
        extension = show.lower()
        st.download_button(f"Download {show} report", exported[1], f"basketball_report.{extension}",
                           "application/pdf" if show == "PDF" else "text/html", on_click="ignore")
