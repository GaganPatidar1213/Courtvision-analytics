from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from utils.aclose import mine_closed
from utils.apriori_analysis import mine, transactions
from utils.decision_tree import evaluate
from utils.feature_engineering import feature_names
from utils.reporting import create_report
from utils.visualization import PLOTS, make_plot

ROOT = Path(__file__).resolve().parents[1]
PAGES = [ROOT / "app.py", *sorted((ROOT / "pages").glob("*.py"))]


@pytest.mark.parametrize("page", PAGES, ids=lambda path: path.stem)
def test_pages_load(page):
    # Exercise the real multipage entry point so Streamlit resolves sidebar links.
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    if page.name != "app.py":
        app.switch_page(f"pages/{page.name}").run()
    assert not list(app.exception)
    assert not list(app.error)
    assert len(app.title) == 1
    sidebar_links = app.sidebar.get("page_link")
    assert len(sidebar_links) == 9
    assert sum(link.proto.help == "Current page" for link in sidebar_links) == 1
    if page.name.startswith(("1_", "2_", "3_", "4_", "5_", "6_", "7_")):
        assert app.get("download_button")
    if page.name.startswith("8_"):
        app.button[0].click().run(timeout=90)
        assert not list(app.exception)
        assert app.get("download_button")


def test_court_stations_match_streamlit_routes():
    from streamlit.source_util import page_icon_and_name

    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    court = next(element.proto.body for element in app.get("html") if "lab-court-map" in element.proto.body
                 and "<svg" in element.proto.body)
    assert court.count('class="lab-marker"') == len(PAGES) - 1
    for page in PAGES[1:]:
        _, route = page_icon_and_name(page)
        assert f'href="{route}" target="_self"' in court


def test_charts_and_reports(basketball, tmp_path):
    raw, processed, audit, missing, windows = basketball
    data = processed.loc[processed.activity.ne("mixed")]
    assert len(set(PLOTS)) == 12
    for kind in PLOTS:
        figure, description, findings = make_plot(kind, data)
        assert figure.data and description and len(findings) >= 3
        assert "Plotly.newPlot" in figure.to_html(include_plotlyjs=False)
    table, _ = transactions(windows)
    args = (raw, data, windows, audit, missing, evaluate(windows, feature_names(windows)),
            mine(table, .15, .55, 1.), mine_closed(table, .15, .55, 1.))
    html = create_report("HTML", *args)
    assert html.count(b"<section>") == 19 and html.count(b"<figure>") == 12
    pdf = create_report("PDF", *args)
    assert pdf.startswith(b"%PDF-") and pdf.strip().endswith(b"%%EOF")


def test_gallery_index_and_every_chart_download(basketball):
    import json

    expected_traces = {
        "Histogram": {"histogram"},
        "Box plot": {"box"},
        "Violin plot": {"violin"},
        "Bar chart": {"bar"},
        "Pie chart": {"pie"},
        "Line chart": {"scattergl", "scatter"},
        "Scatter plot": {"scattergl", "scatter"},
        "Correlation heatmap": {"heatmap"},
        "Pair plot": {"splom"},
        "Density plot": {"scatter"},
        "Area chart": {"scatter"},
        "ECDF": {"scattergl", "scatter"},
    }
    assert list(expected_traces) == PLOTS
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    app.switch_page("pages/4_Visualizations.py").run()
    assert len(app.radio[0].options) == 12
    for kind, types in expected_traces.items():
        app.radio[0].set_value(kind).run()
        assert not app.exception and not app.info, kind
        plot = app.get("plotly_chart")
        assert len(plot) == 1, kind
        spec = json.loads(plot[0].proto.spec)
        assert spec["data"] and {trace["type"] for trace in spec["data"]} <= types, kind
        assert any("Key observations" in element.proto.body for element in app.get("html"))
        assert any("Interpretation" in element.proto.body for element in app.get("html"))
        assert app.get("download_button")[0].proto.label == "Download interactive chart (HTML)"
    app.radio[0].set_value("Pie chart")
    app.multiselect[0].set_value(["hold", "pass"]).run()
    assert not app.exception and not app.info
    pie = json.loads(app.get("plotly_chart")[0].proto.spec)["data"][0]
    data = basketball[1]
    expected = data.loc[data.activity.isin(["hold", "pass"]), "activity"].value_counts().to_dict()
    assert dict(zip(pie["labels"], pie["values"])) == expected
    assert pie["textinfo"] == "label+percent" and pie["showlegend"]
    assert "%{value:,}" in pie["hovertemplate"] and "%{percent:.1%}" in pie["hovertemplate"]
    app.multiselect[0].set_value([]).run()
    assert not app.exception and not app.get("plotly_chart")
    assert "Select at least one activity" in app.info[0].value


def test_pie_keeps_missing_and_blank_activity_labels(basketball):
    data = basketball[1].loc[lambda frame: frame.activity.eq("hold")].head(5).copy()
    data.loc[data.index[:2], "activity"] = [None, " "]
    figure, _, findings = make_plot("Pie chart", data)
    pie = figure.data[0]
    assert dict(zip(pie.labels, pie.values)) == {"hold": 3, "Unlabeled": 2}
    assert any("hold (3 samples" in item for item in findings)
    assert any("Unlabeled (2 samples" in item for item in findings)
    export = figure.to_html(full_html=True, include_plotlyjs=True)
    assert "Plotly.newPlot" in export and "Unlabeled" in export and "label+percent" in export
