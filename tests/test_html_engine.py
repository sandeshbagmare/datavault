"""The HTML engine: correct numbers, flexible output, safe by construction."""

from __future__ import annotations

import json
import re
from collections import defaultdict

import pytest

from conftest import BRANDS, bridge, make_context, sales_rows, saved_path, tools_by_name


def render(ctx, spec=None, **args):
    out = tools_by_name(ctx)["generate_html_viz"].invoke({"data_ref": "sales", **({"spec": spec} if spec else {}), **args})
    assert "saved to" in out, out
    return saved_path(out).read_text(encoding="utf-8"), out


def figures(html):
    return {i: f for i, f in bridge(html)["figures"]}


def only_figure(html):
    figs = list(figures(html).values())
    assert len(figs) == 1
    return figs[0]


def test_bar_chart_values_are_the_real_aggregates(ctx):
    html, _ = render(ctx, chart_type="bar", x_col="brand", y_col="price", aggregation="SUM")
    trace = only_figure(html)["data"][0]
    expected = defaultdict(float)
    for r in sales_rows():
        expected[r["brand"]] += r["price"]
    assert dict(zip(trace["x"], trace["y"], strict=True)) == pytest.approx(dict(expected))
    assert trace["y"] == sorted(trace["y"], reverse=True)  # categories ranked by value


def test_line_chart_is_chronological_regression(ctx):
    """Previously points were ordered by value, producing a scribble instead of a line."""
    html, _ = render(ctx, chart_type="line", x_col="sale_date", y_col="price", aggregation="AVG")
    xs = only_figure(html)["data"][0]["x"]
    assert xs == sorted(xs) and len(xs) == 1000


def test_custom_sql_with_own_column_names_regression(ctx):
    """Previously a query returning 'revenue' crashed because the renderer demanded a 'value' column."""
    html, _ = render(
        ctx, chart_type="line", x_col="sale_date", y_col="revenue",
        sql_query="SELECT DATE_TRUNC('month', sale_date) AS sale_date, SUM(price) AS revenue FROM {ref} GROUP BY 1 ORDER BY 1",
    )
    assert only_figure(html)["data"][0]["y"]


def test_wrong_column_names_fail_loudly_with_the_available_columns(ctx):
    out = tools_by_name(ctx)["generate_html_viz"].invoke({"data_ref": "sales", "chart_type": "bar", "x_col": "colour"})
    assert out.startswith("Error") and "brand" in out


def test_multi_series_secondary_axis_and_stacking(ctx):
    spec = {
        "datasets": {"m": {"group_by": [{"col": "sale_date", "grain": "month", "as": "month"}, "brand"],
                           "metrics": [{"agg": "sum", "col": "price", "as": "revenue"}, {"agg": "count", "as": "units"}],
                           "order_by": ["month"]}},
        "body": [
            {"type": "chart", "chart": "line", "dataset": "m", "x": "month", "y": ["revenue", "units"], "color": "brand", "secondary_y": ["units"]},
            {"type": "chart", "chart": "bar", "dataset": "m", "x": "month", "y": "revenue", "color": "brand", "stack": "percent"},
        ],
    }
    html, _ = render(ctx, spec)
    line, bar = figures(html).values()
    assert len(line["data"]) == 2 * len(BRANDS)  # one line per (series, brand)
    assert {t.get("yaxis") for t in line["data"]} == {None, "y2"} and "yaxis2" in line["layout"]
    assert bar["layout"]["barmode"] == "stack" and bar["layout"]["barnorm"] == "percent"


def test_any_plotly_trace_via_native_figure_with_column_bindings(ctx):
    spec = {
        "datasets": {"flow": {"group_by": ["brand", "vehicle_type"], "metrics": [{"agg": "count", "as": "n"}]}},
        "body": {"type": "chart", "figure": {
            "data": [{"type": "sankey",
                      "node": {"label": ["Toyota", "Honda", "Ford", "Tesla", "Car", "SUV", "Truck"]},
                      "link": {"source": [0, 1, 2, 3], "target": [4, 5, 6, 4], "value": {"$col": "flow.n"}}}],
            "layout": {"title": "Flow"},
        }},
    }
    html, _ = render(ctx, spec)
    fig = only_figure(html)
    assert fig["data"][0]["type"] == "sankey" and len(fig["data"][0]["link"]["value"]) == 12


def test_invalid_plotly_attributes_are_reported_to_the_model(ctx):
    spec = {"datasets": {"d": {"group_by": ["brand"]}}, "body": {"type": "chart", "figure": {"data": [{"type": "bar", "bogusattr": 1}]}}}
    out = tools_by_name(ctx)["generate_html_viz"].invoke({"data_ref": "sales", "spec": spec})
    assert out.startswith("Error") and "Invalid Plotly figure" in out and "bogusattr" in out


@pytest.mark.parametrize("chart", ["pie", "donut", "histogram", "box", "violin", "scatter", "area", "barh", "funnel", "waterfall", "treemap", "sunburst", "heatmap"])
def test_every_simple_chart_type_renders(ctx, chart):
    spec = {
        "datasets": {"d": {"group_by": ["brand", "vehicle_type"], "metrics": [{"agg": "avg", "col": "price", "as": "value"}]},
                     "raw": {"select": ["price", "id", "brand"], "limit": 200}},
        "body": {"type": "chart", "chart": chart, "dataset": "raw" if chart in ("histogram", "box", "violin", "scatter") else "d",
                 "x": {"histogram": "price", "scatter": "id", "box": "brand", "violin": "brand"}.get(chart, "brand"),
                 "y": "price" if chart in ("scatter", "box", "violin") else ("vehicle_type" if chart == "heatmap" else "value"),
                 "z": "value", "path": ["brand", "vehicle_type"]},
    }
    if chart == "heatmap":
        spec["body"]["x"] = "brand"
    html, _ = render(ctx, spec)
    assert only_figure(html)["data"]


def test_layout_blocks_text_markdown_kpis_and_placeholders(ctx):
    spec = {
        "theme": "black",
        "datasets": {"total": {"metrics": [{"agg": "sum", "col": "price", "as": "revenue"}]},
                     "top": {"group_by": ["brand"], "metrics": [{"agg": "sum", "col": "price", "as": "revenue"}], "order_by": ["-revenue"], "limit": 3}},
        "body": [
            {"type": "kpis", "items": [{"label": "Revenue", "dataset": "total", "field": "revenue", "prefix": "$", "format": "compact", "subtext": "across {{top.brand}} and more"}]},
            {"type": "tabs", "tabs": [
                {"title": "Overview", "children": [{"type": "markdown", "text": "## Hello\n\n**{{total.revenue}}** total"}]},
                {"title": "Data", "children": [{"type": "table", "dataset": "top", "columns": ["brand", {"col": "revenue", "format": "currency"}]}]},
            ]},
            {"type": "section", "title": "Notes", "collapsible": True, "children": [{"type": "text", "text": "Plain", "variant": "callout"}]},
        ],
    }
    html, out = render(ctx, spec)
    expected_total = sum(r["price"] for r in sales_rows())
    assert "#000000" in html and "Theme: Black" in html and "stat-card" in html
    assert f"${expected_total / 1e3:,.1f}K" in html or f"${expected_total / 1e6:,.1f}M" in html
    assert "<h2>Hello</h2>" in html and f"{expected_total:,.2f}" in html  # placeholder filled, markdown rendered
    assert "data-tabs" in html and "<details" in html and "$" in html
    assert out.splitlines()[-1].startswith("Dashboard saved to: ")


def test_custom_theme_object(ctx):
    html, _ = render(ctx, chart_type="bar", x_col="brand", theme={"base": "dark", "accent": "#f59e0b", "bg": "#101010", "font": "Georgia, serif", "colors": ["#ff0000", "#00ff00"]})
    assert "#f59e0b" in html and "#101010" in html and "Georgia" in html
    assert only_figure(html)["layout"]["colorway"] == ["#ff0000", "#00ff00"]
    out = tools_by_name(ctx)["generate_html_viz"].invoke({"data_ref": "sales", "chart_type": "bar", "x_col": "brand", "theme": "neon"})
    assert out.startswith("Error") and "Unknown theme" in out


def test_large_data_is_sampled_not_silently_cut(tmp_path):
    ctx = make_context(tmp_path, max_dataset_rows=100)
    spec = {"datasets": {"raw": {"select": ["id", "price"]}}, "body": {"type": "chart", "chart": "scatter", "dataset": "raw", "x": "id", "y": "price"}}
    html, out = render(ctx, spec)
    assert len(only_figure(html)["data"][0]["x"]) == 100
    ids = only_figure(html)["data"][0]["x"]
    assert max(ids) > 500  # a random sample spans the table; a head-cut would stop at 99
    assert "random sample" in html and "Note:" in out


def test_everything_user_controlled_is_escaped(ctx):
    evil = '<img src=x onerror=alert(1)>"</script><script>alert(2)</script>'
    spec = {
        "title": evil, "subtitle": evil,
        "datasets": {"d": {"sql": "SELECT '<script>alert(3)</script>' AS name, 1 AS n"}},
        "body": [
            {"type": "text", "text": evil}, {"type": "markdown", "text": f"{evil} [x](javascript:alert(4)) <b onclick=1>hi</b>"},
            {"type": "table", "dataset": "d", "title": evil},
            {"type": "kpis", "items": [{"label": evil, "value": evil}]},
            {"type": "section", "title": evil, "children": []},
        ],
    }
    html, _ = render(ctx, spec)
    for needle in ("alert(1)", "alert(2)", "alert(3)", "onclick=1", "javascript:alert"):
        # may appear only as inert escaped text, never as live markup
        for m in re.finditer(re.escape(needle), html):
            before = html[max(0, m.start() - 40): m.start()]
            assert "<script>" not in before and "<img" not in before and "<b " not in before, (needle, before)
    assert html.count("<script>") == 2 and "</script><script>alert" not in html  # only our two inline scripts


def test_figure_json_cannot_break_out_of_the_script_tag(ctx):
    spec = {"datasets": {"d": {"sql": "SELECT '</script><script>alert(1)</script>' AS k, 5 AS v"}},
            "body": {"type": "chart", "chart": "bar", "dataset": "d", "x": "k", "y": "v"}}
    html, _ = render(ctx, spec)
    assert "</script><script>alert(1)" not in html
    assert only_figure(html)["data"][0]["x"] == ["</script><script>alert(1)</script>"]


@pytest.mark.parametrize("style", [{"background": "red; } body { display:none"}, {"color": "expression(alert(1))"}, {"x y": "1"}])
def test_style_injection_is_rejected(ctx, style):
    out = tools_by_name(ctx)["generate_html_viz"].invoke({"data_ref": "sales", "spec": {"datasets": {"d": {"select": ["id"], "limit": 1}}, "body": {"type": "table", "dataset": "d", "style": style}}})
    assert out.startswith("Error")


def test_every_page_carries_a_content_security_policy_without_network(ctx):
    html, _ = render(ctx, chart_type="bar", x_col="brand")
    csp = re.search(r'Content-Security-Policy" content="([^"]+)"', html).group(1)
    assert "connect-src &#x27;none&#x27;" in csp or "connect-src 'none'" in csp
    assert "plotly-latest" not in html and re.search(r"cdn\.plot\.ly/plotly-\d+\.\d+\.\d+\.min\.js", html)


def test_plotly_can_be_inlined_for_offline_use(tmp_path):
    ctx = make_context(tmp_path, plotly_js="inline")
    html, _ = render(ctx, chart_type="bar", x_col="brand")
    assert 'src="https://cdn.plot.ly' not in html and len(html) > 1_000_000


def test_custom_code_blocks_get_server_computed_data(ctx):
    spec = {
        "datasets": {"by_brand": {"group_by": ["brand"], "metrics": [{"agg": "count", "as": "n"}], "order_by": ["brand"]}},
        "css": ".mine { color: hotpink }", "js": "document.getElementById('x').textContent = DATAVAULT.data.by_brand.length;",
        "libs": ["https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"],
        "body": {"type": "html", "html": "<div id='x' class='mine'></div>"},
    }
    html, _ = render(ctx, spec)
    assert bridge(html)["data"]["by_brand"] == [{"brand": b, "n": 250} for b in sorted(BRANDS)]
    assert "hotpink" in html and "d3@7" in html and "plotly" in html.lower()


def test_custom_html_with_datasets_and_plotly_latest_rewrite(ctx):
    out = tools_by_name(ctx)["generate_html_viz"].invoke({
        "data_ref": "sales", "title": "Mine",
        "custom_html": "<html><head><title>x</title><script src='https://cdn.plot.ly/plotly-latest.min.js'></script></head><body><h1>Hi</h1></body></html>",
        "datasets": {"t": {"metrics": [{"agg": "count", "as": "n"}]}},
    })
    html = saved_path(out).read_text(encoding="utf-8")
    assert "plotly-latest" not in html and "Content-Security-Policy" in html
    assert bridge(html)["data"]["t"] == [{"n": 1000}] and "<h1>Hi</h1>" in html

    fragment = tools_by_name(ctx)["generate_html_viz"].invoke({"data_ref": "sales", "custom_html": "<div id=a>frag</div><script>Plotly.newPlot('a',[])</script>"})
    page = saved_path(fragment).read_text(encoding="utf-8")
    assert "frag" in page and "cdn.plot.ly/plotly-" in page and "Content-Security-Policy" in page


def test_custom_code_can_be_disabled_by_policy(tmp_path):
    t = tools_by_name(make_context(tmp_path, allow_custom_code=False))["generate_html_viz"]
    assert "disabled by policy" in t.invoke({"data_ref": "sales", "custom_html": "<b>x</b>"})
    assert "disabled by policy" in t.invoke({"data_ref": "sales", "spec": {"datasets": {}, "js": "alert(1)", "body": {"type": "text", "text": "x"}}})
    assert "disabled by policy" in t.invoke({"data_ref": "sales", "spec": {"datasets": {}, "body": {"type": "html", "html": "<i>x</i>"}}})
    assert "saved to" in t.invoke({"data_ref": "sales", "chart_type": "bar", "x_col": "brand"})  # declarative output still works


def test_libs_are_limited_to_known_cdns_unless_network_is_allowed(tmp_path):
    spec = {"datasets": {}, "libs": ["https://evil.example/x.js"], "body": {"type": "html", "html": "x"}}
    assert "not allowed" in tools_by_name(make_context(tmp_path / "a"))["generate_html_viz"].invoke({"data_ref": "sales", "spec": spec})
    open_ctx = make_context(tmp_path / "b", allow_network_in_html=True)
    out = tools_by_name(open_ctx)["generate_html_viz"].invoke({"data_ref": "sales", "spec": spec})
    assert "saved to" in out and "connect-src https:" in saved_path(out).read_text(encoding="utf-8")


def test_spec_may_be_a_json_string_and_errors_are_actionable(ctx):
    spec = {"datasets": {"d": {"group_by": ["brand"]}}, "body": {"type": "chart", "chart": "pie", "dataset": "d", "x": "brand"}}
    html, _ = render(ctx, json.dumps(spec))
    assert only_figure(html)["data"][0]["type"] == "pie"
    t = tools_by_name(ctx)["generate_html_viz"]
    assert "not valid JSON" in t.invoke({"data_ref": "sales", "spec": "{oops"})
    assert "Unknown block type" in t.invoke({"data_ref": "sales", "spec": {"datasets": {}, "body": {"type": "carousel"}}})
    assert "Unknown dataset" in t.invoke({"data_ref": "sales", "spec": {"datasets": {}, "body": {"type": "table", "dataset": "zzz"}}})


def test_legacy_shorthand_charts_and_kpis_still_work(ctx):
    html, out = render(ctx, charts=json.dumps([{"type": "bar", "title": "A", "x": "brand", "y": "price", "agg": "AVG"}, {"type": "pie", "title": "B", "x": "vehicle_type"}]),
                       kpis=json.dumps([{"label": "Units", "sql": "SELECT COUNT(*) FROM sales"}]), layout="comparison", theme="midnight")
    assert out.splitlines()[-1].startswith("Dashboard saved to: ") and "layout-comparison" in html and "Midnight" in html and "1,000" in html
    assert len(figures(html)) == 2
