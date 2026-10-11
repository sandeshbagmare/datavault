"""The Markdown engine: correct numbers, any document shape, safe tables."""

from __future__ import annotations

from collections import defaultdict

import pytest

from conftest import BRANDS, N_ROWS, make_context, sales_rows, saved_path, tools_by_name


def write(ctx, **args) -> str:
    out = tools_by_name(ctx)["generate_markdown"].invoke({"data_ref": "sales", **args})
    assert "Report saved to: " in out, out
    return saved_path(out).read_text(encoding="utf-8")


def table_rows(md: str, header_start: str) -> list[list[str]]:
    lines = md.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(header_start))
    rows = []
    for ln in lines[start + 2:]:
        if not ln.startswith("|"):
            break
        rows.append([c.strip() for c in ln.strip("|").split("|")])
    return rows


def test_breakdown_honors_the_requested_aggregation_regression(ctx):
    """Previously aggregation was accepted and ignored: SUM silently produced AVG/MIN/MAX."""
    md = write(ctx, report_type="breakdown", group_col="brand", metric_col="price", aggregation="SUM")
    sums = defaultdict(float)
    for r in sales_rows():
        sums[r["brand"]] += r["price"]
    rows = {r[0]: r for r in table_rows(md, "| brand")}
    header = [c.strip() for c in next(ln for ln in md.splitlines() if ln.startswith("| brand")).strip("|").split("|")]
    sum_col = header.index("SUM price")
    for brand in BRANDS:
        assert float(rows[brand][sum_col].replace(",", "")) == pytest.approx(sums[brand], abs=0.01)
    assert "**SUM price**" in md  # ranked by the requested metric


def test_percentages_are_shares_of_the_whole_slice_with_other_row_regression(ctx):
    """Previously percentages used only the returned top-N rows as the denominator."""
    md = write(ctx, report_type="summary")
    rows = table_rows(md, "| id") if "| id" in md else table_rows(md, "| vehicle_type")
    pct = {r[0]: float(r[2].rstrip("%")) for r in rows}
    assert pct["Car"] == pytest.approx(sum(1 for r in sales_rows() if r["vehicle_type"] == "Car") / N_ROWS * 100, abs=0.1)
    # notes is 1 distinct value, so no 'Other'; a wide column must add one
    wide = make_context(ctx.output_dir / "w")
    wide.connection.execute("CREATE TABLE wide AS SELECT 'v' || i AS label FROM range(40) t(i)")
    wide.register_table("wide")
    md_wide = tools_by_name(wide)["generate_markdown"].invoke({"data_ref": "wide", "report_type": "summary"})
    body = saved_path(md_wide).read_text(encoding="utf-8")
    other = next(ln for ln in body.splitlines() if ln.startswith("| Other"))
    assert "| 25 |" in other and "62.5%" in other  # 40 rows, top 15 shown -> 25 others = 62.5%


def test_trend_computes_period_change_and_overall_change(ctx):
    md = write(ctx, report_type="trend", date_col="sale_date", metric_col="price", aggregation="AVG", time_grain="quarter")
    rows = table_rows(md, "| Period")
    assert rows[0][2] == "—" and rows[1][2].endswith("%") and len(rows) == 11  # 1000 days ~ 11 quarters
    assert "**Overall Change:**" in md and "Start" in md


def test_trend_handles_zero_start_and_empty_slices(ctx):
    ctx.connection.execute("CREATE TABLE z AS SELECT DATE '2024-01-01' + INTERVAL (i * 40) DAY AS d, (CASE WHEN i = 0 THEN 0 ELSE i END)::DOUBLE AS v FROM range(4) t(i)")
    ctx.register_table("z")
    md = saved_path(tools_by_name(ctx)["generate_markdown"].invoke({"data_ref": "z", "report_type": "trend", "date_col": "d", "metric_col": "v", "aggregation": "SUM"})).read_text(encoding="utf-8")
    assert "—" in md and "Overall Change" in md  # no division by zero
    empty = tools_by_name(ctx)["profile_and_filter"].invoke({"data_ref": "sales", "filter_query": "id < 0"})
    ref = empty.split("`")[1]
    assert "saved to" in tools_by_name(ctx)["generate_markdown"].invoke({"data_ref": ref, "report_type": "summary"})


def test_null_group_keys_are_reported(ctx):
    ctx.connection.execute("CREATE TABLE n AS SELECT CASE WHEN i % 2 = 0 THEN NULL ELSE 'a' END AS k, i AS v FROM range(10) t(i)")
    ctx.register_table("n")
    md = saved_path(tools_by_name(ctx)["generate_markdown"].invoke({"data_ref": "n", "report_type": "breakdown", "group_col": "k", "metric_col": "v", "aggregation": "SUM"})).read_text(encoding="utf-8")
    assert "(null)" in md and "50.0%" in md


def test_summary_rejects_a_wrong_metric_instead_of_substituting(ctx):
    out = tools_by_name(ctx)["generate_markdown"].invoke({"data_ref": "sales", "report_type": "summary", "metric_col": "brand"})
    assert out.startswith("Error") and "numeric" in out


def test_any_document_can_be_composed_from_sections(ctx):
    sections = [
        {"type": "heading", "text": "Q&A: <Brand> | Review", "level": 1},
        {"type": "text", "text": "Top revenue brand is **{{by_brand.brand}}** at {{by_brand.revenue}}."},
        {"type": "table", "dataset": "by_brand", "columns": ["brand", {"col": "revenue", "label": "Revenue ($)", "format": "currency"}], "title": "Revenue"},
        {"type": "metrics", "items": [{"label": "Brands", "value": 4}, {"label": "Top", "dataset": "by_brand", "field": "revenue", "format": "compact"}]},
        {"type": "list", "items": ["one", "two"], "ordered": True},
        {"type": "quote", "text": "line1\nline2"},
        {"type": "code", "language": "sql", "text": "SELECT 1; ``` fenced ```"},
        {"type": "divider"},
        {"type": "image", "url": "https://example.com/a.png", "alt": "chart"},
    ]
    datasets = {"by_brand": {"group_by": ["brand"], "metrics": [{"agg": "sum", "col": "price", "as": "revenue"}], "order_by": ["-revenue"]}}
    md = write(ctx, title="Custom", sections=sections, datasets=datasets)
    sums = defaultdict(float)
    for r in sales_rows():
        sums[r["brand"]] += r["price"]
    best = max(sums.values())
    tied = [b for b, v in sums.items() if v == best]  # several brands can tie; any of them may rank first
    assert md.startswith("# Custom") and any(f"**{b}** at {best:,.2f}" in md for b in tied)
    assert "| Revenue ($) |" in md and "| ---: |" not in md and "|---|---:|" in md
    assert "1. one" in md and "> line2" in md and "![chart](https://example.com/a.png)" in md
    assert "```sql\nSELECT 1; ''' fenced '''\n```" in md  # fences cannot be broken out of
    assert "Source: `sales`" in md  # provenance is automatic


def test_table_cells_cannot_break_the_table(ctx):
    sections = [{"type": "table", "dataset": "d"}]
    md = write(ctx, sections=sections, datasets={"d": {"sql": "SELECT 'a|b\nc' AS txt, 1 AS n"}})
    assert "| a\\|b c | 1 |" in md


def test_sections_and_datasets_may_be_json_strings_and_errors_are_clear(ctx):
    import json

    md = write(ctx, sections=json.dumps([{"type": "text", "text": "hi {{d.n}}"}]), datasets=json.dumps({"d": {"sql": "SELECT 7 AS n"}}))
    assert "hi 7" in md
    t = tools_by_name(ctx)["generate_markdown"]
    assert "Unknown section type" in t.invoke({"data_ref": "sales", "sections": [{"type": "tweet"}]})
    assert "unknown dataset" in t.invoke({"data_ref": "sales", "sections": [{"type": "text", "text": "{{nope.x}}"}]})
    assert "https://" in t.invoke({"data_ref": "sales", "sections": [{"type": "image", "url": "javascript:alert(1)"}]})
    assert "Unknown report type" in t.invoke({"data_ref": "sales", "report_type": "poem"})


def test_truncated_data_is_disclosed_in_the_document(tmp_path):
    ctx = make_context(tmp_path, max_dataset_rows=10)
    md = write(ctx, sections=[{"type": "table", "dataset": "d"}], datasets={"d": {"sql": "SELECT * FROM {ref}"}})
    assert "cut to the first 10 rows" in md
