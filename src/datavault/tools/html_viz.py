"""generate_html_viz — build any chart, dashboard or HTML page from server-side data.

The model writes a small declarative ``spec`` (or raw HTML when it truly needs to);
DuckDB computes the numbers and DataVault renders a safe, self-contained page.
The model never receives the data and never has to write data arrays.
"""

from __future__ import annotations

from typing import Any

import duckdb
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from datavault.context import DataContext, Policy, default_context
from datavault.datasets import load_datasets, parse_json_arg
from datavault.proxy.profiler import DataStore
from datavault.query import QueryError
from datavault.render.html import render_document, wrap_custom_html
from datavault.render.theme import resolve_theme
from datavault.tools._base import ContextGetter, ToolResult, build_tool, write_artifact
from datavault.tools._shorthand import auto_dashboard_spec, charts_spec, single_chart_spec

DESCRIPTION = """Create an interactive HTML visualization or dashboard (saved as a file the user sees). Numbers are computed server-side by DuckDB over the data reference; you describe WHAT to show, never the data itself. Output is a complete page: any theme, any layout, any chart.

TWO WAYS TO CALL
A) Quick: chart_type ('bar','line','area','pie','donut','scatter','histogram','heatmap','box','table', or 'dashboard' for an automatic overview) + x_col, y_col, color_col, aggregation (COUNT/SUM/AVG/MIN/MAX/MEDIAN), optional sql_query, charts=[{type,title,x,y,agg,sql}], kpis=[{label,sql,prefix,suffix}], layout ('grid','comparison','single'), theme.
B) Full control (preferred for anything custom): pass `spec`:
{
 "title": "...", "subtitle": "...", "theme": "dark|light|midnight|black" or {"base":"dark","accent":"#f59e0b","bg":"#000","colors":[...],"font":"Georgia, serif"},
 "datasets": { "<name>": <dataset> },
 "body": <block or [blocks]>
}
DATASET (one read-only query on a reference; use ANY of): {"ref": "ref_x" (optional), "where": "SQL predicate", "group_by": ["brand", {"col":"sale_date","grain":"month","as":"month"}], "metrics": [{"agg":"avg","col":"price_usd","as":"avg_price"}], "order_by": [{"col":"avg_price","desc":true}], "limit": 20, "select": ["a","b"]} or {"sql": "SELECT ... FROM {ref} ..."}. Rows beyond the cap are randomly sampled or cut and noted on the page, so aggregate large data.
BLOCKS ("type"):
- layout: "row" | "column" | "grid"{columns:2 | layout:"comparison"} | "section"{title,collapsible} | "tabs"{tabs:[{title,children}]} | "divider"; all take "children".
- "kpis": {"items":[{"label","dataset","field","prefix","suffix","format":"compact|percent|currency|raw","subtext"}]} (or "value" for a literal).
- "chart": simple {"chart":"bar|barh|line|area|scatter|pie|donut|histogram|box|violin|heatmap|treemap|sunburst|funnel|waterfall","dataset","x","y":"col" or ["colA","colB"],"color","size","stack":true|"percent","secondary_y":["col"],"smooth","labels","log_y","title","height","layout":{...},"trace":{...}} - or ANY Plotly figure: {"figure":{"data":[{"type":"sankey|candlestick|scatter3d|indicator|scatterpolar|...", "x":{"$col":"dataset.column"}, ...}],"layout":{...}}} (use {"$col":"dataset.column"} wherever Plotly expects data; updatemenus/sliders/annotations/shapes work).
- "table": {"dataset","columns":["a",{"col":"b","label":"B","format":"currency"}],"max_rows":50,"title"}.
- "text": {"text","variant":"h1|h2|h3|p|muted|callout|code"}; "markdown": {"text"} - text may embed computed values with {{dataset.column}} or {{dataset.column[2]}}.
- every block accepts "style":{css props}, "class", "id", "span" (grid columns), "height".
CUSTOM CODE (only if enabled; runs sandboxed, no network): "html" block {"html":"<div>..</div>"}, spec-level "css", "js" (reads window.DATAVAULT.data.<dataset> = list of row objects; Plotly is global) and "libs":["https://cdn.jsdelivr.net/..."]. Or pass `custom_html` (a fragment or full page) together with `datasets` to get server-computed data at window.DATAVAULT.data. Use code only for what blocks cannot express (custom interaction, d3/three.js, bespoke SVG).

Example: spec={"datasets":{"m":{"group_by":[{"col":"sale_date","grain":"month","as":"month"}],"metrics":[{"agg":"sum","col":"price_usd","as":"revenue"},{"agg":"count","as":"units"}]}},"body":[{"type":"kpis","items":[{"label":"Months","value":12}]},{"type":"chart","chart":"line","dataset":"m","x":"month","y":["revenue"],"secondary_y":["units"],"title":"Revenue vs units"}]}
Returns a summary and the saved file path."""


class HtmlVizInput(BaseModel):
    """Arguments for generate_html_viz."""

    data_ref: str | None = Field(None, description="Reference id (e.g. 'ref_abc123') or root table name; default dataset for the page.")
    title: str = Field("Data Analytics Dashboard", description="Page title.")
    spec: dict[str, Any] | str | None = Field(None, description="Full document spec (object or JSON string): title, theme, datasets, body. See tool description.")
    datasets: dict[str, Any] | str | None = Field(None, description="Named dataset specs; used with custom_html (window.DATAVAULT.data).")
    custom_html: str | None = Field(None, description="Raw HTML (fragment or full page). Needs custom code enabled. Pair with `datasets` for data.")
    chart_type: str = Field("dashboard", description="Quick mode: dashboard, bar, line, area, pie, donut, scatter, histogram, heatmap, box, table, ...")
    x_col: str = Field("", description="Quick mode: category/dimension column.")
    y_col: str = Field("", description="Quick mode: numeric column to aggregate.")
    color_col: str | None = Field(None, description="Quick mode: column that splits series/colors.")
    aggregation: str = Field("COUNT", description="Quick mode: COUNT, SUM, AVG, MIN, MAX, MEDIAN.")
    sql_query: str | None = Field(None, description="Quick mode: custom read-only SELECT (use {ref} or a table name) for one chart.")
    charts: list[dict[str, Any]] | str | None = Field(None, description="Quick mode: list of chart specs [{type,title,x,y,color,agg,sql}].")
    kpis: list[dict[str, Any]] | str | None = Field(None, description="Quick mode: KPI cards [{label, sql | value, prefix, suffix, subtext}].")
    layout: str = Field("grid", description="Quick mode layout: grid, comparison, single.")
    theme: str | dict[str, Any] | None = Field("dark", description="dark, light, midnight, black, or an object of overrides.")


def _default_ref(ctx: DataContext, data_ref: str | None) -> str | None:
    """The page's default reference. An unknown name is an error, never another dataset."""
    if data_ref and data_ref.strip():
        return ctx.resolve(data_ref).ref_id
    roots = ctx.store.roots()
    return roots[0].ref_id if len(roots) == 1 else None


def _spec_from_args(ctx: DataContext, a: HtmlVizInput, default_ref: str | None) -> dict[str, Any]:
    spec = parse_json_arg(a.spec, "spec")
    if spec:
        if not isinstance(spec, dict):
            raise QueryError("spec must be an object.")
        return dict(spec)
    charts = parse_json_arg(a.charts, "charts")
    kpis = parse_json_arg(a.kpis, "kpis")
    chart_type = (a.chart_type or "dashboard").strip().lower()
    if charts:
        return charts_spec(charts, kpis, a.layout)
    if chart_type == "dashboard":
        if default_ref is None:
            raise QueryError("data_ref is required for an automatic dashboard.")
        return auto_dashboard_spec(ctx, ctx.store.get(default_ref), a.x_col, a.y_col, a.aggregation, kpis, a.layout)
    return single_chart_spec(a.title, chart_type, a.x_col, a.y_col, a.color_col, a.aggregation, a.sql_query)


def _run(ctx: DataContext, a: HtmlVizInput) -> ToolResult:
    default_ref = _default_ref(ctx, a.data_ref)
    if a.custom_html and a.custom_html.strip():
        datasets = load_datasets(ctx, parse_json_arg(a.datasets, "datasets"), default_ref)
        theme = resolve_theme(a.theme)
        page_html = wrap_custom_html(
            ctx, a.custom_html.strip(), title=a.title, theme=theme, data={n: d.records() for n, d in datasets.items()}
        )
        path = write_artifact(ctx.viz_path, "viz_custom", ".html", page_html)
        return _reply("Visualization", path, datasets, [w for d in datasets.values() for w in d.warnings()], 1, 0, default_ref)

    spec = _spec_from_args(ctx, a, default_ref)
    spec.setdefault("title", a.title)
    if a.theme and "theme" not in spec:
        spec["theme"] = a.theme
    if a.datasets and not spec.get("datasets"):
        spec["datasets"] = parse_json_arg(a.datasets, "datasets")
    page = render_document(ctx, spec, default_ref=default_ref, title=a.title)
    path = write_artifact(ctx.viz_path, "dashboard" if page.label == "Dashboard" else "viz", ".html", page.html)
    return _reply(page.label, path, page.datasets, page.warnings, page.blocks, page.charts, default_ref)


def _reply(label: str, path: Any, datasets: dict[str, Any], warnings: list[str], blocks: int, charts: int, default_ref: str | None) -> ToolResult:
    parts = ", ".join(f"{n}({len(d.rows):,} rows x {len(d.columns)} cols)" for n, d in datasets.items()) or "none"
    lines = [f"Rendered {blocks} block(s), {charts} chart(s). Datasets: {parts}."]
    lines += [f"Note: {w}." for w in warnings]
    lines.append(f"{label} saved to: {path}")
    artifact = {
        "kind": "html", "label": label, "path": str(path), "ref_ids": sorted({d.ref_id for d in datasets.values()} | ({default_ref} if default_ref else set())),
        "datasets": {n: {"rows": len(d.rows), "columns": d.columns} for n, d in datasets.items()}, "warnings": warnings,
    }
    return "\n".join(lines), artifact


def make_html_tool(get_context: ContextGetter) -> StructuredTool:
    """Build the tool; ``get_context`` is called on every invocation."""
    return build_tool(
        name="generate_html_viz", description=DESCRIPTION, args_schema=HtmlVizInput,
        failure="generating visualization", run=_run, get_context=get_context,
    )


_ctx: DataContext | None = None


def _ensure_defaults() -> DataContext:
    """The context used by the module-level tool (the default one unless ``configure`` was called)."""
    return _ctx or default_context()


def configure(connection: duckdb.DuckDBPyConnection, data_store: DataStore, output_dir: str = "output/viz") -> None:
    """Bind the module-level ``generate_html_viz`` tool to a connection, store and folder.

    Prefer ``create_tools(DataContext(...))``, which keeps agents isolated.
    """
    global _ctx
    _ctx = DataContext(connection=connection, store=data_store, policy=Policy.from_env(), viz_dir=output_dir)


generate_html_viz = make_html_tool(_ensure_defaults)
