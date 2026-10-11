"""generate_markdown — compose any Markdown document from ordered sections and computed data."""

from __future__ import annotations

from typing import Any

import duckdb
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from datavault.context import DataContext, Policy, default_context
from datavault.datasets import load_datasets, parse_json_arg
from datavault.proxy.profiler import DataStore
from datavault.query import QueryError
from datavault.render.markdown import (
    SECTION_TYPES,
    breakdown_report,
    compose_markdown,
    summary_report,
    trend_report,
)
from datavault.tools._base import ContextGetter, ToolResult, build_tool, write_artifact

DESCRIPTION = """Create a Markdown document (saved as a .md file the user sees) from ordered sections. Numbers and tables are computed server-side by DuckDB over data references; you write the prose and describe the data to show - you never receive or retype the data.

Quick reports: report_type='summary' (overview, metrics, top categories), 'breakdown' (group_col, optional metric_col + aggregation SUM/AVG/...), 'trend' (date_col, metric_col, aggregation, time_grain day/week/month/quarter/year). Every report carries source/lineage info.

Anything else: pass `sections` (and `datasets`) to write ANY document - memo, analysis, comparison, FAQ, README, changelog, runbook:
{
 "datasets": {"by_brand": {"group_by": ["brand"], "metrics": [{"agg":"sum","col":"price_usd","as":"revenue"}], "order_by": [{"col":"revenue","desc":true}], "limit": 10}},
 "sections": [
  {"type":"heading","text":"Executive summary","level":2},
  {"type":"text","text":"Top brand is {{by_brand.brand}} with {{by_brand.revenue}} revenue."},
  {"type":"table","dataset":"by_brand","columns":["brand",{"col":"revenue","format":"currency"}],"max_rows":10},
  {"type":"metrics","items":[{"label":"Top revenue","dataset":"by_brand","field":"revenue","prefix":"$","format":"compact"}]},
  {"type":"list","items":["point one","point two"],"ordered":false},
  {"type":"quote","text":"..."}, {"type":"code","language":"sql","text":"..."}, {"type":"divider"}, {"type":"image","url":"https://...","alt":"..."}
 ]
}
Dataset forms are the same as in generate_html_viz: structured (where/group_by/metrics/order_by/limit/select) or {"sql": "SELECT ... FROM {ref}"}. Placeholders {{dataset.column}} / {{dataset.column[row]}} insert computed values into any text, so prose stays correct without you seeing the numbers. Formats: number, compact, percent, signed_percent, currency, raw.
Returns a summary and the saved file path."""


class MarkdownInput(BaseModel):
    """Arguments for generate_markdown."""

    data_ref: str | None = Field(None, description="Reference id (e.g. 'ref_abc123') or root table name; default dataset.")
    report_type: str = Field("summary", description="'summary', 'breakdown', 'trend' - or 'custom' when `sections` is given.")
    title: str = Field("Data Report", description="Document title.")
    group_col: str | None = Field(None, description="breakdown: column to group by.")
    metric_col: str | None = Field(None, description="summary/breakdown/trend: numeric column to measure.")
    date_col: str | None = Field(None, description="trend: date/timestamp column.")
    aggregation: str = Field("COUNT", description="COUNT, SUM, AVG, MIN, MAX, MEDIAN (breakdown/trend).")
    time_grain: str = Field("month", description="trend: day, week, month, quarter, year.")
    sections: list[Any] | str | None = Field(None, description="Ordered sections for a custom document (see tool description).")
    datasets: dict[str, Any] | str | None = Field(None, description="Named dataset specs that sections bind to.")
    provenance: bool = Field(True, description="Append source/lineage information.")


def _run(ctx: DataContext, a: MarkdownInput) -> ToolResult:
    if a.data_ref and a.data_ref.strip():
        ref = ctx.resolve(a.data_ref)
    else:
        roots = ctx.store.roots()
        ref = roots[0] if len(roots) == 1 else None
    sections = parse_json_arg(a.sections, "sections")
    kind = (a.report_type or "summary").strip().lower()
    if sections:
        if not isinstance(sections, list):
            raise QueryError(f"sections must be a list of section objects. Section types: {list(SECTION_TYPES)}.")
        datasets = load_datasets(ctx, parse_json_arg(a.datasets, "datasets"), ref.ref_id if ref else None)
    else:
        if ref is None:
            raise QueryError("data_ref is required for summary/breakdown/trend reports.")
        if kind == "summary":
            sections, datasets = summary_report(ctx, ref, a.title, a.metric_col)
        elif kind == "breakdown":
            if not a.group_col:
                raise QueryError("'group_col' is required for breakdown reports.")
            sections, datasets = breakdown_report(ctx, ref, a.group_col, a.metric_col, a.aggregation)
        elif kind == "trend":
            if not a.date_col or not a.metric_col:
                raise QueryError("'date_col' and 'metric_col' are required for trend reports.")
            sections, datasets = trend_report(ctx, ref, a.date_col, a.metric_col, a.aggregation, a.time_grain)
        else:
            raise QueryError(f"Unknown report type '{a.report_type}'. Use summary, breakdown, trend, or pass `sections`.")
    text = compose_markdown(ctx, title=a.title, sections=sections, datasets=datasets, provenance=a.provenance)
    path = write_artifact(ctx.report_path, "report", ".md", text)
    parts = ", ".join(f"{n}({len(d.rows):,} rows)" for n, d in datasets.items()) or "none"
    lines = [f"Composed {len(sections)} section(s) from datasets: {parts}."]
    lines += [f"Note: {w}." for d in datasets.values() for w in d.warnings()]
    lines.append(f"Report saved to: {path}")
    artifact = {
        "kind": "markdown", "path": str(path), "sections": len(sections), "chars": len(text),
        "ref_ids": sorted({d.ref_id for d in datasets.values()} | ({ref.ref_id} if ref else set())),
    }
    return "\n".join(lines), artifact


def make_markdown_tool(get_context: ContextGetter) -> StructuredTool:
    """Build the tool; ``get_context`` is called on every invocation."""
    return build_tool(
        name="generate_markdown", description=DESCRIPTION, args_schema=MarkdownInput,
        failure="generating report", run=_run, get_context=get_context,
    )


_ctx: DataContext | None = None


def configure(connection: duckdb.DuckDBPyConnection, data_store: DataStore, output_dir: str = "output/reports") -> None:
    """Bind the module-level ``generate_markdown`` tool to a connection, store and folder.

    Prefer ``create_tools(DataContext(...))``, which keeps agents isolated.
    """
    global _ctx
    _ctx = DataContext(connection=connection, store=data_store, policy=Policy.from_env(), report_dir=output_dir)


generate_markdown = make_markdown_tool(lambda: _ctx or default_context())
