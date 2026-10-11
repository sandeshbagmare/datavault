"""profile_and_filter — explore data safely; returns a bounded Data Profile, never full data."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from datavault.context import DataContext, Policy, default_context
from datavault.proxy.profiler import DataProfiler, DataStore, quote_ident
from datavault.query import QueryError, build_select
from datavault.tools._base import ContextGetter, ToolResult, build_tool

DESCRIPTION = """Explore a dataset safely. Returns a compact Data Profile (schema, statistics, a few sample rows) - never the full data.

Any filter/group/query creates a NEW reference id (see 'Lineage' in the output). Pass that id as data_ref to this tool again, to generate_html_viz, or to generate_markdown. Calling with only data_ref re-profiles the same reference.

Ways to derive a reference (combine freely):
- filter_query: SQL WHERE predicate (no WHERE keyword).
- columns: keep only these columns.
- group_by + metrics: aggregate. group_by items: "brand" or {"col": "sale_date", "grain": "month"} (grains: day, week, month, quarter, year). metrics: [{"agg": "avg", "col": "price_usd", "as": "avg_price"}] (agg: count, sum, avg, min, max, median, stddev, count_distinct, p90, ...).
- order_by + limit: e.g. order_by=[{"col": "avg_price", "desc": true}], limit=10. The sample rows then show the first rows IN ORDER, so this is how you read exact top/bottom answers.
- sql: advanced read-only SELECT; write {ref} for the input reference. Only registered references are queryable.

Example: profile_and_filter(data_ref="ref_abc", filter_query="brand = 'Tesla'", group_by=[{"col":"sale_date","grain":"year"}], metrics=[{"agg":"sum","col":"price_usd","as":"revenue"}])
Privacy: you only get aggregates plus a few bounded sample rows; masked columns show ***."""


class ProfileInput(BaseModel):
    """Arguments for profile_and_filter."""

    data_ref: str | None = Field(None, description="Reference id (e.g. 'ref_abc123') or root table name. Optional only if exactly one dataset exists.")
    filter_query: str | None = Field(None, description="SQL WHERE predicate without the WHERE keyword, e.g. \"vehicle_type = 'Car' AND price_usd > 50000\".")
    columns: list[str] | str | None = Field(None, description="Columns to keep (projection).")
    group_by: list[Any] | str | None = Field(None, description="Group-by columns: names or {'col','grain','as'} objects.")
    metrics: list[Any] | str | None = Field(None, description="Aggregates: [{'agg': 'avg', 'col': 'price', 'as': 'avg_price'}] or strings like 'AVG(price)'.")
    order_by: list[Any] | str | None = Field(None, description="Sort: ['-revenue'] or [{'col': 'revenue', 'desc': true}].")
    limit: int | None = Field(None, description="Maximum rows of the derived reference.")
    sql: str | None = Field(None, description="Advanced: a single read-only SELECT. Use {ref} for the input reference.")
    sample_rows: int | None = Field(None, description="How many sample rows to show (bounded by policy).")


def _listy(value: Any) -> list[Any] | None:
    """Accept a list, a JSON string, a comma-separated string or one name."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("[", "{")):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise QueryError(f"Not valid JSON: {exc.msg}") from exc
            return parsed if isinstance(parsed, list) else [parsed]
        return [p.strip() for p in text.split(",") if p.strip()]
    return list(value)


def _run(ctx: DataContext, a: ProfileInput) -> ToolResult:
    ref = ctx.resolve(a.data_ref)
    spec: dict[str, Any] = {
        "where": a.filter_query, "select": _listy(a.columns), "group_by": _listy(a.group_by),
        "metrics": _listy(a.metrics), "order_by": _listy(a.order_by), "limit": a.limit,
    }
    sql, description, ordered = None, "", False
    if a.sql and a.sql.strip():
        if not ctx.policy.allow_raw_sql:
            raise QueryError("Raw SQL is disabled by policy. Use filter_query/group_by/metrics/order_by/limit.")
        sql = a.sql.replace("{ref}", quote_ident(ref.view_name))
        description = "sql: " + " ".join(a.sql.split())[:160]
        ordered = bool(re.search(r"\border\s+by\b", sql, re.IGNORECASE))
    elif any(v for v in spec.values()):
        built = build_select(ref.view_name, ctx.columns(ref), spec)
        sql, description, ordered = built.sql, built.description, built.ordered
    n = ctx.profiler.sample_size if a.sample_rows is None else max(0, min(a.sample_rows, ctx.policy.max_sample_rows))
    with ctx.lock:
        if sql is None:
            profile = ctx.profiler.describe(ref, n)
        else:
            ctx.validate(sql)
            profile = ctx.profiler.profile(
                ref.view_name, filter_sql=a.filter_query, select_sql=sql, operation=description, ordered=ordered, sample_size=n,
            )
    artifact = {
        "kind": "profile", "ref_id": profile.ref_id, "parent_ref": ref.ref_id if profile.ref_id != ref.ref_id else None,
        "rows": profile.row_count, "columns": [c for c, _ in profile.columns], "lineage": profile.lineage,
    }
    return profile.to_llm_text(), artifact


def make_profile_tool(get_context: ContextGetter) -> StructuredTool:
    """Build the tool; ``get_context`` is called on every invocation."""
    return build_tool(
        name="profile_and_filter", description=DESCRIPTION, args_schema=ProfileInput,
        failure="profiling data", run=_run, get_context=get_context,
    )


_ctx: DataContext | None = None


def configure(profiler: DataProfiler, data_store: DataStore) -> None:
    """Bind the module-level ``profile_and_filter`` tool to a profiler and store.

    Prefer ``create_tools(DataContext(...))``, which keeps agents isolated.
    """
    global _ctx
    _ctx = DataContext(connection=profiler.connection, store=data_store, profiler=profiler, policy=Policy.from_env())


profile_and_filter = make_profile_tool(lambda: _ctx or default_context())
