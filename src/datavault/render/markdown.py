"""Compose Markdown documents from ordered sections bound to named datasets.

The legacy report types (summary / breakdown / trend) are just ready-made section
lists built here; any other document is described with ``sections`` directly.
"""

from __future__ import annotations

import re
from typing import Any

from datavault.context import DataContext
from datavault.datasets import Dataset, fill_placeholders, format_value, load_dataset
from datavault.proxy.profiler import DataReference, column_kind
from datavault.query import QueryError

SECTION_TYPES = ("heading", "text", "list", "table", "metrics", "code", "quote", "divider", "image", "source")
_ALIASES = {"markdown": "text", "paragraph": "text", "p": "text", "bullets": "list", "kpis": "metrics", "stats": "metrics", "h1": "heading", "h2": "heading", "h3": "heading"}
_IMAGE_URL = re.compile(r"^(https://|data:image/)", re.IGNORECASE)


def md_cell(value: Any) -> str:
    """Make a value safe inside a Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _line(value: Any) -> str:
    return " ".join(str(value).split())


class _Composer:
    def __init__(self, ctx: DataContext, datasets: dict[str, Dataset]) -> None:
        self.ctx, self.datasets = ctx, datasets

    def _text(self, value: Any) -> str:
        return fill_placeholders(str(value), self.datasets)

    def _ds(self, name: Any) -> Dataset:
        if not name and len(self.datasets) == 1:
            name = next(iter(self.datasets))
        if name not in self.datasets:
            raise QueryError(f"Unknown dataset '{name}'. Datasets: {list(self.datasets)}.")
        return self.datasets[name]

    def section(self, node: Any) -> str:
        if isinstance(node, str):
            node = {"type": "text", "text": node}
        if not isinstance(node, dict):
            raise QueryError(f"A section must be an object, got {type(node).__name__}.")
        raw = str(node.get("type", "")).lower()
        if raw in ("h1", "h2", "h3"):
            node = {**node, "level": int(raw[1])}
        kind = _ALIASES.get(raw, raw)
        if kind not in SECTION_TYPES:
            raise QueryError(f"Unknown section type '{node.get('type')}'. Section types: {list(SECTION_TYPES)}.")
        return getattr(self, f"_s_{kind}")(node)

    def _s_heading(self, node: dict[str, Any]) -> str:
        level = min(6, max(1, int(node.get("level", 2))))
        return f"{'#' * level} {_line(self._text(node.get('text', '')))}"

    def _s_text(self, node: dict[str, Any]) -> str:
        return self._text(node.get("text", "")).strip()

    def _s_list(self, node: dict[str, Any]) -> str:
        items = node.get("items") or []
        ordered = bool(node.get("ordered"))
        return "\n".join(
            f"{i + 1}. {_line(self._text(it))}" if ordered else f"- {_line(self._text(it))}" for i, it in enumerate(items)
        )

    def _s_code(self, node: dict[str, Any]) -> str:
        text = str(node.get("text", "")).replace("```", "'''")
        return f"```{_line(node.get('language', ''))}\n{text}\n```"

    def _s_quote(self, node: dict[str, Any]) -> str:
        return "\n".join(f"> {ln}" for ln in self._text(node.get("text", "")).splitlines() or [""])

    def _s_divider(self, node: dict[str, Any]) -> str:
        return "---"

    def _s_image(self, node: dict[str, Any]) -> str:
        url = str(node.get("url", ""))
        if not _IMAGE_URL.match(url):
            raise QueryError("image.url must be an https:// URL or a data:image/ URI.")
        return f"![{_line(node.get('alt', ''))}]({url})"

    def _s_source(self, node: dict[str, Any]) -> str:
        refs = list(dict.fromkeys(d.ref_id for d in self.datasets.values()))
        store = self.ctx.store
        assert store is not None
        parts = []
        for ref_id in refs[:5]:
            ref = store.get(ref_id)
            lineage = " -> ".join(store.lineage(ref_id))
            parts.append(f"`{ref_id}` ({ref.row_count:,} rows){': ' + lineage if lineage else ''}")
        return "*Source: " + "; ".join(parts) + "*" if parts else ""

    def _s_metrics(self, node: dict[str, Any]) -> str:
        rows = ["| Metric | Value |", "|--------|-------|"]
        for item in node.get("items", []):
            if "value" in item and "dataset" not in item:
                raw = item["value"]
            else:
                ds = self._ds(item.get("dataset"))
                column = ds.canonical(item.get("field") or ds.columns[0])
                idx = int(item.get("row", 0))
                raw = ds.rows[idx][ds.columns.index(column)] if idx < len(ds.rows) else None
            value = f"{item.get('prefix', '')}{format_value(raw, item.get('format'))}{item.get('suffix', '')}"
            rows.append(f"| {md_cell(self._text(item.get('label', 'Metric')))} | {md_cell(value)} |")
        title = f"**{_line(self._text(node['title']))}**\n\n" if node.get("title") else ""
        return title + "\n".join(rows)

    def _s_table(self, node: dict[str, Any]) -> str:
        ds = self._ds(node.get("dataset"))
        specs = node.get("columns") or ds.columns
        cols = []
        for c in specs:
            name = ds.canonical(c.get("col") if isinstance(c, dict) else c)
            label = (c.get("label") if isinstance(c, dict) else None) or name
            cols.append((ds.columns.index(name), str(label), c.get("format") if isinstance(c, dict) else None))
        limit = int(node.get("max_rows", 50))
        numeric = [column_kind(ds.types[i]) == "numeric" for i, _, _ in cols]
        out = []
        if node.get("title"):
            out.append(f"**{_line(self._text(node['title']))}**\n")
        out.append("| " + " | ".join(md_cell(label) for _, label, _ in cols) + " |")
        out.append("|" + "|".join("---:" if n else "---" for n in numeric) + "|")
        for row in ds.rows[:limit]:
            out.append("| " + " | ".join(md_cell(format_value(row[i], fmt)) for i, _, fmt in cols) + " |")
        if len(ds.rows) > limit:
            out.append(f"\n*Showing {limit:,} of {len(ds.rows):,} rows.*")
        return "\n".join(out)


def compose_markdown(
    ctx: DataContext,
    *,
    title: str,
    sections: list[Any],
    datasets: dict[str, Dataset],
    provenance: bool = True,
) -> str:
    """Render ``sections`` to a Markdown document (title, sections, provenance, footer).

    Raises:
        QueryError: For unknown section types, datasets or columns.
    """
    composer = _Composer(ctx, datasets)
    parts = [f"# {_line(title)}"] if title else []
    parts += [p for p in (composer.section(s) for s in sections) if p]
    if provenance and datasets and not any(isinstance(s, dict) and s.get("type") == "source" for s in sections):
        parts.append(composer._s_source({}))
    notes = [w for d in datasets.values() for w in d.warnings()]
    parts.append("---\n*Generated by DataVault | Server-side aggregation*" + "".join(f"\n\n*Note: {n}*" for n in notes))
    return "\n\n".join(p for p in parts if p) + "\n"


# -- ready-made reports (shorthand for sections) ----------------------------------------


def _mem(name: str, columns: list[str], types: list[str], rows: list[tuple[Any, ...]], ref_id: str) -> Dataset:
    return Dataset(name=name, columns=columns, types=types, rows=rows, ref_id=ref_id)


def _group_rows(
    ctx: DataContext, ref: DataReference, group_col: str, extra: list[dict[str, Any]], limit: int, order: str
) -> Dataset:
    spec = {"group_by": [group_col], "metrics": [{"agg": "count", "as": "Count"}, *extra], "order_by": [{"col": order, "desc": True}], "limit": limit}
    return load_dataset(ctx, f"by_{group_col}", spec, ref.ref_id)


def _with_share(ds: Dataset, ref: DataReference, label: str) -> Dataset:
    """Add a '% of rows' column relative to the whole slice and an 'Other' row for the remainder."""
    total = ref.row_count or sum(r[1] for r in ds.rows) or 1
    rows = [(("(null)" if r[0] is None else r[0]), *r[1:], r[1] / total * 100) for r in ds.rows]
    shown = sum(r[1] for r in ds.rows)
    types = [*ds.types, "DOUBLE"]
    if total > shown and len(ds.rows) >= 1:
        pad = [None] * (len(ds.columns) - 2)
        rows.append(("Other", total - shown, *pad, (total - shown) / total * 100))
    return _mem(ds.name, [*ds.columns, label], types, rows, ds.ref_id)


def summary_report(ctx: DataContext, ref: DataReference, title: str, metric_col: str | None) -> tuple[list[dict[str, Any]], dict[str, Dataset]]:
    """Overview, numeric metrics and top categories for a reference."""
    cols = ctx.columns(ref)
    numeric = [c for c, t in cols if column_kind(t) == "numeric"]
    text = [c for c, t in cols if column_kind(t) == "text"]
    if metric_col:
        match = next((c for c, _ in cols if c.lower() == metric_col.lower()), None)
        if match is None or match not in numeric:
            raise QueryError(f"metric_col '{metric_col}' is not a numeric column. Numeric columns: {numeric}")
        targets = [match]
    else:
        targets = numeric[:4]
    datasets: dict[str, Dataset] = {}
    sections: list[dict[str, Any]] = [
        {"type": "heading", "text": "Overview"},
        {"type": "list", "items": [
            f"**Dataset:** {ref.source_table}", f"**Total Records:** {ref.row_count:,}", f"**Reference:** `{ref.ref_id}`",
            *([f"**Filter:** {ref.filter_sql}"] if ref.filter_sql else []),
        ]},
    ]
    if targets:
        sections.append({"type": "heading", "text": "Key Metrics"})
    for col in targets:
        name = f"metric_{col}"
        spec = {"metrics": [{"agg": a, "col": col, "as": a} for a in ("count", "avg", "min", "max", "stddev")]}
        datasets[name] = load_dataset(ctx, name, spec, ref.ref_id)
        sections.append({"type": "metrics", "title": col, "items": [
            {"label": lbl, "dataset": name, "field": f} for lbl, f in (("Count (non-null)", "count"), ("Average", "avg"), ("Min", "min"), ("Max", "max"), ("Std Dev", "stddev"))
        ]})
    if text:
        sections.append({"type": "heading", "text": "Distribution Summary"})
    for col in text[:3]:
        ds = _with_share(_group_rows(ctx, ref, col, [], 15, "Count"), ref, "% of rows")
        datasets[f"top_{col}"] = ds
        sections.append({"type": "table", "dataset": f"top_{col}", "title": f"Top values in {col}", "columns": [ds.columns[0], "Count", {"col": "% of rows", "format": "percent"}]})
    return sections, datasets


def breakdown_report(
    ctx: DataContext, ref: DataReference, group_col: str, metric_col: str | None, aggregation: str
) -> tuple[list[dict[str, Any]], dict[str, Dataset]]:
    """One table with count, share and (optionally) metric statistics per group."""
    agg = (aggregation or "COUNT").strip().lower()
    extra: list[dict[str, Any]] = []
    if metric_col:
        wanted = [agg] if agg != "count" else []
        for a in dict.fromkeys([*wanted, "avg", "min", "max"]):
            extra.append({"agg": a, "col": metric_col, "as": f"{a.upper()} {metric_col}"})
    order = next((e["as"] for e in extra if e["agg"] == agg), "Count")
    ds = _group_rows(ctx, ref, group_col, extra, 50, order)
    ds = _with_share(ds, ref, "% of rows")
    group_name = ds.columns[0]
    columns = [group_name, "Count", {"col": "% of rows", "format": "percent"}, *[e["as"] for e in extra]]
    sections = [
        {"type": "heading", "text": f"Breakdown by {group_name}"},
        {"type": "text", "text": f"Groups ranked by **{order}**; {len(ds.rows):,} rows shown (remaining groups are summarized as Other)."},
        {"type": "table", "dataset": ds.name, "columns": columns, "max_rows": 60},
    ]
    return sections, {ds.name: ds}


def trend_report(
    ctx: DataContext, ref: DataReference, date_col: str, metric_col: str, aggregation: str, grain: str
) -> tuple[list[dict[str, Any]], dict[str, Dataset]]:
    """Per-period values with period-over-period change and overall change."""
    agg = (aggregation or "COUNT").strip().lower()
    spec = {"group_by": [{"col": date_col, "grain": grain or "month", "as": "Period"}], "metrics": [{"agg": agg, "col": metric_col, "as": "Value"}]}
    raw = load_dataset(ctx, "trend", spec, ref.ref_id)
    rows, prev = [], None
    for period, value in raw.rows:
        change = None if prev in (None, 0) or value is None else (float(value) - float(prev)) / abs(float(prev)) * 100
        value = None if value is None else float(value)
        rows.append((str(period)[:10], value, change))
        prev = value if value is not None else prev
    ds = _mem("trend", ["Period", "Value", "Change"], ["VARCHAR", "DOUBLE", "DOUBLE"], rows, raw.ref_id)
    ds.truncated, ds.sampled = raw.truncated, raw.sampled
    values = [r[1] for r in rows if r[1] is not None]
    sections: list[dict[str, Any]] = [
        {"type": "heading", "text": f"Trend Analysis: {metric_col} over {date_col}"},
        {"type": "text", "text": f"{agg.upper()} of **{metric_col}** per {grain or 'month'}."},
        {"type": "table", "dataset": "trend", "columns": ["Period", {"col": "Value"}, {"col": "Change", "format": "signed_percent"}], "max_rows": 500},
    ]
    if len(values) >= 2:
        first, last = values[0], values[-1]
        overall = f"{(last - first) / abs(first) * 100:+.1f}% ({format_value(first)} -> {format_value(last)})" if first else f"{format_value(first)} -> {format_value(last)}"
        sections += [{"type": "heading", "text": "Summary"}, {"type": "list", "items": [
            f"**Start:** {format_value(first)}", f"**End:** {format_value(last)}", f"**Overall Change:** {overall}", f"**Periods:** {len(rows):,}",
        ]}]
    return sections, {"trend": ds}
