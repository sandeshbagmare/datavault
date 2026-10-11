"""Translate the compact tool arguments (chart_type/x_col/charts/kpis/...) into a document spec."""

from __future__ import annotations

from typing import Any

from datavault.context import DataContext
from datavault.proxy.profiler import DataReference, column_kind
from datavault.query import QueryError

_MAX_CATEGORIES = 1000
_RAW_POINT_CHARTS = ("histogram", "scatter")


def _layout_block(layout: str, children: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "grid", "layout": layout if layout in ("grid", "comparison", "single") else "grid", "children": children}


def _dataset_for(chart_type: str, x: str, y: str, color: str | None, agg: str) -> dict[str, Any]:
    """Structured dataset spec equivalent to the old server-side aggregation for a chart type."""
    if chart_type == "histogram":
        return {"select": [x]}
    if chart_type == "scatter":
        return {"select": [c for c in (x, y, color) if c]}
    if chart_type == "table":
        return {"select": [c for c in (x, y, color) if c], "limit": 50}
    metric = {"agg": agg, "col": y, "as": "value"} if y else {"agg": "count", "as": "value"}
    group = [c for c in (x, color if chart_type != "heatmap" else (color or y)) if c]
    order = [{"col": x}] if chart_type in ("line", "area") else [{"col": "value", "desc": True}]
    return {"group_by": group, "metrics": [metric], "order_by": order, "limit": _MAX_CATEGORIES}


def _chart_block(chart_type: str, dataset: str, x: str, y: str, color: str | None, agg: str, title: str, *, has_value: bool) -> dict[str, Any]:
    if chart_type == "table":
        return {"type": "table", "dataset": dataset, "title": title}
    block: dict[str, Any] = {"type": "chart", "chart": chart_type, "dataset": dataset, "title": title}
    if chart_type == "histogram":
        block["x"] = x
    elif chart_type == "scatter":
        block.update(x=x, y=y, color=color)
    else:
        block.update(x=x, color=color if chart_type != "heatmap" else None)
        if chart_type == "heatmap":
            block.update(y=color or y, z="value")
        elif has_value:
            block.update(y="value", y_title=f"{agg}({y})" if y else "Count")
        elif y:
            block["y"] = y
    return block


def _kpi_blocks(kpis: list[dict[str, Any]] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """(datasets, kpis block) for KPI specs given either as ``sql`` or literal ``value``."""
    datasets: dict[str, Any] = {}
    items: list[dict[str, Any]] = []
    for i, k in enumerate(kpis or []):
        if not isinstance(k, dict):
            raise QueryError("Each KPI must be an object like {'label': ..., 'sql': ... | 'value': ...}.")
        item = {key: k[key] for key in ("label", "prefix", "suffix", "subtext", "format") if key in k}
        if k.get("sql"):
            datasets[f"kpi_{i}"] = {"sql": k["sql"]}
            item["dataset"] = f"kpi_{i}"
        elif "value" in k:
            item["value"] = k["value"]
        else:
            raise QueryError("A KPI needs 'sql' (first value of the first row is used) or 'value'.")
        items.append(item)
    return datasets, ({"type": "kpis", "items": items} if items else {})


def single_chart_spec(title: str, chart_type: str, x: str, y: str, color: str | None, agg: str, sql_query: str | None) -> dict[str, Any]:
    """One chart (or table) over a structured aggregation or a user query."""
    if sql_query and sql_query.strip():
        dataset: dict[str, Any] = {"sql": sql_query}
        block = _chart_block(chart_type, "main", x, y, color, agg, title, has_value=False)
        block = {k: v for k, v in block.items() if v not in ("", None)}
    else:
        if not x:
            raise QueryError("x_col is required for a single chart (the category/dimension column).")
        dataset = _dataset_for(chart_type, x, y, color, agg)
        block = _chart_block(chart_type, "main", x, y, color, agg, title, has_value=chart_type not in _RAW_POINT_CHARTS + ("table",))
    return {"datasets": {"main": dataset}, "body": [block]}


def charts_spec(
    charts: list[dict[str, Any]], kpis: list[dict[str, Any]] | None, layout: str
) -> dict[str, Any]:
    """Several charts (each: type/title/x/y/color/agg/sql) plus optional KPI cards."""
    datasets, kpi_block = _kpi_blocks(kpis)
    blocks: list[dict[str, Any]] = []
    for i, c in enumerate(charts):
        if not isinstance(c, dict):
            raise QueryError("Each chart must be an object like {'type': 'bar', 'x': ..., 'y': ..., 'agg': ...}.")
        c_type = str(c.get("type", "bar")).lower()
        title = c.get("title") or f"{c_type.capitalize()} Analysis"
        name = f"chart_{i}"
        if c.get("sql"):
            datasets[name] = {"sql": c["sql"]}
            block = _chart_block(c_type, name, c.get("x", ""), c.get("y", ""), c.get("color"), c.get("agg", "COUNT"), title, has_value=False)
            blocks.append({k: v for k, v in block.items() if v not in ("", None)})
        else:
            x, y, agg = c.get("x", ""), c.get("y", ""), str(c.get("agg", "COUNT"))
            if not x:
                raise QueryError(f"Chart '{title}' needs 'x' (or use 'sql').")
            datasets[name] = _dataset_for(c_type, x, y, c.get("color"), agg)
            blocks.append(_chart_block(c_type, name, x, y, c.get("color"), agg, title, has_value=c_type not in _RAW_POINT_CHARTS + ("table",)))
    body: list[dict[str, Any]] = ([kpi_block] if kpi_block else []) + [_layout_block(layout, blocks)]
    return {"datasets": datasets, "body": body}


def auto_dashboard_spec(
    ctx: DataContext,
    ref: DataReference,
    focus_x: str,
    focus_y: str,
    aggregation: str,
    kpis: list[dict[str, Any]] | None,
    layout: str,
) -> dict[str, Any]:
    """An automatic overview: KPIs, a focus chart, a breakdown, a trend, a histogram and a table."""
    cols = ctx.columns(ref)
    by_kind = {k: [c for c, t in cols if column_kind(t) == k] for k in ("numeric", "text", "temporal")}
    names = {c.lower(): c for c, _ in cols}
    numeric, text, dates = by_kind["numeric"], by_kind["text"], by_kind["temporal"]
    agg = (aggregation or "COUNT").upper()

    datasets, kpi_block = _kpi_blocks(kpis)
    items: list[dict[str, Any]] = [{"label": "Total Volume", "value": ref.row_count, "subtext": "Active records in scope"}]
    for i, col in enumerate(numeric[:3]):
        datasets[f"num_{i}"] = {"metrics": [{"agg": a, "col": col, "as": a} for a in ("avg", "min", "max")]}
        prefix = "$" if "price" in col.lower() else ""
        items.append({
            "label": f"Avg {col}", "dataset": f"num_{i}", "field": "avg", "prefix": prefix,
            "subtext": f"Min: {{{{num_{i}.min}}}} | Max: {{{{num_{i}.max}}}}",
        })
    for i, col in enumerate(text[:2]):
        datasets[f"cat_{i}"] = {"metrics": [{"agg": "count_distinct", "col": col, "as": "n"}]}
        items.append({"label": f"Distinct {col}s", "dataset": f"cat_{i}", "field": "n", "subtext": f"Unique {col} values"})
    items += kpi_block.get("items", [])

    px = names.get(focus_x.lower()) if focus_x else None
    px = px or (text[0] if text else (numeric[0] if numeric else None))
    py = names.get(focus_y.lower()) if focus_y else None
    py = py if py in numeric else (numeric[0] if numeric else None)
    blocks: list[dict[str, Any]] = []
    if px:
        metric = {"agg": agg, "col": py, "as": "value"} if py else {"agg": "count", "as": "value"}
        datasets["focus"] = {"group_by": [px], "metrics": [metric], "order_by": [{"col": "value", "desc": True}], "limit": 20}
        blocks.append({"type": "chart", "chart": "bar", "dataset": "focus", "x": px, "y": "value",
                       "title": f"{agg} {py} by {px}" if py else f"Distribution by {px}"})
    others = [c for c in text if c != px]
    if others:
        datasets["breakdown"] = {"group_by": [others[0]], "metrics": [{"agg": "count", "as": "count"}], "order_by": [{"col": "count", "desc": True}], "limit": 10}
        blocks.append({"type": "chart", "chart": "donut", "dataset": "breakdown", "x": others[0], "y": "count", "hole": 0.35, "title": f"Breakdown by {others[0]}"})
    value_col = py or (numeric[0] if numeric else None)
    if dates and value_col:
        datasets["trend"] = {"group_by": [{"col": dates[0], "grain": "month", "as": "month"}],
                             "metrics": [{"agg": "avg", "col": value_col, "as": "avg_value"}, {"agg": "count", "as": "volume"}]}
        blocks.append({"type": "chart", "chart": "line", "dataset": "trend", "x": "month", "y": "avg_value", "title": f"Monthly Average {value_col} Trend"})
    if value_col:
        datasets["distribution"] = {"select": [value_col]}
        blocks.append({"type": "chart", "chart": "histogram", "dataset": "distribution", "x": value_col, "nbins": 40, "title": f"Value Distribution: {value_col}"})
    sample_cols = [c for c in (px, py, others[0] if others else None) if c]
    if sample_cols:
        datasets["sample"] = {"select": sample_cols, "limit": 10}
        blocks.append({"type": "table", "dataset": "sample", "title": f"Sample Data View ({px})"})
    return {"datasets": datasets, "body": [{"type": "kpis", "items": items}, _layout_block(layout, blocks)]}
