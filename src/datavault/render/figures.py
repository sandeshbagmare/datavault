"""Build validated Plotly figures from datasets.

Two ways to describe a chart, both producing a plain-JSON figure the browser draws
with plotly.js (so the model never writes data arrays):

* **simple** — ``{"type": "line", "dataset": "sales", "x": "month", "y": ["a", "b"]}``
* **native** — ``{"figure": {"data": [...], "layout": {...}}}`` with any Plotly trace
  type; ``{"$col": "dataset.column"}`` anywhere inside is replaced by that column's values.
"""

from __future__ import annotations

import copy
from typing import Any

import plotly.graph_objects as go

from datavault.datasets import Dataset
from datavault.proxy.profiler import column_kind
from datavault.query import QueryError
from datavault.render.theme import Theme

SIMPLE_TYPES = (
    "bar", "barh", "line", "area", "scatter", "pie", "donut", "histogram", "box", "violin",
    "heatmap", "treemap", "sunburst", "funnel", "waterfall",
)
_DEFAULT_HEIGHT = 380


def deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``extra`` into a copy of ``base`` (``extra`` wins)."""
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_bindings(node: Any, datasets: dict[str, Dataset], default: str | None = None) -> Any:
    """Replace ``{"$col": "dataset.column"}`` objects with the column's values."""
    if isinstance(node, dict):
        if set(node) == {"$col"}:
            return _bound_column(str(node["$col"]), datasets, default)
        return {k: resolve_bindings(v, datasets, default) for k, v in node.items()}
    if isinstance(node, list):
        return [resolve_bindings(v, datasets, default) for v in node]
    return node


def _bound_column(ref: str, datasets: dict[str, Dataset], default: str | None) -> list[Any]:
    name, _, column = ref.partition(".")
    if not column:
        name, column = default or (next(iter(datasets)) if len(datasets) == 1 else ""), ref
    if name not in datasets:
        raise QueryError(f"Binding '{ref}': unknown dataset '{name}'. Datasets: {list(datasets)}. Use 'dataset.column'.")
    return datasets[name].col(column)


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _pick(ds: Dataset, wanted: Any, fallback_kinds: tuple[str, ...], exclude: set[str]) -> str | None:
    if wanted and ds.has(wanted):
        return ds.canonical(wanted)
    for col, typ in zip(ds.columns, ds.types, strict=True):
        if col not in exclude and column_kind(typ) in fallback_kinds:
            return col
    return None


def _groups(ds: Dataset, color: str | None) -> list[tuple[str | None, list[int]]]:
    if not color:
        return [(None, list(range(len(ds.rows))))]
    groups: dict[Any, list[int]] = {}
    for i, value in enumerate(ds.col(color)):
        groups.setdefault(value, []).append(i)
    return [(str(k), idx) for k, idx in groups.items()]


def _take(values: list[Any], idx: list[int]) -> list[Any]:
    return [values[i] for i in idx]


def _sorted_idx(values: list[Any], idx: list[int]) -> list[int]:
    try:
        return sorted(idx, key=lambda i: (values[i] is None, values[i]))
    except TypeError:
        return idx


def _hierarchy(ds: Dataset, path: list[str], value_col: str | None) -> dict[str, list[Any]]:
    cols = [ds.col(p) for p in path]
    values = ds.col(value_col) if value_col else [1] * len(ds.rows)
    nodes: dict[str, dict[str, Any]] = {}
    for r, v in enumerate(values):
        for depth in range(len(path)):
            parts = [str(cols[d][r]) for d in range(depth + 1)]
            node_id = " / ".join(parts)
            node = nodes.setdefault(node_id, {"label": parts[-1], "parent": " / ".join(parts[:-1]), "value": 0})
            node["value"] += v or 0
    ids = list(nodes)
    return {
        "ids": ids,
        "labels": [nodes[i]["label"] for i in ids],
        "parents": [nodes[i]["parent"] for i in ids],
        "values": [nodes[i]["value"] for i in ids],
    }


def _simple(spec: dict[str, Any], ds: Dataset) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    declared = spec.get("chart") or (spec.get("type") if spec.get("type") != "chart" else None)
    kind = str(declared or "bar").lower()
    if kind not in SIMPLE_TYPES:
        raise QueryError(
            f"Unsupported simple chart type '{kind}'. Simple types: {list(SIMPLE_TYPES)}. "
            "For any other Plotly trace (sankey, candlestick, scatter3d, indicator, ...) use "
            "'figure': {'data': [{'type': ..., 'x': {'$col': 'dataset.column'}}]}."
        )
    for key in ("x", "color"):
        if spec.get(key) and not ds.has(spec[key]):
            raise QueryError(f"{key} column '{spec[key]}' not in dataset '{ds.name}'. Columns: {ds.columns}")
    color = ds.canonical(spec["color"]) if spec.get("color") else None
    x = _pick(ds, spec.get("x"), ("temporal", "text", "numeric", "boolean", "other"), set())
    ys = [ds.canonical(y) for y in _as_list(spec.get("y")) if ds.has(y)]
    if kind != "histogram" and _as_list(spec.get("y")) and not ys:
        raise QueryError(f"y column(s) {_as_list(spec.get('y'))} not in dataset '{ds.name}'. Columns: {ds.columns}")
    if not ys:
        y = ds.canonical("value") if ds.has("value") else _pick(ds, None, ("numeric",), {c for c in (x, color) if c})
        ys = [y] if y else []
    if kind == "histogram":
        x = _pick(ds, spec.get("x"), ("numeric",), set()) or x
    layout: dict[str, Any] = {}
    traces: list[dict[str, Any]] = []
    secondary = {s.lower() for s in _as_list(spec.get("secondary_y"))}
    if kind != "histogram" and kind not in ("treemap", "sunburst") and not ys and kind not in ("box", "violin"):
        raise QueryError(f"Chart '{kind}' needs a numeric y column; dataset '{ds.name}' has columns {ds.columns}.")
    xs = ds.col(x) if x else []

    if kind in ("pie", "donut"):
        traces.append({"type": "pie", "labels": xs, "values": ds.col(ys[0]), "hole": spec.get("hole", 0.45 if kind == "donut" else 0)})
    elif kind in ("treemap", "sunburst"):
        path = [ds.canonical(p) for p in _as_list(spec.get("path")) if ds.has(p)] or ([x] if x else [])
        if not path:
            raise QueryError(f"{kind} needs 'path' (list of columns, outer to inner) or 'x'.")
        traces.append({"type": kind, "branchvalues": "total", **_hierarchy(ds, path, ys[0] if ys else None)})
    elif kind == "heatmap":
        y_axis = _pick(ds, spec.get("y"), ("text", "temporal", "numeric"), {x} if x else set())
        z = _pick(ds, spec.get("z") or "value", ("numeric",), {c for c in (x, y_axis) if c})
        if not (x and y_axis and z):
            raise QueryError("heatmap needs x, y and a numeric z column (set 'z').")
        traces.append({"type": "heatmap", "x": xs, "y": ds.col(y_axis), "z": ds.col(z), "colorscale": spec.get("colorscale", "Viridis")})
    elif kind == "histogram":
        for name, idx in _groups(ds, color):
            traces.append({"type": "histogram", "x": _take(xs, idx), "name": name, "nbinsx": spec.get("nbins")})
        if color:
            layout["barmode"] = "overlay"
            for t in traces:
                t["opacity"] = 0.75
    elif kind in ("box", "violin"):
        value = ys[0] if ys else _pick(ds, None, ("numeric",), {c for c in (x, color) if c})
        for name, idx in _groups(ds, color):
            t: dict[str, Any] = {"type": kind, "y": _take(ds.col(value), idx), "name": name or value}
            if x:
                t["x"] = _take(xs, idx)
            traces.append(t)
        if x:
            layout["boxmode" if kind == "box" else "violinmode"] = "group"
    else:
        if kind == "scatter":
            size = ds.canonical(spec["size"]) if spec.get("size") and ds.has(spec["size"]) else None
            text = ds.col(spec["text"]) if spec.get("text") and ds.has(spec["text"]) else None
            color_numeric = bool(color) and column_kind(ds.types[ds.columns.index(color)]) == "numeric"
            for yname in ys:
                for gname, idx in _groups(ds, None if color_numeric else color):
                    marker: dict[str, Any] = {}
                    if size:
                        sizes = _take(ds.col(size), idx)
                        biggest = max((s for s in sizes if isinstance(s, (int, float))), default=1) or 1
                        marker.update(size=sizes, sizemode="area", sizeref=2.0 * biggest / (40.0**2), sizemin=3)
                    if color_numeric and color:
                        marker.update(color=_take(ds.col(color), idx), colorscale="Viridis", showscale=True)
                    traces.append({
                        "type": "scatter", "mode": "markers", "x": _take(xs, idx), "y": _take(ds.col(yname), idx),
                        "name": gname or yname, "marker": marker, **({"text": _take(text, idx)} if text else {}),
                    })
        else:
            horizontal = kind == "barh" or spec.get("orientation") == "h"
            many = len(ys) > 1 or bool(color)
            for yname in ys:
                values = ds.col(yname)
                is_secondary = yname.lower() in secondary
                for gname, idx in _groups(ds, color):
                    use = _sorted_idx(xs, idx) if kind in ("line", "area") or is_secondary else idx
                    xv, yv = _take(xs, use), _take(values, use)
                    label = " - ".join(p for p in (yname if len(ys) > 1 else None, gname) if p) or yname
                    if kind in ("line", "area") or is_secondary:
                        t = {
                            "type": "scatter", "mode": "lines+markers" if len(use) <= 80 else "lines",
                            "x": xv, "y": yv, "name": label,
                            "line": {"shape": "spline" if spec.get("smooth") else "linear"},
                        }
                        if kind == "area" and not is_secondary:
                            if spec.get("stack"):
                                t["stackgroup"] = "one"
                            else:
                                t["fill"] = "tozeroy"
                    else:
                        t = {"type": "bar", "x": yv if horizontal else xv, "y": xv if horizontal else yv, "name": label}
                        if horizontal:
                            t["orientation"] = "h"
                    if is_secondary:
                        t["yaxis"] = "y2"
                    if spec.get("labels") and t["type"] == "bar":
                        t.update(text=yv if not horizontal else xv, textposition="auto")
                    traces.append(t)
            stack = spec.get("stack")
            if any(t["type"] == "bar" for t in traces):
                if stack in ("percent", "percentage"):
                    layout.update(barmode="stack", barnorm="percent")
                elif stack:
                    layout["barmode"] = "relative" if stack == "relative" else "stack"
                elif many:
                    layout["barmode"] = "group"
            if secondary:
                layout["yaxis2"] = {"overlaying": "y", "side": "right", "showgrid": False, "title": {"text": ", ".join(sorted(secondary))}}
    if kind == "waterfall":
        traces = [{"type": "waterfall", "x": xs, "y": ds.col(ys[0])}]
    if kind == "funnel":
        traces = [{"type": "funnel", "y": xs, "x": ds.col(ys[0])}]

    if kind not in ("pie", "donut", "treemap", "sunburst", "heatmap"):
        horizontal = kind == "barh" or spec.get("orientation") == "h"
        x_title, y_title = spec.get("x_title", x), spec.get("y_title", ys[0] if len(ys) == 1 else None)
        if horizontal:
            x_title, y_title = y_title, x_title
        if x_title:
            layout.setdefault("xaxis", {})["title"] = {"text": str(x_title)}
        if y_title:
            layout.setdefault("yaxis", {})["title"] = {"text": str(y_title)}
        layout.setdefault("xaxis", {})["automargin"] = True
        if spec.get("log_y"):
            layout.setdefault("yaxis", {})["type"] = "log"
        if spec.get("log_x"):
            layout.setdefault("xaxis", {})["type"] = "log"
    for t in traces:
        for k in [k for k, v in t.items() if v is None]:
            del t[k]
        if spec.get("trace"):
            t.update(copy.deepcopy(spec["trace"]))
    return traces, layout


def build_figure(
    spec: dict[str, Any],
    datasets: dict[str, Dataset],
    theme: Theme,
    default_dataset: str | None,
) -> dict[str, Any]:
    """Return ``{"data", "layout", "config"[, "frames"]}`` for a chart block.

    Raises:
        QueryError: For unknown datasets/columns/types or figures Plotly rejects.
    """
    if "figure" in spec:
        raw = resolve_bindings(spec["figure"], datasets, spec.get("dataset") or default_dataset)
        if not isinstance(raw, dict) or "data" not in raw:
            raise QueryError("'figure' must be an object with a 'data' list of Plotly traces.")
        data, user_layout = raw["data"], raw.get("layout", {})
        config, frames = raw.get("config", {}), raw.get("frames")
    else:
        name = spec.get("dataset") or default_dataset
        if name not in datasets:
            raise QueryError(f"Chart refers to unknown dataset '{name}'. Datasets: {list(datasets)}.")
        data, built_layout = _simple(spec, datasets[name])
        user_layout, config, frames = deep_merge(built_layout, spec.get("layout", {})), spec.get("config", {}), None
    layout = deep_merge(theme.plotly_layout(), user_layout)
    layout.setdefault("height", int(spec.get("height", _DEFAULT_HEIGHT)))
    if spec.get("title") and "title" not in user_layout:
        layout["title"] = {"text": str(spec["title"]), "x": 0.02, "xanchor": "left"}
    layout.setdefault("autosize", True)
    try:
        go.Figure({"data": data, "layout": layout})
    except (ValueError, TypeError) as exc:
        lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
        raise QueryError(f"Invalid Plotly figure: {' '.join(lines[:4])[:600]}") from exc
    figure: dict[str, Any] = {"data": data, "layout": layout, "config": {"responsive": True, "displaylogo": False, **config}}
    if frames:
        figure["frames"] = frames
    return figure
