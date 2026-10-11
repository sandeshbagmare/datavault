"""Named, bounded datasets that renderers (HTML, Markdown) bind to.

A dataset is the result of one read-only query over a data reference. Renderers
receive the rows; the LLM only ever writes the *spec* that produces them.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

from datavault.context import DataContext
from datavault.proxy.profiler import quote_ident
from datavault.query import QueryError, build_select


def jsonable(value: Any) -> Any:
    """Convert a DuckDB value into something ``json.dumps`` accepts (NaN/Inf -> None)."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return str(value)


@dataclass
class Dataset:
    """Rows produced by one dataset spec."""

    name: str
    columns: list[str]
    types: list[str]
    rows: list[tuple[Any, ...]]
    ref_id: str
    description: str = ""
    truncated: bool = False
    sampled: bool = False
    _index: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._index = {c.lower(): i for i, c in enumerate(self.columns)}

    def has(self, column: str) -> bool:
        """Whether ``column`` exists (case-insensitive)."""
        return str(column).lower() in self._index

    def col(self, column: str) -> list[Any]:
        """All values of a column as JSON-safe Python values.

        Raises:
            QueryError: If the column is not in this dataset.
        """
        i = self._index.get(str(column).lower())
        if i is None:
            raise QueryError(f"Dataset '{self.name}' has no column '{column}'. Columns: {self.columns}")
        return [jsonable(r[i]) for r in self.rows]

    def canonical(self, column: str) -> str:
        """The dataset's own spelling of ``column``.

        Raises:
            QueryError: If the column is not in this dataset.
        """
        i = self._index.get(str(column).lower())
        if i is None:
            raise QueryError(f"Dataset '{self.name}' has no column '{column}'. Columns: {self.columns}")
        return self.columns[i]

    def records(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Rows as JSON-safe dicts."""
        rows = self.rows if limit is None else self.rows[:limit]
        return [{c: jsonable(v) for c, v in zip(self.columns, r, strict=True)} for r in rows]

    def warnings(self) -> list[str]:
        """Human-readable notes about reduced data."""
        notes: list[str] = []
        if self.sampled:
            notes.append(f"dataset '{self.name}' is a random sample of {len(self.rows):,} rows (aggregate for exact totals)")
        elif self.truncated:
            notes.append(f"dataset '{self.name}' was cut to the first {len(self.rows):,} rows (use group_by/metrics/limit)")
        return notes


def format_value(value: Any, fmt: str | None = None) -> str:
    """Format a value for display. ``fmt``: number (default), compact, percent, signed_percent, currency, raw."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float, Decimal)):
        n = float(value)
        if fmt == "raw":
            return str(value)
        if fmt == "percent":
            return f"{n:,.1f}%"
        if fmt in ("signed_percent", "change"):
            return f"{n:+,.1f}%"
        if fmt == "currency":
            return f"${n:,.2f}"
        if fmt == "compact":
            for limit, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
                if abs(n) >= limit:
                    return f"{n / limit:,.1f}{suffix}"
        return f"{value:,}" if isinstance(value, int) else f"{n:,.2f}"
    if isinstance(value, datetime):
        return value.date().isoformat() if value.time() == time(0) else value.isoformat(sep=" ")
    if isinstance(value, (date, time)):
        return value.isoformat()
    return str(value)


_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_]\w*)\.([^\s\[\]{}]+?)(?:\[(\d+)\])?\s*\}\}")


def fill_placeholders(text: str, datasets: dict[str, Dataset]) -> str:
    """Replace ``{{dataset.column}}`` / ``{{dataset.column[2]}}`` with computed values.

    This lets the model write narrative around numbers it never sees.

    Raises:
        QueryError: For unknown datasets/columns or out-of-range rows.
    """

    def repl(match: re.Match[str]) -> str:
        name, column, row = match.group(1), match.group(2), int(match.group(3) or 0)
        if name not in datasets:
            raise QueryError(f"Placeholder '{match.group(0)}': unknown dataset '{name}'. Datasets: {list(datasets)}")
        ds = datasets[name]
        if row >= len(ds.rows):
            raise QueryError(f"Placeholder '{match.group(0)}': dataset '{name}' has only {len(ds.rows)} rows.")
        return format_value(ds.rows[row][ds.columns.index(ds.canonical(column))])

    return _PLACEHOLDER.sub(repl, text)


def parse_json_arg(value: Any, what: str) -> Any:
    """Accept a dict/list or a JSON string (LLMs often send stringified JSON)."""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise QueryError(f"'{what}' is not valid JSON: {exc.msg} (line {exc.lineno}, col {exc.colno}).") from exc
    return value


def load_dataset(ctx: DataContext, name: str, spec: Any, default_ref: str | None) -> Dataset:
    """Run one dataset spec.

    Spec forms: a SQL string, or a dict with either ``sql`` (may contain ``{ref}``)
    or structured fields (``where``, ``select``, ``group_by``, ``metrics``,
    ``order_by``, ``limit``, ``sample``), plus optional ``ref`` and ``max_rows``.
    Non-aggregated data larger than the cap is randomly sampled, not cut.
    """
    if isinstance(spec, str):
        spec = {"sql": spec}
    if not isinstance(spec, dict):
        raise QueryError(f"Dataset '{name}' must be an object or SQL string.")
    ref = ctx.resolve(spec.get("ref") or default_ref)
    cap = min(int(spec.get("max_rows") or ctx.policy.max_dataset_rows), ctx.policy.max_dataset_rows)
    sampled = False
    raw_sql = spec.get("sql") or spec.get("query")
    if raw_sql:
        if not ctx.policy.allow_raw_sql:
            raise QueryError("Raw SQL is disabled by policy. Use where/group_by/metrics/order_by/limit instead.")
        sql = str(raw_sql).replace("{ref}", quote_ident(ref.view_name))
        description = "custom sql"
    else:
        query = dict(spec)
        aggregated = bool(query.get("group_by") or query.get("metrics"))
        if not aggregated and not query.get("sample") and not query.get("limit") and ref.row_count > cap:
            query["sample"] = cap
            sampled = True
        built = build_select(ref.view_name, ctx.columns(ref), query)
        sql, description = built.sql, built.description
    result = ctx.run_select(sql, max_rows=cap)
    return Dataset(
        name=name,
        columns=result.columns,
        types=result.types,
        rows=result.rows,
        ref_id=ref.ref_id,
        description=description,
        truncated=result.truncated and not sampled,
        sampled=sampled,
    )


def load_datasets(ctx: DataContext, specs: dict[str, Any] | None, default_ref: str | None) -> dict[str, Dataset]:
    """Load every named dataset; errors name the dataset that failed."""
    out: dict[str, Dataset] = {}
    for name, spec in (specs or {}).items():
        try:
            out[str(name)] = load_dataset(ctx, str(name), spec, default_ref)
        except (QueryError, KeyError) as exc:
            raise QueryError(f"dataset '{name}': {exc.args[0] if exc.args else exc}") from exc
    return out
