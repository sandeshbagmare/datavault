"""Data profiler — generates lightweight Data Profiles from DuckDB data references.

The Data Profile is the core privacy mechanism: it distills a dataset (potentially
10M+ rows) into a few hundred tokens containing schema, statistics, and a small,
bounded sample. The LLM only ever sees the profile, never the full dataset.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import duckdb

MASK = "***"
_EXACT_DISTINCT_MAX_ROWS = 1_000_000
_NUMERIC_TYPES = {
    "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT", "UINTEGER",
    "UBIGINT", "UHUGEINT", "FLOAT", "DOUBLE", "REAL", "NUMERIC", "INT", "INT4", "INT8", "FLOAT4", "FLOAT8",
}
_TEXT_TYPES = {"VARCHAR", "TEXT", "STRING", "CHAR", "BPCHAR", "UUID"}


def quote_ident(name: str) -> str:
    """Quote a SQL identifier, escaping embedded double quotes."""
    return '"' + str(name).replace('"', '""') + '"'


def column_kind(col_type: str) -> str:
    """Classify a DuckDB type as numeric, temporal, text, boolean or other."""
    t = str(col_type).upper().strip()
    base = t.split("(")[0].strip()
    if t.endswith("]") or base.startswith(("STRUCT", "MAP", "LIST", "UNION", "ENUM")):
        return "other"
    if base in _NUMERIC_TYPES or base.startswith("DECIMAL"):
        return "numeric"
    if base in ("DATE", "TIME") or base.startswith(("TIMESTAMP", "TIME ")):
        return "temporal"
    if base in _TEXT_TYPES:
        return "text"
    if base in ("BOOLEAN", "BOOL"):
        return "boolean"
    return "other"


def _cell(value: Any) -> str:
    """Render a value as a single safe markdown table cell."""
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


@dataclass
class DataProfile:
    """A lightweight summary of a dataset that is safe to show to an LLM.

    Attributes:
        ref_id: Opaque reference identifier for this dataset.
        table_name: The source table or view name.
        row_count: Total number of rows in the dataset.
        columns: List of (column_name, column_type) tuples.
        statistics: Per-column statistics (min, max, mean, nulls, distinct count).
        sample_rows: A small bounded sample of rows for the LLM to understand content.
        filter_applied: SQL WHERE clause applied to derive this profile, if any.
        lineage: Operations applied (root first) to derive this reference.
        hidden_columns: Number of columns omitted to keep the profile small.
    """

    ref_id: str
    table_name: str
    row_count: int
    columns: list[tuple[str, str]]
    statistics: list[dict[str, Any]]
    sample_rows: list[dict[str, Any]]
    filter_applied: str | None = None
    lineage: list[str] = field(default_factory=list)
    hidden_columns: int = 0

    def to_llm_text(self) -> str:
        """Format the profile as compact text suitable for LLM context."""
        lines: list[str] = []
        lines.append(f"## Data Profile: {self.table_name}")
        lines.append(f"**Reference ID:** `{self.ref_id}`")
        lines.append(f"**Total Rows:** {self.row_count:,}")
        if self.filter_applied:
            lines.append(f"**Filter:** `{self.filter_applied}`")
        if self.lineage:
            lines.append("**Lineage:** " + " -> ".join(self.lineage))
        lines.append("")

        lines.append("### Schema")
        lines.append("| Column | Type |")
        lines.append("|--------|------|")
        for col_name, col_type in self.columns:
            lines.append(f"| {_cell(col_name)} | {col_type} |")
        if self.hidden_columns:
            lines.append(f"| ... | +{self.hidden_columns} more columns not shown |")
        lines.append("")

        lines.append("### Statistics")
        lines.append("| Column | Min | Max | Mean | Nulls | Distinct |")
        lines.append("|--------|-----|-----|------|-------|----------|")
        for stat in self.statistics:
            lines.append(
                f"| {_cell(stat['column'])} | {_cell(stat.get('min', 'N/A'))} | "
                f"{_cell(stat.get('max', 'N/A'))} | {stat.get('mean', 'N/A')} | "
                f"{stat['nulls']} | {stat['distinct']} |"
            )
        lines.append("")

        lines.append(f"### Sample Rows ({len(self.sample_rows)} of {self.row_count:,})")
        if self.sample_rows:
            headers = list(self.sample_rows[0].keys())
            lines.append("| " + " | ".join(_cell(h) for h in headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for row in self.sample_rows:
                lines.append("| " + " | ".join(_cell(row[h]) for h in headers) + " |")

        return "\n".join(lines)


@dataclass
class DataReference:
    """A server-side data reference stored in the DataStore.

    Attributes:
        ref_id: Unique reference identifier (also the DuckDB view name).
        view_name: DuckDB view name for this data slice.
        source_table: Table or view this reference was derived from.
        filter_sql: SQL WHERE clause applied, if any.
        row_count: Cached row count.
        parent_ref: The reference this one was derived from (None for roots).
        operation: Human-readable description of how it was derived.
        ordered: True when the view has an ORDER BY the sample should respect.
    """

    ref_id: str
    view_name: str
    source_table: str
    filter_sql: str | None = None
    row_count: int = 0
    parent_ref: str | None = None
    operation: str | None = None
    ordered: bool = False


class DataStore:
    """In-memory store of data references. Maps ref_ids to DuckDB views.

    The DataStore is the bridge between the opaque references the LLM sees
    and the actual data stored in DuckDB. References are live views: they
    reflect the underlying tables as they are now.
    """

    def __init__(self) -> None:
        self._refs: dict[str, DataReference] = {}

    def add(self, ref: DataReference) -> None:
        """Register a new data reference."""
        self._refs[ref.ref_id] = ref

    def get(self, ref_id: str) -> DataReference:
        """Retrieve a data reference by its ID.

        Raises:
            KeyError: If the ref_id is not found.
        """
        if ref_id not in self._refs:
            raise KeyError(f"Data reference '{ref_id}' not found. Available: {self.list_refs()[:10]}")
        return self._refs[ref_id]

    def find(self, name: str) -> DataReference:
        """Look up a reference by ID, or a root reference by its source table name.

        Raises:
            KeyError: If nothing matches. There is deliberately no fallback to
                another dataset.
        """
        name = (name or "").strip()
        if name in self._refs:
            return self._refs[name]
        for ref in self._refs.values():
            if ref.parent_ref is None and ref.source_table.lower() == name.lower():
                return ref
        raise KeyError(f"Data reference '{name}' not found. Available: {self.list_refs()[:10]}")

    def by_view(self, view_name: str) -> DataReference | None:
        """The reference backed by ``view_name``, if any."""
        return next((r for r in self._refs.values() if r.view_name == view_name), None)

    def roots(self) -> list[DataReference]:
        """References that are not derived from another reference."""
        return [r for r in self._refs.values() if r.parent_ref is None]

    def all(self) -> list[DataReference]:
        """All registered references."""
        return list(self._refs.values())

    def lineage(self, ref_id: str) -> list[str]:
        """Operations that produced a reference, root first."""
        chain: list[str] = []
        ref = self._refs.get(ref_id)
        while ref is not None:
            if ref.operation:
                chain.append(ref.operation)
            ref = self._refs.get(ref.parent_ref) if ref.parent_ref else None
        return chain[::-1]

    def list_refs(self) -> list[str]:
        """List all registered reference IDs."""
        return list(self._refs.keys())


class DataProfiler:
    """Generates Data Profiles from DuckDB tables/views and registers references.

    Args:
        connection: A DuckDB connection with the data already loaded.
        data_store: The DataStore to register new references in.
        sample_size: Default number of sample rows in a profile.
        max_cell_chars: Longer sample values are truncated to this length.
        max_columns: Columns beyond this many are summarized, not listed.
        masked_columns: Column names (case-insensitive) whose values are never shown.
    """

    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection,
        data_store: DataStore,
        *,
        sample_size: int = 5,
        max_cell_chars: int = 80,
        max_columns: int = 40,
        masked_columns: frozenset[str] | set[str] = frozenset(),
    ) -> None:
        self._conn = connection
        self._store = data_store
        self.sample_size = sample_size
        self.max_cell_chars = max_cell_chars
        self.max_columns = max_columns
        self.masked = {c.lower() for c in masked_columns}

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """The DuckDB connection this profiler reads from."""
        return self._conn

    def profile(
        self,
        table_name: str,
        filter_sql: str | None = None,
        sample_size: int | None = None,
        *,
        select_sql: str | None = None,
        parent_ref: str | None = None,
        operation: str | None = None,
        ordered: bool = False,
    ) -> DataProfile:
        """Profile a table or view, registering a reference for the result.

        ``table_name`` and ``select_sql`` must come from trusted code or have been
        validated (see ``datavault.query``). Profiling a view that already has a
        reference, with no filter or query, reuses that reference.

        Args:
            table_name: The DuckDB table or view to profile.
            filter_sql: Optional SQL WHERE clause applied to ``table_name``.
            sample_size: Sample rows to include (default: the profiler's setting).
            select_sql: Full SELECT defining the new reference (overrides ``filter_sql``).
            parent_ref: Parent reference ID; inferred from ``table_name`` when omitted.
            operation: Lineage description for the new reference.
            ordered: Whether ``select_sql`` has an ORDER BY samples should respect.
        """
        parent = self._store.get(parent_ref) if parent_ref else self._store.by_view(table_name)
        existing = self._store.by_view(table_name)
        if existing and not filter_sql and select_sql is None:
            ref = existing
        else:
            ref_id = f"ref_{uuid.uuid4().hex[:12]}"
            body = select_sql or (
                f"SELECT * FROM {table_name}" + (f" WHERE {filter_sql}" if filter_sql else "")
            )
            self._conn.execute(f"CREATE OR REPLACE VIEW {ref_id} AS {body}")
            if parent is None:
                operation = operation or f"source: {table_name}"
            elif not operation and filter_sql:
                operation = f"filter: {filter_sql}"
            ref = DataReference(
                ref_id=ref_id,
                view_name=ref_id,
                source_table=table_name,
                filter_sql=filter_sql,
                row_count=self._conn.execute(f"SELECT COUNT(*) FROM {ref_id}").fetchone()[0],
                parent_ref=parent.ref_id if parent else None,
                operation=operation,
                ordered=ordered,
            )
            self._store.add(ref)
        return self.describe(ref, sample_size)

    def describe(self, ref: DataReference, sample_size: int | None = None) -> DataProfile:
        """Build a profile for an existing reference without creating a new one."""
        schema = self._conn.execute(f"DESCRIBE {quote_ident(ref.view_name)}").fetchall()
        all_columns = [(row[0], row[1]) for row in schema]
        shown = all_columns[: self.max_columns]
        n = self.sample_size if sample_size is None else sample_size
        return DataProfile(
            ref_id=ref.ref_id,
            table_name=ref.source_table,
            row_count=ref.row_count,
            columns=shown,
            statistics=self._get_statistics(ref.view_name, shown, ref.row_count),
            sample_rows=self._get_sample_rows(ref.view_name, shown, max(0, n), ref.ordered),
            filter_applied=ref.filter_sql,
            lineage=self._store.lineage(ref.ref_id),
            hidden_columns=len(all_columns) - len(shown),
        )

    def _get_statistics(
        self, view_name: str, columns: list[tuple[str, str]], row_count: int
    ) -> list[dict[str, Any]]:
        """Compute all per-column statistics in a single scan of the view."""
        if not columns:
            return []
        exact = row_count <= _EXACT_DISTINCT_MAX_ROWS
        select: list[str] = []
        for i, (name, col_type) in enumerate(columns):
            q, kind = quote_ident(name), column_kind(col_type)
            select.append(f"COUNT({q}) AS nn{i}")
            if kind != "other":
                select.append(f"{'COUNT(DISTINCT ' + q + ')' if exact else 'approx_count_distinct(' + q + ')'} AS d{i}")
            if kind == "numeric":
                select.append(f"MIN({q}) AS mn{i}, MAX({q}) AS mx{i}, ROUND(AVG({q})::DOUBLE, 2) AS av{i}")
            elif kind == "temporal":
                select.append(f"MIN({q}) AS mn{i}, MAX({q}) AS mx{i}")
        cur = self._conn.execute(f"SELECT {', '.join(select)} FROM {quote_ident(view_name)}")
        row = dict(zip([d[0] for d in cur.description], cur.fetchone(), strict=True))

        stats: list[dict[str, Any]] = []
        for i, (name, col_type) in enumerate(columns):
            kind = column_kind(col_type)
            stat: dict[str, Any] = {
                "column": name,
                "nulls": row_count - row[f"nn{i}"],
                "distinct": row.get(f"d{i}", "N/A"),
                "min": "N/A", "max": "N/A", "mean": "N/A",
            }
            if kind in ("numeric", "temporal"):
                temporal = kind == "temporal"
                stat["min"] = str(row[f"mn{i}"]) if temporal else row[f"mn{i}"]
                stat["max"] = str(row[f"mx{i}"]) if temporal else row[f"mx{i}"]
                stat["mean"] = "N/A" if temporal else row[f"av{i}"]
            if name.lower() in self.masked:
                stat.update(min=MASK, max=MASK, mean=MASK, distinct=MASK)
            stats.append(stat)
        return stats

    def _get_sample_rows(
        self, view_name: str, columns: list[tuple[str, str]], n: int, ordered: bool
    ) -> list[dict[str, Any]]:
        """Fetch at most ``n`` rows: the first rows if the view is ordered, else a random sample."""
        if n == 0 or not columns:
            return []
        cols = ", ".join(quote_ident(c) for c, _ in columns)
        tail = f"LIMIT {n}" if ordered else f"USING SAMPLE {n} ROWS"
        result = self._conn.execute(f"SELECT {cols} FROM {quote_ident(view_name)} {tail}")
        names = [d[0] for d in result.description]
        return [self._safe_row(dict(zip(names, row, strict=True))) for row in result.fetchall()]

    def _safe_row(self, row: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in row.items():
            if key.lower() in self.masked:
                out[key] = MASK
            elif isinstance(value, str) and len(value) > self.max_cell_chars:
                out[key] = value[: self.max_cell_chars] + "…"
            else:
                out[key] = value
        return out


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def check_identifier(name: str) -> str:
    """Validate a plain SQL identifier (used for table names we interpolate)."""
    if not _IDENT.match(name or ""):
        raise ValueError(f"Invalid table name '{name}'. Use letters, digits and underscores.")
    return name
