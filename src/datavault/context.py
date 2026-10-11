"""Per-agent runtime context shared by the three DataVault tools.

A ``DataContext`` owns one DuckDB connection, one reference store, a ``Policy`` and the
output folders. Build one per agent/session and pass it to ``create_tools`` so that
concurrent agents never see each other's data or settings.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import duckdb

from datavault.proxy.profiler import (
    DataProfile,
    DataProfiler,
    DataReference,
    DataStore,
    check_identifier,
    quote_ident,
)
from datavault.query import QueryError, validate_select


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Policy:
    """What the tools may do and how much they may reveal.

    Attributes:
        allow_custom_code: Let the agent supply raw HTML/CSS/JS. Output is always
            confined by a Content-Security-Policy; serve it in a sandboxed iframe.
        allow_raw_sql: Let the agent supply its own SELECT (still read-only and
            restricted to registered references).
        allow_network_in_html: Let generated pages fetch from the network.
        default_sample_rows / max_sample_rows: Rows of real data shown to the LLM.
        max_cell_chars / max_profile_columns: Bounds on profile size.
        max_dataset_rows: Rows pulled into one rendered dataset (browser payload cap).
        query_timeout_s: Seconds before a query is interrupted.
        masked_columns: Column names whose values are never shown to the LLM.
        plotly_js: ``cdn`` (pinned version) or ``inline`` (self-contained, offline).
    """

    allow_custom_code: bool = True
    allow_raw_sql: bool = True
    allow_network_in_html: bool = False
    default_sample_rows: int = 5
    max_sample_rows: int = 20
    max_cell_chars: int = 80
    max_profile_columns: int = 40
    max_dataset_rows: int = 5000
    query_timeout_s: float = 60.0
    masked_columns: frozenset[str] = frozenset()
    plotly_js: Literal["cdn", "inline"] = "cdn"

    @classmethod
    def from_env(cls, **overrides: Any) -> Policy:
        """Policy from ``DATAVAULT_*`` environment variables, then ``overrides``."""
        masked = os.getenv("DATAVAULT_MASKED_COLUMNS", "")
        base = cls(
            allow_custom_code=_flag("DATAVAULT_ALLOW_CUSTOM_CODE", True),
            allow_raw_sql=_flag("DATAVAULT_ALLOW_RAW_SQL", True),
            allow_network_in_html=_flag("DATAVAULT_ALLOW_NETWORK", False),
            masked_columns=frozenset(c.strip().lower() for c in masked.split(",") if c.strip()),
        )
        return replace(base, **overrides)


@dataclass
class QueryResult:
    """Rows returned by a bounded query."""

    columns: list[str]
    types: list[str]
    rows: list[tuple[Any, ...]]
    truncated: bool = False


@dataclass
class DataContext:
    """Connection + references + policy + output folders for one agent session."""

    connection: duckdb.DuckDBPyConnection | None = None
    store: DataStore | None = None
    profiler: DataProfiler | None = None
    policy: Policy = field(default_factory=Policy)
    output_dir: str | Path = "output"
    viz_dir: str | Path | None = None
    report_dir: str | Path | None = None

    def __post_init__(self) -> None:
        if self.connection is None:
            self.connection = self.profiler.connection if self.profiler else duckdb.connect(":memory:")
        self.store = self.store if self.store is not None else DataStore()
        if self.profiler is None:
            self.profiler = DataProfiler(self.connection, self.store)
        p = self.policy
        self.profiler.sample_size = p.default_sample_rows
        self.profiler.max_cell_chars = p.max_cell_chars
        self.profiler.max_columns = p.max_profile_columns
        self.profiler.masked = {c.lower() for c in p.masked_columns} | self.profiler.masked
        self.viz_path = Path(self.viz_dir or Path(self.output_dir) / "viz")
        self.report_path = Path(self.report_dir or Path(self.output_dir) / "reports")
        self.lock = threading.RLock()

    # -- references -------------------------------------------------------------------

    def resolve(self, name: str | None) -> DataReference:
        """Resolve a reference ID or root table name. Never falls back to other data.

        An empty name resolves only when exactly one root dataset is registered.
        """
        assert self.store is not None
        if name and name.strip():
            return self.store.find(name)
        roots = self.store.roots()
        if len(roots) == 1:
            return roots[0]
        raise KeyError(f"data_ref is required. Available: {self.store.list_refs()[:10]}")

    def columns(self, ref: DataReference) -> list[tuple[str, str]]:
        """(name, type) of every column of a reference."""
        assert self.connection is not None
        with self.lock:
            rows = self.connection.execute(f"DESCRIBE {quote_ident(ref.view_name)}").fetchall()
        return [(r[0], r[1]) for r in rows]

    def allowed_tables(self) -> set[str]:
        """Names a SQL statement may reference."""
        assert self.store is not None
        names: set[str] = set()
        for ref in self.store.all():
            names.update((ref.ref_id, ref.view_name, ref.source_table))
        return names

    def register_table(self, name: str, *, profile: bool = False) -> DataProfile | DataReference:
        """Register an existing table/view as a root reference.

        With ``profile=False`` this is a cheap registration (row count only) and the
        reference ID equals ``name``; with ``profile=True`` a full profile is built.
        """
        assert self.connection is not None and self.store is not None and self.profiler is not None
        check_identifier(name)
        with self.lock:
            if profile:
                return self.profiler.profile(name)
            count = self.connection.execute(f"SELECT COUNT(*) FROM {quote_ident(name)}").fetchone()[0]
            ref = DataReference(ref_id=name, view_name=name, source_table=name, row_count=count, operation=f"source: {name}")
            self.store.add(ref)
            return ref

    def mount_parquet(self, path: str | Path, name: str, *, profile: bool = True) -> DataProfile | DataReference:
        """Expose a Parquet/CSV file as table ``name`` and register it as a root reference."""
        assert self.connection is not None
        check_identifier(name)
        escaped = str(Path(path).resolve()).replace("\\", "/").replace("'", "''")
        with self.lock:
            self.connection.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM '{escaped}'")
        return self.register_table(name, profile=profile)

    # -- queries ----------------------------------------------------------------------

    def validate(self, sql: str) -> str:
        """Check ``sql`` is one read-only SELECT over registered references."""
        assert self.connection is not None
        with self.lock:
            return validate_select(self.connection, sql, self.allowed_tables())

    def run_select(self, sql: str, max_rows: int | None = None, *, validated: bool = False) -> QueryResult:
        """Run a read-only query and return at most ``max_rows`` rows.

        Raises:
            QueryError: If the SQL is not allowed, fails, or exceeds the timeout.
        """
        assert self.connection is not None
        cap = min(max_rows or self.policy.max_dataset_rows, self.policy.max_dataset_rows)
        with self.lock:
            statement = sql.strip().rstrip(";") if validated else self.validate(sql)
            timer = threading.Timer(self.policy.query_timeout_s, self.connection.interrupt)
            timer.start()
            try:
                cur = self.connection.execute(f"SELECT * FROM ({statement}) AS _dv LIMIT {cap + 1}")
                columns = [d[0] for d in cur.description]
                types = [str(d[1]) for d in cur.description]
                rows = cur.fetchall()
            except duckdb.InterruptException as exc:
                raise QueryError(f"Query exceeded {self.policy.query_timeout_s:g}s and was stopped.") from exc
            except duckdb.Error as exc:
                raise QueryError(f"Query failed: {str(exc).splitlines()[0]}") from exc
            finally:
                timer.cancel()
        return QueryResult(columns, types, rows[:cap], truncated=len(rows) > cap)


_default: DataContext | None = None
_default_lock = threading.Lock()


def default_context() -> DataContext:
    """Process-wide context used by the module-level tools (``generate_html_viz`` etc.).

    Reads policy from ``DATAVAULT_*`` variables and mounts the dataset named by
    ``PARQUET_PATH`` (or the bundled demo Parquet file, if present) as ``vehicle_sales``.
    Prefer ``create_tools(DataContext(...))`` for anything beyond a quick start.
    """
    global _default
    with _default_lock:
        if _default is None:
            ctx = DataContext(policy=Policy.from_env())
            root = Path(__file__).resolve().parents[2]
            for candidate in (
                os.getenv("PARQUET_PATH"),
                root / "mcp_servers" / "vehicle_sales" / "data" / "vehicle_sales.parquet",
                Path.cwd() / "mcp_servers" / "vehicle_sales" / "data" / "vehicle_sales.parquet",
            ):
                if candidate and Path(candidate).exists():
                    ctx.mount_parquet(candidate, "vehicle_sales", profile=False)
                    break
            _default = ctx
        return _default
