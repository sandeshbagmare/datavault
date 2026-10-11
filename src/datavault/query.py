"""Read-only SQL validation and a structured query builder.

Every query that touches data on behalf of an agent goes through here:

* ``validate_select`` parses SQL with DuckDB's own parser and accepts exactly one
  SELECT whose tables are all registered references (or CTEs). File readers,
  environment access and catalog functions are rejected.
* ``build_select`` turns a small declarative spec (filter / group / metrics / order)
  into SQL with validated, quoted identifiers, so most requests never need raw SQL.

This is a disclosure control, not a complete sandbox: run untrusted workloads in a
separate process or database when you need hard isolation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import duckdb

from datavault.proxy.profiler import quote_ident


class QueryError(ValueError):
    """Raised for invalid or disallowed queries. The message is safe to show to an LLM."""


_TABLE_FUNCTIONS = {"range", "generate_series", "unnest"}
_BLOCKED_PREFIXES = ("read_", "duckdb_", "pragma_", "parquet_", "iceberg_", "delta_", "st_read", "sniff_")
_BLOCKED_SUFFIXES = ("_scan",)
_BLOCKED_NAMES = {
    "glob", "getenv", "current_setting", "getvariable", "query", "query_table",
    "load_extension", "list_files", "write_csv", "write_parquet",
}

GRAINS = ("second", "minute", "hour", "day", "week", "month", "quarter", "year", "decade")
_AGGS = {
    "count": "COUNT", "sum": "SUM", "avg": "AVG", "mean": "AVG", "min": "MIN", "max": "MAX",
    "median": "MEDIAN", "stddev": "STDDEV", "variance": "VARIANCE",
}
_METRIC_STR = re.compile(r"^\s*(\w+)\s*\(\s*(\*|[^)]*?)\s*\)\s*(?:as\s+(\w+))?\s*$", re.IGNORECASE)


def validate_select(conn: duckdb.DuckDBPyConnection, sql: str, allowed_tables: set[str]) -> str:
    """Return ``sql`` (without trailing semicolons) if it is one read-only SELECT.

    Args:
        conn: Connection used only to parse the statement.
        sql: Candidate SQL.
        allowed_tables: Table/view names that may be referenced (case-insensitive).

    Raises:
        QueryError: If the statement is not a single allowed SELECT.
    """
    sql = (sql or "").strip().rstrip(";").strip()
    if not sql:
        raise QueryError("Empty SQL.")
    raw = conn.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0]
    parsed = json.loads(raw)
    if parsed.get("error"):
        msg = str(parsed.get("error_message", "unsupported statement")).splitlines()[0]
        raise QueryError(f"Only a single read-only SELECT is allowed ({msg}).")
    if len(parsed.get("statements", [])) != 1:
        raise QueryError("Only a single SELECT statement is allowed.")

    allowed = {t.lower() for t in allowed_tables}
    problems: list[str] = []

    def visit(node: Any, ctes: frozenset[str]) -> None:
        # A CTE name is visible only to later CTEs and the query body, never to its own
        # definition: otherwise `WITH secret AS (SELECT * FROM secret) ...` would read a real table.
        if isinstance(node, list):
            for item in node:
                visit(item, ctes)
            return
        if not isinstance(node, dict):
            return
        for entry in (node.get("cte_map") or {}).get("map", []):
            visit(entry.get("value"), ctes)
            ctes = ctes | {str(entry.get("key", "")).lower()}
        kind = node.get("type")
        if kind == "BASE_TABLE":
            name = str(node.get("table_name", "")).lower()
            if node.get("catalog_name") or node.get("schema_name") not in ("", "main", None):
                problems.append(f"table '{node.get('schema_name')}.{name}' is not allowed")
            elif name not in allowed and name not in ctes:
                problems.append(f"table '{node.get('table_name')}' is not a known data reference")
        elif kind == "TABLE_FUNCTION":
            name = str((node.get("function") or {}).get("function_name", "")).lower()
            if name not in _TABLE_FUNCTIONS:
                problems.append(f"table function '{name}' is not allowed")
        name = node.get("function_name")
        if isinstance(name, str):
            low = name.lower()
            if low in _BLOCKED_NAMES or low.startswith(_BLOCKED_PREFIXES) or low.endswith(_BLOCKED_SUFFIXES):
                problems.append(f"function '{name}' is not allowed")
        for key, value in node.items():
            if key != "cte_map":
                visit(value, ctes)

    visit(parsed["statements"], frozenset())
    if problems:
        known = sorted(allowed)[:10]
        raise QueryError("; ".join(dict.fromkeys(problems)) + f". Use only these references: {known}")
    return sql


@dataclass
class BuiltQuery:
    """A generated SELECT plus facts needed for lineage and sampling."""

    sql: str
    ordered: bool
    description: str
    aggregated: bool


class _Columns:
    """Case-insensitive column lookup with helpful errors."""

    def __init__(self, columns: list[tuple[str, str]]) -> None:
        self._by_lower = {c.lower(): c for c, _ in columns}
        self.names = [c for c, _ in columns]

    def get(self, name: str) -> str:
        found = self._by_lower.get(str(name).strip().lower())
        if found is None:
            raise QueryError(f"Unknown column '{name}'. Available columns: {self.names[:40]}")
        return found


def _parse_metric(item: Any) -> dict[str, Any]:
    if isinstance(item, str):
        m = _METRIC_STR.match(item)
        if not m:
            raise QueryError(f"Cannot parse metric '{item}'. Use {{'agg': 'AVG', 'col': 'price', 'as': 'avg_price'}}.")
        return {"agg": m.group(1), "col": None if m.group(2) == "*" else m.group(2), "as": m.group(3)}
    if isinstance(item, dict):
        return {"agg": item.get("agg", "count"), "col": item.get("col") or item.get("column"), "as": item.get("as")}
    raise QueryError(f"Invalid metric: {item!r}")


def _metric_sql(item: Any, cols: _Columns) -> tuple[str, str]:
    m = _parse_metric(item)
    agg = str(m["agg"]).strip().lower().replace(" ", "_")
    col = cols.get(m["col"]) if m["col"] else None
    if agg in ("count_distinct", "distinct", "nunique"):
        if not col:
            raise QueryError("count_distinct needs a column.")
        expr, alias = f"COUNT(DISTINCT {quote_ident(col)})", f"distinct_{col}"
    elif re.fullmatch(r"p\d{1,2}", agg):
        if not col:
            raise QueryError(f"{agg} needs a column.")
        expr, alias = f"quantile_cont({quote_ident(col)}, {int(agg[1:]) / 100})", f"{agg}_{col}"
    elif agg in _AGGS:
        fn = _AGGS[agg]
        if fn == "COUNT":
            expr = f"COUNT({quote_ident(col)})" if col else "COUNT(*)"
            alias = f"count_{col}" if col else "count"
        else:
            if not col:
                raise QueryError(f"{fn} needs a column.")
            expr, alias = f"{fn}({quote_ident(col)})", f"{agg}_{col}"
    else:
        raise QueryError(f"Unknown aggregation '{m['agg']}'. Use one of: {sorted(_AGGS)} + count_distinct, pNN.")
    alias = str(m["as"] or alias)[:64]
    return f"{expr} AS {quote_ident(alias)}", alias


def _order_item(item: Any) -> tuple[str, bool]:
    if isinstance(item, dict):
        return str(item.get("col") or item.get("column")), bool(item.get("desc")) or str(item.get("dir", "")).lower() == "desc"
    text = str(item).strip()
    if text.startswith("-"):
        return text[1:].strip(), True
    m = re.match(r"^(.*?)\s+(asc|desc)$", text, re.IGNORECASE)
    return (m.group(1).strip(), m.group(2).lower() == "desc") if m else (text, False)


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def build_select(view: str, columns: list[tuple[str, str]], spec: dict[str, Any]) -> BuiltQuery:
    """Compile a declarative query spec into a SELECT over ``view``.

    Spec keys (all optional): ``where`` (SQL predicate), ``select`` (columns),
    ``group_by`` (column names or ``{"col", "grain", "as"}``), ``metrics``
    (``{"agg", "col", "as"}`` or ``"AVG(price)"``), ``order_by``, ``limit``, ``sample``.

    Raises:
        QueryError: For unknown columns, aggregations or malformed items.
    """
    cols = _Columns(columns)
    where = str(spec.get("where") or spec.get("filter") or "").strip()
    group_items, metric_items = _as_list(spec.get("group_by")), _as_list(spec.get("metrics"))
    select_cols = [cols.get(c) for c in _as_list(spec.get("select") or spec.get("columns"))]
    aggregated = bool(group_items or metric_items)
    parts: list[str] = []
    aliases: list[str] = []
    time_alias: str | None = None
    notes: list[str] = []
    if where:
        notes.append(f"filter: {where}")

    if aggregated:
        for g in group_items:
            col = cols.get(g["col"] if isinstance(g, dict) else g)
            grain = str(g.get("grain", "")).lower() if isinstance(g, dict) else ""
            alias = str((g.get("as") if isinstance(g, dict) else None) or col)
            if grain:
                if grain not in GRAINS:
                    raise QueryError(f"Unknown grain '{grain}'. Use one of {list(GRAINS)}.")
                parts.append(f"DATE_TRUNC('{grain}', {quote_ident(col)}) AS {quote_ident(alias)}")
                time_alias = time_alias or alias
            else:
                parts.append(f"{quote_ident(col)} AS {quote_ident(alias)}")
            aliases.append(alias)
        group_count = len(parts)
        metric_aliases: list[str] = []
        for item in metric_items or ["count(*)"]:
            sql_part, alias = _metric_sql(item, cols)
            parts.append(sql_part)
            metric_aliases.append(alias)
        aliases.extend(metric_aliases)
        notes.append(
            "group_by: " + ", ".join(aliases[:group_count] or ["(all)"]) + "; metrics: " + ", ".join(metric_aliases)
        )
    else:
        chosen = select_cols or cols.names
        parts = [quote_ident(c) for c in chosen]
        aliases = list(chosen)
        if select_cols:
            notes.append("select: " + ", ".join(select_cols))

    sql = f"SELECT {', '.join(parts)} FROM {quote_ident(view)}"
    if where:
        sql += f" WHERE ({where})"
    if aggregated and group_count:
        sql += " GROUP BY " + ", ".join(str(i + 1) for i in range(group_count))
    sample = spec.get("sample")
    if sample and not aggregated:
        sql += f" USING SAMPLE {max(1, int(sample))} ROWS"
        notes.append(f"random sample: {int(sample)} rows")

    order = _as_list(spec.get("order_by") or spec.get("sort"))
    order_sql: list[str] = []
    for item in order:
        name, desc = _order_item(item)
        target = next((a for a in aliases if a.lower() == name.lower()), None) or cols.get(name)
        order_sql.append(f"{quote_ident(target)} {'DESC' if desc else 'ASC'}")
    if not order_sql and aggregated and group_count:
        order_sql = [f"{quote_ident(time_alias)} ASC"] if time_alias else [f"{quote_ident(aliases[group_count])} DESC"]
    if order_sql:
        sql += " ORDER BY " + ", ".join(order_sql)
        notes.append("order_by: " + ", ".join(order_sql))
    limit = spec.get("limit")
    if limit:
        sql += f" LIMIT {max(1, int(limit))}"
        notes.append(f"limit: {int(limit)}")
    return BuiltQuery(sql=sql, ordered=bool(order_sql), description="; ".join(notes) or "all rows", aggregated=aggregated)
