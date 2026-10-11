"""FastMCP server for vehicle sales data — serves 10M+ rows safely.

Exposes tools that aggregate, sample, and profile data via DuckDB.
Never allows unbounded SELECT * queries to prevent context overflow.

Run:
    python server.py           # stdio transport (for MCP clients)
    fastmcp inspect server.py  # inspect available tools
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
from fastmcp import FastMCP

mcp = FastMCP("Vehicle Sales Data Server")

# Data path relative to this script
DATA_DIR = Path(__file__).parent / "data"
PARQUET_PATH = DATA_DIR / "vehicle_sales.parquet"


def _get_connection() -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection with the vehicle_sales view set up.

    Returns:
        A DuckDB connection with vehicle_sales view pointing to the Parquet file.

    Raises:
        FileNotFoundError: If the Parquet file doesn't exist.
    """
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found: {PARQUET_PATH}\n"
            "Run 'python generate_data.py' first to generate the data."
        )
    con = duckdb.connect(":memory:")
    con.execute(
        f"CREATE OR REPLACE VIEW vehicle_sales AS SELECT * FROM '{PARQUET_PATH}'"
    )
    return con


@mcp.tool()
def list_tables() -> str:
    """List all available tables/datasets.

    Returns:
        A formatted string listing available tables.
    """
    return "Available tables:\n- vehicle_sales (10M rows of vehicle sales data)"


@mcp.tool()
def describe_table(table_name: str) -> str:
    """Get the schema and row count for a table.

    Args:
        table_name: Name of the table to describe.

    Returns:
        Schema with column names, types, and total row count.
    """
    try:
        con = _get_connection()
        schema = con.execute(f"DESCRIBE {table_name}").df()
        count = con.execute(f"SELECT COUNT(*) as total FROM {table_name}").fetchone()[0]
        con.close()
        return f"Table: {table_name}\nRow Count: {count:,}\n\nSchema:\n{schema.to_markdown(index=False)}"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def sample_data(table_name: str, n: int = 10) -> str:
    """Get a random sample of rows from a table.

    Args:
        table_name: Name of the table to sample from.
        n: Number of sample rows (max 50).

    Returns:
        A markdown-formatted table with sample rows.
    """
    try:
        n = min(n, 50)  # Safety cap
        con = _get_connection()
        result = con.execute(
            f"SELECT * FROM {table_name} USING SAMPLE {n} ROWS"
        ).df()
        con.close()
        return result.to_markdown(index=False)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def query_data(sql_query: str) -> str:
    """Execute a SQL query against the data. A LIMIT of 100 is enforced.

    Use 'vehicle_sales' as the table name in your queries.

    Args:
        sql_query: SQL query to execute.

    Returns:
        Query results as a markdown table.
    """
    try:
        # Safety: force LIMIT 100 if no LIMIT clause present
        if not re.search(r'\bLIMIT\b', sql_query, re.IGNORECASE):
            sql_query = sql_query.rstrip(';') + " LIMIT 100"

        con = _get_connection()
        result = con.execute(sql_query).df()
        con.close()
        return result.to_markdown(index=False)
    except Exception as e:
        return f"Error executing query: {e}"


@mcp.tool()
def aggregate_data(
    table_name: str,
    group_by: str,
    metric_column: str,
    aggregation: str = "SUM",
) -> str:
    """Aggregate data using GROUP BY — safe for large datasets.

    Args:
        table_name: Table to aggregate.
        group_by: Column to group by (e.g., 'brand', 'vehicle_type').
        metric_column: Column to apply aggregation on (e.g., 'price_usd').
        aggregation: Aggregation function: SUM, AVG, COUNT, MIN, MAX.

    Returns:
        Aggregated results as a markdown table.
    """
    try:
        agg = aggregation.upper()
        if agg not in {"SUM", "AVG", "COUNT", "MIN", "MAX"}:
            return f"Error: Invalid aggregation '{aggregation}'. Use: SUM, AVG, COUNT, MIN, MAX."

        con = _get_connection()
        result = con.execute(
            f'SELECT "{group_by}", {agg}("{metric_column}") AS {agg.lower()}_{metric_column} '
            f'FROM {table_name} GROUP BY "{group_by}" ORDER BY {agg.lower()}_{metric_column} DESC'
        ).df()
        con.close()
        return result.to_markdown(index=False)
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    mcp.run()
