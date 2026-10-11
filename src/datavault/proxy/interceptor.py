"""Data proxy interceptor — intercepts MCP tool responses containing tabular data.

Sits between the MCP server and the LLM. When an MCP tool returns raw tabular data,
the interceptor stores it in DuckDB, registers a DataReference, and returns a
compact DataProfile to the LLM instead. This is the key mechanism that prevents
the LLM from seeing raw data.
"""

from __future__ import annotations

import json
import re

import duckdb
import pandas as pd

from datavault.proxy.profiler import DataProfile, DataProfiler, DataStore


class DataProxyInterceptor:
    """Intercepts MCP tool call responses and replaces raw data with Data Profiles.

    This class wraps any tool that returns tabular data (as markdown tables,
    CSV, or JSON) and substitutes the response with a DataProfile.

    Args:
        db_path: Path to the DuckDB database file (or ":memory:" for in-memory).
    """

    # Tools whose output should be intercepted (returns tabular data)
    INTERCEPT_TOOLS: set[str] = {"query_data", "sample_data", "aggregate_data"}

    # Tools whose output passes through unmodified (metadata only)
    PASSTHROUGH_TOOLS: set[str] = {"list_tables", "describe_table"}

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = duckdb.connect(db_path)
        self._data_store = DataStore()
        self._profiler = DataProfiler(self._conn, self._data_store)
        self._table_counter = 0

    @property
    def data_store(self) -> DataStore:
        """Access the underlying data store for reference lookups."""
        return self._data_store

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Access the underlying DuckDB connection."""
        return self._conn

    @property
    def profiler(self) -> DataProfiler:
        """Access the underlying DataProfiler."""
        return self._profiler

    def intercept(self, tool_name: str, tool_output: str) -> str:
        """Intercept a tool's output and optionally replace it with a DataProfile.

        Args:
            tool_name: The name of the MCP tool that produced the output.
            tool_output: The raw string output from the tool.

        Returns:
            Either the original output (for passthrough tools) or a DataProfile
            text representation (for intercepted tools).
        """
        if tool_name in self.PASSTHROUGH_TOOLS:
            return tool_output

        if tool_name not in self.INTERCEPT_TOOLS:
            return tool_output

        # Try to parse the output as tabular data
        df = self._parse_tabular_output(tool_output)
        if df is None or df.empty:
            return tool_output

        # Store in DuckDB
        table_name = self._store_dataframe(df)

        # Generate and return the profile
        profile = self._profiler.profile(table_name)
        return profile.to_llm_text()

    def register_parquet(self, parquet_path: str, table_name: str) -> DataProfile:
        """Register a Parquet file directly as a data source.

        This is used when the MCP server provides a file path rather than
        inline data. The Parquet file is registered as a DuckDB view.

        Args:
            parquet_path: Path to the Parquet file.
            table_name: Name to use for the DuckDB view.

        Returns:
            A DataProfile for the registered data.
        """
        self._conn.execute(
            f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM '{parquet_path}'"
        )
        return self._profiler.profile(table_name)

    def _store_dataframe(self, df: pd.DataFrame) -> str:
        """Store a pandas DataFrame in DuckDB and return the table name.

        Args:
            df: The DataFrame to store.

        Returns:
            The DuckDB table name where the data was stored.
        """
        self._table_counter += 1
        table_name = f"intercepted_{self._table_counter}"
        self._conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
        return table_name

    def _parse_tabular_output(self, output: str) -> pd.DataFrame | None:
        """Try to parse tool output as tabular data.

        Supports markdown tables, JSON arrays, and CSV.

        Args:
            output: The raw tool output string.

        Returns:
            A pandas DataFrame if parsing succeeded, None otherwise.
        """
        # Try JSON first
        df = self._try_parse_json(output)
        if df is not None:
            return df

        # Try markdown table
        df = self._try_parse_markdown_table(output)
        if df is not None:
            return df

        # Try CSV
        df = self._try_parse_csv(output)
        if df is not None:
            return df

        return None

    def _try_parse_json(self, output: str) -> pd.DataFrame | None:
        """Attempt to parse output as a JSON array of objects."""
        try:
            data = json.loads(output)
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                return pd.DataFrame(data)
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        return None

    def _try_parse_markdown_table(self, output: str) -> pd.DataFrame | None:
        """Attempt to parse output as a markdown table."""
        lines = output.strip().split("\n")
        # Look for lines with | separators
        table_lines = [line for line in lines if "|" in line]
        if len(table_lines) < 3:
            return None

        try:
            # Extract headers
            header_line = table_lines[0]
            headers = [h.strip() for h in header_line.split("|") if h.strip()]
            if not headers:
                return None

            # Skip separator line (---), parse data rows
            data_rows: list[dict[str, str]] = []
            for line in table_lines[2:]:
                values = [v.strip() for v in line.split("|") if v.strip()]
                if len(values) == len(headers):
                    # Skip separator lines
                    if all(re.match(r"^-+$", v) or re.match(r"^:?-+:?$", v) for v in values):
                        continue
                    data_rows.append(dict(zip(headers, values)))

            if data_rows:
                return pd.DataFrame(data_rows)
        except (ValueError, IndexError):
            pass
        return None

    def _try_parse_csv(self, output: str) -> pd.DataFrame | None:
        """Attempt to parse output as CSV data."""
        import io

        try:
            lines = output.strip().split("\n")
            if len(lines) < 2:
                return None
            # Check if first line has commas (CSV-like)
            if "," not in lines[0]:
                return None
            return pd.read_csv(io.StringIO(output))
        except (pd.errors.ParserError, ValueError):
            pass
        return None
