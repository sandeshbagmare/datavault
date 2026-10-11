# DataVault

Privacy-first data agent using LangGraph, DuckDB, and FastMCP.

## Architecture

DataVault safely handles massive datasets (10M+ rows) by ensuring the LLM only ever sees a lightweight **Data Profile** (~264 tokens), while tools operate on the full dataset server-side via DuckDB.

### Core Components

- **FastMCP Server** — Serves vehicle sales data (10M rows) via MCP protocol
- **Data Proxy Layer** — Intercepts MCP responses, replaces raw data with Data Profiles
- **3 Atomic Tools** — `profile_and_filter`, `generate_html_viz`, `generate_markdown`
- **LangGraph Agent** — Orchestrates the tools with state management
- **FastAPI Server** — Serves generated HTML visualizations for iframe embedding

## Use the tools in any agent (LangGraph Deep Agents, LangChain, custom)

```python
from datavault import DataContext, Policy, create_tools

ctx = DataContext(output_dir="out", policy=Policy(masked_columns=frozenset({"email"})))
ctx.mount_parquet("sales.parquet", "sales")      # or ctx.register_table("existing_duckdb_table")
tools = create_tools(ctx)                         # [profile_and_filter, generate_html_viz, generate_markdown]

from deepagents import create_deep_agent          # or langchain.agents.create_agent, create_react_agent, ...
agent = create_deep_agent(model=..., tools=tools)
```

Each `DataContext` is fully isolated (own DuckDB connection, references, policy, output folders), so one process can serve many agents. Tool results carry a machine-readable `artifact` (path, reference ids, lineage) on the `ToolMessage`.

- **`profile_and_filter`** - filter / project / group / aggregate / sort a reference; returns a bounded profile and a new reference id with full lineage. Optional guarded read-only `sql`.
- **`generate_html_viz`** - declarative `spec` (datasets + nested blocks: grid, row, tabs, kpis, chart, table, text, markdown, any theme, any native Plotly `figure` with `{"$col": "dataset.column"}` bindings), quick arguments, or raw `custom_html` / `js` / `css` with server-computed data at `window.DATAVAULT.data`.
- **`generate_markdown`** - any document from ordered `sections` bound to computed datasets (`{{dataset.column}}` placeholders), plus summary / breakdown / trend presets.

**Privacy:** the model never receives full data; only bounded profiles (a few sample rows, truncated cells, optional masked columns). Numbers in charts and reports are computed by DuckDB from the reference the model names.

**Custom code policy:** `Policy(allow_custom_code=True)` (default; env `DATAVAULT_ALLOW_CUSTOM_CODE=0` to disable) lets the agent write raw HTML/CSS/JS. Pages always carry a Content-Security-Policy that blocks network access (`allow_network_in_html` to relax) and are served with `Content-Security-Policy: sandbox`. Other switches: `allow_raw_sql`, `masked_columns` (`DATAVAULT_MASKED_COLUMNS=email,ssn`), `max_dataset_rows`, `plotly_js="inline"` for offline pages.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Generate test data (10M rows)
python mcp_servers/vehicle_sales/generate_data.py

# Start the MCP server
python mcp_servers/vehicle_sales/server.py

# Start the web UI
uvicorn datavault.server:app --reload
```

## Project Structure

```
datavault/
├── mcp_servers/vehicle_sales/   # FastMCP server + data generator
├── src/datavault/
│   ├── proxy/                   # Data Proxy (interceptor + profiler)
│   ├── tools/                   # 3 atomic tools
│   ├── agent.py                 # LangGraph agent
│   └── server.py                # FastAPI server
├── tests/
└── pyproject.toml
```
