# DataVault Handoff Document

**Date:** 2026-10-11
**Project Location:** `C:\Users\bagma\OneDrive\Desktop\tools\datavault`
**Python Environment:** `C:\Users\bagma\AppData\Local\Programs\Python\Python314\python.exe`
**Live Web Server:** `http://127.0.0.1:8000` (Currently running in background)

## Project Context
DataVault is a **privacy-first data agent framework** capable of querying massive datasets (10M+ rows) via MCP servers without exceeding LLM context windows or exposing raw sensitive data.

### Data Proxy Pattern
1. The backend stores the full 10M rows in a DuckDB/Parquet file (`mcp_servers/vehicle_sales/data/vehicle_sales.parquet`, 92.7 MB).
2. The LangGraph agent relies on a `ToolCallInterceptor`. When an MCP tool returns tabular data, the interceptor saves it to DuckDB, and replaces the LLM's context with a lightweight **Data Profile** (~214 tokens) containing schema, summary stats, and 5 sample rows.
3. The LLM interacts with opaque reference IDs (`ref_abc123`) using 3 atomic tools: `profile_and_filter`, `generate_html_viz`, and `generate_markdown`.
4. All aggregations and heavy lifting occur server-side in DuckDB. Visualizations (Plotly + Jinja2) and reports (Jinja2) are rendered into self-contained files served via FastAPI into the frontend iframe.

## Universal LLM Provider & Custom Base URL / API Base Support
The system supports **any** provider, API key, and custom base URL:
- **Commercial Providers**: OpenAI, Google Gemini, Anthropic Claude, OpenRouter (`https://openrouter.ai/api/v1`), Groq (`https://api.groq.com/openai/v1`), DeepSeek (`https://api.deepseek.com/v1`), Together AI (`https://api.together.xyz/v1`), Mistral, Cerebras, Fireworks.
- **Local / Self-Hosted**: Ollama (`http://localhost:11434/v1`), vLLM (`http://localhost:8000/v1`), LM Studio (`http://localhost:1234/v1`), LocalAI.
- **Custom Endpoints**: Any custom API base URL provided via `base_url` / `api_base` / `LLM_BASE_URL` / `API_BASE` / `OPENAI_API_BASE`.
- **UI Settings Modal**: Users can click the **⚙️ Provider & Keys** button on `http://127.0.0.1:8000` to configure their provider, base URL, API key, and model directly in the browser (persisted to `localStorage`).

## What Was Implemented & Tested (124/124 Tests Passing)
1. **Scaffolding & Architecture**: `pyproject.toml`, `.env.example`, `.env`, `.gitignore`, `README.md`, lazy package imports in `src/datavault/__init__.py`.
2. **Data Generator**: `mcp_servers/vehicle_sales/generate_data.py` generated 10,000,000 realistic rows in ~37 seconds into a 92.7 MB Parquet file.
3. **FastMCP Server**: `mcp_servers/vehicle_sales/server.py` exposes 5 tools (`list_tables`, `describe_table`, `sample_data`, `query_data` with enforced LIMIT 100, `aggregate_data`).
4. **Data Profiler & Interceptor**: `src/datavault/proxy/profiler.py` and `interceptor.py` parse raw MCP responses into bounded token summaries.
5. **Pluggable & Isolated Runtime Context**:
   - `DataContext` in `src/datavault/context.py`: Own DuckDB connection, isolated reference store, `Policy`, and separate output folders for concurrent agents.
   - `Policy`: Gates custom code (`allow_custom_code`), raw SQL (`allow_raw_sql`), network access (`allow_network_in_html`), and column masking (`masked_columns`).
   - `create_tools(ctx)`: Returns isolated instances of all 3 tools ready for `create_deep_agent(tools=...)` or LangChain agents.
6. **The 3 Core Tools**:
   - `generate_html_viz` in `src/datavault/tools/html_viz.py` & `render/html.py`: Declarative `spec` with nested layout blocks (grid, row, tabs, kpis, charts, tables, markdown), native Plotly `figure` with column bindings, raw custom HTML/CSS/JS with server-computed data, and backward-compatible quick arguments.
   - `generate_markdown` in `src/datavault/tools/markdown_report.py` & `render/markdown.py`: Composable ordered sections with server-interpolated `{{dataset.column}}` placeholders, plus summary, breakdown, and trend presets.
   - `profile_and_filter` in `src/datavault/tools/profile_filter.py`: Filtering, projection, group-by, aggregations, sorting, and row limits with full reference lineage and bounded sampling.
7. **Security & Guard Rails**:
   - Query guard in `src/datavault/query.py`: Read-only AST validation, CTE isolation, prevention of file read injections (`read_csv`, `read_parquet`, `glob`, etc.).
   - Sandboxed iframe and strict Content Security Policy (CSP) in `src/datavault/server.py`.
8. **LangGraph Agent & FastAPI UI**: Universal provider support (OpenAI, Gemini, Claude, OpenRouter, Groq, DeepSeek, Ollama, custom base URL), chat UI, and dynamic provider configuration modal.

## How to Test and Run
1. The server is currently active on **`http://127.0.0.1:8000`**. Open it in your browser!
2. Click **⚙️ Provider & Keys** in the header to enter your API key (and optionally your custom Base URL if using OpenRouter, Groq, DeepSeek, Ollama, etc.).
3. Click any of the quick-action buttons (e.g. "Dashboard", "Tesla Cars", "Price Trend") or send your own message to test.
