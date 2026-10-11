"""FastAPI server — serves generated HTML visualizations and provides a minimal UI.

Endpoints:
- GET /              — Minimal chat UI for testing the agent
- GET /viz/{file_id} — Serves generated HTML visualizations (for iframe embedding)
- POST /api/query    — Submit a query to the DataVault agent
- GET /api/refs      — List active data references
- GET /api/files     — List generated files
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from langchain_core.messages import AIMessage
from markdown_it import MarkdownIt
from pydantic import BaseModel

# Lazy imports to avoid circular dependency at module level
_agent_graph = None
_interceptor = None
_initial_profile = None
_data_refs: list[str] = []
_generated_files: list[str] = []

app = FastAPI(title="DataVault", version="0.1.0")

_OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
# Generated pages may contain model-written scripts: serve them in an opaque origin.
_VIZ_HEADERS = {
    "Content-Security-Policy": "sandbox allow-scripts allow-downloads",
    "X-Content-Type-Options": "nosniff",
}
_REPORT_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; img-src data: https:",
    "X-Content-Type-Options": "nosniff",
}
_MARKDOWN = MarkdownIt("commonmark", {"html": False}).enable("table")


def _output_file(directory: Path, file_name: str, kind: str) -> Path:
    """Resolve ``file_name`` inside ``directory`` only (no path traversal)."""
    folder = directory.resolve()
    path = (folder / file_name).resolve()
    if Path(file_name).name != file_name or path.parent != folder or not path.is_file():
        raise HTTPException(status_code=404, detail=f"{kind} not found: {file_name}")
    return path


class QueryRequest(BaseModel):
    """Request body for the /api/query endpoint."""

    query: str
    data_refs: list[str] | None = None
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None


class QueryResponse(BaseModel):
    """Response body for the /api/query endpoint."""

    response: str
    data_refs: list[str]
    generated_files: list[str]


_current_agent_config: tuple[Any, ...] | None = None


def _get_agent(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> tuple[Any, Any]:
    """Get the active agent graph, creating or re-creating it if custom settings are provided."""
    global _agent_graph, _interceptor, _initial_profile, _data_refs, _current_agent_config

    has_custom = any([provider, model, api_key, base_url])
    requested_config = (provider, model, api_key, base_url) if has_custom else None

    # If already set (e.g., monkeypatched in tests or already initialized) and no new custom config
    if _agent_graph is not None:
        if not has_custom or requested_config == _current_agent_config:
            return _agent_graph, _interceptor

    from datavault.agent import create_agent

    parquet_path = os.getenv(
        "PARQUET_PATH",
        str(Path(__file__).parent.parent.parent / "mcp_servers" / "vehicle_sales" / "data" / "vehicle_sales.parquet"),
    )

    _agent_graph, _interceptor, _initial_profile = create_agent(
        parquet_path=parquet_path,
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        output_dir=Path(__file__).parent.parent.parent / "output",
    )
    _current_agent_config = requested_config

    if _initial_profile and _initial_profile.ref_id not in _data_refs:
        _data_refs.append(_initial_profile.ref_id)

    return _agent_graph, _interceptor


def _ensure_agent() -> None:
    """Lazily initialize the agent on first request using environment settings."""
    _get_agent()


# --- API Endpoints ---

@app.post("/api/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest) -> QueryResponse:
    """Submit a query to the DataVault agent with optional provider/base_url/api_key overrides."""
    agent_graph, _ = _get_agent(
        provider=request.provider,
        model=request.model,
        api_key=request.api_key,
        base_url=request.base_url,
    )

    from datavault.agent import run_query

    refs = request.data_refs or _data_refs
    result = run_query(agent_graph, request.query, data_refs=refs, generated_files=_generated_files)

    # Extract the final AI response
    messages = result.get("messages", [])
    ai_response = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            ai_response = msg.text
            break

    # Update tracked state
    new_refs = result.get("data_refs", [])
    new_files = result.get("generated_files", [])
    for ref in new_refs:
        if ref not in _data_refs:
            _data_refs.append(ref)
    for f in new_files:
        if f not in _generated_files:
            _generated_files.append(f)

    return QueryResponse(
        response=ai_response,
        data_refs=_data_refs,
        generated_files=_generated_files,
    )


@app.get("/api/refs")
async def list_refs() -> JSONResponse:
    """List all active data references."""
    _ensure_agent()
    return JSONResponse({"data_refs": _data_refs})


@app.get("/api/files")
async def list_files() -> JSONResponse:
    """List all generated files."""
    return JSONResponse({"generated_files": _generated_files})


@app.get("/viz/{file_name}")
async def serve_viz(file_name: str) -> HTMLResponse:
    """Serve a generated HTML visualization file."""
    file_path = _output_file(_OUTPUT_DIR / "viz", file_name, "Visualization")
    return HTMLResponse(content=file_path.read_text(encoding="utf-8"), headers=_VIZ_HEADERS)


@app.get("/report/{file_name}")
async def serve_report(file_name: str) -> HTMLResponse:
    """Serve a generated markdown report as HTML."""
    file_path = _output_file(_OUTPUT_DIR / "reports", file_name, "Report")
    body = _MARKDOWN.render(file_path.read_text(encoding="utf-8"))  # raw HTML in the source is escaped
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; line-height: 1.6; }}
table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #f5f5f5; }}
code, pre {{ background: #f5f5f5; padding: 2px 6px; border-radius: 4px; }}
img {{ max-width: 100%; }}
</style></head><body>{body}</body></html>"""
    return HTMLResponse(content=html, headers=_REPORT_HEADERS)


# --- Minimal Chat UI ---

_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataVault - Privacy-First Data Agent</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f0f;
            color: #e0e0e0;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px 25px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .header h1 {
            font-size: 1.3em;
            color: white;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .header .badge {
            background: rgba(255,255,255,0.2);
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.7em;
        }
        .main {
            flex: 1;
            display: flex;
            overflow: hidden;
        }
        .chat-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        .message {
            margin-bottom: 20px;
            max-width: 85%;
        }
        .message.user {
            margin-left: auto;
            background: #1a1a2e;
            border: 1px solid #333;
            border-radius: 16px 16px 4px 16px;
            padding: 12px 18px;
        }
        .message.assistant {
            background: #16213e;
            border: 1px solid #2a2a4a;
            border-radius: 16px 16px 16px 4px;
            padding: 12px 18px;
        }
        .message.assistant pre {
            background: #0a0a1a;
            padding: 10px;
            border-radius: 6px;
            overflow-x: auto;
            margin: 8px 0;
            font-size: 0.85em;
        }
        .message .role {
            font-size: 0.75em;
            color: #888;
            margin-bottom: 6px;
        }
        .input-area {
            padding: 15px 20px;
            background: #1a1a1a;
            border-top: 1px solid #333;
            display: flex;
            gap: 10px;
        }
        .input-area input {
            flex: 1;
            background: #2a2a2a;
            border: 1px solid #444;
            color: white;
            padding: 12px 18px;
            border-radius: 8px;
            font-size: 1em;
            outline: none;
        }
        .input-area input:focus { border-color: #667eea; }
        .input-area button {
            background: linear-gradient(135deg, #667eea, #764ba2);
            border: none;
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
        }
        .input-area button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .sidebar {
            width: 300px;
            background: #1a1a1a;
            border-left: 1px solid #333;
            padding: 20px;
            overflow-y: auto;
        }
        .sidebar h3 {
            color: #999;
            font-size: 0.8em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }
        .sidebar .ref {
            background: #2a2a2a;
            padding: 8px 12px;
            border-radius: 6px;
            margin-bottom: 6px;
            font-family: monospace;
            font-size: 0.8em;
            color: #667eea;
        }
        .sidebar .file-link {
            display: block;
            padding: 6px 10px;
            margin-bottom: 4px;
            background: #2a2a2a;
            border-radius: 4px;
            color: #aaa;
            text-decoration: none;
            font-size: 0.8em;
        }
        .sidebar .file-link:hover { background: #333; color: #667eea; }
        .viz-frame {
            width: 100%;
            border: none;
            border-radius: 8px;
            margin-top: 10px;
            background: white;
            min-height: 400px;
        }
        .loading { color: #667eea; font-style: italic; }
        .quick-actions {
            padding: 10px 20px;
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        .quick-actions button {
            background: #2a2a2a;
            border: 1px solid #444;
            color: #ccc;
            padding: 6px 14px;
            border-radius: 16px;
            cursor: pointer;
            font-size: 0.85em;
        }
        .quick-actions button:hover { border-color: #667eea; color: #667eea; }
        .settings-btn {
            background: rgba(255,255,255,0.15);
            border: 1px solid rgba(255,255,255,0.3);
            color: white;
            padding: 6px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85em;
            transition: all 0.2s;
        }
        .settings-btn:hover {
            background: rgba(255,255,255,0.25);
        }
        .modal {
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.7);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }
        .modal-content {
            background: #1e1e2f;
            border: 1px solid #444;
            border-radius: 12px;
            width: 90%;
            max-width: 520px;
            padding: 24px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.5);
            color: #eee;
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 18px;
        }
        .modal-header h2 { font-size: 1.2em; color: white; }
        .modal-header .close { cursor: pointer; font-size: 1.4em; color: #aaa; }
        .modal-header .close:hover { color: white; }
        .form-group { margin-bottom: 14px; text-align: left; }
        .form-group label { display: block; font-size: 0.85em; margin-bottom: 5px; color: #bbb; }
        .form-group select, .form-group input {
            width: 100%;
            padding: 9px 12px;
            background: #2a2a3e;
            border: 1px solid #4a4a6a;
            border-radius: 6px;
            color: white;
            font-size: 0.9em;
            outline: none;
            box-sizing: border-box;
        }
        .form-group select:focus, .form-group input:focus { border-color: #667eea; }
        .form-group small { display: block; font-size: 0.75em; color: #888; margin-top: 4px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>
            DataVault
            <span class="badge">Privacy-First</span>
        </h1>
        <div style="display: flex; align-items: center; gap: 15px;">
            <div style="color: rgba(255,255,255,0.7); font-size: 0.85em;">
                LLM sees ~214 tokens per dataset
            </div>
            <button onclick="toggleSettingsModal()" class="settings-btn">⚙️ Provider & Keys</button>
        </div>
    </div>

    <div id="settingsModal" class="modal" style="display:none;">
        <div class="modal-content">
            <div class="modal-header">
                <h2>LLM Provider & API Settings</h2>
                <span class="close" onclick="toggleSettingsModal()">&times;</span>
            </div>
            <div class="form-group">
                <label>Provider</label>
                <select id="settingProvider" onchange="onProviderSelect()">
                    <option value="">Default (from server .env)</option>
                    <option value="google">Google Gemini</option>
                    <option value="openai">OpenAI</option>
                    <option value="anthropic">Anthropic Claude</option>
                    <option value="openrouter">OpenRouter (openrouter.ai)</option>
                    <option value="groq">Groq (api.groq.com)</option>
                    <option value="deepseek">DeepSeek (api.deepseek.com)</option>
                    <option value="ollama">Ollama (localhost:11434)</option>
                    <option value="together">Together AI</option>
                    <option value="vllm">vLLM / LM Studio / Local</option>
                    <option value="custom">Custom (OpenAI-compatible / Any URL)</option>
                </select>
            </div>
            <div class="form-group">
                <label>Custom Base URL / API Base (Optional)</label>
                <input type="text" id="settingBaseUrl" placeholder="e.g. https://api.deepseek.com/v1 or https://openrouter.ai/api/v1">
                <small>Accepts any provider endpoint (api_base or base_url)</small>
            </div>
            <div class="form-group">
                <label>API Key</label>
                <input type="password" id="settingApiKey" placeholder="Paste your API key here (sk-... or AIza...)">
                <small>Saved only in your local browser storage</small>
            </div>
            <div class="form-group">
                <label>Model Name (Optional override)</label>
                <input type="text" id="settingModel" placeholder="e.g. gpt-4o, gemini-2.0-flash, deepseek-chat">
            </div>
            <div style="display: flex; justify-content: flex-end; gap: 10px; margin-top: 20px;">
                <button onclick="clearSettings()" style="background:#333; border:1px solid #555; color:#ccc; padding:8px 16px; border-radius:6px; cursor:pointer;">Reset</button>
                <button onclick="saveSettings()" style="background:linear-gradient(135deg, #667eea, #764ba2); border:none; color:white; padding:8px 18px; border-radius:6px; cursor:pointer; font-weight:600;">Save & Apply</button>
            </div>
        </div>
    </div>

    <div class="quick-actions">
        <button onclick="sendQuery('Show me a dashboard of the vehicle sales data')">Dashboard</button>
        <button onclick="sendQuery('Profile the data and filter for Tesla cars only')">Tesla Cars</button>
        <button onclick="sendQuery('Generate a trend report of average price over time')">Price Trend</button>
        <button onclick="sendQuery('Create a pie chart of sales by vehicle type')">Vehicle Types</button>
        <button onclick="sendQuery('Generate a breakdown report by brand with price stats')">Brand Breakdown</button>
    </div>

    <div class="main">
        <div class="chat-panel">
            <div class="messages" id="messages">
                <div class="message assistant">
                    <div class="role">DataVault Agent</div>
                    <div>Welcome! I'm DataVault, a privacy-first data analysis agent. I can analyze your vehicle sales dataset (10M rows) without ever seeing the raw data. Configure your API key/base_url via ⚙️ Provider & Keys, or use the quick actions above!</div>
                </div>
            </div>
            <div class="input-area">
                <input type="text" id="queryInput" placeholder="Ask about the data..." onkeypress="if(event.key==='Enter')sendQuery()">
                <button id="sendBtn" onclick="sendQuery()">Send</button>
            </div>
        </div>
        <div class="sidebar">
            <h3>Data References</h3>
            <div id="refs"><div class="ref">Loading...</div></div>

            <h3 style="margin-top: 20px;">Generated Files</h3>
            <div id="files"><span style="color: #666; font-size: 0.85em;">None yet</span></div>

            <h3 style="margin-top: 20px;">Live Preview</h3>
            <div id="preview"><span style="color: #666; font-size: 0.85em;">Send a query to generate visualizations</span></div>
        </div>
    </div>

    <script>
        function toggleSettingsModal() {
            const m = document.getElementById('settingsModal');
            m.style.display = m.style.display === 'none' ? 'flex' : 'none';
        }

        function loadSavedSettings() {
            document.getElementById('settingProvider').value = localStorage.getItem('dv_provider') || '';
            document.getElementById('settingBaseUrl').value = localStorage.getItem('dv_base_url') || '';
            document.getElementById('settingApiKey').value = localStorage.getItem('dv_api_key') || '';
            document.getElementById('settingModel').value = localStorage.getItem('dv_model') || '';
        }

        function saveSettings() {
            const p = document.getElementById('settingProvider').value.trim();
            const b = document.getElementById('settingBaseUrl').value.trim();
            const k = document.getElementById('settingApiKey').value.trim();
            const m = document.getElementById('settingModel').value.trim();

            if (p) localStorage.setItem('dv_provider', p); else localStorage.removeItem('dv_provider');
            if (b) localStorage.setItem('dv_base_url', b); else localStorage.removeItem('dv_base_url');
            if (k) localStorage.setItem('dv_api_key', k); else localStorage.removeItem('dv_api_key');
            if (m) localStorage.setItem('dv_model', m); else localStorage.removeItem('dv_model');

            toggleSettingsModal();
            addMessage('System', '⚙️ API & Model settings saved and will be applied to your queries.', 'assistant');
        }

        function clearSettings() {
            localStorage.removeItem('dv_provider');
            localStorage.removeItem('dv_base_url');
            localStorage.removeItem('dv_api_key');
            localStorage.removeItem('dv_model');
            loadSavedSettings();
            toggleSettingsModal();
            addMessage('System', '⚙️ Settings reset to server defaults (.env).', 'assistant');
        }

        function onProviderSelect() {
            const p = document.getElementById('settingProvider').value;
            const b = document.getElementById('settingBaseUrl');
            const m = document.getElementById('settingModel');
            if (p === 'openrouter' && !b.value) { b.value = 'https://openrouter.ai/api/v1'; if (!m.value) m.value = 'meta-llama/llama-3.3-70b-instruct'; }
            else if (p === 'groq' && !b.value) { b.value = 'https://api.groq.com/openai/v1'; if (!m.value) m.value = 'llama-3.3-70b-versatile'; }
            else if (p === 'deepseek' && !b.value) { b.value = 'https://api.deepseek.com/v1'; if (!m.value) m.value = 'deepseek-chat'; }
            else if (p === 'ollama' && !b.value) { b.value = 'http://localhost:11434/v1'; if (!m.value) m.value = 'llama3.2'; }
            else if (p === 'together' && !b.value) { b.value = 'https://api.together.xyz/v1'; }
            else if (p === 'vllm' && !b.value) { b.value = 'http://localhost:8000/v1'; }
        }

        async function sendQuery(text) {
            const input = document.getElementById('queryInput');
            const query = text || input.value.trim();
            if (!query) return;

            input.value = '';
            const btn = document.getElementById('sendBtn');
            btn.disabled = true;

            // Add user message
            addMessage('You', query, 'user');
            addMessage('DataVault Agent', '<span class="loading">Thinking...</span>', 'assistant', 'thinking');

            const payload = {
                query,
                provider: localStorage.getItem('dv_provider') || undefined,
                base_url: localStorage.getItem('dv_base_url') || undefined,
                api_key: localStorage.getItem('dv_api_key') || undefined,
                model: localStorage.getItem('dv_model') || undefined
            };

            try {
                const res = await fetch('/api/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                const data = await res.json();

                // Remove thinking message
                document.getElementById('thinking')?.remove();

                // Add response
                const formatted = esc(data.response || 'No response generated').replace(/```([\\s\\S]*?)```/g, '<pre>$1</pre>')
                    .replace(/\\n/g, '<br>');
                addMessage('DataVault Agent', formatted, 'assistant');

                // Update sidebar
                updateRefs(data.data_refs || []);
                updateFiles(data.generated_files || []);
            } catch (err) {
                document.getElementById('thinking')?.remove();
                addMessage('DataVault Agent', `Error: ${esc(err.message)}`, 'assistant');
            }
            btn.disabled = false;
        }

        function esc(s) {
            return String(s).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
        }

        function addMessage(role, content, type, id) {
            const div = document.createElement('div');
            div.className = `message ${type}`;
            if (id) div.id = id;
            div.innerHTML = `<div class="role">${role}</div><div>${content}</div>`;
            document.getElementById('messages').appendChild(div);
            div.scrollIntoView({behavior: 'smooth'});
        }

        function updateRefs(refs) {
            const el = document.getElementById('refs');
            el.innerHTML = refs.map(r => `<div class="ref">${esc(r)}</div>`).join('');
        }

        function updateFiles(files) {
            const el = document.getElementById('files');
            const preview = document.getElementById('preview');
            if (!files.length) return;

            el.innerHTML = files.map(f => {
                const name = f.split(/[\\\\/]/).pop();
                if (name.endsWith('.html')) {
                    return `<a class="file-link" href="/viz/${encodeURIComponent(name)}" target="_blank">${esc(name)}</a>`;
                }
                return `<a class="file-link" href="/report/${encodeURIComponent(name)}" target="_blank">${esc(name)}</a>`;
            }).join('');

            // Show latest HTML viz in preview
            const htmlFiles = files.filter(f => f.endsWith('.html'));
            if (htmlFiles.length) {
                const latest = htmlFiles[htmlFiles.length - 1].split(/[\\\\/]/).pop();
                preview.innerHTML = `<iframe class="viz-frame" sandbox="allow-scripts allow-downloads" src="/viz/${encodeURIComponent(latest)}"></iframe>`;
            }
        }

        // Initialize settings and refs on load
        loadSavedSettings();
        fetch('/api/refs').then(r => r.json()).then(d => updateRefs(d.data_refs)).catch(() => {});
    </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def serve_ui() -> HTMLResponse:
    """Serve the minimal chat UI."""
    return HTMLResponse(content=_UI_HTML)


def create_app() -> FastAPI:
    """Factory function for the FastAPI app.

    Returns:
        The configured FastAPI application.
    """
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
