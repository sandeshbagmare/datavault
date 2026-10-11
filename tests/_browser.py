"""Tiny headless-browser driver (Chrome DevTools Protocol) for rendering tests."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_CANDIDATES = [
    os.environ.get("DATAVAULT_TEST_BROWSER", ""),
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
]
BROWSER = next((p for p in _CANDIDATES if p and Path(p).exists()), None)


@dataclass
class PageResult:
    """What the browser saw after loading a page."""

    html: str = ""
    errors: list[str] = field(default_factory=list)
    console: list[str] = field(default_factory=list)
    values: dict[str, Any] = field(default_factory=dict)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _http_json(port: int, path: str, timeout: float = 20) -> Any:
    deadline = time.time() + timeout
    while True:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001 - browser still starting
            if time.time() > deadline:
                raise
            time.sleep(0.3)


async def _drive(port: int, url: str, probes: dict[str, str], settle: float) -> PageResult:
    import websockets

    result = PageResult()
    targets = _http_json(port, "/json")
    page = next(t for t in targets if t.get("type") == "page")
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
        counter = 0
        pending: dict[int, asyncio.Future[Any]] = {}

        async def reader() -> None:
            async for raw in ws:
                msg = json.loads(raw)
                if "id" in msg and msg["id"] in pending:
                    pending.pop(msg["id"]).set_result(msg)
                elif msg.get("method") == "Runtime.exceptionThrown":
                    d = msg["params"]["exceptionDetails"]
                    result.errors.append(d.get("exception", {}).get("description") or d.get("text", ""))
                elif msg.get("method") == "Runtime.consoleAPICalled":
                    result.console.append(" ".join(str(a.get("value", a.get("description", ""))) for a in msg["params"]["args"]))
                elif msg.get("method") == "Log.entryAdded":
                    result.console.append(msg["params"]["entry"].get("text", ""))

        async def call(method: str, **params: Any) -> dict[str, Any]:
            nonlocal counter
            counter += 1
            fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
            pending[counter] = fut
            await ws.send(json.dumps({"id": counter, "method": method, "params": params}))
            return (await asyncio.wait_for(fut, 60)).get("result", {})

        task = asyncio.create_task(reader())
        for domain in ("Page", "Runtime", "Log"):
            await call(f"{domain}.enable")
        await call("Page.navigate", url=url)
        await asyncio.sleep(settle)
        evaluate = lambda expr: call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)  # noqa: E731
        result.html = (await evaluate("document.documentElement.outerHTML"))["result"].get("value", "")
        for name, expr in probes.items():
            result.values[name] = (await evaluate(expr))["result"].get("value")
        task.cancel()
    return result


def render(path: Path, probes: dict[str, str] | None = None, settle: float = 4.0) -> PageResult:
    """Load ``path`` in a headless browser and return the final DOM, JS errors and probe values."""
    assert BROWSER, "no browser available"
    port = _free_port()
    profile = tempfile.mkdtemp(prefix="dv-browser-")
    proc = subprocess.Popen(
        [BROWSER, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
         f"--remote-debugging-port={port}", f"--user-data-dir={profile}", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        return asyncio.run(_drive(port, path.as_uri(), probes or {}, settle))
    finally:
        try:
            import websockets.sync.client as sync_client

            version = _http_json(port, "/json/version", timeout=3)
            with sync_client.connect(version["webSocketDebuggerUrl"]) as ws:
                ws.send(json.dumps({"id": 1, "method": "Browser.close"}))
                time.sleep(0.5)
        except Exception:  # noqa: BLE001 - best effort shutdown
            pass
        proc.kill()
