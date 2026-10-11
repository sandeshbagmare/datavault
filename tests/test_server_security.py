"""Generated files are served from an opaque origin, reports are escaped, traversal is blocked."""

from fastapi.testclient import TestClient

from datavault import server


def client(tmp_path, monkeypatch):
    (tmp_path / "viz").mkdir(exist_ok=True)
    (tmp_path / "reports").mkdir(exist_ok=True)
    monkeypatch.setattr(server, "_OUTPUT_DIR", tmp_path)
    return TestClient(server.app)


def test_viz_pages_are_served_sandboxed(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    (tmp_path / "viz" / "dashboard_1.html").write_text("<html>ok</html>", encoding="utf-8")
    r = c.get("/viz/dashboard_1.html")
    assert r.status_code == 200 and r.text == "<html>ok</html>"
    assert r.headers["content-security-policy"].startswith("sandbox allow-scripts")
    assert "allow-same-origin" not in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"


def test_path_traversal_is_rejected(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    (tmp_path / "secret.txt").write_text("top secret", encoding="utf-8")
    for url in ("/viz/..%2Fsecret.txt", "/viz/%2e%2e/secret.txt", "/viz/..%5Csecret.txt", "/report/..%2Fsecret.txt", "/viz/nope.html"):
        r = c.get(url)
        assert r.status_code == 404 and "top secret" not in r.text, url


def test_reports_render_markdown_and_never_run_embedded_html(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    (tmp_path / "reports" / "r.md").write_text(
        "# Title\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n<script>alert(1)</script> <img src=x onerror=alert(2)> [x](javascript:alert(3))\n",
        encoding="utf-8",
    )
    r = c.get("/report/r.md")
    assert r.status_code == 200 and "<h1>Title</h1>" in r.text and "<table>" in r.text
    assert "<script>" not in r.text and "<img" not in r.text and 'href="javascript' not in r.text
    assert "default-src 'none'" in r.headers["content-security-policy"]


def test_ui_escapes_model_output_and_sandboxes_the_preview():
    ui = TestClient(server.app).get("/").text
    assert 'sandbox="allow-scripts allow-downloads"' in ui and "function esc(" in ui
    assert "esc(data.response" in ui and "encodeURIComponent(latest)" in ui
