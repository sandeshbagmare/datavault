"""Compile a declarative document spec into one self-contained HTML page.

The model describes *what* to show (blocks bound to named datasets); this module owns
*how* it is drawn, so pages are safe, consistent and cheap to produce. Raw HTML/CSS/JS
is available as an explicit, policy-gated escape hatch and always runs under a
Content-Security-Policy.
"""

from __future__ import annotations

import html as _html
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

from markdown_it import MarkdownIt

from datavault.context import DataContext
from datavault.datasets import Dataset, fill_placeholders, format_value, jsonable, load_datasets
from datavault.query import QueryError
from datavault.render.figures import SIMPLE_TYPES, build_figure
from datavault.render.theme import Theme, resolve_theme

CDN_HOSTS = ("https://cdn.plot.ly", "https://cdn.jsdelivr.net", "https://unpkg.com", "https://cdnjs.cloudflare.com")
_FONT_STYLE, _FONT_FILES = "https://fonts.googleapis.com", "https://fonts.gstatic.com"
_MD = MarkdownIt("commonmark", {"html": False, "linkify": False}).enable("table")
_CSS_PROP = re.compile(r"^[a-zA-Z-]{2,40}$")
_CSS_VALUE = re.compile(r"^[^;{}<>\\]{0,200}$")
_CLASS = re.compile(r"[^A-Za-z0-9_ -]")

BLOCK_TYPES = (
    "row", "column", "grid", "section", "tabs", "text", "markdown", "kpis", "table", "chart", "html", "divider",
)


def esc(value: Any) -> str:
    """HTML-escape any value."""
    return _html.escape(str(value), quote=True)


@lru_cache(maxsize=1)
def _plotly_version() -> str:
    from plotly.offline.offline import get_plotlyjs_version

    return get_plotlyjs_version()


@lru_cache(maxsize=1)
def _plotly_inline() -> str:
    from plotly.offline import get_plotlyjs

    return get_plotlyjs()


def js_json(value: Any) -> str:
    """JSON that is safe inside a ``<script>`` element."""
    try:
        text = json.dumps(value, default=jsonable, allow_nan=False)
    except ValueError as exc:
        raise QueryError(f"Value cannot be serialized to JSON (NaN/Infinity?): {exc}") from exc
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


@dataclass
class RenderedPage:
    """A compiled page plus facts for the tool's reply."""

    html: str
    label: str
    datasets: dict[str, Dataset]
    warnings: list[str] = field(default_factory=list)
    blocks: int = 0
    charts: int = 0

    @property
    def points(self) -> int:
        """Total rows delivered to the browser."""
        return sum(len(d.rows) for d in self.datasets.values())


def csp(allow_network: bool) -> str:
    """Content-Security-Policy for generated pages."""
    cdn = " ".join(CDN_HOSTS)
    if allow_network:
        return (
            "default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval' https:; style-src 'unsafe-inline' https:; "
            "img-src data: blob: https:; font-src data: https:; connect-src https:; media-src data: blob: https:; "
            "base-uri 'none'; form-action 'none'"
        )
    return (
        f"default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval' {cdn}; "
        f"style-src 'unsafe-inline' {_FONT_STYLE}; img-src data: blob:; font-src data: {_FONT_FILES}; "
        "connect-src 'none'; base-uri 'none'; form-action 'none'"
    )


def plotly_script(policy_mode: str) -> str:
    """The <script> tag (or inline source) that loads plotly.js matching the installed Plotly."""
    if policy_mode == "inline":
        return f"<script>{_plotly_inline()}</script>"
    return f'<script src="{plotly_cdn_url()}" charset="utf-8"></script>'


class _Compiler:
    def __init__(self, ctx: DataContext, theme: Theme, datasets: dict[str, Dataset], default_dataset: str | None):
        self.ctx, self.theme, self.datasets = ctx, theme, datasets
        self.default_dataset = default_dataset
        self.figures: list[tuple[str, dict[str, Any]]] = []
        self.uses_code = False
        self.counts: dict[str, int] = {}
        self._ids = 0

    # -- helpers ----------------------------------------------------------------------

    def _id(self, prefix: str = "dv") -> str:
        self._ids += 1
        return f"{prefix}{self._ids}"

    def _attrs(self, node: dict[str, Any], base: str, extra_style: str = "") -> str:
        classes = " ".join(filter(None, [base, _CLASS.sub("", str(node.get("class", "")))]))
        styles = [extra_style] if extra_style else []
        style = node.get("style")
        if isinstance(style, dict):
            for prop, value in style.items():
                if not _CSS_PROP.match(str(prop)) or not _CSS_VALUE.match(str(value)):
                    raise QueryError(f"Unsupported CSS declaration {prop!r}: {value!r}")
                low = str(value).lower()
                if "expression(" in low or "javascript:" in low or ("url(" in low and not self.ctx.policy.allow_custom_code):
                    raise QueryError(f"CSS value for {prop!r} is not allowed (url() requires custom code to be enabled).")
                styles.append(f"{prop}:{value}")
        if node.get("span"):
            styles.append(f"grid-column:span {int(node['span'])}")
        if node.get("height") and node.get("type") not in ("chart",):
            styles.append(f"min-height:{int(node['height'])}px")
        ident = f' id="{esc(_CLASS.sub("", str(node["id"])))}"' if node.get("id") else ""
        return f'{ident} class="{esc(classes)}"' + (f' style="{esc(";".join(styles))}"' if styles else "")

    def _text(self, text: Any) -> str:
        return fill_placeholders(str(text), self.datasets)

    # -- dispatch ---------------------------------------------------------------------

    def block(self, node: Any) -> str:
        if node is None:
            return ""
        if isinstance(node, list):
            return "".join(self.block(n) for n in node)
        if isinstance(node, str):
            node = {"type": "markdown", "text": node}
        if not isinstance(node, dict):
            raise QueryError(f"A block must be an object, got {type(node).__name__}.")
        kind = str(node.get("type", "")).lower()
        if kind == "heading":
            node = {**node, "variant": node.get("variant") or f"h{int(node.get('level', 2))}"}
        kind = {"stack": "column", "card": "section", "kpi": "kpis", "heading": "text"}.get(kind, kind)
        if kind in SIMPLE_TYPES:
            node, kind = {**node, "chart": kind}, "chart"
        if kind not in BLOCK_TYPES:
            raise QueryError(
                f"Unknown block type '{node.get('type')}'. Block types: {list(BLOCK_TYPES)}; "
                f"chart types ({list(SIMPLE_TYPES)}) can also be used directly as the block type."
            )
        self.counts[kind] = self.counts.get(kind, 0) + 1
        return getattr(self, f"_b_{kind}")(node)

    def _children(self, node: dict[str, Any]) -> str:
        return self.block(node.get("children") if "children" in node else node.get("blocks"))

    # -- layout -----------------------------------------------------------------------

    def _b_row(self, node: dict[str, Any]) -> str:
        min_w = int(node.get("min_width", 280))
        return f'<div{self._attrs(node, "dv-row", f"--dv-min:{min_w}px")}>{self._children(node)}</div>'

    def _b_column(self, node: dict[str, Any]) -> str:
        return f'<div{self._attrs(node, "dv-col")}>{self._children(node)}</div>'

    def _b_grid(self, node: dict[str, Any]) -> str:
        layout = str(node.get("layout", "")).lower()
        cols = node.get("columns")
        if layout == "comparison":
            base, style = "layout-comparison", ""
        elif layout == "single":
            base, style = "layout-single", ""
        elif cols:
            template = f"repeat({int(cols)},minmax(0,1fr))" if str(cols).isdigit() else str(cols)
            if not _CSS_VALUE.match(template):
                raise QueryError("grid.columns must be a number or a CSS grid-template-columns value.")
            base, style = "dv-grid", f"grid-template-columns:{template}"
        else:
            base, style = "dashboard-grid", ""
        return f'<div{self._attrs(node, base, style)}>{self._children(node)}</div>'

    def _b_section(self, node: dict[str, Any]) -> str:
        title = f'<h3 class="dv-title">{esc(self._text(node["title"]))}</h3>' if node.get("title") else ""
        if node.get("collapsible"):
            summary = f"<summary>{esc(self._text(node.get('title', 'Details')))}</summary>"
            open_attr = "" if node.get("collapsed") else " open"
            return f'<details{self._attrs(node, "dv-card")}{open_attr}>{summary}{self._children(node)}</details>'
        return f'<section{self._attrs(node, "dv-card")}>{title}{self._children(node)}</section>'

    def _b_tabs(self, node: dict[str, Any]) -> str:
        tabs = node.get("tabs")
        if not isinstance(tabs, list) or not tabs:
            raise QueryError("tabs block needs 'tabs': [{'title': ..., 'children': [...]}].")
        group = self._id("tabs")
        buttons, panels = [], []
        for i, tab in enumerate(tabs):
            buttons.append(
                f'<button type="button" class="dv-tab{" active" if i == 0 else ""}" data-tab="{group}-{i}">'
                f'{esc(self._text(tab.get("title", f"Tab {i + 1}")))}</button>'
            )
            panels.append(f'<div class="dv-panel" id="{group}-{i}"{"" if i == 0 else " hidden"}>{self.block(tab.get("children"))}</div>')
        return f'<div{self._attrs(node, "dv-tabs")} data-tabs="{group}"><div class="dv-tabbar">{"".join(buttons)}</div>{"".join(panels)}</div>'

    def _b_divider(self, node: dict[str, Any]) -> str:
        return f"<hr{self._attrs(node, 'dv-hr')}>"

    # -- content ----------------------------------------------------------------------

    def _b_text(self, node: dict[str, Any]) -> str:
        variant = str(node.get("variant") or node.get("level") or "p").lower()
        variant = {"1": "h1", "2": "h2", "3": "h3", "4": "h4", "heading": "h2", "title": "h1"}.get(variant, variant)
        tag = variant if variant in ("h1", "h2", "h3", "h4", "p") else ("pre" if variant == "code" else "p")
        extra = {"muted": "dv-muted", "callout": "dv-callout", "caption": "dv-muted dv-small"}.get(variant, "")
        return f'<{tag}{self._attrs(node, f"dv-text {extra}".strip())}>{esc(self._text(node.get("text", "")))}</{tag}>'

    def _b_markdown(self, node: dict[str, Any]) -> str:
        return f'<div{self._attrs(node, "dv-md")}>{_MD.render(self._text(node.get("text", "")))}</div>'

    def _dataset(self, name: Any) -> Dataset:
        name = name or self.default_dataset
        if name not in self.datasets:
            raise QueryError(f"Unknown dataset '{name}'. Datasets: {list(self.datasets)}. Define it under 'datasets'.")
        return self.datasets[name]

    def _b_kpis(self, node: dict[str, Any]) -> str:
        items = node.get("items")
        if items is None:
            items = [node]
        cards = []
        for item in items:
            if "value" in item and "dataset" not in item:
                raw = item["value"]
            else:
                ds = self._dataset(item.get("dataset"))
                field_name = item.get("field") or ds.columns[0]
                row = int(item.get("row", 0))
                raw = ds.rows[row][ds.columns.index(ds.canonical(field_name))] if row < len(ds.rows) else None
            value = f'{item.get("prefix", "")}{format_value(raw, item.get("format"))}{item.get("suffix", "")}'
            sub = self._text(item.get("subtext", ""))
            cards.append(
                '<div class="stat-card">'
                f'<div class="label">{esc(self._text(item.get("label", "Metric")))}</div>'
                f'<div class="value">{esc(value)}</div>'
                + (f'<div class="subtext">{esc(sub)}</div>' if sub else "")
                + "</div>"
            )
        return f'<div{self._attrs(node, "stat-cards")}>{"".join(cards)}</div>'

    def _b_table(self, node: dict[str, Any]) -> str:
        ds = self._dataset(node.get("dataset"))
        specs = node.get("columns") or ds.columns
        columns: list[tuple[str, str, str | None]] = []
        for c in specs:
            name = c.get("col") if isinstance(c, dict) else c
            label = (c.get("label") if isinstance(c, dict) else None) or ds.canonical(name)
            columns.append((ds.canonical(name), str(label), c.get("format") if isinstance(c, dict) else None))
        limit = int(node.get("max_rows", 50))
        index = [ds.columns.index(c) for c, _, _ in columns]
        head = "".join(f"<th>{esc(label)}</th>" for _, label, _ in columns)
        body = []
        for row in ds.rows[:limit]:
            cells = []
            for i, (_, _, fmt) in zip(index, columns, strict=True):
                v = row[i]
                num = isinstance(v, (int, float)) and not isinstance(v, bool)
                cells.append(f'<td{" class=num" if num else ""}>{esc(format_value(v, fmt))}</td>')
            body.append(f'<tr>{"".join(cells)}</tr>')
        title = f'<h3 class="dv-title">{esc(self._text(node["title"]))}</h3>' if node.get("title") else ""
        more = f'<div class="dv-muted dv-small">Showing {min(limit, len(ds.rows)):,} of {len(ds.rows):,} rows</div>' if len(ds.rows) > limit else ""
        return (
            f'<div{self._attrs(node, "chart-container")}>{title}<div class="table-widget"><table>'
            f'<thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>{more}</div>'
        )

    def _b_chart(self, node: dict[str, Any]) -> str:
        figure = build_figure(node, self.datasets, self.theme, self.default_dataset)
        div_id = self._id("plot")
        self.figures.append((div_id, figure))
        return f'<div{self._attrs(node, "chart-container")}><div id="{div_id}" class="dv-plot"></div></div>'

    def _b_html(self, node: dict[str, Any]) -> str:
        if not self.ctx.policy.allow_custom_code:
            raise QueryError("Raw HTML blocks are disabled by policy (allow_custom_code=False).")
        self.uses_code = True
        return f'<div{self._attrs(node, "dv-html")}>{node.get("html", "")}</div>'


_BASE_CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
:root { /*THEME*/ --gap: 20px; }
body { font-family: var(--font); background: var(--bg); color: var(--text-main); min-height: 100vh; display: flex; flex-direction: column; }
.header { background: var(--header-bg); color: var(--header-text); padding: 24px 32px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
.header h1 { font-size: 1.6em; font-weight: 700; margin-bottom: 6px; }
.header .sub { opacity: 0.85; margin-bottom: 8px; }
.header .meta { font-size: 0.85em; opacity: 0.9; display: flex; flex-wrap: wrap; gap: 12px; }
.header .meta span { background: rgba(255,255,255,0.15); padding: 2px 10px; border-radius: 12px; }
.content-area { padding: 24px 32px; flex: 1; display: flex; flex-direction: column; gap: var(--gap); }
.stat-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }
.stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; box-shadow: var(--card-shadow); }
.stat-card .label { font-size: 0.85em; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
.stat-card .value { font-size: 1.8em; font-weight: 700; color: var(--accent); line-height: 1.1; }
.stat-card .subtext { font-size: 0.8em; color: var(--text-muted); margin-top: 6px; }
.chart-container, .dv-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; box-shadow: var(--card-shadow); min-width: 0; }
.dv-card > summary { cursor: pointer; font-weight: 600; margin-bottom: 12px; }
.dv-title { font-size: 1.05em; margin-bottom: 12px; }
.dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(520px, 100%), 1fr)); gap: 24px; }
.layout-comparison { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
.layout-single { display: flex; flex-direction: column; gap: 24px; }
.dv-grid { display: grid; gap: 24px; }
.dv-row { display: flex; flex-wrap: wrap; gap: var(--gap); }
.dv-row > * { flex: 1 1 var(--dv-min, 280px); min-width: 0; }
.dv-col { display: flex; flex-direction: column; gap: var(--gap); }
.dv-plot { width: 100%; }
.dv-text { line-height: 1.5; }
.dv-muted { color: var(--text-muted); } .dv-small { font-size: 0.85em; }
.dv-callout { border-left: 4px solid var(--accent); background: var(--surface); padding: 12px 16px; border-radius: 6px; }
.dv-md { line-height: 1.6; } .dv-md table { border-collapse: collapse; margin: 10px 0; }
.dv-md th, .dv-md td { border: 1px solid var(--border); padding: 6px 10px; } .dv-md code, pre { background: var(--surface-hover); padding: 2px 6px; border-radius: 4px; }
.dv-md a { color: var(--accent); }
.dv-hr { border: 0; border-top: 1px solid var(--border); }
.dv-tabbar { display: flex; gap: 6px; border-bottom: 1px solid var(--border); margin-bottom: 16px; flex-wrap: wrap; }
.dv-tab { background: none; border: 0; color: var(--text-muted); padding: 10px 16px; cursor: pointer; font: inherit; border-bottom: 2px solid transparent; }
.dv-tab.active { color: var(--accent); border-bottom-color: var(--accent); }
.table-widget { overflow-x: auto; margin-top: 10px; }
.table-widget table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
.table-widget th, .table-widget td { padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--border); white-space: nowrap; }
.table-widget td.num { text-align: right; font-variant-numeric: tabular-nums; }
.table-widget th { background: var(--surface-hover); color: var(--text-muted); font-weight: 600; }
.table-widget tr:hover td { background: var(--surface-hover); }
.footer { text-align: center; padding: 20px; font-size: 0.85em; color: var(--text-muted); border-top: 1px solid var(--border); background: var(--surface); margin-top: auto; }
.dv-error { background: #7f1d1d; color: #fff; padding: 10px 16px; font-size: 0.9em; }
@media (max-width: 800px) { .layout-comparison { grid-template-columns: 1fr; } .header, .content-area { padding-left: 16px; padding-right: 16px; } }
"""

_RUNTIME_JS = """
(function () {
  var D = window.DATAVAULT;
  function fail(el, msg) { el.innerHTML = '<div style="color:#f87171;padding:12px">' + msg + '</div>'; }
  (D.figures || []).forEach(function (f) {
    var el = document.getElementById(f[0]);
    if (typeof Plotly === 'undefined') { return fail(el, 'plotly.js did not load (offline or blocked). Regenerate with plotly_js="inline".'); }
    try { Plotly.newPlot(el, f[1].data, f[1].layout, f[1].config).then(function () { if (f[1].frames) Plotly.addFrames(el, f[1].frames); }); }
    catch (e) { fail(el, 'Chart error: ' + e.message); }
  });
  document.querySelectorAll('[data-tabs]').forEach(function (box) {
    box.querySelectorAll('.dv-tab').forEach(function (btn) {
      btn.addEventListener('click', function () {
        box.querySelectorAll('.dv-tab').forEach(function (b) { b.classList.toggle('active', b === btn); });
        box.querySelectorAll(':scope > .dv-panel').forEach(function (p) { p.hidden = p.id !== btn.dataset.tab; });
        window.dispatchEvent(new Event('resize'));
      });
    });
  });
})();
"""


def _header(title: str, subtitle: str, chips: list[str]) -> str:
    sub = f'<div class="sub">{esc(subtitle)}</div>' if subtitle else ""
    return f'<div class="header"><h1>{esc(title)}</h1>{sub}<div class="meta">{"".join(f"<span>{c}</span>" for c in chips)}</div></div>'


def _check_libs(ctx: DataContext, libs: Any) -> list[str]:
    urls = [str(u) for u in (libs or [])]
    for url in urls:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if parsed.scheme != "https" or (origin not in CDN_HOSTS and not ctx.policy.allow_network_in_html):
            raise QueryError(f"Library '{url}' is not allowed. Use https URLs from: {list(CDN_HOSTS)}.")
    return urls


def plotly_cdn_url() -> str:
    """Versioned CDN URL matching the installed Plotly (``plotly-latest`` is frozen at 1.x)."""
    return f"https://cdn.plot.ly/plotly-{_plotly_version()}.min.js"


def wrap_custom_html(
    ctx: DataContext, source: str, *, title: str, theme: Theme, data: dict[str, Any]
) -> str:
    """Prepare model-written HTML: CSP, a working plotly.js, and ``window.DATAVAULT.data``.

    A full document gets the additions injected into its ``<head>``; a fragment is placed
    in a themed page. Raises ``QueryError`` if custom code is disabled by policy.
    """
    if not ctx.policy.allow_custom_code:
        raise QueryError("Raw HTML is disabled by policy (allow_custom_code=False). Use 'spec' with block types instead.")
    text = re.sub(r"https://cdn\.plot\.ly/plotly-latest(?:\.min)?\.js", plotly_cdn_url(), source)
    has_plotly_tag = re.search(r"<script[^>]+src=[^>]*plotly", text, re.IGNORECASE) is not None
    needs_plotly = "plotly" in text.lower() and not has_plotly_tag
    if "<html" not in text.lower():
        return page_shell(ctx, title=title, theme=theme, body=text, needs_plotly=needs_plotly, data=data, bare=True)
    bridge = {"data": data, "figures": [], "theme": {k: v for k, v in theme.tokens.items() if k != "dark"}}
    inject = (
        f'<meta http-equiv="Content-Security-Policy" content="{esc(csp(ctx.policy.allow_network_in_html))}">'
        + (plotly_script(ctx.policy.plotly_js) if needs_plotly else "")
        + f"<script>window.DATAVAULT = {js_json(bridge)};</script>"
    )
    if re.search(r"<head[^>]*>", text, re.IGNORECASE):
        return re.sub(r"(<head[^>]*>)", lambda m: m.group(1) + inject, text, count=1, flags=re.IGNORECASE)
    return re.sub(r"(<html[^>]*>)", lambda m: m.group(1) + f"<head>{inject}</head>", text, count=1, flags=re.IGNORECASE)


def page_shell(
    ctx: DataContext,
    *,
    title: str,
    theme: Theme,
    body: str,
    needs_plotly: bool,
    figures: list[tuple[str, dict[str, Any]]] | None = None,
    data: dict[str, Any] | None = None,
    css: str = "",
    js: str = "",
    libs: list[str] | None = None,
    header: str = "",
    footer: str = "",
    bare: bool = False,
) -> str:
    """Assemble a complete HTML document (shared by compiled specs and raw fragments).

    With ``bare=True`` the body is not wrapped in the padded content area.
    """
    bridge = {"data": data or {}, "figures": [[i, f] for i, f in (figures or [])], "theme": {k: v for k, v in theme.tokens.items() if k != "dark"}}
    user_js = (
        "<script>try {\n" + js.replace("</script", "<\\/script") + "\n} catch (e) { var b = document.createElement('div'); "
        "b.className = 'dv-error'; b.textContent = 'Script error: ' + e.message; document.body.prepend(b); }</script>"
        if js.strip() else ""
    )
    plotly = plotly_script(ctx.policy.plotly_js) if (needs_plotly or figures) else ""
    lib_tags = "".join(f'<script src="{esc(u)}"></script>' for u in libs or [])
    style = _BASE_CSS.replace("/*THEME*/", theme.css_vars()) + css.replace("</style", "<\\/style")
    main = body if bare else '<div class="content-area">' + body + "</div>"
    return (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        f'<meta http-equiv="Content-Security-Policy" content="{esc(csp(ctx.policy.allow_network_in_html))}">'
        f"<title>{esc(title)}</title>{plotly}{lib_tags}<style>{style}</style></head><body>"
        f"{header}{main}{footer}"
        f"<script>window.DATAVAULT = {js_json(bridge)};</script><script>{_RUNTIME_JS}</script>{user_js}</body></html>"
    )


def render_document(ctx: DataContext, spec: dict[str, Any], *, default_ref: str | None, title: str = "") -> RenderedPage:
    """Compile ``spec`` to HTML.

    Spec keys: ``title``, ``subtitle``, ``theme``, ``datasets`` (name -> dataset spec),
    ``body`` (a block or list of blocks), ``header``/``footer`` (bool), and the gated
    ``css``, ``js``, ``libs``. See ``BLOCK_TYPES``.

    Raises:
        QueryError: For anything invalid or disallowed; the message tells the model how to fix it.
    """
    if not isinstance(spec, dict):
        raise QueryError("spec must be an object with 'datasets' and 'body'.")
    code_keys = [k for k in ("css", "js", "libs") if spec.get(k)]
    if code_keys and not ctx.policy.allow_custom_code:
        raise QueryError(f"Custom code ({', '.join(code_keys)}) is disabled by policy (allow_custom_code=False).")
    theme = resolve_theme(spec.get("theme"))
    datasets = load_datasets(ctx, spec.get("datasets"), default_ref)
    default_dataset = next(iter(datasets)) if len(datasets) == 1 else spec.get("dataset")
    body_spec = spec.get("body", spec.get("blocks", spec.get("children")))
    if not body_spec:
        raise QueryError("spec needs a non-empty 'body': a block or list of blocks. Block types: " + str(list(BLOCK_TYPES)))

    compiler = _Compiler(ctx, theme, datasets, default_dataset)
    body = compiler.block(body_spec)
    leaves = {k: v for k, v in compiler.counts.items() if k in ("chart", "table", "html", "kpis", "markdown", "text")}
    single = sum(leaves.values()) == 1 and any(k in leaves for k in ("chart", "table", "html"))
    label = "Visualization" if single else "Dashboard"

    refs = list(dict.fromkeys(d.ref_id for d in datasets.values())) or ([ctx.resolve(default_ref).ref_id] if default_ref or len(ctx.store.roots()) == 1 else [])
    chips = []
    if refs:
        chips.append("Reference: " + ", ".join(f"<code>{esc(r)}</code>" for r in refs[:3]))
        if len(refs) == 1:
            ref = ctx.store.get(refs[0])
            chips.append(f"Dataset Rows: {ref.row_count:,}")
            if ref.filter_sql:
                chips.append(f"Filter: {esc(ref.filter_sql)}")
    chips.append(f"Theme: {esc(theme.name.capitalize())}")
    warnings = [w for d in datasets.values() for w in d.warnings()]
    page_title = str(spec.get("title") or title or "DataVault")
    header = _header(page_title, str(spec.get("subtitle", "")), chips) if spec.get("header", True) else ""
    notes = "".join(f"<div>{esc(w)}</div>" for w in warnings)
    footer = (
        f'<div class="footer">{notes}Generated by DataVault | Server-side DuckDB analytics | '
        f"{sum(len(d.rows) for d in datasets.values()):,} data points rendered</div>"
        if spec.get("footer", True) else ""
    )
    needs_plotly = "plotly" in str(spec.get("js", "")).lower() or "plotly" in body.lower()
    page = page_shell(
        ctx, title=page_title, theme=theme, body=body, needs_plotly=needs_plotly, figures=compiler.figures,
        data={n: d.records() for n, d in datasets.items()} if (compiler.uses_code or spec.get("js")) else {},
        css=str(spec.get("css", "")), js=str(spec.get("js", "")), libs=_check_libs(ctx, spec.get("libs")),
        header=header, footer=footer,
    )
    return RenderedPage(page, label, datasets, warnings, sum(compiler.counts.values()), len(compiler.figures))
