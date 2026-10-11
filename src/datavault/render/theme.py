"""Theme tokens shared by the HTML renderer and the chart builder."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from datavault.query import QueryError

_FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
_PALETTE = ["#6366f1", "#22d3ee", "#f59e0b", "#10b981", "#f43f5e", "#a78bfa", "#84cc16", "#fb923c", "#38bdf8", "#e879f9"]
_LIGHT_PALETTE = ["#4f46e5", "#0891b2", "#d97706", "#059669", "#e11d48", "#7c3aed", "#65a30d", "#ea580c", "#0284c7", "#c026d3"]

THEMES: dict[str, dict[str, Any]] = {
    "dark": {
        "bg": "#0f1117", "surface": "#181b24", "surface_hover": "#222634", "border": "#2a2e3d",
        "text": "#f1f5f9", "muted": "#94a3b8", "accent": "#6366f1",
        "header_bg": "linear-gradient(135deg, #4338ca 0%, #6d28d9 100%)", "header_text": "#ffffff",
        "shadow": "0 4px 14px rgba(0,0,0,0.35)", "colors": _PALETTE, "dark": True,
    },
    "midnight": {
        "bg": "#0b0f19", "surface": "#111827", "surface_hover": "#1f2937", "border": "#1f2937",
        "text": "#f9fafb", "muted": "#9ca3af", "accent": "#38bdf8",
        "header_bg": "linear-gradient(135deg, #0369a1 0%, #1e1b4b 100%)", "header_text": "#ffffff",
        "shadow": "0 4px 12px rgba(0,0,0,0.4)", "colors": _PALETTE, "dark": True,
    },
    "black": {
        "bg": "#000000", "surface": "#0a0a0a", "surface_hover": "#141414", "border": "#222222",
        "text": "#f5f5f5", "muted": "#a3a3a3", "accent": "#22d3ee",
        "header_bg": "#000000", "header_text": "#ffffff",
        "shadow": "none", "colors": _PALETTE, "dark": True,
    },
    "light": {
        "bg": "#f4f6f9", "surface": "#ffffff", "surface_hover": "#f8fafc", "border": "#e2e8f0",
        "text": "#1e293b", "muted": "#64748b", "accent": "#4f46e5",
        "header_bg": "linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)", "header_text": "#ffffff",
        "shadow": "0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.05)", "colors": _LIGHT_PALETTE, "dark": False,
    },
}

_CSS_VARS = {
    "bg": "--bg", "surface": "--surface", "surface_hover": "--surface-hover", "border": "--border",
    "text": "--text-main", "muted": "--text-muted", "accent": "--accent", "header_bg": "--header-bg",
    "header_text": "--header-text", "shadow": "--card-shadow", "font": "--font", "radius": "--radius",
}
_SAFE_VALUE = re.compile(r"^[#\w\s%.,()\-/'\"+]*$")


@dataclass(frozen=True)
class Theme:
    """Resolved theme: display name, CSS variables and Plotly defaults."""

    name: str
    tokens: dict[str, Any]

    def css_vars(self) -> str:
        """``:root`` custom properties for the page."""
        out = []
        for key, var in _CSS_VARS.items():
            if key in self.tokens:
                out.append(f"{var}: {self.tokens[key]};")
        return " ".join(out)

    def plotly_layout(self) -> dict[str, Any]:
        """Base Plotly layout: transparent canvas, theme-colored text and grid."""
        t = self.tokens
        axis = {"gridcolor": t["border"], "zerolinecolor": t["border"], "linecolor": t["border"]}
        return {
            "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
            "font": {"family": t["font"], "color": t["text"]},
            "colorway": list(t["colors"]),
            "margin": {"l": 50, "r": 30, "t": 50, "b": 50},
            "xaxis": dict(axis), "yaxis": dict(axis),
            "legend": {"bgcolor": "rgba(0,0,0,0)"},
            "hoverlabel": {"font": {"family": t["font"]}},
        }


def resolve_theme(theme: Any) -> Theme:
    """Resolve a theme name or an object of token overrides.

    An object may set ``base`` (dark/midnight/black/light) and any of: bg, surface,
    border, text, muted, accent, header_bg, header_text, font, radius, shadow, and
    ``colors`` (a list of chart colors).

    Raises:
        QueryError: For unknown names or unsafe values.
    """
    if theme in (None, ""):
        theme = "dark"
    overrides: dict[str, Any] = {}
    if isinstance(theme, dict):
        overrides = dict(theme)
        base = str(overrides.pop("base", "dark")).lower()
        name = str(overrides.pop("name", "custom"))
    else:
        base = name = str(theme).lower()
    if base not in THEMES:
        raise QueryError(f"Unknown theme '{theme}'. Use dark, light, midnight, black, or an object with a 'base' and color overrides.")
    tokens = {"font": _FONT, "radius": "10px", **THEMES[base]}
    for key, value in overrides.items():
        if key == "colors":
            colors = [str(c) for c in value] if isinstance(value, list) else []
            if not colors or any(not _SAFE_VALUE.match(c) for c in colors):
                raise QueryError("theme.colors must be a list of CSS colors.")
            tokens["colors"] = colors
        elif key in _CSS_VARS or key == "dark":
            if key != "dark" and not _SAFE_VALUE.match(str(value)):
                raise QueryError(f"theme.{key} contains unsupported characters.")
            tokens[key] = value
        else:
            raise QueryError(f"Unknown theme key '{key}'. Allowed: {sorted(_CSS_VARS) + ['colors', 'base']}.")
    return Theme(name=name if isinstance(theme, dict) else base, tokens=tokens)
