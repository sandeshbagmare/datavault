"""DataVault — Privacy-first data tools and agent using LangGraph, DuckDB, and FastMCP.

Attach the tools to any agent (including a LangGraph Deep Agent)::

    from datavault import DataContext, create_tools
    tools = create_tools(DataContext(output_dir="out"))

Importing this package is cheap; the agent and web server load on first use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datavault.agent import create_agent
    from datavault.context import DataContext, Policy
    from datavault.server import create_app
    from datavault.tools import create_tools

__all__ = ["DataContext", "Policy", "create_agent", "create_app", "create_tools"]

__version__ = "0.1.0"

_LAZY = {
    "create_agent": "datavault.agent",
    "create_app": "datavault.server",
    "create_tools": "datavault.tools",
    "DataContext": "datavault.context",
    "Policy": "datavault.context",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'datavault' has no attribute {name!r}")
