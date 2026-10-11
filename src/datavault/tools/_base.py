"""Shared plumbing for the three tools: context lookup, error handling, artifacts."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from datavault.context import DataContext, default_context
from datavault.query import QueryError

ToolResult = tuple[str, dict[str, Any] | None]
ContextGetter = Callable[[], DataContext]


def fixed_context(ctx: DataContext | None) -> ContextGetter:
    """A getter bound to ``ctx`` (or the process-wide default context)."""
    return (lambda: ctx) if ctx is not None else default_context


def build_tool(
    *,
    name: str,
    description: str,
    args_schema: type[BaseModel],
    failure: str,
    run: Callable[[DataContext, Any], ToolResult],
    get_context: ContextGetter,
) -> StructuredTool:
    """Create a LangChain tool whose behavior depends only on ``get_context()``.

    The tool returns text for the model plus a machine-readable artifact (available on
    the ``ToolMessage`` when called through an agent). Failures become short, actionable
    ``Error ...`` strings instead of exceptions so the model can correct itself.
    """

    def invoke(**kwargs: Any) -> ToolResult:
        try:
            return run(get_context(), args_schema(**kwargs))
        except KeyError as exc:
            return f"Error: {exc.args[0] if exc.args else exc}", None
        except QueryError as exc:
            return f"Error {failure}: {exc}", None
        except duckdb.Error as exc:
            return f"Error {failure}: {type(exc).__name__}: {str(exc).splitlines()[0]}", None
        except Exception as exc:  # noqa: BLE001 - tools must never raise into the agent loop
            return f"Error {failure}: {type(exc).__name__}: {exc}", None

    return StructuredTool.from_function(
        func=invoke,
        name=name,
        description=description,
        args_schema=args_schema,
        response_format="content_and_artifact",
        infer_schema=False,
    )


def write_artifact(directory: Path, prefix: str, suffix: str, content: str) -> Path:
    """Write ``content`` to a new uniquely named file and return its absolute path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prefix}_{uuid.uuid4().hex[:8]}{suffix}"
    path.write_text(content, encoding="utf-8")
    return path.resolve()
