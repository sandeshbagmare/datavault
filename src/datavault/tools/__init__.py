"""The three DataVault tools, plus ``create_tools`` to attach them to any agent.

    from datavault.tools import create_tools
    from datavault.context import DataContext

    ctx = DataContext(output_dir="out")
    ctx.mount_parquet("sales.parquet", "sales")          # or ctx.register_table(...)
    tools = create_tools(ctx)                            # -> list[BaseTool]
    agent = create_deep_agent(model=..., tools=tools)    # or any LangGraph / LangChain agent
"""

from langchain_core.tools import BaseTool

from datavault.context import DataContext, Policy
from datavault.tools.html_viz import generate_html_viz, make_html_tool
from datavault.tools.markdown_report import generate_markdown, make_markdown_tool
from datavault.tools.profile_filter import make_profile_tool, profile_and_filter


def create_tools(ctx: DataContext | None = None, *, policy: Policy | None = None) -> list[BaseTool]:
    """Create ``[profile_and_filter, generate_html_viz, generate_markdown]`` bound to ``ctx``.

    Tools from different contexts never share data, references, files or settings, so one
    process can serve many agents. Needs no LLM, graph or web server.

    Args:
        ctx: The session context (connection, references, policy, output folders).
            A fresh empty context is created when omitted.
        policy: Convenience for ``DataContext(policy=...)`` when ``ctx`` is omitted.
    """
    if ctx is None:
        ctx = DataContext(policy=policy or Policy.from_env())
    elif policy is not None:
        raise ValueError("Pass either ctx or policy, not both; set the policy on the DataContext.")
    getter = lambda: ctx  # noqa: E731 - tiny closure shared by all three tools
    return [make_profile_tool(getter), make_html_tool(getter), make_markdown_tool(getter)]


__all__ = [
    "DataContext",
    "Policy",
    "create_tools",
    "generate_html_viz",
    "generate_markdown",
    "profile_and_filter",
]
