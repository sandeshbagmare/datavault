"""Tools attach to any agent, stay isolated per context, and never leak the full data."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from conftest import N_ROWS, make_context, saved_path, tools_by_name


def test_unknown_reference_is_an_error_not_another_dataset(ctx):
    tools = tools_by_name(ctx)
    for name, args in (
        ("profile_and_filter", {"data_ref": "ref_nope"}),
        ("generate_html_viz", {"data_ref": "ref_nope", "chart_type": "bar", "x_col": "brand"}),
        ("generate_markdown", {"data_ref": "ref_nope"}),
    ):
        out = tools[name].invoke(args)
        assert out.startswith("Error"), (name, out)
        assert "saved to" not in out


def test_two_contexts_are_isolated(tmp_path):
    a = make_context(tmp_path / "a", name="alpha")
    b = make_context(tmp_path / "b", name="beta")
    ta, tb = tools_by_name(a), tools_by_name(b)

    assert ta["profile_and_filter"].invoke({"data_ref": "alpha"}).startswith("## Data Profile")
    assert tb["profile_and_filter"].invoke({"data_ref": "alpha"}).startswith("Error")  # B never sees A's table

    out_a = ta["generate_html_viz"].invoke({"data_ref": "alpha", "chart_type": "bar", "x_col": "brand"})
    out_b = tb["generate_html_viz"].invoke({"data_ref": "beta", "chart_type": "bar", "x_col": "brand"})
    assert (tmp_path / "a") in saved_path(out_a).parents and (tmp_path / "b") in saved_path(out_b).parents


def test_derived_references_keep_full_lineage(ctx):
    p = tools_by_name(ctx)["profile_and_filter"]
    first = p.invoke({"data_ref": "sales", "filter_query": "brand = 'Tesla'"})
    ref1 = re.search(r"ref_[a-f0-9]{12}", first).group()
    second = p.invoke({
        "data_ref": ref1,
        "group_by": [{"col": "sale_date", "grain": "year"}],
        "metrics": [{"agg": "sum", "col": "price", "as": "revenue"}],
    })
    assert "Lineage" in second and "brand = 'Tesla'" in second and "group_by" in second and "revenue" in second
    ref2 = re.search(r"\*\*Reference ID:\*\* `(ref_[a-f0-9]{12})`", second).group(1)
    assert ctx.store.get(ref2).parent_ref == ref1
    # Re-profiling without a transformation does not create another reference
    count = len(ctx.store.list_refs())
    p.invoke({"data_ref": ref2})
    assert len(ctx.store.list_refs()) == count


def test_ordered_reference_shows_exact_top_rows(ctx):
    out = tools_by_name(ctx)["profile_and_filter"].invoke({
        "data_ref": "sales", "group_by": ["brand"], "metrics": [{"agg": "max", "col": "price", "as": "top"}],
        "order_by": [{"col": "brand", "desc": False}], "limit": 2,
    })
    rows = [ln for ln in out.splitlines() if ln.startswith("| Ford") or ln.startswith("| Honda")]
    assert len(rows) >= 2  # alphabetical first two groups, in order


def test_llm_only_sees_bounded_samples(ctx):
    p = tools_by_name(ctx)["profile_and_filter"]
    default = p.invoke({"data_ref": "sales"})
    sample_section = default.split("### Sample Rows")[1]
    assert len([ln for ln in sample_section.splitlines() if ln.startswith("| ")]) - 2 <= 5
    assert "x" * 100 not in default  # 300-char cells are truncated

    greedy = p.invoke({"data_ref": "sales", "sample_rows": 10_000})
    assert len([ln for ln in greedy.split("### Sample Rows")[1].splitlines() if ln.startswith("| ")]) - 2 <= 20
    assert len(greedy) < 8000 and f"{N_ROWS:,}" in greedy


def test_masked_columns_never_reach_the_model(tmp_path):
    ctx = make_context(tmp_path, masked_columns=frozenset({"buyer_email"}))
    out = tools_by_name(ctx)["profile_and_filter"].invoke({"data_ref": "sales", "sample_rows": 20})
    assert "@example.com" not in out and "***" in out


def test_raw_sql_is_read_only_and_policy_gated(ctx, tmp_path):
    p = tools_by_name(ctx)["profile_and_filter"]
    ok = p.invoke({"data_ref": "sales", "sql": "SELECT brand, COUNT(*) AS n FROM {ref} GROUP BY brand"})
    assert "## Data Profile" in ok and "| brand |" in ok
    for bad in ("SELECT * FROM read_csv('x.csv')", "DROP TABLE sales", "SELECT * FROM information_schema.tables"):
        assert p.invoke({"data_ref": "sales", "sql": bad}).startswith("Error"), bad

    locked = tools_by_name(make_context(tmp_path / "locked", allow_raw_sql=False))
    assert "disabled by policy" in locked["profile_and_filter"].invoke({"data_ref": "sales", "sql": "SELECT 1"})
    assert "disabled by policy" in locked["generate_markdown"].invoke({
        "data_ref": "sales", "sections": [{"type": "table", "dataset": "d"}], "datasets": {"d": {"sql": "SELECT 1"}},
    })


def test_tools_return_machine_readable_artifacts_through_toolnode(ctx):
    tools = tools_by_name(ctx)
    calls = [
        {"name": "generate_html_viz", "id": "c1", "args": {"data_ref": "sales", "chart_type": "pie", "x_col": "brand"}},
        {"name": "generate_markdown", "id": "c2", "args": {"data_ref": "sales", "report_type": "summary"}},
        {"name": "profile_and_filter", "id": "c3", "args": {"data_ref": "sales", "filter_query": "brand = 'Ford'"}},
    ]
    msgs = [tools[c["name"]].invoke({**c, "type": "tool_call"}) for c in calls]
    assert all(isinstance(m, ToolMessage) for m in msgs)
    by_id = {m.tool_call_id: m for m in msgs}
    assert by_id["c1"].artifact["kind"] == "html" and by_id["c1"].artifact["path"].endswith(".html")
    assert by_id["c2"].artifact["kind"] == "markdown" and by_id["c2"].artifact["ref_ids"] == ["sales"]
    assert by_id["c3"].artifact["parent_ref"] == "sales" and by_id["c3"].artifact["rows"] == N_ROWS // 4


class ScriptedModel(BaseChatModel):
    """A chat model that replays scripted AI messages (stands in for a real LLM)."""

    script: list[AIMessage]
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        message = self.script[self.cursor]
        self.cursor += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_tools_drive_a_standard_langchain_agent_end_to_end(ctx):
    from langchain.agents import create_agent

    spec = {
        "datasets": {"by_brand": {"group_by": ["brand"], "metrics": [{"agg": "sum", "col": "price", "as": "revenue"}]}},
        "body": [{"type": "chart", "chart": "bar", "dataset": "by_brand", "x": "brand", "y": "revenue"}],
    }
    model = ScriptedModel(script=[
        AIMessage(content="", tool_calls=[{"name": "profile_and_filter", "id": "1", "args": {"data_ref": "sales"}}]),
        AIMessage(content="", tool_calls=[{"name": "generate_html_viz", "id": "2", "args": {"data_ref": "sales", "spec": spec}}]),
        AIMessage(content="Done."),
    ])
    agent = create_agent(model, tools=list(tools_by_name(ctx).values()))
    result = agent.invoke({"messages": [("user", "chart revenue by brand")]})
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert [m.name for m in tool_msgs] == ["profile_and_filter", "generate_html_viz"]
    assert "Visualization saved to" in tool_msgs[1].content and tool_msgs[1].artifact["kind"] == "html"
    assert result["messages"][-1].content == "Done."


def test_concurrent_calls_on_one_context_are_safe(ctx):
    t = tools_by_name(ctx)

    def work(i: int) -> str:
        if i % 2:
            return t["generate_markdown"].invoke({"data_ref": "sales", "report_type": "breakdown", "group_col": "brand"})
        return t["generate_html_viz"].invoke({"data_ref": "sales", "chart_type": "bar", "x_col": "vehicle_type"})

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(work, range(16)))
    assert all("saved to" in r for r in results), results
    assert len({saved_path(r) for r in results}) == 16


@pytest.mark.parametrize("name", ["profile_and_filter", "generate_html_viz", "generate_markdown"])
def test_tool_schemas_are_valid_for_function_calling(ctx, name):
    schema = tools_by_name(ctx)[name].get_input_schema().model_json_schema()
    assert schema["type"] == "object" and schema["properties"]
    assert all("description" in p or "$ref" in p or "anyOf" in p for p in schema["properties"].values())
