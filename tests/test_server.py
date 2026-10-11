from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from datavault import server


@pytest.mark.parametrize(
    ("replies", "expected"),
    [
        pytest.param(
            [AIMessage(content="Here is the dashboard summary.")],
            "Here is the dashboard summary.",
            id="assistant-not-user",
        ),
        pytest.param(
            [
                AIMessage(
                    content="I will generate the dashboard.",
                    tool_calls=[
                        {"name": "generate_html_viz", "args": {}, "id": "dashboard-call"}
                    ],
                ),
                ToolMessage(
                    content="Dashboard saved to: output/viz/dashboard.html",
                    tool_call_id="dashboard-call",
                ),
                AIMessage(content="The dashboard is ready."),
            ],
            "The dashboard is ready.",
            id="assistant-not-tool-output",
        ),
        pytest.param(
            [
                AIMessage(
                    content=[
                        {"type": "thinking", "thinking": "Private reasoning"},
                        {"type": "text", "text": "Sales increased. "},
                        {"type": "text", "text": "The dashboard is ready."},
                    ]
                )
            ],
            "Sales increased. The dashboard is ready.",
            id="structured-text-without-reasoning",
        ),
        pytest.param([AIMessage(content="")], "", id="empty-assistant-not-user"),
    ],
)
def test_query_returns_final_assistant_text(
    monkeypatch: pytest.MonkeyPatch, replies: list[BaseMessage], expected: str
) -> None:
    graph = Mock()
    graph.invoke.return_value = {
        "messages": [HumanMessage(content="Show me a dashboard"), *replies],
        "data_refs": ["ref_123456789abc"],
        "generated_files": ["output/viz/dashboard.html"],
    }
    monkeypatch.setattr(server, "_agent_graph", graph)
    monkeypatch.setattr(server, "_data_refs", [])
    monkeypatch.setattr(server, "_generated_files", [])

    with TestClient(server.app) as client:
        response = client.post("/api/query", json={"query": "Show me a dashboard"})

    assert response.status_code == 200
    assert response.json() == {
        "response": expected,
        "data_refs": ["ref_123456789abc"],
        "generated_files": ["output/viz/dashboard.html"],
    }
