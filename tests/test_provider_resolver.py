"""Test suite for universal LLM provider, base_url, and api_key resolution."""

import os
from unittest.mock import Mock, patch
import pytest
from fastapi.testclient import TestClient
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from datavault import server
from datavault.agent import _get_llm


def test_openrouter_preset():
    llm = _get_llm(provider="openrouter", api_key="sk-or-test", model="meta-llama/llama-3.3-70b-instruct")
    assert isinstance(llm, ChatOpenAI)
    assert llm.openai_api_base == "https://openrouter.ai/api/v1"
    assert llm.model_name == "meta-llama/llama-3.3-70b-instruct"


def test_groq_preset():
    llm = _get_llm(provider="groq", api_key="gsk-test")
    assert isinstance(llm, ChatOpenAI)
    assert llm.openai_api_base == "https://api.groq.com/openai/v1"


def test_deepseek_preset():
    llm = _get_llm(provider="deepseek", api_key="sk-test")
    assert isinstance(llm, ChatOpenAI)
    assert llm.openai_api_base == "https://api.deepseek.com/v1"


def test_ollama_preset():
    llm = _get_llm(provider="ollama", model="llama3.2")
    assert isinstance(llm, ChatOpenAI)
    assert llm.openai_api_base == "http://localhost:11434/v1"
    assert llm.openai_api_key.get_secret_value() == "not-needed"


def test_custom_api_base_url():
    custom_url = "https://custom-proxy.enterprise.org/v1"
    llm = _get_llm(provider="custom", api_key="secret-key-123", base_url=custom_url, model="custom-model")
    assert isinstance(llm, ChatOpenAI)
    assert llm.openai_api_base == custom_url
    assert llm.openai_api_key.get_secret_value() == "secret-key-123"
    assert llm.model_name == "custom-model"


def test_google_with_base_url():
    llm = _get_llm(provider="google", api_key="aiza-dummy", base_url="https://generativelanguage.googleapis.com")
    assert isinstance(llm, ChatGoogleGenerativeAI)


def test_anthropic_with_base_url():
    llm = _get_llm(provider="anthropic", api_key="sk-ant-dummy", base_url="https://api.anthropic.com")
    assert isinstance(llm, ChatAnthropic)


def test_api_query_with_custom_provider_payload(monkeypatch: pytest.MonkeyPatch):
    """Test that POST /api/query accepts provider, base_url, api_key overrides."""
    mock_run_query = Mock(return_value={
        "messages": [AIMessage(content="Verified with custom provider!")],
        "data_refs": ["ref_test123"],
        "generated_files": []
    })
    monkeypatch.setattr("datavault.agent.run_query", mock_run_query)

    # Patch create_agent so we don't hit external APIs during endpoint test
    mock_create_agent = Mock(return_value=(Mock(), Mock(), None))
    monkeypatch.setattr("datavault.agent.create_agent", mock_create_agent)

    client = TestClient(server.app)
    response = client.post("/api/query", json={
        "query": "Show me sales summary",
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "sk-or-test-key",
        "model": "anthropic/claude-3.5-sonnet"
    })

    assert response.status_code == 200
    assert response.json()["response"] == "Verified with custom provider!"

    # Verify create_agent received the custom parameters
    mock_create_agent.assert_called_once()
    _, kwargs = mock_create_agent.call_args
    assert kwargs["provider"] == "openrouter"
    assert kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert kwargs["api_key"] == "sk-or-test-key"
    assert kwargs["model"] == "anthropic/claude-3.5-sonnet"
