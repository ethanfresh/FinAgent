import sys
import types

import pytest
from fastapi.testclient import TestClient

from finagent import web
from finagent.runner import AgentResult, ToolCall

client = TestClient(web.app)


def _install_fake_runner_module():
    module = types.ModuleType("_fake_web_runner")

    class FakeRunner:
        def run(self, question):
            return AgentResult(
                answer=f"fake answer: {question}", tool_calls=[ToolCall(name="fake_tool", args={})]
            )

    module.FakeRunner = FakeRunner
    sys.modules["_fake_web_runner"] = module


@pytest.fixture
def fake_runner(monkeypatch):
    _install_fake_runner_module()
    monkeypatch.setenv("FINAGENT_RUNNER", "_fake_web_runner:FakeRunner")
    web._runner.cache_clear()
    yield
    web._runner.cache_clear()


def test_stats_endpoint_shape():
    res = client.get("/api/stats")
    assert res.status_code == 200
    data = res.json()
    assert "requests_by_status" in data
    assert "tool_calls" in data
    assert "latency" in data


def test_ask_endpoint_uses_configured_runner(fake_runner):
    res = client.post("/api/ask", json={"question": "hi"})
    assert res.status_code == 200
    data = res.json()
    assert data["answer"] == "fake answer: hi"
    assert data["tool_calls"][0]["name"] == "fake_tool"


def test_ask_rejects_empty_question(fake_runner):
    res = client.post("/api/ask", json={"question": "   "})
    assert res.status_code == 400
