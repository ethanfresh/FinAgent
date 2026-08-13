import json
import sys
import types

import pytest
from fastapi.testclient import TestClient

from finagent import web
from finagent.runner import AgentResult, ToolCall

client = TestClient(web.app)


def _install_fake_runner_module(received_history=None):
    module = types.ModuleType("_fake_web_runner")

    class FakeRunner:
        def run(self, question, history=None):
            if received_history is not None:
                received_history.append(history)
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


def test_ask_forwards_history_to_runner(monkeypatch):
    received = []
    _install_fake_runner_module(received_history=received)
    monkeypatch.setenv("FINAGENT_RUNNER", "_fake_web_runner:FakeRunner")
    web._runner.cache_clear()

    res = client.post(
        "/api/ask",
        json={"question": "and now?", "history": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
    )

    web._runner.cache_clear()
    assert res.status_code == 200
    assert received == [[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]]


def test_ask_defaults_to_empty_history(fake_runner):
    res = client.post("/api/ask", json={"question": "hi"})
    assert res.status_code == 200


def test_redteam_personas_lists_defaults():
    res = client.get("/api/redteam/personas")
    assert res.status_code == 200
    names = [p["name"] for p in res.json()]
    assert names  # non-empty, and each entry has the Persona shape
    assert all({"name", "description", "goal"} <= p.keys() for p in res.json())


def test_redteam_run_starts_a_job_and_status_reflects_it(monkeypatch):
    ran = {}

    def fake_run_redteam(turns, base_url):
        ran["turns"] = turns
        ran["base_url"] = base_url
        return {"base_url": base_url, "turns_per_session": turns, "sessions": []}

    def fake_write_report(report):
        ran["written"] = report
        return None, None

    monkeypatch.setattr(web, "run_redteam", fake_run_redteam)
    monkeypatch.setattr(web, "write_report", fake_write_report)
    with web._redteam_lock:
        web._redteam_state.update(status="idle", turns=None, started_at=None, finished_at=None, error=None)

    res = client.post("/api/redteam/run", json={"turns": 2})
    assert res.status_code == 200
    assert res.json()["status"] == "started"

    for _ in range(50):
        with web._redteam_lock:
            if web._redteam_state["status"] != "running":
                break
        import time as _time

        _time.sleep(0.02)

    status = client.get("/api/redteam/status").json()
    assert status["status"] == "done"
    assert ran["turns"] == 2


def test_redteam_run_rejects_concurrent_run(monkeypatch):
    with web._redteam_lock:
        web._redteam_state.update(status="running", turns=4, started_at=1, finished_at=None, error=None)
    res = client.post("/api/redteam/run", json={})
    assert res.json()["status"] == "already_running"
    with web._redteam_lock:
        web._redteam_state.update(status="idle", turns=None, started_at=None, finished_at=None, error=None)


def test_redteam_reports_lists_and_fetches(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "REDTEAM_DIR", tmp_path)
    report = {"base_url": "http://x", "turns_per_session": 1, "sessions": [{"persona": "P", "goal": "g", "transcript": [], "issues": [{"turn": 1}]}]}
    (tmp_path / "20260101T000000Z.json").write_text(json.dumps(report))

    listed = client.get("/api/redteam/reports").json()
    assert len(listed) == 1
    assert listed[0]["issues"] == 1
    assert listed[0]["name"] == "20260101T000000Z"

    fetched = client.get(f"/api/redteam/reports/{listed[0]['name']}").json()
    assert fetched == report

    missing = client.get("/api/redteam/reports/does-not-exist")
    assert missing.status_code == 404


def test_redteam_fixes_empty_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(web, "REDTEAM_FIXES_PATH", tmp_path / "does-not-exist.json")
    res = client.get("/api/redteam/fixes")
    assert res.status_code == 200
    assert res.json() == []


def test_redteam_fixes_reads_the_curated_file(monkeypatch, tmp_path):
    fixes = [{"id": "example-fix", "title": "Example", "severity": "high"}]
    path = tmp_path / "redteam_fixes.json"
    path.write_text(json.dumps(fixes))
    monkeypatch.setattr(web, "REDTEAM_FIXES_PATH", path)
    res = client.get("/api/redteam/fixes")
    assert res.status_code == 200
    assert res.json() == fixes
