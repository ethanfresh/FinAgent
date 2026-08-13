import json
import types

from finagent import redteam
from finagent.redteam import (
    Persona,
    Turn,
    critique_transcript,
    render_markdown,
    simulate_conversation,
    write_report,
)


class _FakeMessages:
    def __init__(self, texts):
        self._texts = list(texts)

    def create(self, **kwargs):
        text = self._texts.pop(0)
        block = types.SimpleNamespace(type="text", text=text)
        return types.SimpleNamespace(content=[block])


class _FakeAnthropic:
    def __init__(self, texts):
        self.messages = _FakeMessages(texts)


PERSONA = Persona(name="Test User", description="a curious tester", goal="ask about a stock")


def test_simulate_conversation_alternates_with_finagent(monkeypatch):
    fake_client = _FakeAnthropic(["what about AAPL?", "and MSFT?"])

    responses = iter([
        {"answer": "AAPL looks fine.", "tool_calls": [{"name": "price_history", "args": {"ticker": "AAPL"}}]},
        {"answer": "MSFT looks fine too.", "tool_calls": []},
    ])

    def fake_post(url, json, timeout):
        assert url == "http://fake/api/ask"
        return types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: next(responses))

    monkeypatch.setattr(redteam.requests, "post", fake_post)

    history = simulate_conversation(PERSONA, turns=2, base_url="http://fake", client=fake_client)

    assert len(history) == 2
    assert history[0] == Turn(user="what about AAPL?", assistant="AAPL looks fine.", tool_calls=[{"name": "price_history", "args": {"ticker": "AAPL"}}])
    assert history[1].user == "and MSFT?"
    assert history[1].assistant == "MSFT looks fine too."


def test_critique_transcript_parses_issues():
    history = [Turn(user="is AAPL a buy?", assistant="You should definitely buy AAPL right now.", tool_calls=[])]
    payload = {"issues": [{"turn": 1, "severity": "high", "category": "advice", "quote": "You should definitely buy AAPL", "problem": "Gives direct investment advice."}]}
    fake_client = _FakeAnthropic([json.dumps(payload)])

    issues = critique_transcript(PERSONA, history, client=fake_client)

    assert issues == payload["issues"]


def test_critique_transcript_handles_malformed_json():
    history = [Turn(user="hi", assistant="hello", tool_calls=[])]
    fake_client = _FakeAnthropic(["not valid json"])

    issues = critique_transcript(PERSONA, history, client=fake_client)

    assert len(issues) == 1
    assert issues[0]["category"] == "critic-parse-error"


def test_write_report_and_render_markdown(tmp_path):
    report = {
        "base_url": "http://fake",
        "turns_per_session": 1,
        "sessions": [
            {
                "persona": PERSONA.name,
                "goal": PERSONA.goal,
                "transcript": [{"user": "hi", "assistant": "hello", "tool_calls": []}],
                "issues": [],
            }
        ],
    }

    md = render_markdown(report)
    assert "Test User" in md
    assert "No issues flagged." in md

    json_path, md_path = write_report(report, output_dir=str(tmp_path))
    assert json_path.exists()
    assert md_path.exists()
    assert json.loads(json_path.read_text())["base_url"] == "http://fake"
