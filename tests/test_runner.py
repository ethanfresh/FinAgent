import sys
import types

from finagent.runner import AgentResult, ToolCall, load_runner


def _install_fake_runner_module():
    module = types.ModuleType("_fake_finagent_runner")

    class FakeRunner:
        def run(self, question):
            return AgentResult(
                answer=f"fake: {question}", tool_calls=[ToolCall(name="fake_tool", args={"q": question})]
            )

    module.FakeRunner = FakeRunner
    sys.modules["_fake_finagent_runner"] = module


def test_default_runner_is_finagent_runner(monkeypatch):
    monkeypatch.delenv("FINAGENT_RUNNER", raising=False)
    from finagent.agent.runner import FinAgentRunner

    runner = load_runner()
    assert isinstance(runner, FinAgentRunner)


def test_load_runner_respects_env_override(monkeypatch):
    _install_fake_runner_module()
    monkeypatch.setenv("FINAGENT_RUNNER", "_fake_finagent_runner:FakeRunner")

    runner = load_runner()
    result = runner.run("hello")

    assert result.answer == "fake: hello"
    assert result.tool_calls[0].name == "fake_tool"
