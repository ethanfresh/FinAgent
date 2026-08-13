import pytest

pytest.importorskip("langchain_aws")


def test_bedrock_runner_constructs_without_credentials():
    from finagent.agent.bedrock_runner import BedrockAgentRunner

    runner = BedrockAgentRunner()
    assert runner is not None


def test_bedrock_runner_selectable_via_env(monkeypatch):
    monkeypatch.setenv("FINAGENT_RUNNER", "finagent.agent.bedrock_runner:BedrockAgentRunner")

    from finagent.agent.bedrock_runner import BedrockAgentRunner
    from finagent.runner import load_runner

    runner = load_runner()
    assert isinstance(runner, BedrockAgentRunner)
