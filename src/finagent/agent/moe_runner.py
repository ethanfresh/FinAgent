import os

from finagent.agent.moe_graph import build_moe_graph
from finagent.observability import langfuse_callbacks
from finagent.runner import AgentResult


class MoEAgentRunner:
    """AgentRunner backed by the mixture-of-experts graph (agent/moe_graph.py)
    instead of the single flat FinAgentRunner. Same AgentRunner protocol
    (`run(question: str) -> AgentResult`), different internal architecture —
    proves the platform is agent-agnostic with a second real, non-trivial agent,
    not just a trivial dummy swapped in for a test.

    Select it with:
        FINAGENT_RUNNER=finagent.agent.moe_runner:MoEAgentRunner
    """

    def __init__(self, model_name: str = "claude-sonnet-5"):
        self._graph = build_moe_graph(model_name)

    def run(self, question: str) -> AgentResult:
        environment = os.environ.get("FINAGENT_ENV", "runner")
        result = self._graph.invoke(
            {"question": question, "active_experts": [], "expert_reports": {}, "tool_calls": [], "answer": ""},
            config={"callbacks": langfuse_callbacks(), "metadata": {"environment": environment, "backend": "moe"}},
        )
        return AgentResult(answer=result["answer"], tool_calls=result["tool_calls"])
