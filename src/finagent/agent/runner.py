from finagent.agent.graph import build_graph, run_graph
from finagent.runner import AgentResult


class FinAgentRunner:
    """Default AgentRunner: FinAgent's own LangGraph agent (router → tools → synthesizer),
    calling Claude directly via the Anthropic API.

    Satisfies the AgentRunner protocol: `run(question: str, history=None) -> AgentResult`.
    """

    def __init__(self, model_name: str = "claude-sonnet-5"):
        self._graph = build_graph(model_name)

    def run(self, question: str, history: list[dict] | None = None) -> AgentResult:
        return run_graph(self._graph, question, backend="anthropic", history=history)
