import os

from langchain_core.messages import AIMessage, HumanMessage

from finagent.agent.graph import build_graph, extract_text
from finagent.observability import langfuse_callbacks
from finagent.runner import AgentResult, ToolCall


class FinAgentRunner:
    """Default AgentRunner: FinAgent's own LangGraph agent (router → tools → synthesizer).

    Satisfies the AgentRunner protocol with just `run(question: str) -> AgentResult`.
    LangFuse traces are tagged with the FINAGENT_ENV env var (set by the CLI/web/eval
    entry points), so tracing granularity doesn't leak into the protocol's call signature.
    """

    def __init__(self, model_name: str = "claude-sonnet-5"):
        self._graph = build_graph(model_name)

    def run(self, question: str) -> AgentResult:
        environment = os.environ.get("FINAGENT_ENV", "runner")
        result = self._graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"callbacks": langfuse_callbacks(), "metadata": {"environment": environment}},
        )
        messages = result["messages"]

        tool_calls = [
            ToolCall(name=call["name"], args=call["args"])
            for m in messages
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
            for call in m.tool_calls
        ]

        final = next((m for m in reversed(messages) if isinstance(m, AIMessage) and m.content), None)
        if final is None:
            raise RuntimeError("agent produced no answer")

        return AgentResult(answer=extract_text(final.content), tool_calls=tool_calls)
