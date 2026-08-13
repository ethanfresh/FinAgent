import operator
import os
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from finagent.observability import langfuse_callbacks
from finagent.runner import AgentResult, ToolCall
from finagent.tools import ALL_TOOLS

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}

SYSTEM_PROMPT = (
    "You are FinAgent, a financial research assistant. Use the available tools to look up "
    "SEC filings, price history, and fundamental ratios before answering. Cite the figures "
    "you used. Do not give investment advice. Answer in plain text with no markdown "
    "formatting (no asterisks, headers, or bullet characters) since your response is "
    "rendered in a plain-text chat bubble — with one exception: when citing a source URL "
    "(e.g. an EDGAR filing), write it as a markdown link `[title](url)` where the title is "
    "a short, human-readable label describing the source (e.g. 'NVDA 10-Q — filed "
    "2026-05-20'), never the raw URL or link text like 'here' or 'this filing'."
)


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


def extract_text(content) -> str:
    """Flatten an AIMessage's content into plain text.

    Anthropic responses can return content as a list of blocks (text,
    thinking, citations, ...) instead of a plain string; only the text
    blocks matter for display.
    """
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def build_graph(model_name: str = "claude-sonnet-5", llm=None):
    """Build the router/tools/synthesizer graph around a chat model.

    Defaults to Claude via the Anthropic API. Pass a different LangChain
    chat model (e.g. ChatBedrock) to run the same graph against a different
    backend — see agent/bedrock_runner.py.
    """
    llm = (llm or ChatAnthropic(model=model_name)).bind_tools(ALL_TOOLS)

    def router(state: AgentState) -> dict:
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = llm.invoke(messages)
        return {"messages": [response]}

    def tool_execution(state: AgentState) -> dict:
        last = state["messages"][-1]
        outputs = []
        for call in last.tool_calls:
            tool = TOOLS_BY_NAME[call["name"]]
            result = tool.invoke(call["args"])
            outputs.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        return {"messages": outputs}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "end"

    graph = StateGraph(AgentState)
    graph.add_node("router", router)
    graph.add_node("tools", tool_execution)
    graph.add_node("synthesizer", router)
    graph.set_entry_point("router")
    graph.add_conditional_edges("router", should_continue, {"tools": "tools", "end": END})
    graph.add_conditional_edges("synthesizer", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "synthesizer")
    return graph.compile()


def run_graph(graph, question: str, backend: str = "anthropic") -> AgentResult:
    """Invoke a compiled graph and package the result as an AgentResult.

    Shared by every AgentRunner implementation (FinAgentRunner, BedrockAgentRunner, ...)
    so LangFuse tagging and tool-call extraction stay in one place.
    """
    environment = os.environ.get("FINAGENT_ENV", "runner")
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"callbacks": langfuse_callbacks(), "metadata": {"environment": environment, "backend": backend}},
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
