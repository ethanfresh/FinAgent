import operator
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from finagent.tools import ALL_TOOLS

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}

SYSTEM_PROMPT = (
    "You are FinAgent, a financial research assistant. Use the available tools to look up "
    "SEC filings, price history, and fundamental ratios before answering. Cite the figures "
    "you used. Do not give investment advice. Answer in plain text with no markdown "
    "formatting (no asterisks, headers, or bullet characters) since your response is "
    "rendered in a plain-text chat bubble."
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


def build_graph(model_name: str = "claude-sonnet-5"):
    llm = ChatAnthropic(model=model_name).bind_tools(ALL_TOOLS)

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
