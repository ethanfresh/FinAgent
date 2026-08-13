"""Mixture-of-experts financial analysis graph.

Instead of one agent with every tool, a dispatcher gates which of several
scoped experts are relevant to the question (financials / news / executives),
runs only those in parallel, and a synthesizer weaves their reports into one
narrative — closer to how an analyst actually builds "the story" of a company
than a single flat tool-calling loop.
"""

import operator
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from finagent.agent.graph import extract_text
from finagent.runner import ToolCall
from finagent.tools.edgar import edgar_filings
from finagent.tools.executives import executive_profile
from finagent.tools.market_data import fundamental_ratios, price_history
from finagent.tools.news import company_news

EXPERTS = {
    "financials": {
        "tools": [price_history, fundamental_ratios, edgar_filings],
        "description": (
            "up-to-date financial figures: stock price history, fundamental ratios "
            "(margins, P/E, ROE), and SEC filings"
        ),
    },
    "news": {
        "tools": [company_news],
        "description": "recent company news and events",
    },
    "executives": {
        "tools": [executive_profile, company_news],
        "description": (
            "company leadership — who runs the company, their background and compensation, "
            "and how they're reflected in recent news coverage (do not invent a sentiment "
            "score; ground any reputation comment in what the tools actually returned)"
        ),
    },
}

DISPATCH_PROMPT = f"""You gate a team of financial-analysis experts. Given the user's \
question, decide which experts are relevant:

{chr(10).join(f"- {name}: {cfg['description']}" for name, cfg in EXPERTS.items())}

Respond with ONLY a comma-separated list of relevant expert names from \
{list(EXPERTS)} — nothing else. Pick every expert that would meaningfully \
contribute to a complete answer; for a broad "tell me about this company" \
question, that's usually all of them."""

SYNTHESIZER_PROMPT = """You are a financial analyst weaving specialist reports into one \
cohesive answer for the user's question. You'll be given reports from a financials \
expert, a news expert, and/or an executives expert (only the ones judged relevant \
were run) — combine them into a single narrative that tells the story the question \
is asking for, not three disconnected paragraphs. Only use figures, dates, and facts \
that appear in the reports below — do not add anything the experts didn't report. \
Cite source URLs as markdown links `[title](url)` with a short descriptive title, never \
raw URLs. No other markdown (no asterisks, headers, bullets) since this renders in a \
plain-text chat bubble. Do not give investment advice. Never name an internal tool or \
function as a source, even if an expert report mentions one — attribute data to the real \
provider instead (Yahoo Finance for price/fundamentals/news/executive data, SEC EDGAR for \
filings)."""


class MoEState(TypedDict):
    question: str
    history: list[dict]
    active_experts: list[str]
    expert_reports: Annotated[dict, operator.or_]
    tool_calls: Annotated[list, operator.add]
    answer: str


def _parse_expert_list(text: str) -> list[str]:
    named = [name.strip().lower() for name in text.split(",")]
    active = [name for name in named if name in EXPERTS]
    return active or ["financials"]


def _question_with_history(question: str, history: list[dict] | None) -> str:
    """Prefix the question with prior turns so every node in the graph — dispatch,
    each expert, and the synthesizer — sees the same conversation a human reading
    the chat transcript would, instead of just the newest message in isolation."""
    if not history:
        return f"Question: {question}"
    lines = [f"{'User' if t.get('role') == 'user' else 'Assistant'}: {t.get('content', '')}" for t in history]
    return "Conversation so far:\n" + "\n".join(lines) + f"\n\nQuestion: {question}"


def _make_expert_node(name: str, tools: list, description: str, model_name: str):
    tools_by_name = {t.name: t for t in tools}
    system = (
        f"You are the {name} expert on a financial-analysis team. Your job: gather "
        f"{description}. Use your tools to answer the question, then write a focused "
        "2-4 sentence report grounded only in what the tools returned, including any "
        "source URLs the tools gave you. If your area turns out not to be relevant "
        "to this specific question, say so briefly instead of forcing an answer. Never "
        "name an internal tool or function as your source — attribute data to the real "
        "provider instead (Yahoo Finance for price/fundamentals/news/executive data, SEC "
        "EDGAR for filings)."
    )
    llm = ChatAnthropic(model=model_name).bind_tools(tools)

    def node(state: MoEState) -> dict:
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=_question_with_history(state["question"], state.get("history"))),
        ]
        response = llm.invoke(messages)

        tool_calls_made = []
        if getattr(response, "tool_calls", None):
            tool_messages = []
            for call in response.tool_calls:
                result = tools_by_name[call["name"]].invoke(call["args"])
                tool_calls_made.append(ToolCall(name=call["name"], args=call["args"]))
                tool_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
            response = llm.invoke([*messages, response, *tool_messages])

        return {"expert_reports": {name: extract_text(response.content)}, "tool_calls": tool_calls_made}

    return node


def build_moe_graph(model_name: str = "claude-sonnet-5"):
    dispatch_llm = ChatAnthropic(model=model_name)
    synth_llm = ChatAnthropic(model=model_name)

    def dispatch(state: MoEState) -> dict:
        response = dispatch_llm.invoke(
            [
                SystemMessage(content=DISPATCH_PROMPT),
                HumanMessage(content=_question_with_history(state["question"], state.get("history"))),
            ]
        )
        return {"active_experts": _parse_expert_list(extract_text(response.content))}

    def route_to_experts(state: MoEState) -> list[Send]:
        return [Send(f"expert_{name}", state) for name in state["active_experts"]]

    def synthesize(state: MoEState) -> dict:
        reports_text = "\n\n".join(
            f"[{name.upper()} EXPERT]\n{report}" for name, report in state["expert_reports"].items()
        )
        question_block = _question_with_history(state["question"], state.get("history"))
        messages = [
            SystemMessage(content=SYNTHESIZER_PROMPT),
            HumanMessage(content=f"{question_block}\n\nExpert reports:\n{reports_text}"),
        ]
        response = synth_llm.invoke(messages)
        return {"answer": extract_text(response.content)}

    graph = StateGraph(MoEState)
    graph.add_node("dispatch", dispatch)
    graph.add_node("synthesize", synthesize)
    for name, cfg in EXPERTS.items():
        graph.add_node(f"expert_{name}", _make_expert_node(name, cfg["tools"], cfg["description"], model_name))
        graph.add_edge(f"expert_{name}", "synthesize")

    graph.set_entry_point("dispatch")
    graph.add_conditional_edges("dispatch", route_to_experts, [f"expert_{n}" for n in EXPERTS])
    graph.add_edge("synthesize", END)
    return graph.compile()
