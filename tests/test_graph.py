from langchain_core.messages import AIMessage, HumanMessage

from finagent.agent.graph import TOOLS_BY_NAME, _seed_messages, build_graph


def test_tools_registered():
    assert set(TOOLS_BY_NAME) == {
        "price_history",
        "fundamental_ratios",
        "edgar_filings",
        "company_news",
        "executive_profile",
        "filing_search",
    }


def test_graph_compiles():
    graph = build_graph()
    assert graph is not None
    node_names = set(graph.get_graph().nodes)
    assert {"router", "tools", "synthesizer"} <= node_names


def test_seed_messages_with_no_history_is_just_the_question():
    messages = _seed_messages("what about AAPL?", None)
    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "what about AAPL?"


def test_seed_messages_replays_prior_turns_before_the_question():
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    messages = _seed_messages("and now?", history)
    assert [type(m) for m in messages] == [HumanMessage, AIMessage, HumanMessage]
    assert [m.content for m in messages] == ["hi", "hello", "and now?"]


def test_seed_messages_ignores_unknown_roles():
    history = [{"role": "system", "content": "ignored"}]
    messages = _seed_messages("hi", history)
    assert len(messages) == 1
    assert messages[0].content == "hi"
