from finagent.agent.moe_graph import (
    EXPERTS,
    _parse_expert_list,
    _question_with_history,
    build_moe_graph,
)


def test_experts_have_scoped_tools():
    assert set(EXPERTS) == {"financials", "news", "executives"}
    assert {t.name for t in EXPERTS["financials"]["tools"]} == {
        "price_history",
        "fundamental_ratios",
        "edgar_filings",
        "filing_search",
    }
    assert {t.name for t in EXPERTS["news"]["tools"]} == {"company_news"}


def test_parse_expert_list_filters_to_known_experts():
    assert _parse_expert_list("financials, news") == ["financials", "news"]
    assert _parse_expert_list("financials, nonsense, executives") == ["financials", "executives"]


def test_parse_expert_list_falls_back_when_empty():
    assert _parse_expert_list("") == ["financials"]
    assert _parse_expert_list("nonsense") == ["financials"]


def test_moe_graph_compiles_with_expert_and_synth_nodes():
    graph = build_moe_graph()
    node_names = set(graph.get_graph().nodes)
    assert {"dispatch", "expert_financials", "expert_news", "expert_executives", "synthesize"} <= node_names


def test_question_with_history_prefixes_prior_turns():
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    text = _question_with_history("and now?", history)
    assert "User: hi" in text
    assert "Assistant: hello" in text
    assert text.endswith("Question: and now?")


def test_question_with_history_is_bare_question_when_empty():
    assert _question_with_history("hi", None) == "Question: hi"
    assert _question_with_history("hi", []) == "Question: hi"


def test_moe_runner_selectable_via_env(monkeypatch):
    monkeypatch.setenv("FINAGENT_RUNNER", "finagent.agent.moe_runner:MoEAgentRunner")

    from finagent.agent.moe_runner import MoEAgentRunner
    from finagent.runner import load_runner

    runner = load_runner()
    assert isinstance(runner, MoEAgentRunner)
