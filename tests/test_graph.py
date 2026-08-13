from finagent.agent.graph import TOOLS_BY_NAME, build_graph


def test_tools_registered():
    assert set(TOOLS_BY_NAME) == {
        "price_history",
        "fundamental_ratios",
        "edgar_filings",
        "company_news",
        "executive_profile",
    }


def test_graph_compiles():
    graph = build_graph()
    assert graph is not None
    node_names = set(graph.get_graph().nodes)
    assert {"router", "tools", "synthesizer"} <= node_names
