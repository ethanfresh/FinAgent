import pytest
import requests

pytest.importorskip("chromadb")
pytest.importorskip("sentence_transformers")


def _require_network():
    try:
        requests.get("https://www.sec.gov", timeout=5)
    except requests.RequestException:
        pytest.skip("no network access in this environment")


@pytest.fixture
def isolated_index(tmp_path, monkeypatch):
    from finagent.rag import store

    monkeypatch.setattr(store, "CHROMA_DIR", tmp_path / "chroma")
    store._client.cache_clear()
    yield
    store._client.cache_clear()


def test_index_filing_and_search_round_trip(isolated_index):
    _require_network()
    from finagent.rag.ingest import index_filing
    from finagent.rag.store import is_indexed
    from finagent.tools.filing_search import filing_search

    result = index_filing("NVDA", form_type="10-K", limit=1)
    assert "error" not in result
    assert result["filings_indexed"]
    assert result["filings_indexed"][0]["chunks"] > 0
    assert is_indexed("NVDA", "10-K")

    search = filing_search.invoke({"ticker": "NVDA", "query": "risk factors related to competition", "form_type": "10-K", "limit": 3})
    assert "error" not in search
    assert search["passages"]
    assert all(p["url"] for p in search["passages"])


def test_filing_search_lazily_indexes_on_first_call(isolated_index):
    _require_network()
    from finagent.rag.store import is_indexed
    from finagent.tools.filing_search import filing_search

    assert not is_indexed("AAPL", "10-K")
    search = filing_search.invoke({"ticker": "AAPL", "query": "supplier risk", "form_type": "10-K", "limit": 2})
    assert "error" not in search
    assert search["passages"]
    assert is_indexed("AAPL", "10-K")


def test_filing_search_unknown_ticker_returns_error(isolated_index):
    _require_network()
    from finagent.tools.filing_search import filing_search

    result = filing_search.invoke({"ticker": "NOT_A_REAL_TICKER", "query": "anything", "form_type": "10-K"})
    assert "error" in result
