import pytest
import requests

from finagent.tools.edgar import edgar_filings
from finagent.tools.market_data import fundamental_ratios, price_history


def _require_network():
    try:
        requests.get("https://www.sec.gov", timeout=5)
    except requests.RequestException:
        pytest.skip("no network access in this environment")


def test_price_history_returns_quarterly_closes():
    _require_network()
    result = price_history.invoke({"ticker": "NVDA", "period": "1y"})
    assert result["ticker"] == "NVDA"
    assert "error" not in result
    assert len(result["quarterly_close"]) > 0


def test_fundamental_ratios_returns_known_fields():
    _require_network()
    result = fundamental_ratios.invoke({"ticker": "AAPL"})
    assert result["ticker"] == "AAPL"
    assert "error" not in result
    assert "grossMargins" in result


def test_edgar_filings_returns_recent_10q():
    _require_network()
    result = edgar_filings.invoke({"ticker": "NVDA", "form_type": "10-Q", "limit": 2})
    assert result["ticker"] == "NVDA"
    assert "error" not in result
    assert len(result["filings"]) > 0
    assert result["filings"][0]["form"] == "10-Q"


def test_edgar_filings_unknown_ticker():
    _require_network()
    result = edgar_filings.invoke({"ticker": "NOT_A_REAL_TICKER"})
    assert "error" in result
