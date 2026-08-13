import datetime as dt

import pytest
import requests

from finagent.tools.edgar import edgar_filings
from finagent.tools.executives import executive_profile
from finagent.tools.market_data import fundamental_ratios, price_history
from finagent.tools.news import company_news


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
    today = dt.datetime.now(tz=dt.UTC).date()
    for date_str in result["quarterly_close"]:
        assert dt.date.fromisoformat(date_str) <= today, "quarterly_close must not label data with a future date"


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


def test_company_news_returns_articles():
    _require_network()
    result = company_news.invoke({"ticker": "NVDA", "limit": 3})
    assert result["ticker"] == "NVDA"
    assert "error" not in result
    assert 0 < len(result["articles"]) <= 3
    assert result["articles"][0]["title"]


def test_executive_profile_returns_officers():
    _require_network()
    result = executive_profile.invoke({"ticker": "NVDA"})
    assert result["ticker"] == "NVDA"
    assert "error" not in result
    assert len(result["officers"]) > 0
    assert result["officers"][0]["name"]
