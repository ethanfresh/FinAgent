from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

from finagent.tools.edgar import edgar_filings as _edgar_filings
from finagent.tools.executives import executive_profile as _executive_profile
from finagent.tools.market_data import (
    fundamental_ratios as _fundamental_ratios,
)
from finagent.tools.market_data import price_history as _price_history
from finagent.tools.news import company_news as _company_news

mcp = FastMCP("finagent")


@mcp.tool()
def price_history(ticker: str, period: str = "1y") -> dict:
    """Get historical quarterly closing prices for a stock ticker."""
    return _price_history.invoke({"ticker": ticker, "period": period})


@mcp.tool()
def fundamental_ratios(ticker: str) -> dict:
    """Get key fundamental ratios for a stock ticker (margins, P/E, ROE, etc)."""
    return _fundamental_ratios.invoke({"ticker": ticker})


@mcp.tool()
def edgar_filings(ticker: str, form_type: str = "10-Q", limit: int = 4) -> dict:
    """Get a company's most recent SEC EDGAR filings of a given form type."""
    return _edgar_filings.invoke({"ticker": ticker, "form_type": form_type, "limit": limit})


@mcp.tool()
def company_news(ticker: str, limit: int = 5) -> dict:
    """Get recent news headlines for a company."""
    return _company_news.invoke({"ticker": ticker, "limit": limit})


@mcp.tool()
def executive_profile(ticker: str) -> dict:
    """Get a company's leadership: names, titles, ages, and compensation."""
    return _executive_profile.invoke({"ticker": ticker})


def run() -> None:
    mcp.run()


if __name__ == "__main__":
    run()
