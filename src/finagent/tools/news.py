import yfinance as yf
from langchain_core.tools import tool


@tool
def company_news(ticker: str, limit: int = 5) -> dict:
    """Get recent news headlines for a company.

    Args:
        ticker: Stock ticker symbol, e.g. "NVDA".
        limit: Max number of articles to return.
    """
    raw = yf.Ticker(ticker).news or []
    articles = []
    for item in raw[:limit]:
        content = item.get("content", {})
        articles.append(
            {
                "title": content.get("title"),
                "summary": content.get("summary"),
                "published": content.get("pubDate"),
                "publisher": (content.get("provider") or {}).get("displayName"),
                "url": (content.get("canonicalUrl") or {}).get("url"),
            }
        )
    if not articles:
        return {"ticker": ticker, "error": "no news found"}
    return {"ticker": ticker, "articles": articles}
