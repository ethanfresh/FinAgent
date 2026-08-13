import yfinance as yf
from langchain_core.tools import tool


@tool
def executive_profile(ticker: str) -> dict:
    """Get a company's leadership: names, titles, ages, and compensation.

    Args:
        ticker: Stock ticker symbol, e.g. "NVDA".
    """
    info = yf.Ticker(ticker).info
    officers = info.get("companyOfficers", [])
    if not officers:
        return {"ticker": ticker, "error": "no executive data found"}
    profiles = [
        {
            "name": o.get("name"),
            "title": o.get("title"),
            "age": o.get("age"),
            "totalPay": o.get("totalPay"),
        }
        for o in officers
    ]
    return {"ticker": ticker, "officers": profiles}
