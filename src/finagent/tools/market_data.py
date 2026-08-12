import yfinance as yf
from langchain_core.tools import tool


@tool
def price_history(ticker: str, period: str = "1y") -> dict:
    """Get historical closing prices for a stock ticker.

    Args:
        ticker: Stock ticker symbol, e.g. "NVDA".
        period: Lookback window, e.g. "1mo", "6mo", "1y", "5y".
    """
    hist = yf.Ticker(ticker).history(period=period)
    if hist.empty:
        return {"ticker": ticker, "error": "no price data found"}
    closes = hist["Close"].resample("QE").last().dropna()
    return {
        "ticker": ticker,
        "period": period,
        "quarterly_close": {str(idx.date()): round(val, 2) for idx, val in closes.items()},
    }


@tool
def fundamental_ratios(ticker: str) -> dict:
    """Get key fundamental ratios for a stock ticker (margins, P/E, etc).

    Args:
        ticker: Stock ticker symbol, e.g. "NVDA".
    """
    info = yf.Ticker(ticker).info
    if not info or info.get("regularMarketPrice") is None and info.get("trailingPE") is None:
        return {"ticker": ticker, "error": "no fundamental data found"}
    fields = [
        "grossMargins",
        "operatingMargins",
        "profitMargins",
        "trailingPE",
        "forwardPE",
        "returnOnEquity",
        "debtToEquity",
    ]
    return {"ticker": ticker, **{f: info.get(f) for f in fields}}
