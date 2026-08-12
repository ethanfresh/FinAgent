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
    closes = hist["Close"]
    closes.index = closes.index.tz_localize(None)
    # Group by calendar quarter but key each value by the actual last trading
    # date observed in that quarter (not pandas' resample bin-edge label,
    # which would tag an in-progress quarter with its future end date).
    quarterly = closes.groupby(closes.index.to_period("Q")).apply(lambda s: (s.index[-1].date(), s.iloc[-1]))
    return {
        "ticker": ticker,
        "period": period,
        "quarterly_close": {str(date): round(price, 2) for date, price in quarterly},
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
