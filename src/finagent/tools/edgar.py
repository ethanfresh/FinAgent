import os
from functools import lru_cache

import requests
from langchain_core.tools import tool

SEC_CONTACT = os.environ.get("SEC_EDGAR_CONTACT", "FinAgent research contact@example.com")
HEADERS = {"User-Agent": SEC_CONTACT}


@lru_cache(maxsize=1)
def _ticker_to_cik() -> dict:
    resp = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return {row["ticker"].upper(): str(row["cik_str"]).zfill(10) for row in resp.json().values()}


@tool
def edgar_filings(ticker: str, form_type: str = "10-Q", limit: int = 4) -> dict:
    """Get a company's most recent SEC EDGAR filings of a given form type.

    Args:
        ticker: Stock ticker symbol, e.g. "NVDA".
        form_type: SEC form type, e.g. "10-K", "10-Q", "8-K".
        limit: Max number of filings to return.
    """
    cik = _ticker_to_cik().get(ticker.upper())
    if cik is None:
        return {"ticker": ticker, "error": "ticker not found in SEC EDGAR company list"}

    resp = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=HEADERS, timeout=10)
    resp.raise_for_status()
    recent = resp.json()["filings"]["recent"]

    matches = []
    for form, accession, date, doc in zip(
        recent["form"], recent["accessionNumber"], recent["filingDate"], recent["primaryDocument"]
    ):
        if form == form_type:
            accession_nodash = accession.replace("-", "")
            matches.append(
                {
                    "form": form,
                    "filed": date,
                    "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}/{doc}",
                }
            )
        if len(matches) >= limit:
            break

    return {"ticker": ticker, "cik": cik, "form_type": form_type, "filings": matches}
