"""Ingest a company's SEC filings into the local vector index for filing_search.

Unlike edgar_filings (which only returns filing metadata and a link),
this fetches the filing's actual document, strips it to plain text, chunks it
(see chunking.py), embeds each chunk, and upserts into the persistent local
Chroma collection — so filing_search can retrieve real passages instead of
just pointing at the document.
"""

import warnings

import requests

from finagent.rag.chunking import chunk_filing
from finagent.rag.store import RagDependencyError, embed_texts, get_collection
from finagent.tools.edgar import HEADERS, edgar_filings


def fetch_filing_text(url: str) -> str:
    try:
        from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
    except ImportError as exc:
        raise RagDependencyError("beautifulsoup4 is not installed — run `uv sync --extra rag`") from exc

    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    # SEC filing documents mix inline XBRL tags into otherwise-ordinary HTML,
    # which trips bs4's "this looks like XML" heuristic — parsing with the
    # HTML parser is still correct here since we only want the readable text.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    lines = (line.strip() for line in soup.get_text(separator="\n").splitlines())
    return "\n".join(line for line in lines if line)


def index_filing(ticker: str, form_type: str = "10-K", limit: int = 1) -> dict:
    """Fetch, chunk, embed, and store a company's most recent filing(s) of a
    given form type. Returns a summary of what was indexed."""
    listing = edgar_filings.invoke({"ticker": ticker, "form_type": form_type, "limit": limit})
    if "error" in listing:
        return {"ticker": ticker, "form_type": form_type, "error": listing["error"]}
    if not listing["filings"]:
        return {"ticker": ticker, "form_type": form_type, "error": f"no {form_type} filings found"}

    collection = get_collection()
    indexed = []
    for filing in listing["filings"]:
        text = fetch_filing_text(filing["url"])
        chunks = chunk_filing(text)
        if not chunks:
            continue

        ids = [f"{ticker.upper()}:{form_type}:{filing['filed']}:{i}" for i in range(len(chunks))]
        embeddings = embed_texts([c["text"] for c in chunks])
        metadatas = [
            {
                "ticker": ticker.upper(),
                "form_type": form_type,
                "filed": filing["filed"],
                "url": filing["url"],
                "section": c["section"] or "",
            }
            for c in chunks
        ]
        collection.upsert(ids=ids, embeddings=embeddings, documents=[c["text"] for c in chunks], metadatas=metadatas)
        indexed.append({"filed": filing["filed"], "url": filing["url"], "chunks": len(chunks)})

    return {"ticker": ticker.upper(), "form_type": form_type, "filings_indexed": indexed}
