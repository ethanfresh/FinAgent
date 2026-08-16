import os

from langchain_core.tools import tool

from finagent.rag.store import RagDependencyError


def _index_on_demand() -> bool:
    """Whether a filing that isn't in the index yet may be fetched and embedded
    inline. True locally, where waiting a couple of minutes is fine. Turned off
    on the hosted demo, where indexing a full 10-K on a shared CPU would stall
    the request past any sensible timeout — there, the index shipped in the
    image is the whole corpus.
    """
    return os.environ.get("FINAGENT_RAG_INDEX_ON_DEMAND", "1").strip().lower() not in {"0", "false", "no", "off"}


@tool
def filing_search(ticker: str, query: str, form_type: str = "10-K", limit: int = 5) -> dict:
    """Semantically search the full text of a company's SEC filing for passages relevant to a question or topic.

    Indexes the filing on first use for a given ticker/form_type (fetches, chunks, and embeds it locally), then
    searches it. Use this for questions about a filing's actual content — risk factors, MD&A, specific
    disclosures — not just which filings exist or their metadata (use edgar_filings for that).

    Args:
        ticker: Stock ticker symbol, e.g. "NVDA".
        query: Natural-language question or topic to search for within the filing text.
        form_type: SEC form type to search within, e.g. "10-K", "10-Q".
        limit: Max number of passages to return.
    """
    from finagent.rag.ingest import index_filing
    from finagent.rag.store import (
        embed_texts,
        get_collection,
        indexed_filings,
        is_indexed,
    )

    try:
        if not is_indexed(ticker, form_type):
            if not _index_on_demand():
                available = ", ".join(f"{f['ticker']} {f['form_type']}" for f in indexed_filings())
                return {
                    "ticker": ticker.upper(),
                    "query": query,
                    "error": (
                        f"{ticker.upper()} {form_type} is not in this deployment's filing index, and "
                        f"indexing new filings is disabled here. Searchable filings: {available or 'none'}."
                    ),
                }
            result = index_filing(ticker, form_type=form_type, limit=1)
            if "error" in result:
                return {"ticker": ticker, "query": query, "error": result["error"]}

        collection = get_collection()
        query_embedding = embed_texts([query])[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where={"$and": [{"ticker": ticker.upper()}, {"form_type": form_type}]},
        )
    except RagDependencyError as exc:
        return {"ticker": ticker, "query": query, "error": str(exc)}

    if not results["ids"][0]:
        return {"ticker": ticker, "query": query, "error": "no indexed content found"}

    passages = [
        {"text": doc, "section": meta.get("section") or None, "filed": meta.get("filed"), "url": meta.get("url")}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0], strict=True)
    ]
    return {"ticker": ticker.upper(), "form_type": form_type, "query": query, "passages": passages}
