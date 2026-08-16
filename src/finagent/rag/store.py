"""Local vector store for filing_search: a persistent Chroma collection,
embedded with a small local sentence-transformers model — no external vector
DB service or embeddings API key required, so it's runnable and verifiable
the same way the rest of this project is.

chromadb/sentence-transformers are only imported inside these functions, not
at module load time, so importing this module (and anything that imports
finagent.tools, which is most of the app) doesn't require the `rag` extra —
only actually calling filing_search does. Mirrors how BedrockAgentRunner
fails cleanly rather than being a hard dependency of the whole app.
"""

import os
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Relative by default so local dev keeps its index in the repo root, but
# overridable because the deployed container needs it on a writable path that
# survives restarts (a mounted volume) rather than under the process CWD.
CHROMA_DIR = Path(os.environ.get("FINAGENT_CHROMA_DIR", ".chroma"))
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "sec_filings"


class RagDependencyError(RuntimeError):
    """Raised when the `rag` extra (chromadb, sentence-transformers) isn't installed."""


@lru_cache
def _embedder():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RagDependencyError("sentence-transformers is not installed — run `uv sync --extra rag`") from exc
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    return _embedder().encode(texts, show_progress_bar=False).tolist()


@lru_cache
def _client():
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError as exc:
        raise RagDependencyError("chromadb is not installed — run `uv sync --extra rag`") from exc
    return chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))


def get_collection():
    return _client().get_or_create_collection(COLLECTION_NAME)


def is_indexed(ticker: str, form_type: str) -> bool:
    result = get_collection().get(where={"$and": [{"ticker": ticker.upper()}, {"form_type": form_type}]}, limit=1)
    return len(result["ids"]) > 0


def indexed_filings() -> list[dict]:
    """The distinct filings currently in the collection.

    Used to tell the caller what's actually searchable when on-demand indexing
    is turned off (see filing_search), so a miss produces a useful answer
    rather than a bare "not found".
    """
    result = get_collection().get(include=["metadatas"])
    seen: dict[tuple[str, str], dict] = {}
    for meta in result.get("metadatas") or []:
        ticker, form_type = meta.get("ticker"), meta.get("form_type")
        if ticker and form_type and (ticker, form_type) not in seen:
            seen[(ticker, form_type)] = {"ticker": ticker, "form_type": form_type, "filed": meta.get("filed")}
    return sorted(seen.values(), key=lambda d: (d["ticker"], d["form_type"]))
