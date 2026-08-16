FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# HF_HOME points the model cache inside the image so a cold start never reaches
# Hugging Face. PORT is read by `finagent serve`, so the same image runs on any
# platform that assigns a port at runtime. FINAGENT_CHROMA_DIR is absolute
# because the vector store must not depend on the process working directory.
ENV HF_HOME=/app/.hf-cache \
    TOKENIZERS_PARALLELISM=false \
    ANONYMIZED_TELEMETRY=False \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    FINAGENT_CHROMA_DIR=/app/.chroma \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Dependencies first, so editing application code doesn't invalidate this layer.
# --extra rag pulls chromadb + sentence-transformers so filing_search works;
# torch resolves to the CPU-only wheel via the index pinned in pyproject.toml.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --extra rag

# Bake the embedding model into the image. Without this, the first
# filing_search call pays a ~90MB download, and pays it again after every
# restart because the container filesystem is ephemeral.
RUN /app/.venv/bin/python -c \
    "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .
RUN uv sync --frozen --no-dev --extra rag

# Non-root at runtime. The app still needs write access to the vector store
# (tickers not baked into the image get indexed on demand) and to the red-team
# report directory.
RUN useradd --create-home --uid 10001 finagent \
    && mkdir -p /app/.chroma \
    && chown -R finagent:finagent /app
USER finagent

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD /app/.venv/bin/python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT','8000') + '/health').read()"

CMD ["sh", "-c", "uv run finagent serve --host 0.0.0.0 --port ${PORT:-8000}"]
