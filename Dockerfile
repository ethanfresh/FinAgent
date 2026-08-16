# Multi-stage on purpose. A single-stage build of this image came out at
# 7.7GB: uv's download cache shared a layer with the venv, and the `chown -R`
# that handed /app to the runtime user rewrote every file into a second 2.2GB
# copy. Building in one stage and copying only the finished tree into a clean
# runtime avoids both.

# ---- build ----------------------------------------------------------------
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/opt/hf-cache

# Dependencies first, so editing application code doesn't invalidate this layer.
# --extra rag pulls chromadb + sentence-transformers so filing_search works;
# torch resolves to the CPU-only wheel via the index pinned in pyproject.toml.
# The cache mount keeps uv's downloads out of the image entirely.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra rag

# Bake the embedding model into the image. Without this, the first
# filing_search call pays a ~90MB download, and pays it again after every
# restart because the container filesystem is ephemeral.
RUN /app/.venv/bin/python -c \
    "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra rag

# ---- runtime --------------------------------------------------------------
FROM python:3.11-slim AS runtime

# PATH puts the venv first so the console script and python resolve without uv.
# FINAGENT_CHROMA_DIR is absolute because the vector store must not depend on
# the process working directory.
ENV HF_HOME=/opt/hf-cache \
    TOKENIZERS_PARALLELISM=false \
    ANONYMIZED_TELEMETRY=False \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH \
    PORT=8000 \
    FINAGENT_CHROMA_DIR=/app/.chroma

# Create the user before the tree arrives so COPY --chown can hand it over in
# place. The app needs write access to the vector store and the red-team report
# directory, so it owns /app rather than merely reading it.
RUN useradd --create-home --uid 10001 finagent

WORKDIR /app
COPY --from=builder --chown=finagent:finagent /app /app
COPY --from=builder --chown=finagent:finagent /opt/hf-cache /opt/hf-cache

USER finagent

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT','8000') + '/health').read()"

# The venv's console script directly: uv isn't needed at runtime, and skipping
# it avoids re-checking the lockfile on every container start.
CMD ["sh", "-c", "finagent serve --host 0.0.0.0 --port ${PORT:-8000}"]
