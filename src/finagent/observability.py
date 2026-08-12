import os
from functools import lru_cache


@lru_cache
def langfuse_callbacks() -> list:
    """Return LangFuse tracing callbacks, or an empty list if no keys are configured."""
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return []
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]
