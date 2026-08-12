import os
import time
from functools import lru_cache
from pathlib import Path

import sentry_sdk
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

load_dotenv()

from finagent.agent.graph import build_graph, extract_text
from finagent.observability import langfuse_callbacks

sentry_sdk.init(dsn=os.environ.get("SENTRY_DSN"), traces_sample_rate=0.1)

REQUEST_COUNT = Counter("finagent_requests_total", "Total /api/ask requests", ["status"])
REQUEST_LATENCY = Histogram("finagent_request_latency_seconds", "Latency of /api/ask requests")
TOOL_CALLS = Counter("finagent_tool_calls_total", "Tool invocations by tool name", ["tool"])

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"

app = FastAPI(title="FinAgent")


class AskRequest(BaseModel):
    question: str


class ToolCallOut(BaseModel):
    name: str
    args: dict


class AskResponse(BaseModel):
    answer: str
    tool_calls: list[ToolCallOut]


@lru_cache
def _graph():
    return build_graph()


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    start = time.perf_counter()
    try:
        question = req.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="question must not be empty")

        result = _graph().invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"callbacks": langfuse_callbacks(), "metadata": {"environment": "web"}},
        )
        messages = result["messages"]

        tool_calls = [
            ToolCallOut(name=call["name"], args=call["args"])
            for m in messages
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
            for call in m.tool_calls
        ]
        for call in tool_calls:
            TOOL_CALLS.labels(tool=call.name).inc()

        final = next((m for m in reversed(messages) if isinstance(m, AIMessage) and m.content), None)
        if final is None:
            raise HTTPException(status_code=502, detail="agent produced no answer")

        REQUEST_COUNT.labels(status="ok").inc()
        return AskResponse(answer=extract_text(final.content), tool_calls=tool_calls)
    except HTTPException:
        REQUEST_COUNT.labels(status="error").inc()
        raise
    except Exception:
        REQUEST_COUNT.labels(status="error").inc()
        raise
    finally:
        REQUEST_LATENCY.observe(time.perf_counter() - start)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _counter_by_label(counter: Counter, label: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for family in counter.collect():
        for sample in family.samples:
            if sample.name.endswith("_total") and label in sample.labels:
                out[sample.labels[label]] = sample.value
    return out


def _histogram_stats(hist: Histogram) -> dict[str, float]:
    count = total = 0.0
    for family in hist.collect():
        for sample in family.samples:
            if sample.name.endswith("_count"):
                count = sample.value
            elif sample.name.endswith("_sum"):
                total = sample.value
    return {"count": count, "avg_latency_seconds": round(total / count, 3) if count else 0.0}


@app.get("/api/stats")
def stats() -> dict:
    return {
        "requests_by_status": _counter_by_label(REQUEST_COUNT, "status"),
        "tool_calls": _counter_by_label(TOOL_CALLS, "tool"),
        "latency": _histogram_stats(REQUEST_LATENCY),
    }


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")
