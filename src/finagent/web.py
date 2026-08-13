import json
import os
import threading
import time
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Literal

import sentry_sdk
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

load_dotenv()
os.environ.setdefault("FINAGENT_ENV", "web")

from finagent.redteam import DEFAULT_PERSONAS, run_redteam, write_report
from finagent.runner import load_runner

sentry_sdk.init(dsn=os.environ.get("SENTRY_DSN"), traces_sample_rate=0.1)

REQUEST_COUNT = Counter("finagent_requests_total", "Total /api/ask requests", ["status"])
REQUEST_LATENCY = Histogram("finagent_request_latency_seconds", "Latency of /api/ask requests")
TOOL_CALLS = Counter("finagent_tool_calls_total", "Tool invocations by tool name", ["tool"])

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"

app = FastAPI(title="FinAgent")


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str
    history: list[HistoryMessage] = []


class ToolCallOut(BaseModel):
    name: str
    args: dict


class AskResponse(BaseModel):
    answer: str
    tool_calls: list[ToolCallOut]


@lru_cache
def _runner():
    return load_runner()


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    start = time.perf_counter()
    try:
        question = req.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="question must not be empty")

        try:
            history = [h.model_dump() for h in req.history]
            result = _runner().run(question, history=history)
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        tool_calls = [ToolCallOut(name=c.name, args=c.args) for c in result.tool_calls]
        for call in tool_calls:
            TOOL_CALLS.labels(tool=call.name).inc()

        REQUEST_COUNT.labels(status="ok").inc()
        return AskResponse(answer=result.answer, tool_calls=tool_calls)
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


REDTEAM_DIR = Path("artifacts/redteam")
REDTEAM_FIXES_PATH = Path("redteam_fixes.json")

_redteam_lock = threading.Lock()
_redteam_state: dict = {"status": "idle", "turns": None, "started_at": None, "finished_at": None, "error": None}


class RedteamRunRequest(BaseModel):
    turns: int = 4


@app.get("/api/redteam/personas")
def redteam_personas() -> list[dict]:
    return [asdict(p) for p in DEFAULT_PERSONAS]


def _run_redteam_job(turns: int, base_url: str) -> None:
    try:
        report = run_redteam(turns=turns, base_url=base_url)
        write_report(report)
        with _redteam_lock:
            _redteam_state.update(status="done", finished_at=time.time())
    except Exception as exc:  # noqa: BLE001 — top-level boundary for a background thread; must not crash silently
        with _redteam_lock:
            _redteam_state.update(status="error", error=str(exc), finished_at=time.time())


@app.post("/api/redteam/run")
def redteam_run(req: RedteamRunRequest, request: Request) -> dict:
    with _redteam_lock:
        if _redteam_state["status"] == "running":
            return {"status": "already_running"}
        _redteam_state.update(status="running", turns=req.turns, started_at=time.time(), finished_at=None, error=None)

    base_url = str(request.base_url).rstrip("/")
    thread = threading.Thread(target=_run_redteam_job, args=(req.turns, base_url), daemon=True)
    thread.start()
    return {"status": "started"}


@app.get("/api/redteam/status")
def redteam_status() -> dict:
    with _redteam_lock:
        return dict(_redteam_state)


@app.get("/api/redteam/reports")
def redteam_reports() -> list[dict]:
    if not REDTEAM_DIR.exists():
        return []
    out = []
    for path in sorted(REDTEAM_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        issue_count = sum(len(s.get("issues", [])) for s in data.get("sessions", []))
        out.append(
            {
                "name": path.stem,
                "sessions": len(data.get("sessions", [])),
                "issues": issue_count,
                "turns_per_session": data.get("turns_per_session"),
            }
        )
    return out


@app.get("/api/redteam/reports/{name}")
def redteam_report(name: str) -> dict:
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_")
    path = REDTEAM_DIR / f"{safe_name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="report not found")
    return json.loads(path.read_text())


@app.get("/api/redteam/fixes")
def redteam_fixes() -> list[dict]:
    """Hand-curated record of real issues the red-team tester found and that were
    then actually fixed in code — not auto-inferred, since reliably detecting
    "this issue class no longer occurs" from LLM critic output across separate
    live runs isn't something to trust unsupervised; an engineer records each
    fix here once it's verified by a real post-fix run."""
    if not REDTEAM_FIXES_PATH.exists():
        return []
    return json.loads(REDTEAM_FIXES_PATH.read_text())


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")
