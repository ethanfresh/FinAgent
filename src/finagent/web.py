import json
import os
import threading
import time
from collections import defaultdict, deque
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
from pydantic import BaseModel, Field

load_dotenv()
os.environ.setdefault("FINAGENT_ENV", "web")

from finagent.redteam import DEFAULT_PERSONAS, run_redteam, write_report
from finagent.runner import load_runner

sentry_sdk.init(dsn=os.environ.get("SENTRY_DSN"), traces_sample_rate=0.1)

REQUEST_COUNT = Counter("finagent_requests_total", "Total /api/ask requests", ["status"])
REQUEST_LATENCY = Histogram("finagent_request_latency_seconds", "Latency of /api/ask requests")
TOOL_CALLS = Counter("finagent_tool_calls_total", "Tool invocations by tool name", ["tool"])

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# Public-demo hardening. The hosted demo answers questions on the maintainer's
# own Anthropic API key, so every endpoint that spends tokens needs a ceiling
# the caller can't raise. All of this is off/unbounded-enough by default so
# local dev, the CLI, and the eval harness behave exactly as before.
PUBLIC_DEMO = _env_flag("FINAGENT_PUBLIC_DEMO")

# These ceilings exist to bound what a stranger can spend, so they only bind
# when the app is actually exposed as the public demo. Applying them locally
# would throttle the project's own tooling: `finagent redteam` drives four
# personas through several turns each in quick succession, which trips a
# 6-per-minute limit partway through a run. A zero means "no limit"; every
# value stays individually overridable by environment variable.
MAX_QUESTION_CHARS = _env_int("FINAGENT_MAX_QUESTION_CHARS", 1000 if PUBLIC_DEMO else 100_000)
MAX_HISTORY_MESSAGES = _env_int("FINAGENT_MAX_HISTORY_MESSAGES", 20 if PUBLIC_DEMO else 200)
RATE_LIMIT_PER_MINUTE = _env_int("FINAGENT_RATE_LIMIT_PER_MINUTE", 6 if PUBLIC_DEMO else 0)
RATE_LIMIT_PER_HOUR = _env_int("FINAGENT_RATE_LIMIT_PER_HOUR", 40 if PUBLIC_DEMO else 0)
GLOBAL_DAILY_CAP = _env_int("FINAGENT_GLOBAL_DAILY_CAP", 500 if PUBLIC_DEMO else 0)
MAX_REDTEAM_TURNS = 8


class _SlidingWindowLimiter:
    """Request ceilings for /api/ask, held in process memory.

    Deliberately not backed by Redis: the demo runs as a single container, and
    a limiter that resets on redeploy is the right amount of machinery for
    capping a portfolio demo's spend. The global daily counter is the actual
    budget backstop — per-IP limits alone don't help when traffic arrives from
    many addresses.
    """

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._global: deque[float] = deque()
        self._lock = threading.Lock()

    @staticmethod
    def _prune(window: deque[float], now: float, span: float) -> None:
        while window and now - window[0] > span:
            window.popleft()

    def _sweep(self, now: float) -> None:
        """Drop per-IP windows that can no longer reject anything, so a long
        tail of one-off visitors doesn't grow the dict without bound."""
        for key in [k for k, v in self._hits.items() if not v or now - v[-1] > 3600]:
            del self._hits[key]

    def check(self, key: str) -> str | None:
        """Record a request. Returns None to allow, or a reason to reject."""
        now = time.time()
        with self._lock:
            self._prune(self._global, now, 86400)
            if GLOBAL_DAILY_CAP and len(self._global) >= GLOBAL_DAILY_CAP:
                return "This demo has reached its daily question limit. Please try again tomorrow."

            if len(self._hits) > 5000:
                self._sweep(now)

            hits = self._hits[key]
            self._prune(hits, now, 3600)
            if RATE_LIMIT_PER_HOUR and len(hits) >= RATE_LIMIT_PER_HOUR:
                return "Hourly limit reached for this demo. Please try again later."
            if RATE_LIMIT_PER_MINUTE and sum(1 for t in hits if now - t <= 60) >= RATE_LIMIT_PER_MINUTE:
                return "You're sending questions a little too quickly — wait a moment and try again."

            hits.append(now)
            self._global.append(now)
            return None


_limiter = _SlidingWindowLimiter()


def _client_key(request: Request) -> str:
    """Best-effort caller identity for rate limiting.

    Behind a PaaS proxy the socket peer is always the proxy, so prefer the
    forwarded-address headers. These are spoofable when the app is exposed
    directly, which is why GLOBAL_DAILY_CAP (which no header can bypass) is
    the limit that actually protects the API budget.
    """
    for header in ("fly-client-ip", "x-forwarded-for"):
        value = request.headers.get(header)
        if value:
            return value.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


app = FastAPI(title="FinAgent")


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=MAX_QUESTION_CHARS * 20)


class AskRequest(BaseModel):
    question: str = Field(max_length=MAX_QUESTION_CHARS)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=MAX_HISTORY_MESSAGES)


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
def ask(req: AskRequest, request: Request) -> AskResponse:
    limit_reason = _limiter.check(_client_key(request))
    if limit_reason:
        REQUEST_COUNT.labels(status="rate_limited").inc()
        raise HTTPException(status_code=429, detail=limit_reason)

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


@app.get("/health")
def health() -> dict:
    """Liveness probe for the platform's health checks. Deliberately does not
    touch the model or the vector store — it answers whether the process is
    up, not whether every downstream dependency is."""
    return {"status": "ok"}


@app.get("/api/config")
def config() -> dict:
    """Feature flags the frontend needs so it can disable controls the server
    would reject anyway (currently: triggering a red-team run)."""
    return {"public_demo": PUBLIC_DEMO, "redteam_run_enabled": not PUBLIC_DEMO}


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
    turns: int = Field(default=4, ge=1, le=MAX_REDTEAM_TURNS)


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
    # A run is 4 personas x N turns of LLM calls against the app's own API key,
    # so on the public demo this is view-only: visitors read the saved reports
    # instead of triggering new runs. The pydantic bound on `turns` is the
    # second half of that — the UI clamps it client-side, which a direct POST
    # would otherwise ignore.
    if PUBLIC_DEMO:
        raise HTTPException(
            status_code=403,
            detail="Red-team runs are disabled on the public demo — browse the saved reports instead.",
        )

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
