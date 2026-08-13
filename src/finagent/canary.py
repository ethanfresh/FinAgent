import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import ray
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from finagent.evals.run import _git_sha, _load_cases, _run_case

os.environ.setdefault("FINAGENT_ENV", "canary")

HISTORY_PATH = Path(".finagent/canary_history.json")
ROLLING_WINDOW = 5
PUSHGATEWAY_ADDR = os.environ.get("PUSHGATEWAY_ADDR", "localhost:9091")


def _push_metrics(pass_rate: float, ok: bool) -> None:
    """Push the canary result to Prometheus Pushgateway so Prometheus can alert
    on it — canary is a batch job, not a scrapeable long-running process, so it
    reports its result rather than being scraped (the standard Prometheus
    pattern for batch/cron jobs)."""
    registry = CollectorRegistry()
    Gauge("finagent_canary_pass_rate", "Most recent canary run pass rate", registry=registry).set(pass_rate)
    Gauge("finagent_canary_ok", "1 if the most recent canary run passed threshold, else 0", registry=registry).set(
        1 if ok else 0
    )
    try:
        push_to_gateway(PUSHGATEWAY_ADDR, job="finagent_canary", registry=registry)
    except OSError as exc:
        print(f"(pushgateway unreachable at {PUSHGATEWAY_ADDR}, skipping metric push: {exc})")


def _load_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    return json.loads(HISTORY_PATH.read_text())


def _save_history(history: list[dict]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, indent=2))


def run_canary(dataset_path: str, threshold: float, subset_size: int = 5) -> bool:
    cases = _load_cases(dataset_path)[:subset_size]
    git_sha = _git_sha()

    if not ray.is_initialized():
        ray.init(num_cpus=max(len(cases), 1), ignore_reinit_error=True, log_to_driver=False)

    results = ray.get([_run_case.remote(c) for c in cases])
    pass_rate = sum(1 for r in results if r["score"] == "1") / len(results) if results else 0.0

    history = _load_history()
    recent = [h["pass_rate"] for h in history[-ROLLING_WINDOW:]]
    baseline = sum(recent) / len(recent) if recent else pass_rate

    history.append(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "git_sha": git_sha,
            "pass_rate": pass_rate,
            "n_cases": len(results),
        }
    )
    _save_history(history)

    print(f"Canary pass rate: {pass_rate:.0%} ({len(results)} cases, git {git_sha})")
    print(f"Rolling baseline (last {len(recent)} runs): {baseline:.0%}")
    print(f"Threshold: {threshold:.0%}")

    ok = pass_rate >= threshold
    if not ok:
        print(f"FAIL: pass rate {pass_rate:.0%} is below threshold {threshold:.0%}")
    elif recent and pass_rate < baseline - 0.15:
        print(f"WARN: pass rate {pass_rate:.0%} dropped sharply vs rolling baseline {baseline:.0%}")
    else:
        print("OK")

    _push_metrics(pass_rate, ok)
    return ok


def main(dataset_path: str, threshold: float, subset_size: int = 5) -> None:
    ok = run_canary(dataset_path, threshold, subset_size)
    sys.exit(0 if ok else 1)
