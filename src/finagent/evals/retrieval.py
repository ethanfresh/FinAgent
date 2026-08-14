"""Retrieval-quality eval for filing_search: recall@k against a small,
hand-verified golden set (evals/retrieval_golden.jsonl) — does at least one
of the top-k retrieved passages actually contain the expected content, in
the expected section?

Deterministic substring matching, not an LLM judge — "does this passage
contain X" doesn't need one, and staying deterministic makes a regression
unambiguous. Each case's expected_substring was taken from a real passage
this pipeline actually retrieved, not guessed at.

Runs sequentially, not Ray-parallelized like the QA eval harness in run.py:
each case may fetch and index a real SEC filing on first use, and hammering
SEC EDGAR with concurrent requests isn't considerate of their service.
"""

import json
import os

import wandb

from finagent.evals.run import _git_sha
from finagent.tools.filing_search import filing_search


def _load_cases(dataset_path: str) -> list[dict]:
    with open(dataset_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _run_case(case: dict) -> dict:
    result = filing_search.invoke(
        {
            "ticker": case["ticker"],
            "query": case["query"],
            "form_type": case.get("form_type", "10-K"),
            "limit": case.get("k", 5),
        }
    )
    passages = result.get("passages", [])
    expected = case["expected_substring"].lower()
    hit = any(expected in p["text"].lower() for p in passages)
    section_ok = any(
        case.get("expected_section_contains", "").lower() in (p["section"] or "").lower() for p in passages
    )
    return {**case, "hit": hit, "section_ok": section_ok, "passages_returned": len(passages)}


def _log_to_wandb(results: list[dict], recall: float, git_sha: str) -> None:
    if not os.environ.get("WANDB_API_KEY"):
        return
    run = wandb.init(project="finagent-evals", config={"git_sha": git_sha, "n_cases": len(results), "eval": "retrieval"})
    table = wandb.Table(columns=["ticker", "query", "hit", "section_ok", "passages_returned"])
    for r in results:
        table.add_data(r["ticker"], r["query"], r["hit"], r["section_ok"], r["passages_returned"])
    run.log({"retrieval_recall": recall, "cases": table})
    run.finish()


def run_retrieval_eval(dataset_path: str = "evals/retrieval_golden.jsonl") -> list[dict]:
    cases = _load_cases(dataset_path)
    results = [_run_case(c) for c in cases]

    for r in results:
        status = "HIT " if r["hit"] else "MISS"
        section_flag = "" if r["section_ok"] else " (wrong section)"
        print(f"[{status}] {r['ticker']} {r['form_type']}: {r['query']}{section_flag}")

    recall = sum(1 for r in results if r["hit"]) / len(results) if results else 0.0
    print(f"\nrecall@k: {recall:.0%} ({len(results)} cases)")

    _log_to_wandb(results, recall, _git_sha())
    return results
