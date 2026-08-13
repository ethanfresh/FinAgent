"""Automated prompt optimization: propose system-prompt variants with the LLM,
score each against the eval harness, and adopt whichever scores best — a small,
real implementation of the "prompt optimization" pattern (OPRO/APE-style
hill-climbing), not a stub.
"""

import os

import ray
import wandb
from anthropic import Anthropic

from finagent.agent.graph import PROMPT_PATH, build_graph, load_system_prompt, run_graph
from finagent.evals.run import MODEL_ID, _git_sha, _load_cases, judge_score

CANDIDATE_MARKER = "===CANDIDATE==="

META_PROMPT = """You are optimizing the system prompt for a financial research agent. \
The agent has access to tools for SEC filings, stock price history, and fundamental \
ratios, and its answers are graded by an LLM judge for correctness and groundedness.

Current system prompt:
---
{current_prompt}
---

Cases where this prompt produced an answer the judge scored 0 (fail):
{failure_text}

Propose {n} improved, complete, self-contained system prompts that would fix the \
failure patterns above. Each candidate must still: use the tools before answering, \
cite figures, avoid investment advice, answer in plain text with no markdown except \
`[title](url)` links for sources. Do not just describe changes — write out each full \
prompt. Separate candidates with the exact line:
{marker}

No other commentary before, between, or after the candidates."""


@ray.remote
def _score_case(case: dict, system_prompt: str) -> dict:
    graph = build_graph(MODEL_ID, system_prompt=system_prompt)
    result = run_graph(graph, case["question"], backend="optimize")
    judge = Anthropic()
    score = judge_score(judge, case["question"], case["reference"], result.answer)
    return {"question": case["question"], "reference": case["reference"], "answer": result.answer, "score": score}


def _score_prompt(system_prompt: str, cases: list[dict]) -> tuple[float, list[dict]]:
    results = ray.get([_score_case.remote(c, system_prompt) for c in cases])
    pass_rate = sum(1 for r in results if r["score"] == "1") / len(results) if results else 0.0
    return pass_rate, results


def propose_variants(current_prompt: str, failures: list[dict], n: int) -> list[str]:
    client = Anthropic()
    if failures:
        failure_text = "\n\n".join(
            f"Question: {f['question']}\nReference: {f['reference']}\nAgent answer: {f['answer']}"
            for f in failures
        )
    else:
        failure_text = "(none — the prompt passed every case; propose variants that might generalize even better)"

    prompt = META_PROMPT.format(current_prompt=current_prompt, failure_text=failure_text, n=n, marker=CANDIDATE_MARKER)
    resp = client.messages.create(model=MODEL_ID, max_tokens=4000, messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in resp.content if b.type == "text")
    candidates = [c.strip() for c in text.split(CANDIDATE_MARKER) if c.strip()]
    return candidates[:n]


def optimize(dataset_path: str = "evals/golden.jsonl", rounds: int = 2, candidates_per_round: int = 3) -> dict:
    cases = _load_cases(dataset_path)
    if not ray.is_initialized():
        ray.init(num_cpus=max(len(cases), 1), ignore_reinit_error=True, log_to_driver=False)

    current_prompt = load_system_prompt()
    current_score, current_results = _score_prompt(current_prompt, cases)
    baseline_score = current_score
    print(f"baseline: {current_score:.0%} ({len(cases)} cases)")

    history = [{"round": 0, "score": current_score, "prompt": current_prompt}]

    for round_num in range(1, rounds + 1):
        failures = [r for r in current_results if r["score"] != "1"]
        candidates = propose_variants(current_prompt, failures, candidates_per_round)
        print(f"round {round_num}: proposed {len(candidates)} candidates ({len(failures)} failures to fix)")

        scored = []
        for i, cand in enumerate(candidates, 1):
            score, results = _score_prompt(cand, cases)
            print(f"  candidate {i}: {score:.0%}")
            scored.append((score, cand, results))

        if not scored:
            print("  no usable candidates returned, stopping")
            break

        best_score, best_prompt, best_results = max(scored, key=lambda x: x[0])
        history.append({"round": round_num, "score": best_score, "prompt": best_prompt})

        if best_score > current_score:
            print(f"  -> {best_score:.0%} beats current {current_score:.0%}, adopting")
            current_score, current_prompt, current_results = best_score, best_prompt, best_results
        else:
            print(f"  -> best candidate {best_score:.0%} does not beat current {current_score:.0%}, keeping current")

    improved = current_score > baseline_score
    if improved:
        PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROMPT_PATH.write_text(current_prompt.strip() + "\n")
        print(f"Wrote new champion prompt to {PROMPT_PATH} ({baseline_score:.0%} -> {current_score:.0%})")
    else:
        print(f"No improvement over baseline ({baseline_score:.0%}); prompt file left unchanged")

    _log_to_wandb(history, baseline_score, current_score)
    return {
        "baseline_score": baseline_score,
        "final_score": current_score,
        "improved": improved,
        "rounds_run": len(history) - 1,
    }


def _log_to_wandb(history: list[dict], baseline_score: float, final_score: float) -> None:
    if not os.environ.get("WANDB_API_KEY"):
        return

    run = wandb.init(project="finagent-optimize", config={"git_sha": _git_sha(), "model": MODEL_ID})
    table = wandb.Table(columns=["round", "score", "prompt"])
    for h in history:
        table.add_data(h["round"], h["score"], h["prompt"])
    run.log({"baseline_score": baseline_score, "final_score": final_score, "rounds": table})
    run.finish()
