import json
import subprocess

import ray
import wandb
from anthropic import Anthropic
from langchain_core.messages import AIMessage, HumanMessage

from finagent.agent.graph import build_graph, extract_text
from finagent.observability import langfuse_callbacks

JUDGE_PROMPT = """You are grading a financial research agent's answer for a golden eval set.

Question: {question}
Reference (what a correct, grounded answer should cover): {reference}
Agent's answer: {answer}

Score the agent's answer 1 if it is consistent with the reference and grounded in specific
figures (not vague or evasive), otherwise 0. Respond with only the digit 1 or 0."""

MODEL_ID = "claude-sonnet-5"


def _load_cases(dataset_path: str) -> list[dict]:
    with open(dataset_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@ray.remote
def _run_case(case: dict, git_sha: str) -> dict:
    graph = build_graph(MODEL_ID)
    judge = Anthropic()

    out = graph.invoke(
        {"messages": [HumanMessage(content=case["question"])]},
        config={"callbacks": langfuse_callbacks(), "metadata": {"environment": "eval", "git_sha": git_sha}},
    )
    answer_msg = next(m for m in reversed(out["messages"]) if isinstance(m, AIMessage) and m.content)
    answer = extract_text(answer_msg.content)

    verdict = judge.messages.create(
        model=MODEL_ID,
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    question=case["question"], reference=case["reference"], answer=answer
                ),
            }
        ],
    )
    text_block = next((b for b in verdict.content if b.type == "text"), None)
    score = text_block.text.strip() if text_block else "0"
    return {"question": case["question"], "answer": answer, "score": score}


def run_eval(dataset_path: str, parallelism: int = 4) -> list[dict]:
    cases = _load_cases(dataset_path)
    git_sha = _git_sha()

    if not ray.is_initialized():
        ray.init(num_cpus=max(parallelism, 1), ignore_reinit_error=True, log_to_driver=False)

    futures = [_run_case.remote(case, git_sha) for case in cases]
    results = ray.get(futures)

    for r in results:
        print(f"[{r['score']}] {r['question']}")

    pass_rate = sum(1 for r in results if r["score"] == "1") / len(results) if results else 0.0
    print(f"\n{pass_rate:.0%} pass rate ({len(results)} cases)")

    _log_to_wandb(results, pass_rate, git_sha)
    return results


def _log_to_wandb(results: list[dict], pass_rate: float, git_sha: str) -> None:
    import os

    if not os.environ.get("WANDB_API_KEY"):
        return

    run = wandb.init(
        project="finagent-evals",
        config={"git_sha": git_sha, "model": MODEL_ID, "n_cases": len(results)},
    )
    table = wandb.Table(columns=["question", "answer", "score"])
    for r in results:
        table.add_data(r["question"], r["answer"], r["score"])
    run.log({"pass_rate": pass_rate, "cases": table})
    run.finish()
