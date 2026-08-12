import json

from anthropic import Anthropic
from langchain_core.messages import AIMessage, HumanMessage

from finagent.agent.graph import build_graph

JUDGE_PROMPT = """You are grading a financial research agent's answer for a golden eval set.

Question: {question}
Reference (what a correct, grounded answer should cover): {reference}
Agent's answer: {answer}

Score the agent's answer 1 if it is consistent with the reference and grounded in specific
figures (not vague or evasive), otherwise 0. Respond with only the digit 1 or 0."""


def _load_cases(dataset_path: str) -> list[dict]:
    with open(dataset_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_eval(dataset_path: str) -> list[dict]:
    graph = build_graph()
    judge = Anthropic()
    cases = _load_cases(dataset_path)
    results = []

    for case in cases:
        out = graph.invoke({"messages": [HumanMessage(content=case["question"])]})
        answer_msg = next(m for m in reversed(out["messages"]) if isinstance(m, AIMessage) and m.content)
        answer = answer_msg.content

        verdict = judge.messages.create(
            model="claude-sonnet-5",
            max_tokens=5,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        question=case["question"], reference=case["reference"], answer=answer
                    ),
                }
            ],
        )
        score = verdict.content[0].text.strip()
        results.append({"question": case["question"], "answer": answer, "score": score})
        print(f"[{score}] {case['question']}")

    pass_rate = sum(1 for r in results if r["score"] == "1") / len(results) if results else 0.0
    print(f"\n{pass_rate:.0%} pass rate ({len(results)} cases)")
    return results
