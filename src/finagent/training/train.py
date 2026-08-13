"""Fine-tune a small local classifier on judge-graded transcripts (SFT).

This is a real, CPU-runnable PyTorch training loop — tokenize, forward pass,
cross-entropy loss, backward, optimizer step — used as a local stand-in for
the SageMaker/GPU fine-tuning path described in the README, which needs
cloud training infra this dev environment doesn't have.

The task: given a (question, answer) pair, predict whether the eval judge
would grade it a pass (1) or fail (0) — i.e. distill the judge's signal into
a small, fast local model. That's the "SFT on graded transcripts" mentioned
throughout the README's eval/fine-tuning story, made concrete.
"""

from pathlib import Path

import torch
from anthropic import Anthropic
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from finagent.evals.run import JUDGE_PROMPT, MODEL_ID, _load_cases
from finagent.runner import load_runner

BASE_MODEL = "prajjwal1/bert-tiny"
CHECKPOINT_DIR = Path("artifacts/judge-checkpoint")


class _TranscriptDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["question"],
            ex["answer"],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        label = 1 if str(ex["score"]).strip() == "1" else 0
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def collect_training_examples(dataset_path: str) -> list[dict]:
    """Run the agent against the golden set and grade each answer with the same
    judge the eval harness uses — the graded transcripts become training data."""
    cases = _load_cases(dataset_path)
    runner = load_runner()
    judge = Anthropic()
    examples = []

    for case in cases:
        answer = runner.run(case["question"]).answer
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
        examples.append({"question": case["question"], "answer": answer, "score": score})
        print(f"[{score}] {case['question']}")

    return examples


def train_judge(dataset_path: str = "evals/golden.jsonl", epochs: int = 3, lr: float = 2e-5) -> dict:
    examples = collect_training_examples(dataset_path)
    if len(examples) < 2:
        raise ValueError("need at least 2 graded examples to fine-tune")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    model.train()

    dataset = _TranscriptDataset(examples, tokenizer)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        epoch_losses = []
        for batch in loader:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            out.loss.backward()
            optimizer.step()
            epoch_losses.append(out.loss.item())
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        print(f"epoch {epoch + 1}/{epochs} — loss {avg_loss:.4f}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CHECKPOINT_DIR)
    tokenizer.save_pretrained(CHECKPOINT_DIR)

    return {"n_examples": len(examples), "losses": losses, "checkpoint_dir": str(CHECKPOINT_DIR)}
