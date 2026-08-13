import os

import click
import sentry_sdk
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("FINAGENT_ENV", "cli")

sentry_sdk.init(dsn=os.environ.get("SENTRY_DSN"), traces_sample_rate=0.1)


@click.group()
def main():
    """FinAgent: a financial research agent and its ops harness."""


@main.command()
@click.argument("question")
def ask(question: str):
    """Ask the agent a financial research question."""
    from finagent.runner import load_runner

    result = load_runner().run(question)
    click.echo(result.answer)


@main.command()
@click.option("--dataset", default="evals/golden.jsonl", show_default=True)
@click.option("--parallelism", default=4, show_default=True)
def eval(dataset: str, parallelism: int):
    """Run the agent against the golden eval dataset."""
    from finagent.evals.run import run_eval

    run_eval(dataset, parallelism=parallelism)


@main.command()
@click.option("--dataset", default="evals/golden.jsonl", show_default=True)
@click.option("--threshold", default=0.85, show_default=True)
@click.option("--subset-size", default=5, show_default=True)
def canary(dataset: str, threshold: float, subset_size: int):
    """Re-run a fixed case subset and fail if the pass rate drops below --threshold."""
    from finagent.canary import main as canary_main

    canary_main(dataset, threshold, subset_size)


@main.command()
@click.option("--dataset", default="evals/golden.jsonl", show_default=True)
@click.option("--epochs", default=3, show_default=True)
def train(dataset: str, epochs: int):
    """Fine-tune a small local judge classifier on graded transcripts (requires: uv sync --extra training)."""
    from finagent.training.train import train_judge

    result = train_judge(dataset, epochs=epochs)
    click.echo(f"Trained on {result['n_examples']} examples, checkpoint at {result['checkpoint_dir']}")


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True)
def serve(host: str, port: int):
    """Run the FinAgent web app."""
    import uvicorn

    uvicorn.run("finagent.web:app", host=host, port=port)


if __name__ == "__main__":
    main()
