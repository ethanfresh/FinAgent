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
@click.option("--dataset", default="evals/golden.jsonl", show_default=True)
@click.option("--rounds", default=2, show_default=True)
@click.option("--candidates", default=3, show_default=True)
def optimize(dataset: str, rounds: int, candidates: int):
    """Optimize the agent's system prompt against the eval harness (LLM proposes, eval scores)."""
    from finagent.optimize import optimize as run_optimize

    result = run_optimize(dataset, rounds=rounds, candidates_per_round=candidates)
    click.echo(
        f"baseline {result['baseline_score']:.0%} -> final {result['final_score']:.0%} "
        f"(improved: {result['improved']}, {result['rounds_run']} rounds)"
    )


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True)
def serve(host: str, port: int):
    """Run the FinAgent web app."""
    import uvicorn

    uvicorn.run("finagent.web:app", host=host, port=port)


@main.command(name="index-filings")
@click.argument("ticker")
@click.option("--form-type", default="10-K", show_default=True)
@click.option("--limit", default=1, show_default=True, help="How many recent filings of this form type to index.")
def index_filings(ticker: str, form_type: str, limit: int):
    """Fetch, chunk, embed, and store a company's filing text for filing_search (requires: uv sync --extra rag)."""
    from finagent.rag.ingest import index_filing

    result = index_filing(ticker, form_type=form_type, limit=limit)
    if "error" in result:
        raise click.ClickException(result["error"])
    for f in result["filings_indexed"]:
        click.echo(f"indexed {result['ticker']} {form_type} filed {f['filed']}: {f['chunks']} chunks")


@main.command(name="retrieval-eval")
@click.option("--dataset", default="evals/retrieval_golden.jsonl", show_default=True)
def retrieval_eval(dataset: str):
    """Run the filing-search retrieval pipeline against its golden dataset (requires: uv sync --extra rag)."""
    from finagent.evals.retrieval import run_retrieval_eval

    run_retrieval_eval(dataset)


@main.command()
@click.option("--turns", default=4, show_default=True, help="Chat turns per simulated persona.")
@click.option("--base-url", default="http://localhost:8000", show_default=True, help="FinAgent web app to talk to (must already be running).")
def redteam(turns: int, base_url: str):
    """Simulate real users chatting with FinAgent live and report problems found in its replies."""
    from finagent.redteam import run_redteam, write_report

    report = run_redteam(turns=turns, base_url=base_url)
    json_path, md_path = write_report(report)
    total_issues = sum(len(s["issues"]) for s in report["sessions"])
    click.echo(f"{total_issues} issue(s) found across {len(report['sessions'])} persona sessions")
    click.echo(f"report: {md_path}")
    click.echo(f"raw:    {json_path}")


if __name__ == "__main__":
    main()
