import click
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()


@click.group()
def main():
    """FinAgent: a financial research agent and its ops harness."""


@main.command()
@click.argument("question")
def ask(question: str):
    """Ask the agent a financial research question."""
    from finagent.agent.graph import build_graph, extract_text

    graph = build_graph()
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    final = next(m for m in reversed(result["messages"]) if isinstance(m, AIMessage) and m.content)
    click.echo(extract_text(final.content))


@main.command()
@click.option("--dataset", default="evals/golden.jsonl", show_default=True)
def eval(dataset: str):
    """Run the agent against the golden eval dataset."""
    from finagent.evals.run import run_eval

    run_eval(dataset)


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True)
def serve(host: str, port: int):
    """Run the FinAgent web app."""
    import uvicorn

    uvicorn.run("finagent.web:app", host=host, port=port)


if __name__ == "__main__":
    main()
