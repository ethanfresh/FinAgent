import os

from langchain_aws import ChatBedrock

from finagent.agent.graph import build_graph, run_graph
from finagent.runner import AgentResult


class BedrockAgentRunner:
    """AgentRunner backed by Amazon Bedrock instead of the Anthropic API directly.

    Same LangGraph router/tools/synthesizer graph as FinAgentRunner — only the
    chat model changes. Requires `uv sync --extra bedrock` plus AWS credentials
    with Bedrock model access to actually answer a question; this dev environment
    has neither, so the live call path is unverified here. What's verified: the
    graph wires up correctly against ChatBedrock, and construction/invocation
    fail with a clean, expected AWS error rather than crashing unexpectedly —
    the same standard applied to the Sentry integration (safe, real code;
    live behavior unverified without a live account).

    Select it with:
        FINAGENT_RUNNER=finagent.agent.bedrock_runner:BedrockAgentRunner
    """

    def __init__(self, model_id: str | None = None, region_name: str | None = None):
        model_id = model_id or os.environ.get(
            "BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"
        )
        region_name = region_name or os.environ.get("AWS_REGION", "us-east-1")
        llm = ChatBedrock(model_id=model_id, region_name=region_name)
        self._graph = build_graph(llm=llm)

    def run(self, question: str, history: list[dict] | None = None) -> AgentResult:
        return run_graph(self._graph, question, backend="bedrock", history=history)
