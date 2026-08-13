import importlib
import os
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class ToolCall:
    name: str
    args: dict


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)


@runtime_checkable
class AgentRunner(Protocol):
    def run(self, question: str, history: list[dict] | None = None) -> AgentResult:
        """Answer a question, given optional prior turns as
        [{"role": "user"|"assistant", "content": str}, ...], returning the final
        answer and any tool calls made."""


def load_runner() -> AgentRunner:
    """Instantiate the configured AgentRunner.

    Defaults to FinAgent's own LangGraph agent. Set FINAGENT_RUNNER to a
    "module.path:ClassName" dotted path to point the eval harness, canary
    job, and web app at a different agent without touching their code —
    only the class needs to implement `run(question: str, history=None) -> AgentResult`.
    """
    dotted_path = os.environ.get("FINAGENT_RUNNER", "finagent.agent.runner:FinAgentRunner")
    module_path, _, class_name = dotted_path.partition(":")
    module = importlib.import_module(module_path)
    runner_cls = getattr(module, class_name)
    return runner_cls()
