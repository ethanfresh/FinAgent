from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

load_dotenv()

from finagent.agent.graph import build_graph, extract_text  # noqa: E402

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"

app = FastAPI(title="FinAgent")


class AskRequest(BaseModel):
    question: str


class ToolCallOut(BaseModel):
    name: str
    args: dict


class AskResponse(BaseModel):
    answer: str
    tool_calls: list[ToolCallOut]


@lru_cache
def _graph():
    return build_graph()


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    result = _graph().invoke({"messages": [HumanMessage(content=question)]})
    messages = result["messages"]

    tool_calls = [
        ToolCallOut(name=call["name"], args=call["args"])
        for m in messages
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
        for call in m.tool_calls
    ]

    final = next((m for m in reversed(messages) if isinstance(m, AIMessage) and m.content), None)
    if final is None:
        raise HTTPException(status_code=502, detail="agent produced no answer")

    return AskResponse(answer=extract_text(final.content), tool_calls=tool_calls)


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")
