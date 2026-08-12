# FinAgent Platform

**A reference platform for building, evaluating, and observing LLM agents in financial workflows.**

FinAgent is a financial research agent — it answers questions about public companies using SEC EDGAR filings and market data — but the agent is intentionally simple. The point of this project is everything *around* the agent: the evaluation harness, observability layer, orchestration, and deployment tooling that turn an agent prototype into something an ML team could actually operate in production.

Think of it as the harness an ML platform team would hand to engineers and say: *"Onboard your agent here, and you get evals, tracing, drift detection, and CI for free."*

---

## Status

Everything below is marked **built** (implemented and independently verified — hitting live APIs, a real Kubernetes cluster, a real LangFuse/W&B account, or a running Docker container) or **roadmap** (designed, not implemented — usually because it needs infrastructure this dev environment doesn't have, like a GPU or an AWS account).

| Capability | Status |
|---|---|
| LangGraph agent (router → tools → synthesizer) | **Built** |
| Tools: `edgar_filings`, `price_history`, `fundamental_ratios` | **Built** |
| CLI (`ask`, `eval`, `canary`, `serve`) | **Built** |
| Web chat UI (FastAPI + `web/`) | **Built** |
| MCP stdio server (`finagent-mcp`) | **Built** — tested with a real MCP client |
| Eval harness (LLM-as-judge, golden dataset) | **Built** |
| Ray fan-out for eval/canary parallelism | **Built** |
| W&B experiment logging | **Built** — logs to a real W&B project |
| LangFuse tracing | **Built** — traces confirmed via LangFuse's API |
| Prometheus `/metrics` endpoint | **Built** |
| Sentry error capture | **Built**, but unverified against a live project (no `SENTRY_DSN` configured) |
| Drift detection / canary command | **Built** — rolling baseline persisted locally |
| Dockerfile | **Built** — image builds and runs correctly |
| Terraform (EKS, IAM/IRSA, SageMaker role, S3) | **Written, `terraform validate`-clean** — never applied (no AWS account) |
| Jsonnet → Kubernetes manifests | **Built** — rendered manifests were actually deployed to a local `kind` cluster and answered real questions through the live pod |
| Airflow DAG for canary scheduling | **Roadmap** — canary scheduling today is a Kubernetes `CronJob` (see `infra/jsonnet/`), not Airflow |
| Grafana dashboard | **Roadmap** — the Prometheus endpoint exists, no dashboard is built on top of it yet |
| Fine-tuning / RLHF (PyTorch, SageMaker) | **Roadmap** — not implemented; needs GPU/cloud training infra this environment doesn't have |
| Bedrock as a model backend | **Roadmap** — the agent only calls the Anthropic API directly today |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Clients                              │
│ CLI · Web UI · Claude Desktop (via MCP) · Batch eval runner │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  LangGraph Agent                            │
│   router → tool execution → synthesizer                     │
│   (traced end-to-end with LangFuse)                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  MCP Tool Server                            │
│   edgar_filings · price_history · fundamental_ratios        │
│   (same tools serve the agent and external MCP clients)     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                         Eval Layer                          │
│          LLM-as-judge harness (Ray fan-out) → W&B           │
│      Canary CronJob → rolling-baseline drift detection      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                         Infra & Ops                         │
│ Terraform (EKS, unapplied) · Jsonnet → k8s · GH Actions CI  │
│               Prometheus · Sentry · LangFuse                │
└─────────────────────────────────────────────────────────────┘
```

## Stack

| Development | Infrastructure | Ops |
|---|---|---|
| Python | Ray (eval fan-out) | Git, GitHub Actions (CI) |
| LangGraph (agent graph) | Docker | LangFuse (tracing) |
| MCP (tool interface) | Kubernetes (kind locally; verified) | Sentry (error tracking) |
| FastAPI (web + API) | Jsonnet (manifest templating) | Prometheus (metrics) |
| | Terraform (EKS, written/validated, unapplied) | Weights & Biases (experiment tracking) |

## Tools in detail

Every tool below is wired into real code, not just listed — file references point at the actual integration.

**Agent & LLM**

| Tool | How it's used | Purpose |
|---|---|---|
| Anthropic API (`anthropic`, `langchain-anthropic`) | `ChatAnthropic` in [agent/graph.py](src/finagent/agent/graph.py) is bound to the 3 tools and invoked in the router/synthesizer loop | The model that decides which tool to call and writes the final answer |
| LangGraph | `StateGraph` in `agent/graph.py` wires `router → tools → synthesizer` nodes with conditional edges based on whether the model requested a tool call | Explicit, inspectable agent control flow instead of an opaque agent loop |
| LangChain (`langchain`, `langchain-core`) | Supplies the `@tool` decorator (schema generation from type hints/docstrings) used in `tools/`, the message types (`HumanMessage`/`AIMessage`/`ToolMessage`), and the LangFuse callback integration | Shared plumbing between the agent, its tools, and tracing |
| MCP (`mcp` SDK) | `FastMCP` in [mcp_server/server.py](src/finagent/mcp_server/server.py) wraps the same three tool functions as an MCP stdio server | Exposes the tools to any MCP client (Claude Desktop, another agent) without duplicating tool logic |

**Data sources**

| Tool | How it's used | Purpose |
|---|---|---|
| yfinance | `tools/market_data.py` — `price_history` and `fundamental_ratios` pull from Yahoo Finance | The market-data half of "grounded in filings and live market data" |
| SEC EDGAR (via `requests`, no SDK) | `tools/edgar.py` calls SEC's public `company_tickers.json` and `submissions/CIK....json` endpoints directly with a required identifying `User-Agent` | The filings half of the grounding story — real 10-K/10-Q/8-K data |

**Web app & CLI**

| Tool | How it's used | Purpose |
|---|---|---|
| FastAPI | [web.py](src/finagent/web.py) defines `POST /api/ask`, `GET /metrics`, and serves the static `web/` UI | HTTP layer between the browser and the agent |
| uvicorn | ASGI server running the FastAPI app, launched via `finagent serve` and inside the Docker container | Production-grade server (not a dev-only one) |
| Pydantic | `AskRequest` / `AskResponse` / `ToolCallOut` models in `web.py` | Request/response validation and typed JSON schemas |
| Click | [cli.py](src/finagent/cli.py) defines the `finagent` command group (`ask`, `eval`, `canary`, `serve`) | Primary way to drive the agent/eval harness without the web UI |
| python-dotenv | `load_dotenv()` at the top of `cli.py`, `web.py`, `mcp_server/server.py` | Loads `.env` so local dev doesn't need vars exported manually |

**Evaluation & parallelism**

| Tool | How it's used | Purpose |
|---|---|---|
| Ray | [evals/run.py](src/finagent/evals/run.py) and [canary.py](src/finagent/canary.py) wrap the per-case agent+judge call in `@ray.remote` and fan it out with `ray.get()` | Parallel eval/canary execution instead of running cases sequentially |
| Weights & Biases | `_log_to_wandb()` in `evals/run.py` logs pass rate and a per-case table to a `finagent-evals` W&B project | Experiment tracking so eval runs are comparable across prompt/model changes |

**Observability**

| Tool | How it's used | Purpose |
|---|---|---|
| LangFuse | [observability.py](src/finagent/observability.py)'s `langfuse_callbacks()` returns a `CallbackHandler` passed into every `graph.invoke(..., config={"callbacks": [...]})` call (CLI, web, eval, canary) | Full tracing of every LLM call and tool invocation, tagged by environment |
| prometheus-client | `web.py` defines `Counter`/`Histogram` metrics (`finagent_requests_total`, `finagent_request_latency_seconds`, `finagent_tool_calls_total`) exposed at `/metrics` | Operational metrics a real Prometheus server would scrape |
| Sentry SDK | `sentry_sdk.init()` in both `cli.py` and `web.py` | Exception capture/grouping in production; safe no-op without a DSN |

**Testing & quality**

| Tool | How it's used | Purpose |
|---|---|---|
| pytest | [tests/](tests/) — tool-level tests hitting live SEC EDGAR/Yahoo Finance, plus a graph-construction test | Regression safety net, run in CI |
| ruff | Linter, default rule set | Catches unused imports, unsafe exception handling, stale patterns before they ship; runs in CI |

**Infra & deployment**

| Tool | How it's used | Purpose |
|---|---|---|
| Docker | [Dockerfile](Dockerfile) builds the app image via `uv sync`; `.dockerignore` keeps `.venv`/`.git`/`.env` out of the build context | Portable, reproducible runtime packaging |
| Terraform | [infra/terraform/](infra/terraform/) provisions EKS, IAM (including IRSA), a SageMaker execution role, and an S3 bucket | Infrastructure as code for the production AWS deployment |
| Jsonnet | [infra/jsonnet/](infra/jsonnet/) templates the Kubernetes manifests (`lib/finagent.libsonnet` + per-environment files) | One shared definition instead of copy-pasted YAML per environment |
| kubectl / kind | `deploy/k8s/` holds the rendered manifests; `kind` runs a local cluster for testing | The orchestration layer the app runs on — EKS in production, kind locally |
| GitHub Actions | [.github/workflows/ci.yml](.github/workflows/ci.yml) runs `ruff check` and `pytest` on every push/PR | CI gate |

## Quickstart

```bash
git clone https://github.com/<you>/finagent-platform.git && cd finagent-platform
uv sync
cp .env.example .env   # ANTHROPIC_API_KEY, LANGFUSE keys, WANDB_API_KEY
uv run finagent ask "How did NVDA's gross margin trend over the last 4 quarters?"
```

LangFuse tracing uses LangFuse Cloud (`LANGFUSE_BASE_URL` in `.env`) — no local stack to stand up.

## Web app

A minimal chat UI lives in `web/` and is served by the same FastAPI process that exposes `/api/ask`:

```bash
uv run finagent serve   # http://127.0.0.1:8000
```

A second page at `/architecture.html` ("Backend" in the nav) visualizes the request pipeline (router → tool execution → synthesizer), the three tools and what they call out to, and live counters — polled from a real `/api/stats` endpoint backed by the same Prometheus metrics as `/metrics`, not mock data.

## MCP tool server

The agent's tools (`edgar_filings`, `price_history`, `fundamental_ratios`) are exposed as a standalone MCP server, so any MCP client — Claude Desktop, another agent, a notebook — can use them directly:

```bash
uv run finagent-mcp   # stdio MCP server
```

```json
{ "mcpServers": { "finagent": { "command": "uv", "args": ["run", "finagent-mcp"], "cwd": "/path/to/finagent-platform" } } }
```

The same tool implementations back both the LangGraph agent and the MCP server — one source of truth, two consumption patterns.

## Evaluation harness

Every change to prompts, tools, or the graph runs against a golden dataset (`evals/golden.jsonl`, currently 8 cases — small on purpose for a reference project, structured so growing it is just appending JSONL lines). Each case is scored by an LLM judge on:

- **Correctness** — does the answer match the reference?
- **Groundedness** — is every claim traceable to retrieved filing/market data?
- **Hallucination** — penalizes fabricated figures or citations.

```bash
# Parallelized across Ray workers, logged to W&B
uv run finagent eval --dataset evals/golden.jsonl --parallelism 4
```

Runs are logged to W&B with the git SHA and model ID as metadata, so any two runs are directly comparable.

## Drift detection

`finagent canary` re-runs a fixed case subset, scores it, and compares the pass rate against a rolling baseline persisted in `.finagent/canary_history.json`:

```bash
uv run finagent canary --threshold 0.85
```

If the pass rate drops below `--threshold`, the command exits nonzero — wire it into a scheduler (a Kubernetes `CronJob` manifest is included in `infra/jsonnet/`; Airflow is the natural swap-in for teams already standardized on it, but that integration isn't built here) and it becomes an alert. This catches silent regressions from upstream model updates, data source changes, or prompt edits that skipped the full eval.

## Observability

- **Tracing** — every LLM call, tool invocation, latency, and token count is traced in LangFuse, tagged by environment (`cli`, `web`, `eval`).
- **Metrics** — `/metrics` on the web app exports Prometheus counters/histograms: request count by status, request latency, tool-call counts by tool name.
- **Errors** — Sentry is wired into both the CLI and web app (`sentry_sdk.init`); it's a safe no-op without a `SENTRY_DSN`, and live capture is untested since this project doesn't have a Sentry project configured.

This makes "why did this answer get worse last Tuesday" a query instead of an archaeology project.

## Infrastructure as code

- **Terraform** (`infra/terraform/`) provisions an EKS cluster, IAM roles (including IRSA for the app's AWS access), a SageMaker execution role, and an artifacts S3 bucket. It's `terraform validate`-clean but has never been applied — this project has no AWS account.
- **Jsonnet** (`infra/jsonnet/`) templates the Kubernetes manifests (Namespace, ServiceAccount, Deployment, Service, canary CronJob) so `local` and `prod` environments share one library instead of copy-pasted YAML.

Both were tested for real, not just syntax-checked: the Jsonnet output was applied to a local `kind` cluster, the pod came up healthy behind its readiness probe, and `/api/ask` answered real questions through the live Service.

```bash
kind create cluster --name finagent
jsonnet -m deploy/k8s infra/jsonnet/environments/local.jsonnet
kubectl create secret generic finagent-secrets -n finagent --from-env-file=.env
kubectl apply -f deploy/k8s/
```

## Onboarding a new agent

The platform is agent-agnostic: implement the `AgentRunner` protocol (`run(question: str) -> AgentResult`), register a golden dataset in `evals/`, and point the eval/canary jobs at your runner via config. You inherit tracing, evals, and drift detection without writing any of it.

## Project structure

```
finagent-platform/
├── src/finagent/
│   ├── agent/           # LangGraph graph, nodes, prompts
│   ├── tools/            # EDGAR, market data, ratios (shared with MCP)
│   ├── mcp_server/       # MCP stdio server exposing tools/
│   ├── evals/             # judge prompt, scoring, Ray runner
│   ├── canary.py          # drift detection against a rolling baseline
│   ├── observability.py   # LangFuse callback wiring
│   ├── web.py              # FastAPI app: /api/ask, /metrics, static web/ UI
│   └── cli.py               # ask / eval / canary / serve
├── web/                     # chat UI (HTML/CSS/JS) served by web.py
├── evals/golden.jsonl        # golden Q&A dataset
├── infra/
│   ├── terraform/             # EKS, IAM/IRSA, SageMaker, S3 (unapplied)
│   └── jsonnet/                # k8s manifest templates (verified on kind)
├── deploy/k8s/                  # rendered manifests from jsonnet
├── Dockerfile
├── .github/workflows/ci.yml      # lint (ruff) + test (pytest)
└── tests/
```

## Disclaimer

FinAgent is a research and engineering demonstration. Its output is not investment advice.
