# FinAgent Platform

**A reference platform for building, evaluating, and observing LLM agents in financial workflows.**

FinAgent is a financial research agent — it answers questions about public companies using SEC EDGAR filings and market data — but the agent is intentionally simple. The point of this project is everything *around* the agent: the evaluation harness, observability layer, orchestration, and deployment tooling that turn an agent prototype into something an ML team could actually operate in production.

Think of it as the harness an ML platform team would hand to engineers and say: *"Onboard your agent here, and you get evals, tracing, drift detection, and CI for free."*

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Clients                              │
│   CLI  ·  Claude Desktop (via MCP)  ·  Batch eval runner    │
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
│                  Platform Layer                             │
│   Eval harness (LLM-as-judge, Ray fan-out) → W&B            │
│   Nightly canary job → drift detection → alerting           │
│   Docker · Kubernetes (kind) · GitHub Actions CI            │
└─────────────────────────────────────────────────────────────┘
```

## Stack

| Concern | Tool |
|---|---|
| Agent framework | LangGraph |
| Tool interface | MCP (Model Context Protocol) server |
| Data sources | SEC EDGAR, yfinance |
| Evaluation | Custom LLM-as-judge harness, golden dataset |
| Parallelism | Ray (eval fan-out) |
| Experiment tracking | Weights & Biases |
| Observability | LangFuse (self-hosted, Docker Compose) |
| Scheduling | Cron / Airflow DAG (canary evals) |
| Packaging & deploy | Docker, Kubernetes (kind), Helm-style manifests |
| CI | GitHub Actions (ruff, mypy, pytest, eval smoke test) |

## Quickstart

```bash
# 1. Install
git clone https://github.com/<you>/finagent-platform.git
cd finagent-platform
uv sync

# 2. Configure
cp .env.example .env   # add ANTHROPIC_API_KEY, LANGFUSE keys, WANDB_API_KEY

# 3. Start observability stack
docker compose up -d langfuse

# 4. Ask the agent something
uv run finagent ask "How did NVDA's gross margin trend over the last 4 quarters?"
```

## Using the tools from Claude Desktop (MCP)

The agent's tools are exposed as a standalone MCP server, so any MCP client can use them directly:

```bash
uv run finagent-mcp   # stdio MCP server
```

Add to Claude Desktop's config:

```json
{
  "mcpServers": {
    "finagent": {
      "command": "uv",
      "args": ["run", "finagent-mcp"],
      "cwd": "/path/to/finagent-platform"
    }
  }
}
```

The same tool implementations back both the LangGraph agent and the MCP server — one source of truth, two consumption patterns.

## Evaluation harness

Every change to prompts, tools, or the graph runs against a golden dataset of ~50 financial Q&A pairs (`evals/golden.jsonl`). Each case is scored by an LLM judge on:

- **Correctness** — does the answer match the reference?
- **Groundedness** — is every claim traceable to retrieved filing/market data?
- **Hallucination** — penalizes fabricated figures or citations.

```bash
# Full eval run, parallelized across Ray workers, logged to W&B
uv run finagent eval --dataset evals/golden.jsonl --parallelism 8
```

Runs are logged to W&B with the prompt version, git SHA, and model ID as metadata, so any two runs are directly comparable.

## Drift detection

A nightly canary job re-runs a fixed 10-case subset and compares scores against a rolling baseline:

```bash
uv run finagent canary --threshold 0.85
```

If the aggregate score drops below threshold, the job exits nonzero and fires an alert. This catches silent regressions from upstream model updates, data source changes, or prompt edits that skipped the full eval.

## Observability

All agent runs — interactive, eval, and canary — are traced in LangFuse: every LLM call, tool invocation, latency, and token count, tagged by prompt version and environment. This makes "why did this answer get worse last Tuesday" a query instead of an archaeology project.

## Deployment

The full stack runs on Kubernetes. Locally, that's [kind](https://kind.sigs.k8s.io/):

```bash
kind create cluster --name finagent
kubectl apply -f deploy/k8s/
```

Manifests include readiness/liveness probes, resource requests and limits, and a ConfigMap-driven configuration pattern. `deploy/README.md` documents how each piece maps to a production EKS deployment (IRSA for AWS credentials, ALB ingress, Karpenter for eval-job autoscaling).

## Onboarding a new agent onto this harness

The platform is agent-agnostic by design. To onboard a new agent:

1. Implement the `AgentRunner` protocol (`run(question: str) -> AgentResult`).
2. Register your golden dataset in `evals/`.
3. Point the eval and canary jobs at your runner via config.

You inherit tracing, evals, drift detection, and CI without writing any of it.

## Project structure

```
finagent-platform/
├── src/finagent/
│   ├── agent/          # LangGraph graph, nodes, prompts
│   ├── tools/          # EDGAR, market data, ratios (shared with MCP)
│   ├── mcp_server/     # MCP stdio server exposing tools/
│   ├── evals/          # judge prompts, scoring, Ray runner
│   └── observability/  # LangFuse instrumentation
├── evals/golden.jsonl  # golden Q&A dataset
├── deploy/
│   ├── docker/
│   └── k8s/
├── .github/workflows/  # CI: lint, typecheck, test, eval smoke
└── docker-compose.yml  # LangFuse stack
```

## Roadmap

- Airflow DAG replacing cron for canary orchestration
- Terraform module for the equivalent EKS deployment
- Multi-model eval matrix (compare judge/actor model combinations)
- Prometheus metrics endpoint + Grafana dashboard for agent latency/error rates

## Disclaimer

FinAgent is a research and engineering demonstration. Its output is not investment advice.
