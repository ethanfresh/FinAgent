# FinAgent Platform

**A reference platform for building, evaluating, and observing LLM agents in financial workflows.**

FinAgent is a financial research agent — it answers questions about public companies using SEC EDGAR filings and market data — but the agent is intentionally simple. The point of this project is everything *around* the agent: the evaluation harness, fine-tuning loop, observability layer, orchestration, and deployment tooling that turn an agent prototype into something an ML team could actually operate in production.

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
│                    Eval & Training Layer                    │
│          LLM-as-judge harness (Ray fan-out) → W&B           │
│     Fine-tuning / RLHF loop (PyTorch, SageMaker) → W&B      │
│    Nightly canary (Airflow) → drift detection → alerting    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                         Infra & Ops                         │
│   Terraform (EKS) · Jsonnet manifests · GitHub Actions CI   │
│          Prometheus + Grafana · Sentry · LangFuse           │
└─────────────────────────────────────────────────────────────┘
```

## Stack

| Development | Infrastructure | Ops |
|---|---|---|
| Python | Ray (eval + training fan-out) | Git, GitHub Actions (CI) |
| Bash (job scripts) | Amazon EKS (kind locally) | AWS (Bedrock, SageMaker) |
| LangGraph (agent graph) | Airflow (canary scheduling) | LangFuse (tracing) |
| PyTorch (fine-tuning) | Jsonnet (manifest templating) | Sentry (error tracking) |
| MCP (tool interface) | Terraform (EKS provisioning) | Prometheus + Grafana (metrics) |
| | | Weights & Biases (experiment tracking) |

## Quickstart

```bash
git clone https://github.com/<you>/finagent-platform.git && cd finagent-platform
uv sync
cp .env.example .env   # ANTHROPIC_API_KEY or AWS creds for Bedrock, LANGFUSE keys, WANDB_API_KEY
docker compose up -d langfuse
uv run finagent ask "How did NVDA's gross margin trend over the last 4 quarters?"
```

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

Every change to prompts, tools, or the graph runs against a golden dataset of ~50 financial Q&A pairs (`evals/golden.jsonl`). Each case is scored by an LLM judge on:

- **Correctness** — does the answer match the reference?
- **Groundedness** — is every claim traceable to retrieved filing/market data?
- **Hallucination** — penalizes fabricated figures or citations.

```bash
# Full eval run, parallelized across Ray workers, logged to W&B
uv run finagent eval --dataset evals/golden.jsonl --parallelism 8
```

Runs are logged to W&B with the prompt version, git SHA, and model ID as metadata, so any two runs are directly comparable.

## Fine-tuning & RL

The router and judge models can be tuned instead of run off-the-shelf: `finagent train` launches a PyTorch fine-tuning job (SFT on graded transcripts, or RLAIF using the eval harness's judge score as reward) on SageMaker or an EKS GPU node pool.

```bash
uv run finagent train --base-model judge-v1 --method rlaif --backend sagemaker
```

Tuned checkpoints are versioned and logged to W&B alongside the golden-dataset eval run that qualifies them, so a checkpoint is only promoted if it beats the current baseline — no manual comparison.

## Drift detection

An Airflow DAG re-runs a fixed 10-case canary subset nightly and compares scores against a rolling baseline:

```bash
uv run finagent canary --threshold 0.85
```

If the aggregate score drops below threshold, the job exits nonzero and fires an alert. This catches silent regressions from upstream model updates, data source changes, or prompt edits that skipped the full eval.

## Observability

- **Tracing** — every LLM call, tool invocation, latency, and token count is traced in LangFuse, tagged by prompt version and environment.
- **Metrics** — a Prometheus endpoint exports request rate, latency, and eval/canary pass rate; a Grafana dashboard visualizes it.
- **Errors** — Sentry captures and groups exceptions across the agent, MCP server, and eval/training workers.

This makes "why did this answer get worse last Tuesday" a query instead of an archaeology project.

## Infrastructure as code

- **Terraform** provisions the EKS cluster, IAM/IRSA roles, and supporting AWS resources (S3 for artifacts, SageMaker for training jobs).
- **Jsonnet** templates the Kubernetes manifests (agent, MCP server, eval/canary CronJobs) so environments share a base config instead of copy-pasted YAML.

Locally, the same manifests run on [kind](https://kind.sigs.k8s.io/):

```bash
kind create cluster --name finagent
kubectl apply -f deploy/k8s/
```

`deploy/README.md` maps each piece to production EKS (IRSA for AWS credentials, ALB ingress, Karpenter for eval/training autoscaling).

## Onboarding a new agent

The platform is agent-agnostic: implement the `AgentRunner` protocol (`run(question: str) -> AgentResult`), register a golden dataset in `evals/`, and point the eval/canary/training jobs at your runner via config. You inherit tracing, evals, drift detection, fine-tuning, and CI without writing any of it.

## Project structure

```
finagent-platform/
├── src/finagent/
│   ├── agent/          # LangGraph graph, nodes, prompts
│   ├── tools/          # EDGAR, market data, ratios (shared with MCP)
│   ├── mcp_server/     # MCP stdio server exposing tools/
│   ├── evals/          # judge prompts, scoring, Ray runner
│   ├── training/        # PyTorch fine-tuning + RLAIF loop
│   └── observability/  # LangFuse, Prometheus, Sentry instrumentation
├── evals/golden.jsonl  # golden Q&A dataset
├── infra/
│   ├── terraform/      # EKS, IAM/IRSA, SageMaker resources
│   └── jsonnet/        # k8s manifest templates
├── deploy/
│   ├── docker/
│   └── k8s/             # rendered manifests from jsonnet
├── .github/workflows/  # CI: lint, typecheck, test, eval smoke
└── docker-compose.yml  # LangFuse stack
```

## Disclaimer

FinAgent is a research and engineering demonstration. Its output is not investment advice.
