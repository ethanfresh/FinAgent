# FinAgent Platform

**A reference platform for building, evaluating, and observing LLM agents in financial workflows.**

FinAgent is a financial research agent — it answers questions about public companies using SEC EDGAR filings and market data — but the agent is intentionally simple. The point of this project is everything *around* the agent: the evaluation harness, fine-tuning loop, observability layer, orchestration, and deployment tooling that turn an agent prototype into something an ML team could actually operate in production.

Think of it as the harness an ML platform team would hand to engineers and say: *"Onboard your agent here, and you get evals, tracing, drift detection, and CI for free."*

---

## Status

Everything below is marked **built** (implemented and independently verified — hitting live APIs, a real Kubernetes cluster, a real Airflow/Prometheus/Grafana stack, a real LangFuse/W&B account, or a running Docker container) or **roadmap** (designed, not implemented — usually because it needs infrastructure this dev environment doesn't have, like a GPU cluster or a billable AWS account).

| Capability | Status |
|---|---|
| LangGraph agent (router → tools → synthesizer) | **Built** |
| Tools: `edgar_filings`, `price_history`, `fundamental_ratios`, `company_news`, `executive_profile` | **Built** |
| Mixture-of-experts agent (dispatcher → parallel experts → synthesizer) | **Built** — a second, structurally different agent (see [Mixture-of-experts agent](#mixture-of-experts-agent)); gating verified to actually discriminate, not just run everything |
| Automated prompt optimization (`finagent optimize`) | **Built** — a real run improved the eval pass rate from 50% → 75%, independently re-confirmed at 88% on a fresh eval (see [Prompt optimization](#prompt-optimization)) |
| Prometheus alerting (Alertmanager + Pushgateway) | **Built** — a real canary failure fired a `CanaryFailing` alert end-to-end through Prometheus → Alertmanager → a webhook receiver (see [Alerting](#alerting)) |
| CLI (`ask`, `eval`, `canary`, `train`, `optimize`, `serve`) | **Built** |
| Web chat UI + Backend visualization page (FastAPI + `web/`) | **Built** |
| `AgentRunner` protocol + `FINAGENT_RUNNER` swap mechanism | **Built** — proven with a real dummy agent swapped in via env var, not just written |
| MCP stdio server (`finagent-mcp`) | **Built** — tested with a real MCP client |
| Eval harness (LLM-as-judge, golden dataset) | **Built** |
| Ray fan-out for eval/canary parallelism | **Built** |
| W&B experiment logging | **Built** — logs to a real W&B project |
| LangFuse tracing | **Built** — traces confirmed via LangFuse's API |
| Prometheus `/metrics` endpoint | **Built** |
| Sentry error capture | **Built**, but unverified against a live project (no `SENTRY_DSN` configured) |
| Drift detection / canary command | **Built** — rolling baseline persisted locally |
| Dockerfile | **Built** — image builds and runs correctly |
| Terraform (EKS, IAM/IRSA, SageMaker role, S3) | **Written, `terraform validate`-clean** — never applied (no AWS account; provisioning real billable infra needs an explicit go-ahead this project doesn't have) |
| Jsonnet → Kubernetes manifests | **Built** — rendered manifests were actually deployed to a local `kind` cluster and answered real questions through the live pod |
| Airflow DAG for canary scheduling | **Built** — a real `apache-airflow` install ran the DAG via `airflow dags test`, executing the actual `finagent canary` command and correctly propagating both a pass and a threshold failure as task success/failure |
| Grafana dashboard on Prometheus | **Built** — Prometheus (scraping the live app) + a provisioned Grafana dashboard run via Docker Compose; confirmed real request/tool-call counts rendered in the dashboard, matching actual API calls made during testing |
| PyTorch fine-tuning (local SFT) | **Built** — `finagent train` runs a real CPU training loop (tokenize → forward → cross-entropy loss → backward → optimizer step) on judge-graded transcripts, saving a real checkpoint |
| Fine-tuning on SageMaker / full RLHF (PPO) | **Roadmap** — the local PyTorch loop is real; submitting it as a SageMaker training job and a full PPO/RLHF loop (vs. this project's SFT) both need infra/scope this environment doesn't have |
| Bedrock model backend | **Built, live call unverified** — `BedrockAgentRunner` is real and swappable via `FINAGENT_RUNNER`; running it gets all the way to a `botocore` call to Bedrock and fails only on `NoCredentialsError` (no AWS account) — same standard as the Sentry integration |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Clients                              │
│ CLI · Web UI · Claude Desktop (via MCP) · Batch eval runner │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                           Agents                             │
│    FinAgentRunner: router → tool execution → synthesizer    │
│ MoEAgentRunner: dispatcher → parallel experts → synthesizer │
│ (swappable via FINAGENT_RUNNER, both traced with LangFuse)  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                       MCP Tool Server                       │
│     edgar_filings · price_history · fundamental_ratios      │
│              company_news · executive_profile               │
│   (same tools serve the agents and external MCP clients)    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                         Eval Layer                          │
│          LLM-as-judge harness (Ray fan-out) → W&B           │
│       Prompt optimizer: propose → score → adopt best        │
│       PyTorch SFT on graded transcripts → checkpoint        │
│   Airflow DAG → Canary → rolling-baseline drift detection   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                         Infra & Ops                         │
│ Terraform (EKS, unapplied) · Jsonnet → k8s · GH Actions CI  │
│   Prometheus + Grafana + Alertmanager · Sentry · LangFuse   │
│   Model backends: Anthropic API · Bedrock (code-verified)   │
└─────────────────────────────────────────────────────────────┘
```

## Stack

| Development | Infrastructure | Ops |
|---|---|---|
| Python | Ray (eval/canary fan-out) | Git, GitHub Actions (CI) |
| Bash (Airflow's BashOperator) | Docker | AWS (Bedrock — code-verified; SageMaker role) |
| LangGraph (2 agents: flat + MoE) | Kubernetes (kind locally; verified) | LangFuse (tracing) |
| PyTorch (`finagent train`) | Airflow (canary DAG; verified) | Sentry (error tracking) |
| MCP (tool interface) | Jsonnet (manifest templating) | Prometheus + Grafana (metrics; verified) |
| FastAPI (web + API) | Terraform (EKS, written/validated, unapplied) | Alertmanager + Pushgateway (alerting; verified) |
| | | Weights & Biases (experiment + optimizer tracking) |

## Tools in detail

Every tool below is wired into real code, not just listed — file references point at the actual integration.

**Agent & LLM**

| Tool | How it's used | Purpose |
|---|---|---|
| Anthropic API (`anthropic`, `langchain-anthropic`) | `ChatAnthropic` in [agent/graph.py](src/finagent/agent/graph.py) is bound to tools and invoked in the router/synthesizer loop | The model that decides which tool to call and writes the final answer |
| LangGraph | `StateGraph` in `agent/graph.py` wires `router → tools → synthesizer`; a second, structurally different graph in [agent/moe_graph.py](src/finagent/agent/moe_graph.py) wires `dispatch → {experts} → synthesize` with dynamic `Send`-based fan-out | Explicit, inspectable agent control flow — proven with two real architectures, not just one |
| LangChain (`langchain`, `langchain-core`) | Supplies the `@tool` decorator (schema generation from type hints/docstrings) used in `tools/`, the message types (`HumanMessage`/`AIMessage`/`ToolMessage`), and the LangFuse callback integration | Shared plumbing between the agents, their tools, and tracing |
| MCP (`mcp` SDK) | `FastMCP` in [mcp_server/server.py](src/finagent/mcp_server/server.py) wraps all five tool functions as an MCP stdio server | Exposes the tools to any MCP client (Claude Desktop, another agent) without duplicating tool logic |
| langchain-aws (`ChatBedrock`) | [agent/bedrock_runner.py](src/finagent/agent/bedrock_runner.py)'s `BedrockAgentRunner` passes a `ChatBedrock` instance into the same `build_graph()` used by the default runner | A second model backend selectable via `FINAGENT_RUNNER`, matching an AWS-native (Bedrock) deployment |

**Data sources**

| Tool | How it's used | Purpose |
|---|---|---|
| yfinance | `tools/market_data.py` (`price_history`, `fundamental_ratios`), `tools/news.py` (`company_news`), `tools/executives.py` (`executive_profile`) — all pull from Yahoo Finance | Market data, recent news, and leadership/compensation data — the "up-to-date financial information, company news, executive reputation" story |
| SEC EDGAR (via `requests`, no SDK) | `tools/edgar.py` calls SEC's public `company_tickers.json` and `submissions/CIK....json` endpoints directly with a required identifying `User-Agent` | The filings half of the grounding story — real 10-K/10-Q/8-K data |

**Web app & CLI**

| Tool | How it's used | Purpose |
|---|---|---|
| FastAPI | [web.py](src/finagent/web.py) defines `POST /api/ask`, `GET /api/stats`, `GET /metrics`, and serves the static `web/` UI | HTTP layer between the browser and the agent |
| uvicorn | ASGI server running the FastAPI app, launched via `finagent serve` and inside the Docker container | Production-grade server (not a dev-only one) |
| Pydantic | `AskRequest` / `AskResponse` / `ToolCallOut` models in `web.py` | Request/response validation and typed JSON schemas |
| Click | [cli.py](src/finagent/cli.py) defines the `finagent` command group (`ask`, `eval`, `canary`, `train`, `optimize`, `serve`) | Primary way to drive the agent/eval harness without the web UI |
| python-dotenv | `load_dotenv()` at the top of `cli.py`, `web.py`, `mcp_server/server.py` | Loads `.env` so local dev doesn't need vars exported manually |

**Evaluation & parallelism**

| Tool | How it's used | Purpose |
|---|---|---|
| Ray | [evals/run.py](src/finagent/evals/run.py), [canary.py](src/finagent/canary.py), and [optimize.py](src/finagent/optimize.py) all wrap their per-case scoring calls in `@ray.remote` and fan out with `ray.get()` | Parallel eval/canary/optimizer scoring instead of running cases sequentially |
| Weights & Biases | `_log_to_wandb()` in `evals/run.py` and `optimize.py` logs pass rate / round-by-round scores to `finagent-evals` and `finagent-optimize` W&B projects | Experiment tracking so eval and optimization runs are comparable across prompt/model changes |
| Apache Airflow | [infra/airflow/dags/canary_dag.py](infra/airflow/dags/canary_dag.py) — a `BashOperator` runs `finagent canary` inside the app's own `uv` environment; ran end-to-end via `airflow dags test`, propagating both a pass and a real threshold failure | Scheduled orchestration for the nightly drift-detection job |

**Fine-tuning**

| Tool | How it's used | Purpose |
|---|---|---|
| PyTorch | [training/train.py](src/finagent/training/train.py) — a manual training loop (tokenize → forward → cross-entropy loss → `backward()` → `optimizer.step()`) fine-tuning `prajjwal1/bert-tiny` to predict judge pass/fail from (question, answer) pairs | Real, CPU-runnable SFT on judge-graded transcripts — a local stand-in for the SageMaker/GPU path |
| Transformers (`AutoModelForSequenceClassification`) | Same file — base model + classification head, checkpoint saved to `artifacts/judge-checkpoint/` | Pretrained tokenizer/model loading and checkpoint I/O |

**Observability**

| Tool | How it's used | Purpose |
|---|---|---|
| LangFuse | [observability.py](src/finagent/observability.py)'s `langfuse_callbacks()` returns a `CallbackHandler` passed into every `graph.invoke(..., config={"callbacks": [...]})` call (CLI, web, eval, canary) | Full tracing of every LLM call and tool invocation, tagged by environment |
| prometheus-client | `web.py` defines `Counter`/`Histogram` metrics (`finagent_requests_total`, `finagent_request_latency_seconds`, `finagent_tool_calls_total`) exposed at `/metrics` | Operational metrics a real Prometheus server would scrape |
| Prometheus + Grafana | [infra/observability/](infra/observability/) — `docker-compose.yml` runs Prometheus (scraping the live app's `/metrics`) and a provisioned Grafana with an 8-panel dashboard; confirmed real counts render, matching actual API calls | The metrics side of "detecting performance, decay and drift issues" |
| Alertmanager + Pushgateway | [alert_rules.yml](infra/observability/alert_rules.yml) defines 4 alert rules; [canary.py](src/finagent/canary.py) pushes its result to Pushgateway since it's a batch job, not scrapeable. Verified end-to-end: a real canary failure fired `CanaryFailing`, Alertmanager picked it up, and a webhook receiver logged the delivered payload | Gets paged about drift/errors instead of having to notice a dashboard |
| Sentry SDK | `sentry_sdk.init()` in both `cli.py` and `web.py` | Exception capture/grouping in production; safe no-op without a DSN |

**Testing & quality**

| Tool | How it's used | Purpose |
|---|---|---|
| pytest | [tests/](tests/) — tool-level tests hitting live SEC EDGAR/Yahoo Finance, graph-construction tests for both agents, the MoE dispatcher's gating logic, the `FINAGENT_RUNNER` swap mechanism, and the FastAPI layer (`/api/ask`, `/api/stats`) via `TestClient` with a fake runner so it doesn't need a live LLM call | Regression safety net, run in CI |
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

A second page at `/architecture.html` ("Backend" in the nav) is the full tour: the complete stack broken out by Development/Infrastructure/Ops with what each tool actually does in this codebase, the request pipeline (router → tool execution → synthesizer), the three tools and what they call out to, and live counters polled from a real `/api/stats` endpoint backed by the same Prometheus metrics as `/metrics` — not mock data.

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

## Prompt optimization

`finagent optimize` closes the loop the eval harness opens: it asks Claude to read the *current* system prompt plus the specific cases it just failed, propose full rewrites aimed at those failures, scores every candidate against the same golden set (parallelized the same way as `eval`), and adopts whichever one scores best — a small, real hill-climbing implementation of the "prompt optimization" pattern, not a stub:

```bash
uv run finagent optimize --rounds 2 --candidates 3
```

Run for real during development, starting from the original hand-written prompt:

```
baseline: 50% (8 cases)
round 1: proposed 2 candidates (4 failures to fix)
  candidate 1: 50%
  candidate 2: 75%
  -> 75% beats current 50%, adopting
Wrote new champion prompt to prompts/system_prompt.txt (50% -> 75%)
```

The candidate that won specifically fixed a real bug this project had already hit once before (the "most recent quarterly closing price" mid-quarter-vs-quarter-end ambiguity) — the optimizer rediscovered and fixed it on its own from eval failures, without being told what the bug was. A completely separate, freshly-run `finagent eval` afterward independently confirmed **88%** — up from the 50% baseline.

The live prompt is file-backed (`prompts/system_prompt.txt`, loaded by [agent/graph.py](src/finagent/agent/graph.py)'s `load_system_prompt()`), not hardcoded — so a winning candidate takes effect on the next run without a code change, and every round is logged to a `finagent-optimize` W&B project.

## Fine-tuning

`finagent train` fine-tunes a small local model (`prajjwal1/bert-tiny`, 4.4M params) to predict whether the eval judge would grade a (question, answer) pair a pass or fail — distilling the judge's signal into something fast and local. It's a genuine PyTorch training loop, not a wrapper around a one-liner:

```bash
uv sync --extra training   # torch + transformers
uv run finagent train --epochs 3
```

```
[1] What is AAPL's gross margin?
[0] What was AAPL's stock price trend over the last year?
...
epoch 1/3 — loss 0.6894
epoch 2/3 — loss 0.6745
epoch 3/3 — loss 0.6734
Trained on 8 examples, checkpoint at artifacts/judge-checkpoint
```

The training data is collected live: the agent answers every golden-set question, the same judge the eval harness uses grades each answer, and those graded transcripts become the fine-tuning set — SFT on judge-graded transcripts, made concrete instead of just described. A real `model.safetensors` checkpoint is saved to `artifacts/judge-checkpoint/`.

This runs on CPU as a stand-in for the SageMaker/GPU path — submitting the same job to SageMaker, and a full PPO/RLHF loop rather than SFT, are both roadmap (this dev environment has no AWS account or GPU).

## Drift detection

`finagent canary` re-runs a fixed case subset, scores it, and compares the pass rate against a rolling baseline persisted in `.finagent/canary_history.json`:

```bash
uv run finagent canary --threshold 0.85
```

If the pass rate drops below `--threshold`, the command exits nonzero. Scheduling is real, not just described — [infra/airflow/dags/canary_dag.py](infra/airflow/dags/canary_dag.py) is a genuine Airflow DAG whose `BashOperator` runs this exact command:

```bash
cd infra/airflow && export AIRFLOW_HOME="$(pwd)"
airflow db migrate
airflow dags test finagent_canary 2026-08-12
```

Run for real during development: the DAG loaded cleanly, executed the real `finagent canary` command (Ray fan-out, live LLM calls, a real pass rate), and correctly marked the task **SUCCESS** at a lenient threshold and **FAILED** at the production one (`0.85`) — proving both the pass and fail paths propagate correctly through Airflow, not just the happy path. A Kubernetes `CronJob` (`infra/jsonnet/`) is the equivalent for teams that schedule via k8s instead of Airflow.

Canary is a batch job, not a scrapeable process, so it *pushes* its result (`finagent_canary_pass_rate`, `finagent_canary_ok`) to a Prometheus Pushgateway rather than waiting to be scraped — the standard Prometheus pattern for cron/batch jobs. That metric is what the `CanaryFailing` alert (see [Alerting](#alerting)) watches.

## Observability

- **Tracing** — every LLM call, tool invocation, latency, and token count is traced in LangFuse, tagged by environment (`cli`, `web`, `eval`, `canary`) via the `FINAGENT_ENV` variable each entry point sets.
- **Metrics** — `/metrics` on the web app exports Prometheus counters/histograms: request count by status, request latency, tool-call counts by tool name.
- **Dashboards** — [infra/observability/](infra/observability/) runs Prometheus + a provisioned Grafana dashboard via Docker Compose:
  ```bash
  cd infra/observability && docker compose up -d
  # Prometheus: http://localhost:9090  ·  Grafana: http://localhost:3000 (admin/finagent)  ·  Alertmanager: http://localhost:9093
  ```
  Verified live: Prometheus's target for the app showed `up`, and Grafana's dashboard rendered real request/tool-call/canary/alert counts that matched actual API calls and canary runs made during testing — not placeholder panels.
- **Errors** — Sentry is wired into both the CLI and web app (`sentry_sdk.init`); it's a safe no-op without a `SENTRY_DSN`, and live capture is untested since this project doesn't have a Sentry project configured.

This makes "why did this answer get worse last Tuesday" a query instead of an archaeology project.

## Alerting

Metrics and dashboards tell you *that* something's wrong if you happen to be looking; alerting tells you *without* looking. [infra/observability/alert_rules.yml](infra/observability/alert_rules.yml) defines four real Prometheus alert rules:

| Alert | Fires when |
|---|---|
| `FinAgentDown` | the app hasn't been scrapeable for 1m |
| `HighErrorRate` | >10% of `/api/ask` requests error over 5m |
| `HighLatency` | average request latency exceeds 5s over 5m |
| `CanaryFailing` | the pushed canary pass rate drops below 0.85 |

Prometheus evaluates these and forwards firing alerts to Alertmanager ([alertmanager.yml](infra/observability/alertmanager.yml)), which routes them to a receiver — in production that's `slack_configs`/`pagerduty_configs`/etc.; here it's a tiny local webhook ([webhook_receiver.py](infra/observability/webhook_receiver.py)) that stands in for one so the full pipeline can be proven end-to-end without a real integration configured.

It was proven end-to-end, not just config-validated: a real canary run scored 67%, `CanaryFailing` transitioned to `firing` in Prometheus, Alertmanager picked it up (`GET /api/v2/alerts` showed it active), and the webhook receiver logged the delivered payload with the correct annotation (*"The most recent canary run scored 66.67%, below the 0.85 threshold"*). `promtool check rules` also passes clean.

## Model backends

The agent's LLM is swappable through the same `FINAGENT_RUNNER` mechanism used for onboarding a different agent entirely — because a different model backend *is* just a different `AgentRunner`:

```bash
# Default: Anthropic API
uv run finagent ask "..."

# Bedrock (requires: uv sync --extra bedrock, plus AWS credentials)
FINAGENT_RUNNER="finagent.agent.bedrock_runner:BedrockAgentRunner" uv run finagent ask "..."
```

[agent/bedrock_runner.py](src/finagent/agent/bedrock_runner.py) passes a `ChatBedrock` instance into the exact same `build_graph()` the default runner uses — same router/tools/synthesizer graph, different chat model. This is real, wired code: running it without AWS credentials gets all the way to `botocore`'s API call and fails with a clean `NoCredentialsError`, not an import error or a crash — the same bar applied to the Sentry integration elsewhere in this README.

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

The platform is agent-agnostic by construction, not just by description: `web.py`, `cli.py`, and the eval/canary harness never import FinAgent's own agent directly — they all call [runner.py](src/finagent/runner.py)'s `load_runner()`, which instantiates whatever class `FINAGENT_RUNNER` points at (a `"module.path:ClassName"` dotted string), defaulting to `finagent.agent.runner:FinAgentRunner`.

To onboard a different agent: implement a class with `run(question: str) -> AgentResult` (see [runner.py](src/finagent/runner.py) for the tiny `AgentResult`/`ToolCall` dataclasses), point `FINAGENT_RUNNER` at it, and register a golden dataset in `evals/`. You inherit tracing, the web UI, the eval harness, and drift detection without touching any of their code.

```bash
FINAGENT_RUNNER="my_package.agent:MyAgentRunner" uv run finagent ask "..."
```

[tests/test_runner.py](tests/test_runner.py) proves this with a real (if trivial) second agent swapped in via the env var — not just a written claim.

### Mixture-of-experts agent

A second, *non-trivial* agent proving the same point — not just the swap-in dummy above, but a genuinely different LangGraph architecture, selectable the same way:

```bash
FINAGENT_RUNNER="finagent.agent.moe_runner:MoEAgentRunner" uv run finagent ask "Tell me about NVIDIA — how's it doing financially, any recent news, and who runs it?"
```

Instead of one agent with every tool, [agent/moe_graph.py](src/finagent/agent/moe_graph.py) gates three scoped experts:

- **financials** — `price_history`, `fundamental_ratios`, `edgar_filings`
- **news** — `company_news` (Yahoo Finance headlines via `yfinance`)
- **executives** — `executive_profile` (real leadership names, titles, compensation) + `company_news`, instructed to ground any reputation comment in actual recent coverage rather than invent a sentiment score

A **dispatcher** node reads the question and decides which experts are relevant — verified to actually discriminate, not just always run everything: `"What is AAPL's gross margin?"` → `["financials"]` only; `"Who is the CEO of Tesla?"` → `["executives"]` only. Only the selected experts run, in parallel (LangGraph's `Send` API for dynamic fan-out), each with its own scoped toolset and its own focused report. A **synthesizer** then weaves whichever reports ran into one narrative — "the story" of the company — rather than three disconnected paragraphs.

Run for real: a broad question correctly triggered all three experts, each pulled real data (NVDA's actual margins and price range, its actual 10-K filing link, real recent news about its Wall Street financing deal, Jensen Huang's real name/age/compensation), and the synthesizer produced one coherent multi-paragraph story citing all of it — not three tool dumps stapled together.

## Project structure

```
finagent-platform/
├── src/finagent/
│   ├── agent/
│   │   ├── graph.py           # flat agent graph, run_graph() helper, prompt loader
│   │   ├── moe_graph.py       # MoE graph: dispatch -> parallel experts -> synthesize
│   │   ├── runner.py          # FinAgentRunner (flat agent, Anthropic)
│   │   ├── moe_runner.py      # MoEAgentRunner
│   │   └── bedrock_runner.py  # BedrockAgentRunner: either graph, ChatBedrock
│   ├── tools/                  # EDGAR, market data, news, executives (shared with MCP)
│   ├── mcp_server/              # MCP stdio server exposing tools/
│   ├── evals/                    # judge prompt, scoring, Ray runner
│   ├── training/                  # PyTorch SFT loop on judge-graded transcripts
│   ├── runner.py                   # AgentRunner protocol + FINAGENT_RUNNER loader
│   ├── optimize.py                  # prompt optimizer: propose -> score -> adopt
│   ├── canary.py                     # drift detection + Pushgateway metric push
│   ├── observability.py               # LangFuse callback wiring
│   ├── web.py                          # FastAPI app: /api/ask, /api/stats, /metrics
│   └── cli.py                           # ask / eval / canary / train / optimize / serve
├── prompts/system_prompt.txt              # live system prompt (optimizer writes here)
├── web/                                     # chat UI + Backend visualization page
├── evals/golden.jsonl                        # golden Q&A dataset
├── infra/
│   ├── terraform/                              # EKS, IAM/IRSA, SageMaker, S3 (unapplied)
│   ├── jsonnet/                                  # k8s manifest templates (verified on kind)
│   ├── airflow/dags/canary_dag.py                 # canary DAG (verified with airflow dags test)
│   └── observability/                              # Prometheus + Grafana + Alertmanager (verified)
│       ├── alert_rules.yml, alertmanager.yml, webhook_receiver.py
│       └── grafana/
├── deploy/k8s/                                       # rendered manifests from jsonnet
├── artifacts/judge-checkpoint/                          # fine-tuned model output (gitignored)
├── Dockerfile
├── .github/workflows/ci.yml                               # lint (ruff) + test (pytest)
└── tests/                                                   # tools, both agents, runner swap, FastAPI, Bedrock
```

## Disclaimer

FinAgent is a research and engineering demonstration. Its output is not investment advice.
