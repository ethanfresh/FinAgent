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
| `filing_search` — RAG over real SEC filing text (chunk → embed → Chroma → retrieve) | **Built** — a real run against a live NVDA/AAPL 10-K got 100% recall@k on a hand-verified golden set, and the agent used it live in conversation with zero tool-name leakage (see [Retrieval-augmented filing search](#retrieval-augmented-filing-search)) |
| Mixture-of-experts agent (dispatcher → parallel experts → synthesizer) | **Built** — a second, structurally different agent (see [Mixture-of-experts agent](#mixture-of-experts-agent)); gating verified to actually discriminate, not just run everything |
| Automated prompt optimization (`finagent optimize`) | **Built** — a real run improved the eval pass rate from 50% → 75%, independently re-confirmed at 88% on a fresh eval (see [Prompt optimization](#prompt-optimization)) |
| Prometheus alerting (Alertmanager + Pushgateway) | **Built** — a real canary failure fired a `CanaryFailing` alert end-to-end through Prometheus → Alertmanager → a webhook receiver (see [Alerting](#alerting)) |
| CLI (`ask`, `eval`, `canary`, `train`, `optimize`, `redteam`, `index-filings`, `retrieval-eval`, `serve`) | **Built** |
| Web chat UI + Backend visualization page (FastAPI + `web/`) | **Built** |
| `AgentRunner` protocol + `FINAGENT_RUNNER` swap mechanism | **Built** — proven with a real dummy agent swapped in via env var, not just written |
| MCP stdio server (`finagent-mcp`) | **Built** — tested with a real MCP client |
| Eval harness (LLM-as-judge, golden dataset) | **Built** |
| Adversarial user simulation (`finagent redteam`) | **Built** — a real run against the live app found 8 genuine issues across 4 simulated-user sessions, including a stateless-conversation UX bug that's since been fixed and re-verified (see [Adversarial user simulation](#adversarial-user-simulation-red-team) / [Fixed issues](#fixed-issues)) |
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
| Chroma (local vector store) | | Weights & Biases (experiment + optimizer tracking) |
| sentence-transformers (local embeddings) | | |

## Tools in detail

Every tool below is wired into real code, not just listed — file references point at the actual integration.

**Agent & LLM**

| Tool | How it's used | Purpose |
|---|---|---|
| Anthropic API (`anthropic`, `langchain-anthropic`) | `ChatAnthropic` in [agent/graph.py](src/finagent/agent/graph.py) is bound to tools and invoked in the router/synthesizer loop | The model that decides which tool to call and writes the final answer |
| LangGraph | `StateGraph` in `agent/graph.py` wires `router → tools → synthesizer`; a second, structurally different graph in [agent/moe_graph.py](src/finagent/agent/moe_graph.py) wires `dispatch → {experts} → synthesize` with dynamic `Send`-based fan-out | Explicit, inspectable agent control flow — proven with two real architectures, not just one |
| LangChain (`langchain`, `langchain-core`) | Supplies the `@tool` decorator (schema generation from type hints/docstrings) used in `tools/`, the message types (`HumanMessage`/`AIMessage`/`ToolMessage`), and the LangFuse callback integration | Shared plumbing between the agents, their tools, and tracing |
| MCP (`mcp` SDK) | `FastMCP` in [mcp_server/server.py](src/finagent/mcp_server/server.py) wraps all six tool functions as an MCP stdio server | Exposes the tools to any MCP client (Claude Desktop, another agent) without duplicating tool logic |
| langchain-aws (`ChatBedrock`) | [agent/bedrock_runner.py](src/finagent/agent/bedrock_runner.py)'s `BedrockAgentRunner` passes a `ChatBedrock` instance into the same `build_graph()` used by the default runner | A second model backend selectable via `FINAGENT_RUNNER`, matching an AWS-native (Bedrock) deployment |

**Data sources**

| Tool | How it's used | Purpose |
|---|---|---|
| yfinance | `tools/market_data.py` (`price_history`, `fundamental_ratios`), `tools/news.py` (`company_news`), `tools/executives.py` (`executive_profile`) — all pull from Yahoo Finance | Market data, recent news, and leadership/compensation data — the "up-to-date financial information, company news, executive reputation" story |
| SEC EDGAR (via `requests`, no SDK) | `tools/edgar.py` calls SEC's public `company_tickers.json` and `submissions/CIK....json` endpoints directly with a required identifying `User-Agent` | The filings half of the grounding story — real 10-K/10-Q/8-K data |

**Retrieval (RAG)**

| Tool | How it's used | Purpose |
|---|---|---|
| BeautifulSoup + lxml | [rag/ingest.py](src/finagent/rag/ingest.py)'s `fetch_filing_text()` strips a real filing document (fetched from SEC EDGAR, not a summary) down to plain text | Turns a raw SEC HTML/inline-XBRL document into text worth chunking |
| sentence-transformers (`all-MiniLM-L6-v2`) | [rag/store.py](src/finagent/rag/store.py)'s `embed_texts()`, run locally on CPU | Embeddings with no external API key/service — same "actually runnable in this dev environment" bar as everything else |
| Chroma | `rag/store.py`'s `get_collection()` — a persistent local `PersistentClient` collection (`.chroma/`, gitignored) | A real vector database, not a hand-rolled cosine-similarity loop |
| `finagent.tools.filing_search` | [tools/filing_search.py](src/finagent/tools/filing_search.py) — lazily indexes a ticker/form_type on first use, then does a metadata-filtered vector query | The retrieval tool itself, wired into both agents and the MCP server |

**Web app & CLI**

| Tool | How it's used | Purpose |
|---|---|---|
| FastAPI | [web.py](src/finagent/web.py) defines `POST /api/ask`, `GET /api/stats`, `GET /metrics`, and serves the static `web/` UI | HTTP layer between the browser and the agent |
| uvicorn | ASGI server running the FastAPI app, launched via `finagent serve` and inside the Docker container | Production-grade server (not a dev-only one) |
| Pydantic | `AskRequest` / `AskResponse` / `ToolCallOut` models in `web.py` | Request/response validation and typed JSON schemas |
| Click | [cli.py](src/finagent/cli.py) defines the `finagent` command group (`ask`, `eval`, `canary`, `train`, `optimize`, `redteam`, `index-filings`, `retrieval-eval`, `serve`) | Primary way to drive the agent/eval harness without the web UI |
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

## Retrieval-augmented filing search

Every other tool here either returns a link to a filing (`edgar_filings`) or a pre-computed ratio (`fundamental_ratios`) — none of them let the agent actually read what a filing *says*. `filing_search` closes that gap: real chunking, real local embeddings, a real vector database, and a real retrieval eval, not a `grep` dressed up as RAG.

```bash
uv sync --extra rag   # chromadb + sentence-transformers + beautifulsoup4/lxml
uv run finagent index-filings NVDA --form-type 10-K   # optional pre-warm; filing_search also indexes lazily on first use
uv run finagent retrieval-eval
```

**Ingestion** ([rag/ingest.py](src/finagent/rag/ingest.py)): fetches a filing's actual document from SEC EDGAR (not just its metadata), strips it to plain text with BeautifulSoup, and hands it to the chunker.

**Chunking** ([rag/chunking.py](src/finagent/rag/chunking.py)): section-aware — 10-Ks and 10-Qs are organized into numbered "Item" sections (`Item 1A. Risk Factors`, `Item 2. Properties`, ...), and when those headers are detected, a chunk never straddles a section boundary, so every retrieved passage carries correct section context. Falls back to a fixed-size sliding window with overlap when no section structure is found. Pure text logic with no model/network dependency, so it's unit-tested unconditionally (`tests/test_chunking.py`) rather than gated behind the `rag` extra.

**Embedding + storage** ([rag/store.py](src/finagent/rag/store.py)): `sentence-transformers` (`all-MiniLM-L6-v2`) runs locally on CPU — no embeddings API key, same "actually runnable in this dev environment" bar as the rest of the project — into a persistent local Chroma collection (`.chroma/`, gitignored).

**The tool** ([tools/filing_search.py](src/finagent/tools/filing_search.py)): `filing_search(ticker, query, form_type, limit)` lazily indexes a ticker/form_type combination the first time it's asked about (fetch → chunk → embed → store, ~5-15s), then serves a metadata-filtered vector query. Wired into both agents (the flat runner's tool list and the MoE agent's `financials` expert) and the MCP server. Fails cleanly with a clear error — not an `ImportError` crash — if the `rag` extra isn't installed, so the rest of the app (including `finagent ask` with zero RAG usage) works fine without it; verified by simulating `chromadb`/`sentence-transformers` as absent and confirming the tool registry and graph still build.

**Retrieval eval** ([evals/retrieval.py](src/finagent/evals/retrieval.py), `evals/retrieval_golden.jsonl`): recall@k against 4 hand-verified cases spanning two real companies (NVDA, AAPL) and three filing sections (Risk Factors, Properties). Deterministic substring + section match, not an LLM judge — "does this passage contain X" doesn't need one, and staying deterministic makes a regression unambiguous. Every `expected_substring` was taken from a passage this pipeline actually retrieved, not guessed at. A real run: **100% recall@k (4/4)**, logged to the same `finagent-evals` W&B project as the QA eval harness. Runs sequentially, not Ray-parallelized like `evals/run.py` — cases fetch real SEC filings, and hammering EDGAR concurrently isn't considerate of their service.

Verified live in conversation, not just via direct tool calls: asked the running app "What does Nvidia's 10-K say about risks from competition?" and got back a multi-paragraph answer quoting and paraphrasing real retrieved passages from the actual Item 1A section, correctly cited, with the `filing_search`/`edgar_filings` tool names never appearing in the visible answer (see [Tools in detail](#tools-in-detail) above for the "never name the tool" rule).

One environment-specific note kept for the next session: `chromadb`'s default dependency on `onnxruntime` has no macOS x86_64 wheels past `1.23.x`, and `sentence-transformers`' torch dependency needs `numpy<2` on this platform (same Rosetta/x86_64 situation as the `training` extra) — both pinned in the `rag` extra in `pyproject.toml`.

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

A third page at `/redteam.html` ("Red Team" in the nav) runs the adversarial user simulation described below from a button in the browser, no CLI needed (see [Adversarial user simulation](#adversarial-user-simulation-red-team)).

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

## Adversarial user simulation (red-team)

The golden-dataset eval above catches regressions against known reference answers; `finagent redteam` is the complementary, open-ended probe — no reference answers, just an LLM playing a realistic end user and a second LLM (an unbiased-financial-advisor QA persona) critiquing what FinAgent actually said:

```bash
uv run finagent serve &                # needs a live app to talk to
uv run finagent redteam --turns 4
```

Four personas (a nervous first-time investor, a pushy user trying to get direct buy/sell advice out of it, a detail-oriented analyst stress-testing exact figures and dates, and someone poking at edge-case/nonexistent tickers) each hold a live multi-turn conversation against the real `/api/ask` endpoint. The conversation is genuinely reactive — each next user message is generated from the real prior FinAgent reply, not a fixed script. A critic pass then reads the full transcript and flags concrete problems (hallucinated claims, guardrail breaks, bias, evasiveness, factual/date errors, ungraceful tool failures) with a turn number and verbatim quote, writing both a JSON and a Markdown report to `artifacts/redteam/`.

A real run against the live app found **8 genuine issues across 4 sessions**, the standout being a UX-breaking one: `/api/ask` was stateless (no conversation memory), so when a persona referenced something FinAgent had said one turn earlier, FinAgent flatly denied having said it ("I don't have any earlier turn in this conversation...") instead of acknowledging the product doesn't retain history — read as the assistant contradicting itself and actively eroded the simulated user's trust ("this is making me trust it less, not more"). Smaller findings: presenting a filing/price date without hedging that it could be an anomalous data point until the user pushed back, and one instance of asserting general knowledge (that `ZVZZT` is a NASDAQ test symbol) as if it were tool-verified fact in the same breath as genuinely grounded figures.

**That statelessness bug has since been fixed** — see [Fixed issues](#fixed-issues) below.

A third page in the web app, `/redteam.html` ("Red Team" in the nav), puts this in front of a browser instead of just a CLI: it lists the live personas from `/api/redteam/personas`, a "Run new test" button kicks off a real run in a background thread (`/api/redteam/run`, polled via `/api/redteam/status`) — so it survives the request finishing without a job queue — and every past report is browsable, transcript included, via `/api/redteam/reports`. Verified live end-to-end: triggered a run from the actual button in a browser, watched the status banner track it to completion, and confirmed the new report rendered with its real issues.

### Fixed issues

The Red Team page also tracks a hand-curated list of issues the tester found that were then actually fixed — `GET /api/redteam/fixes`, backed by the tracked `redteam_fixes.json` (not gitignored, unlike the raw run reports in `artifacts/redteam/`). An entry is only added once a fix is made *and* re-verified by a fresh live run — never auto-inferred from one run's absence of a finding.

One entry so far: the stateless-conversation bug above. The fix threads real conversation history through the whole stack — `/api/ask` now accepts a `history` field, `agent/graph.py`'s `run_graph()` replays it as message turns before the new question (the mixture-of-experts graph does the equivalent via a formatted prefix, since it doesn't use a message-list state), every `AgentRunner` (`FinAgentRunner`, `MoEAgentRunner`, `BedrockAgentRunner`) forwards it, and the chat UI (`web/app.js`) now maintains and sends the running conversation — the red-team tester was updated the same way, so it now exercises the product exactly as a real browser session would. A fresh post-fix run found 2 issues total (down from 8-11 pre-fix) with **zero** in the context-failure/self-contradiction category across all 4 sessions, including one persona proactively cross-referencing its own earlier turn correctly ("consistent with the same forward-dated data issue you flagged earlier"). One known caveat, disclosed rather than hidden: history is replayed as plain text only (no tool-call internals), so under sufficiently skeptical pushback FinAgent can't always prove a prior reply was tool-grounded — in one manual test it re-verified correctly but mischaracterized an already-correct earlier answer as "fabricated." Not in scope for this fix; left as a known limitation.

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

## Public demo deployment

The hosted demo runs as a single Fly.io machine behind `finagent.freshbuilds.dev`. It's the same image and the same code path as local development — the difference is entirely in configuration, because a public endpoint spends the maintainer's own Anthropic API budget on behalf of anyone who finds it.

**What public mode changes** (`FINAGENT_PUBLIC_DEMO=1`):

| Guard | Default | Why |
|---|---|---|
| `FINAGENT_RATE_LIMIT_PER_MINUTE` | 6 | Stops a single visitor (or a tab left on refresh) from running up a bill |
| `FINAGENT_RATE_LIMIT_PER_HOUR` | 40 | Same, over a longer window |
| `FINAGENT_GLOBAL_DAILY_CAP` | 500 | The real budget backstop — per-IP limits don't help against distributed traffic, and unlike the per-IP counters no request header can bypass this one |
| `FINAGENT_MAX_QUESTION_CHARS` | 1000 | Caps input tokens per request |
| `FINAGENT_MAX_HISTORY_MESSAGES` | 20 | Caps how much conversation gets replayed to the model each turn |
| Red-team runs | disabled | A run is 4 personas × N turns of live LLM calls. The demo serves the saved reports in `artifacts/redteam/` read-only; `/api/config` tells the frontend to disable the button, and `POST /api/redteam/run` returns 403 regardless |
| `FINAGENT_RAG_INDEX_ON_DEMAND=0` | — | Indexing a fresh 10-K takes minutes on a shared CPU, which a visitor experiences as a hang. `filing_search` answers from the index baked into the image and names the filings it does have when asked for one it doesn't |

None of this is on by default, so local development, the CLI, and the eval harness are unaffected.

**Image notes.** `sentence-transformers` depends on torch, whose default Linux wheel bundles ~2.5GB of CUDA libraries that a CPU-only demo will never execute. `pyproject.toml` pins torch to PyTorch's CPU index on Linux, which is what keeps the image at a sane size. The embedding model is downloaded at build time rather than first request — otherwise every restart would re-fetch it onto an ephemeral filesystem.

**Sizing.** The app measures ~455MB resident with the web stack, CPU torch, and the embedder all loaded, so `fly.toml` asks for 1GB. 512MB would run at the edge with no headroom for request handling.

```bash
# 1. Build the filing index that ships in the image. The demo can only search
#    filings indexed here, since on-demand indexing is off in production.
./scripts/prewarm-index.sh                       # or: TICKERS="AAPL NVDA" ./scripts/prewarm-index.sh

# 2. Create the app and give it the one secret it needs.
fly launch --no-deploy --name finagent
fly secrets set ANTHROPIC_API_KEY=sk-ant-...     # plus LANGFUSE_*/SENTRY_DSN/WANDB_API_KEY if wanted

# 3. Ship it.
fly deploy

# 4. Point the subdomain at it (adds the DNS record and provisions TLS).
fly certs add finagent.freshbuilds.dev
```

Then add a CNAME for `finagent` at the registrar pointing to the app's `.fly.dev` hostname, and link to it from the portfolio page at `freshbuilds.dev/finagent`.

To preview exactly what a visitor sees before deploying, run the demo-mode config locally on port 8001:

```bash
FINAGENT_PUBLIC_DEMO=1 FINAGENT_RAG_INDEX_ON_DEMAND=0 uv run finagent serve --port 8001
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
