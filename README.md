# CareerAgent

CareerAgent is an agentic RAG backend for technical job matching, resume analysis, project relevance ranking, and interview-prep reporting. It combines deterministic scoring, lexical/vector retrieval, optional LLM augmentation, MCP tool exposure, and an evaluation harness.

The project is intended as a serious AI backend / RAG engineering prototype: runnable through Docker, covered by tests, and designed so LLM output explains structured evidence rather than replacing deterministic scores.

## Status

- Backend: FastAPI service with versioned APIs under `/api/v1`
- Retrieval: BM25, vector, hybrid fusion, and optional reranking
- Agent: ReAct loop (Reason → Act → Observe) over the tools (incl. KB search, advice, bullet rewriting, multi-JD comparison), with step tracing and an `ask_user` pause/resume action; exposed for one-shot Q&A (`POST /api/v1/career/ask`) and a streamed multi-turn chat with persistent state + slash commands (`POST /api/v1/career/chat/stream`)
- LLM: optional OpenAI-compatible client with deterministic fallback
- MCP: stdio server exposing the core tools
- Knowledge base: interview-question KB in PostgreSQL + pgvector, powering RAG interview prep
- Evaluation: retrieval metrics, ablation runner, LLM-as-judge, latency/cost utilities
- Frontend: minimal static chat UI served at `/ui/` — paste a resume + one or more JDs, then chat with the agent (free text; reasoning steps stream in live and the answer types out token-by-token) or use slash commands with autocomplete (`/match`, `/report`, `/prep`, `/audit`, `/compare`), with the reasoning trace expandable per reply
- Tests: `490 passed` with Postgres up (`489 passed`, 1 DB integration test skipped without it), `96%` coverage on Python 3.11.15

Current boundary: this is a backend-first prototype. It does not yet include a production UI, database persistence, authentication, rate limiting, or production observability.

## Study & Interview Notes (`docs/`)

Companion notes for learning the codebase and explaining its engineering
decisions in an interview. **AI-generated study aids — verify against the
source before relying on them.**

- [`docs/interview_engineering_notes.md`](docs/interview_engineering_notes.md) — decision-by-decision deep dive (13 sections): architecture, the agent + memory design, parsing/scoring, every model/parameter/threshold choice (with "what if I tune it" and the industry alternatives), retrieval, RAG, the rule-based audit, evaluation, MCP, and the recurring engineering patterns.
- [`docs/interview_mock_qa.md`](docs/interview_mock_qa.md) — mock interview, ~160 questions across 48 topics in four layers: (1) project-level Q&A with multi-layer follow-up chains; (2) **fundamentals deep dives** (transformer/attention, sentence-embedding training & distillation, BM25 probabilistic roots, HNSW/PQ internals, LLM sampling & KV-cache, async/GIL); (3) **debugging & system-design scenarios** (retrieval-quality regressions, scaling to multi-tenant SaaS, i18n, real-time KB updates); (4) **trap questions** that probe trade-offs to bedrock.
- [`docs/interview_self_intro.md`](docs/interview_self_intro.md) — a 30-second / 2-minute project pitch script, with hooks that steer follow-up questions toward prepared ground.

## Quick Start

Docker is the recommended way to run the project because the app targets Python 3.11+ and depends on ML packages that are easier to reproduce in a container.

```bash
cp .env.example .env
docker compose up --build          # starts the backend + a pgvector Postgres
docker compose exec backend python -m scripts.ingest_kb   # load the knowledge base
```

The API will be available at:

- Service index: `http://localhost:8000/`
- Web UI: `http://localhost:8000/ui/`
- Health check: `http://localhost:8000/health`
- Swagger docs: `http://localhost:8000/docs`

Run the test suite inside the container:

```bash
docker compose exec backend pytest -q
docker compose exec backend pytest --cov=app --cov-report=term-missing -q
```

Run the MCP server:

```bash
docker compose exec backend python -m app.mcp.server
```

Run the end-to-end career match skill:

```bash
docker compose exec backend python -m app.skills.career_match --jd JD.txt --resume RESUME.txt
```

Use `--no-llm` for a fully deterministic run.

## Architecture

Entry points sit on top of one shared core — they are alternative ways to
invoke the same services, not stages of a single pipeline:

```
Inputs (JD / Resume)
        │
   Entry points:  REST API (/api/v1) · ReAct agent · MCP server · career-match skill
        │
        ▼
 Deterministic core  (owns all scores / rankings / findings)
   JD & resume parsing · BM25 + vector + hybrid retrieval (+ optional rerank)
   skill matching · project-relevance ranking · rule-based risk audit
        │
        ▼
 Optional LLM layer  (schema-validated, deterministic fallback)
   field extraction · risk advice · grounded narrative report
        │
        ▼
 Outputs: match score · skill gaps · ranked experience · risk findings · report
```

The deterministic core always computes the numbers; the LLM layer only
extracts, advises, and narrates on top of them. The ReAct agent and MCP server
are orchestration/exposure layers over the same core services.

## Project Structure

```
career-agent-rag/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── api/
│   │   ├── router.py        # Route aggregator
│   │   ├── errors.py        # Consistent {status, error} exception handlers
│   │   ├── deps.py          # Shared lazy singletons (embedding service, LLM)
│   │   └── v1/              # API v1 endpoints
│   │       ├── jd.py        # POST /api/v1/jd/parse
│   │       ├── resume.py    # POST /api/v1/resume/parse
│   │       ├── match.py     # POST /api/v1/match, /api/v1/match/report
│   │       ├── audit.py     # POST /api/v1/audit
│   │       ├── career.py    # POST /api/v1/career-match (end-to-end), /career/ask (agent), /career/chat/stream (multi-turn SSE)
│   │       └── interview.py # POST /api/v1/interview-prep (RAG)
│   ├── models/              # Pydantic data schemas
│   │   ├── jd.py
│   │   ├── resume.py
│   │   ├── match.py
│   │   ├── audit.py
│   │   └── common.py        # ErrorResponse / ErrorDetail
│   ├── services/            # Business logic
│   │   ├── embedding.py          # EmbeddingService
│   │   ├── _embedding_helpers.py # Shared embedding utilities
│   │   ├── jd_parser.py          # JDParser (rule + embedding)
│   │   ├── resume_parser.py      # ResumeParser (rule + embedding)
│   │   ├── keyword_matcher.py    # KeywordMatcher (keyword + vector)
│   │   ├── vector_matcher.py     # VectorMatcher (semantic match)
│   │   ├── report_generator.py   # ReportGenerator (template-based report)
│   │   ├── match_pipeline.py     # Rank resume items vs JD via retrieval
│   │   ├── project_auditor.py    # Rule-based resume authenticity / risk checks
│   │   ├── llm_client.py         # LLMClient (OpenAI-compatible chat wrapper)
│   │   ├── llm_support.py        # LLM helpers w/ deterministic fallback + schema validation
│   │   ├── usage.py              # TokenUsage / UsageTracker / estimate_cost
│   │   ├── knowledge.py          # KB loading + in-memory KB retriever
│   │   ├── interview_prep.py     # RAG interview prep (retrieve KB → grounded guide)
│   │   ├── agent/               # ReAct agent
│   │   │   ├── react_controller.py # ReactAgent (Thought→Action→Observation loop, ask_user pause/resume)
│   │   │   ├── tools.py            # Default ReAct tools over shared state (parse/match/rank/audit/report + kb_search/interview_prep/advise/rewrite_bullet + compare_jds/select_jd)
│   │   │   ├── slash.py            # Deterministic slash commands (/match, /report, /prep, /audit, /compare)
│   │   │   ├── sessions.py         # In-memory chat sessions (persistent ReactState + history + rolling summary)
│   │   │   ├── schemas.py          # ReactState / ReactTool / ReactStep / ReactResult / ReactDecision
│   │   │   └── trace.py            # Scratchpad rendering + step serialization
│   │   └── retrieval/            # Retrieval backends (shared interface)
│   │       ├── base.py             # Retriever protocol, tokenizer, corpus builder
│   │       ├── bm25_retriever.py   # BM25Retriever (Okapi BM25, lexical recall)
│   │       ├── vector_retriever.py # VectorRetriever (embedding, semantic recall)
│   │       ├── hybrid_retriever.py # HybridRetriever (RRF / weighted fusion)
│   │       ├── reranker.py         # Reranker + RerankingRetriever (cross-encoder)
│   │       ├── factory.py          # build_retriever(method, ...) ablation switch
│   │       └── pgvector_retriever.py # PgVectorRetriever (KB search via pgvector)
│   ├── db/                    # PostgreSQL + pgvector
│   │   └── connection.py        # connection + knowledge_doc schema
│   ├── mcp/                   # MCP server + client
│   │   ├── tools.py             # MCP tool implementations (dict in/out)
│   │   ├── server.py            # FastMCP server exposing the tools
│   │   └── client.py            # MCPClient (connect + call tools over stdio)
│   ├── skills/                # Packaged end-to-end capabilities
│   │   └── career_match.py      # run_career_match() full pipeline + CLI
│   ├── eval/                  # Evaluation harnesses
│   │   ├── metrics.py           # recall@k, MRR, nDCG@k
│   │   ├── datasets.py          # fixture loaders (corpus + labeled queries)
│   │   ├── runner.py            # evaluate_retriever() → EvalReport
│   │   ├── ablation.py          # run_ablation() across retrieval methods
│   │   ├── llm_judge.py         # LLM-as-Judge: grounding + quality scoring
│   │   └── perf.py              # LatencyRecorder + p50/p95/p99 stats
│   ├── core/
│       └── config.py        # pydantic-settings configuration
│   └── Dockerfile               # Backend Docker image
├── tests/                   # Pytest test suite
│   ├── conftest.py          # TestClient fixture
│   ├── test_health.py
│   ├── api/
│   │   ├── test_errors.py
│   │   ├── test_deps.py
│   │   └── v1/
│   │       ├── test_jd.py
│   │       ├── test_resume.py
│   │       ├── test_match.py
│   │       ├── test_audit.py
│   │       ├── test_career.py
│   │       └── test_interview.py
│   ├── db/
│   │   └── test_connection.py       # schema helper (fake conn)
│   ├── fixtures/            # Evaluation dataset
│   │   ├── retrieval_documents.json # Pooled resume corpus (with stable ids)
│   │   ├── relevance_queries.json   # Queries + graded relevance labels
│   │   ├── sample_jobs.json         # Eval jobs (job_id link + requirements)
│   │   └── job_descriptions.json    # Same jobs in JobDescription model shape
│   ├── eval/
│   │   ├── test_metrics.py          # recall@k, MRR, nDCG@k
│   │   ├── test_datasets.py         # fixture loaders
│   │   ├── test_runner.py           # evaluate_retriever over fixtures
│   │   ├── test_ablation.py         # ablation harness
│   │   ├── test_llm_judge.py        # LLM-as-Judge
│   │   └── test_perf.py             # latency stats
│   ├── mcp/
│   │   ├── test_tools.py            # MCP tool implementations
│   │   ├── test_server.py           # FastMCP server (list/call tools)
│   │   └── test_client.py           # MCPClient (unit + live server)
│   ├── skills/
│   │   └── test_career_match.py     # career-match pipeline
│   └── services/
│       ├── test_jd_parser.py
│       ├── test_resume_parser.py
│       ├── test_keyword_matcher.py
│       ├── test_vector_matcher.py
│       ├── test_report_generator.py
│       ├── test_match_pipeline.py
│       ├── test_project_auditor.py
│       ├── test_llm_client.py
│       ├── test_llm_support.py
│       ├── test_usage.py
│       ├── test_knowledge.py
│       ├── test_interview_prep.py
│       ├── agent/
│       │   ├── test_react.py
│       │   ├── test_tools.py
│       │   ├── test_slash.py
│       │   ├── test_sessions.py
│       │   └── test_trace.py
│       └── retrieval/
│           ├── test_bm25_retriever.py
│           ├── test_vector_retriever.py
│           ├── test_hybrid_retriever.py
│           ├── test_reranker.py
│           ├── test_factory.py
│           └── test_pgvector_retriever.py
├── data/knowledge/          # Curated KB (interview_questions.json)
├── scripts/                 # ingest_kb.py (embed + upsert KB into pgvector)
├── frontend/                # Minimal static UI (index.html)
├── docs/                    # AI-generated study & interview notes (verify against source)
│   ├── interview_engineering_notes.md # Decision/parameter/threshold deep dive
│   ├── interview_mock_qa.md           # Mock interview w/ multi-layer follow-up chains
│   └── interview_self_intro.md        # 30s/2min project pitch script
├── docker-compose.yml       # Multi-service orchestration
└── requirements.txt
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service index (name, version, docs link) |
| GET | `/health` | Health check |
| POST | `/api/v1/jd/parse` | Parse job description (LLM extraction, rule fallback) |
| POST | `/api/v1/resume/parse` | Parse resume (LLM extraction, rule fallback) |
| POST | `/api/v1/match` | Match JD against resume (skills, relevance, risk audit) |
| POST | `/api/v1/match/report` | Generate matching report (incl. risk audit) |
| POST | `/api/v1/audit` | Audit a resume for authenticity / quality risks |
| POST | `/api/v1/career-match` | End-to-end: parse + match + rank + audit + report from raw text |
| POST | `/api/v1/career/ask` | One-shot Q&A over a resume + one or more JDs: the ReAct agent picks tools dynamically (incl. multi-JD comparison); returns answer + reasoning trace |
| POST | `/api/v1/career/chat/stream` | Multi-turn chat over Server-Sent Events: persistent session (parsed JD/resume/match + rolling history), slash commands for the deterministic pipeline, and an agent that can pause to ask the user (`awaiting_user`) and resume. Emits a `step` event per ReAct step, then the answer as `token` events, then `done` |
| POST | `/api/v1/interview-prep` | RAG: retrieve interview questions from the KB + grounded prep guide |
| GET | `/ui/` | Minimal browser UI |

Swagger docs: `http://localhost:8000/docs`

Successful responses use a `{"status": "success", "data": ...}` envelope; errors use the mirror shape `{"status": "error", "error": {type, message, detail}}`, applied consistently to validation (422), HTTP, and unhandled (500) errors via `app/api/errors.py` (500s don't leak internals). CORS is enabled for the browser frontend (`CORS_ORIGINS`).

## Current Matching Signals

The current matching pipeline combines:

- Exact and rule-based skill overlap
- Raw resume text skill recovery
- Embedding-based semantic skill matching
- JD responsibility to resume experience alignment
- Document-level semantic similarity

The overall score is skill-first:

- With experience alignment: `75% skill coverage + 15% experience alignment + 10% document similarity`
- Without experience alignment: `85% skill coverage + 15% document similarity`
- Keyword-only fallback: `90% skill coverage`

`POST /api/v1/match/report` turns the match result into a template-based markdown report with skill gaps, experience alignment, and recommendations.

## Retrieval

The `app/services/retrieval/` package treats the resume (its experiences and projects) as a searchable corpus and ranks documents against a JD-derived query. All backends share one interface — `search(query, k) -> list[RetrievalResult]` — so they can be swapped or composed:

- **BM25Retriever** — Okapi BM25 implemented from scratch (lexical/keyword recall). Scores documents by IDF-weighted, length-normalized term frequency with configurable `k1` / `b`. Matches exact tech terms (PyTorch, Docker, Kubernetes) that embeddings tend to blur.
- **VectorRetriever** — embedding-based semantic recall. The corpus is embedded once at construction; each query is embedded and ranked by cosine similarity, with an optional `min_score` floor. Matches experiences phrased differently from the JD but close in meaning.
- **HybridRetriever** — fuses BM25 and vector recall over one corpus, exposing each arm as `.bm25` / `.vector` for inspection or ablation. Default fusion is **Reciprocal Rank Fusion** (rank-based, so the two score scales never need normalizing); a **weighted** min-max sum is available via `method="weighted"`. Per-arm weights are configurable.
- **Reranker / RerankingRetriever** — a cross-encoder re-scoring stage. The bi-encoder retrievers above score query and document independently (cheap, coarse); the cross-encoder feeds the (query, document) pair through the model together (accurate, expensive), so it re-ranks only a small candidate pool: *recall top ~20 → cross-encoder rescore → top-k*. `RerankingRetriever` wraps any base retriever behind the same `search(query, k)` interface. The model loads lazily on first use and can be injected for testing.

`corpus_from_resume(resume)` builds a list of `RetrievalDocument`s (one per experience and project), each carrying a stable id (`exp:0`, `proj:1`), source type/index, and display metadata for provenance. Retrievers consume plain text via `document_texts(docs)` and identify hits by integer index, so a result's `doc_id` maps straight back to `docs[doc_id]` for reporting and evaluation. A tech-aware tokenizer preserves tokens like `c++`, `c#`, and `node.js`.

`build_retriever(method, corpus, ...)` constructs any backend by name — `"bm25"`, `"vector"`, `"hybrid"`, `"hybrid+rerank"` — which is the switch used for ablations.

## Project Relevance

`match_pipeline.rank_resume_projects(jd, resume, ...)` scores each resume experience/project by retrieval relevance to a JD-derived query (skills + responsibilities). It returns `ProjectRelevance` entries — stable `doc_id`, human-readable `label`, the raw retriever `score`, and a min-max `normalized_score` (best = 1.0) — attributed back to the source item via provenance. The `method` argument selects the retrieval backend, so the same call powers ablation comparisons.

## Project Risk Audit

`project_auditor.audit_resume(resume)` runs transparent, rule-based authenticity checks (no LLM, so every finding is explainable) and returns a `ProjectAuditReport` with `RiskFinding`s, a normalized `risk_score`, and a summary. It detects:

- **unsupported_skill** — a skill is listed but never appears in any experience/project. Advanced/high-claim skills (RAG, agents, MCP, LoRA, …) are flagged *high* and must be backed by actual prose, since listing an impressive term as a "technology" is just another unverified claim.
- **vague_experience** — a highlight claims impact or effort without quantification (no numbers/percentages).
- **unsupported_project_claim** — a project lists advanced technologies its description does not substantiate (too thin, or never mentions them).

This gives the LLM layer a structured starting point for risk analysis instead of auditing from scratch. The audit is exposed standalone at `POST /api/v1/audit`, attached to every `/match` result as `project_audit`, and rendered as a "Project Risk Audit" section in the generated report.

## Agent

`app/services/agent/` is a **ReAct agent** (`ReactAgent`): instead of routing a task to a single tool, it runs a Reason→Act→Observe loop. At each step the LLM emits a thought and either an action (tool call) or a final answer; the tool runs, its observation is fed back, and the loop continues until the LLM finishes or a step budget is hit.

- `build_default_agent(llm)` wires the services as ReAct tools — `parse_jd`, `parse_resume`, `match`, `rank_projects`, `audit`, `generate_report`, plus the agentic extensions `kb_search`, `interview_prep`, `advise`, `rewrite_bullet`, and the multi-JD pair `compare_jds` / `select_jd`.
- Tools operate on a shared `ReactState` (working memory), so the model passes only small inputs and reads concise observations rather than echoing large objects between steps.
- Tool preconditions surface as error observations (e.g. "parse the JD first"), which the agent reasons about and recovers from — genuine self-correction.
- `run(task, state)` returns a `ReactResult` with the final answer, the recorded `steps` (thought/action/observation), and whether it completed within the budget.

Where the deterministic `/career-match` pipeline always runs the *same* fixed steps, the agent shines on **open-ended, branch-by-result** tasks where the path isn't known up front — e.g. *"why isn't my match higher, and how do I fix it?"* The agent diagnoses (`match` / `audit` / `rank_projects`), then decides what to do about it: `kb_search` to pull background on a missing skill (looping one query per gap), `advise` for targeted recommendations grounded on the diagnosis, or `rewrite_bullet` to retarget a resume line — and it can fold interview prep (`interview_prep`, RAG over the KB) into the same loop. The KB-backed tools degrade gracefully when no knowledge base or LLM is available.

For **multi-JD** questions (*"which of these roles should I apply to?"*), the agent calls `compare_jds` — it parses and matches every candidate JD (seeded in `jd_inputs`) against the resume and ranks them best-first — then `select_jd` promotes the chosen role into the active `jd` / `match` so the single-JD tools (`rank_projects`, `advise`, `rewrite_bullet`, …) tailor the resume to it. The number of comparison rounds is data-driven, which is exactly what a fixed endpoint can't express.

This is surfaced as `POST /api/v1/career/ask` (`{question, resume_text, jd_text}` or `{question, resume_text, jds: [{text, label}]}` → `{answer, completed, steps}`): the endpoint seeds a `ReactState` (candidate JDs, embedding service, KB retriever, LLM) and returns the agent's answer plus its full Thought/Action/Observation trace for transparency. The deterministic endpoints stay untouched — the agent is an additional entry point for free-form questions, not a replacement for the fixed pipeline.

### Conversational chat (`/career/chat/stream`)

`POST /api/v1/career/chat/stream` turns the agent into a multi-turn assistant, streamed over Server-Sent Events. A process-local session (`sessions.py`) holds the **persistent `ReactState`** — so parsing/matching done on one turn (by a tool *or* a slash command) is reused on the next instead of recomputed — plus the conversation history. The first turn seeds the resume/JD(s) and returns a `session_id`; later turns send just the id and a message.

**Conversation memory** is two-tier so the agent can follow back-references ("why that one?") without the context growing unbounded: the **last 3 rounds** are passed verbatim, and **older turns are folded into a rolling LLM summary** (`fold_old_turns`, with a bounded plain-text fallback when no LLM is configured). The summary plus recent turns are injected into the agent's prompt as a `Conversation so far:` block — separate from the structured working-memory summary (parsed JD/resume/match), which the agent already had.

Two input styles share that one session state:

- **Slash commands** (`slash.py`) — `/match`, `/report`, `/prep [role] [difficulty]`, `/audit`, `/compare`, `/help` — call the **deterministic pipeline directly**. They're cheap, reproducible, and work even with no LLM configured. This keeps the deterministic-core invariant visible in the product: shortcuts for the fixed analyses, the agent for everything open-ended.
- **Free text** drives the **ReAct agent**, which can now also emit an `ask_user` action: it **pauses** mid-loop (`state == "awaiting_user"`, returning the question), and the next user message **resumes** the same run with the reply folded in as the observation. This is what makes genuinely interactive flows possible — mock-interview follow-ups, or rewriting a resume bullet after asking the user for the missing metric — rather than one-shot answers.

**Streaming** is two-phase. The agent gathers via tools, then signals it's ready with a `finish` action — it does *not* write the user-facing answer in the decision JSON. Instead the answer is **composed in a separate step** that streams token-by-token (`ReactAgent.iter_run` yields each step for live `step` events; `stream_answer` streams the composed reply as `token` events; then a `done` event carries the full reply, state, session id, and history). Splitting "decide to finish" from "write the answer" is what lets the reply stream like a typing assistant while keeping the per-step reasoning visible. The static chat UI consumes this stream — reasoning steps appear live, the answer types out — and offers slash-command autocomplete.

The agent is LLM-driven by design (it requires a configured LLM); decisions are parsed as strict JSON via the same `extract_json` helper, and a malformed/unknown reply becomes a recoverable observation rather than a crash. Note the deterministic-core invariant still holds: the tools call the same services, so scores/rankings/findings are computed deterministically — the LLM only decides *which* steps to take.

`LLMClient` (`llm_client.py`) is a thin wrapper over any **OpenAI-compatible** chat endpoint, reading `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` from settings. The SDK client is created lazily and can be injected for testing.

## LLM augmentation (deterministic core + LLM layer)

The architecture keeps a **deterministic core** (rules, retrieval, vector/BM25 scoring, rule-based audit) that owns all scores, rankings, and findings, with the **LLM as an optional layer** that explains, extracts, and synthesizes on top of that structured evidence — never overriding the numbers. `llm_support.py` makes this safe and testable:

- `generate_text(llm, prompt, system, fallback)` — free-form text (e.g. a narrative report) that **falls back to a deterministic string** if the LLM is unconfigured, errors, or returns nothing.
- `generate_model(llm, prompt, model_cls, fallback)` — asks for JSON, validates it into a **Pydantic model**, and on failure does a **corrective retry**: it feeds the bad output, the error, and the target schema back, and tightens the system prompt to demand JSON-only (no prose/fences) — not a blind re-prompt. After the retries it falls back to a deterministic object (the reference pattern for future LLM-based parsers).

`generate_report(jd, resume, result, llm=...)` is the first application: structured fields are always computed deterministically; the narrative `full_report` is rendered from a template by default, or generated by the LLM grounded strictly on those fields when a client is supplied — falling back to the template if the call fails. Explanations are tested by structure/fallback behavior; scores stay deterministically tested.

`POST /api/v1/match/report` uses the LLM-generated narrative when `LLM_API_KEY` / `LLM_MODEL` are configured, and transparently falls back to the template report otherwise.

The same pattern is applied across tools, deterministic-core-first:

- **`parse_jd` / `parse_resume`** accept an optional `llm=`: when configured, fields are **extracted by the LLM** (validated into `JobDescription` / `Resume` via `generate_model`, with the rule-based parse as the fallback) — better on messy inputs. Skills are normalized to lowercase to match the rule convention. The `/api/v1/jd/parse` and `/api/v1/resume/parse` endpoints use the LLM when configured.
- **`audit_resume`** accepts an optional `llm=`: rule findings and the risk score stay deterministic; the LLM only adds **how-to-fix `advice`** grounded on those findings. Exposed on the standalone `POST /api/v1/audit` path (kept out of the hot `/match` path to keep matching fast).
- `resume_matcher` and `project_ranker` stay **pure scoring** — their explanations live in the LLM report, not per-tool, so the score/ranking the eval harness depends on never goes through the model.

## MCP Server

`app/mcp/` exposes CareerAgent's capabilities as **MCP** tools via a FastMCP server, so any MCP host (e.g. Claude Desktop) or MCP client can discover and call them: `parse_jd`, `parse_resume`, `match_resume`, `audit_resume`, `rank_projects`.

- `tools.py` — the tool implementations (dict in, dict out), kept dependency-free and unit-tested.
- `server.py` — registers them on a FastMCP server with JSON-Schema input schemas derived from the type hints, and lazily attaches the embedding service.

Run it as a stdio MCP server:

```bash
docker compose exec backend python -m app.mcp.server
```

`client.py` provides `MCPClient`, an async context manager that launches/connects to an MCP server (this project's by default) and calls tools over the protocol:

```python
async with MCPClient() as client:
    names = await client.list_tools()
    result = await client.call_tool("parse_jd", {"raw_text": "..."})
```

It decodes JSON tool output to dicts/lists and raises `MCPToolError` when the server reports a failed call.

## Knowledge base & RAG

This is where retrieval-augmented *generation* actually happens. The other retrieval in the project ranks the resume's own items; here the LLM's output is **grounded on documents retrieved from an external knowledge base**.

- **Store:** a curated interview-question / skill-note KB (`data/knowledge/`, ~145 docs across ~30 topics) lives in **PostgreSQL + pgvector**. Every entry carries `skill` / `type` plus `role` / `difficulty` / `tags` / `answer_outline`, which the loader puts into the `metadata` (`jsonb`) column (the loader treats these as optional, so bare entries still load). `scripts/ingest_kb.py` embeds each document and upserts it into `knowledge_doc` (`vector(384)`, GIN index on `metadata`, HNSW index on `embedding` for cosine ANN search).
- **Retriever:** `PgVectorRetriever.search(query, k, filters=...)` implements the same `Retriever` interface but ranks DB-side with pgvector's cosine operator (`<=>`), and supports **metadata-filtered vector search** — scalar equality (`{"difficulty": "mid"}`) and jsonb array overlap (`{"role": ["backend"]}`) applied before ranking. That filtering + persistence is the concrete reason to use pgvector over an in-memory index. Because it's behind the `Retriever` protocol, the in-memory BM25/vector retriever is used for the offline test suite while pgvector backs production — wrapped in a `RerankingRetriever` so the pgvector candidate pool is re-scored by the cross-encoder before grounding (filters and candidate metadata are forwarded through the rerank stage). A single integration test exercises the real DB (including a metadata filter) and self-skips when Postgres isn't available.
- **Generation:** `interview_prep.generate_interview_prep(jd, resume, kb_retriever, llm, role=, difficulty=)` retrieves **per JD skill** (each skill contributes its top results, deduped, so common skills don't crowd out the rest) — **optionally metadata-filtered by `role` / `difficulty`** — then asks the LLM to write a prep guide **grounded strictly on the retrieved questions and their `answer_outline`s**, highlighting the candidate's skill gaps. With no LLM it falls back to listing the retrieved questions (with outlines) + gaps. Exposed at `POST /api/v1/interview-prep` (which accepts optional `role` / `difficulty`).

This keeps the architecture's invariant: retrieval and the structured match stay deterministic; the LLM only synthesizes over retrieved context, with a fallback.

## Skill

`app/skills/career_match.py` packages the whole pipeline as one capability — `run_career_match(jd_text, resume_text, ...)` chains parse → match → rank → audit → report and returns a `CareerMatchResult`. It powers a Claude Code skill at `.claude/skills/career-match/SKILL.md` and a CLI:

```bash
python -m app.skills.career_match --jd JD.txt --resume RESUME.txt   # --no-llm for the deterministic path
```

Where MCP exposes the tools *individually* for an external host to orchestrate, the skill is the *in-process, end-to-end* version that chains them into a single report.

## Evaluation

`app/eval/` measures retrieval quality on a labeled dataset:

- **metrics.py** — `recall_at_k` (was the right doc recalled?), `mrr` (how high is the first hit?), `ndcg_at_k` (graded ranking quality, the one that judges reranking).
- **datasets.py** — loads the fixtures in `tests/fixtures/`: a pooled corpus of resume documents (`retrieval_documents.json`) and queries with graded relevance labels (`relevance_queries.json`). Pooling several resumes into one index gives realistic distractors, which is what makes the metrics meaningful.
- **runner.py** — `evaluate_retriever(retriever, documents, queries, k)` runs each query, maps results back to stable ids, and returns mean recall@k / MRR / nDCG@k in an `EvalReport`.

Swapping the retrieval method through `build_retriever` and re-running the harness is the basis for the ablation below.

### Ablation

`app/eval/ablation.py` runs every retrieval method over the same labeled fixture (`run_ablation` + `format_ablation_table`, or `python -m app.eval.ablation`). Results on the bundled dataset (k=10):

| Method | Recall@10 | MRR | nDCG@10 |
|--------|-----------|-----|----------|
| bm25 | 0.800 | 0.917 | 0.771 |
| vector | 0.922 | 0.903 | 0.864 |
| hybrid | 0.901 | 0.917 | 0.842 |
| hybrid+rerank | 0.901 | 1.000 | 0.873 |

Takeaways on this set: **vector retrieval wins on Recall@10** — the queries include paraphrases that lexical search misses, which embeddings recover. **hybrid+rerank wins on MRR and nDCG@10** after the cross-encoder re-scores the hybrid candidate pool, showing that reranking is useful for promoting the strongest relevant hit even when the fused retriever already has good recall. The cross-encoder model downloads on first use and is then cached.

### LLM-as-Judge

`app/eval/llm_judge.py` evaluates *generated* text (e.g. the LLM report) against the structured evidence it should be grounded on. `judge_report(llm, output_text, evidence)` returns a `JudgeVerdict` scoring **groundedness / coverage / clarity** (1–5) and listing **unsupported claims** — the automated hallucination check for the report generator. It runs through the same `generate_model` path (schema-validated, corrective retry), so a judge failure yields an *unevaluated* verdict rather than raising.

This catches real drift: on a live grounded report the judge scored coverage and clarity 5 but groundedness 3, flagging the coaching-style suggestions the report had added beyond the structured data — exactly the kind of "fluent but unsupported" addition the deterministic-core architecture is meant to keep in check.

### Latency & cost

- `app/eval/perf.py` — `LatencyRecorder.measure()` times code blocks and `stats()` summarizes them as **p50 / p95 / p99 / mean / max** (percentiles describe tail behavior the mean hides).
- `app/services/usage.py` — `TokenUsage` / `UsageTracker` accumulate token counts, and `estimate_cost(usage, input_per_mtok, output_per_mtok)` converts them to dollars. `LLMClient` captures `response.usage` into `last_usage` and feeds an optional `usage_tracker`, so cost is tracked transparently across calls.

Example over 3 live LLM calls: p50 ≈ 2.8s, p95 ≈ 4.2s; 269 total tokens ≈ $0.0003 at example pricing. (LLM latency dominates; the deterministic retrieval/audit paths are sub-second.)

## Development

### Prerequisites

- Docker & Docker Compose
- Python 3.11+ if you run the project outside Docker

The container image uses `python:3.11-slim-bookworm`. Local Python versions older than 3.10 will fail on modern type syntax such as `Any | None`; using Docker avoids that mismatch.

### Docker

```bash
cp .env.example .env
docker compose up --build
```

The development server uses `uvicorn --reload` with volume-mounted code. Edit any file on your host — the server restarts automatically.

### Running Tests

```bash
docker compose exec backend pytest -q
docker compose exec backend pytest --cov=app --cov-report=term-missing -q
```

## Retrieval Roadmap

Further improvements to the RAG retrieval stack, not yet implemented. Listed in
priority order with the reasoning and expected effect.

### DB-side hybrid retrieval (BM25 + vector)

**Now:** the pgvector path recalls semantically (then reranks) but has no
lexical arm; the BM25 arm and RRF fusion run only over the in-memory resume
corpus.
**Change:** add a `tsvector` full-text column (+ GIN index) to `knowledge_doc`
and fuse lexical and vector rankings DB-side (e.g. RRF).
**Why:** pure vector search drifts on exact technical tokens (`gRPC`, `k8s`,
version strings); BM25 nails them.
**Effect:** higher recall on rare/exact terms, especially for short keyword
queries — complementing the cross-encoder reranker, which only re-orders what
was already recalled.

### Query rewriting / expansion (HyDE, multi-query)

**Now:** the KB query is just the JD's skill terms joined together; per-skill
retrieval already spreads coverage across skills.
**Change:** before retrieval, expand the query — HyDE (generate a hypothetical
answer, embed that) or multi-query (LLM rewrites into several phrasings, union
the results).
**Why:** short keyword queries embed poorly and miss paraphrased passages.
**Effect:** better recall on under-specified or jargon-light queries, at the
cost of one extra LLM call per query.

## License

MIT
