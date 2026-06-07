# CareerAgent

CareerAgent is an agentic RAG backend for technical job matching, resume analysis, project relevance ranking, and interview-prep reporting. It combines deterministic scoring, lexical/vector retrieval, optional LLM augmentation, MCP tool exposure, and an evaluation harness.

The project is intended as a serious AI backend / RAG engineering prototype: runnable through Docker, covered by tests, and designed so LLM output explains structured evidence rather than replacing deterministic scores.

## Status

- Backend: FastAPI service with versioned APIs under `/api/v1`
- Retrieval: BM25, vector, hybrid fusion, and optional reranking
- Agent: ReAct loop (Reason → Act → Observe) that calls the tools, with step tracing
- LLM: optional OpenAI-compatible client with deterministic fallback
- MCP: stdio server exposing the core tools
- Knowledge base: interview-question KB in PostgreSQL + pgvector, powering RAG interview prep
- Evaluation: retrieval metrics, ablation runner, LLM-as-judge, latency/cost utilities
- Frontend: minimal static UI served at `/ui/` (paste JD + resume → report)
- Tests: `414 passed` with Postgres up (`413 passed`, 1 DB integration test skipped without it), `97%` coverage on Python 3.11.15

Current boundary: this is a backend-first prototype. It does not yet include a production UI, database persistence, authentication, rate limiting, or production observability.

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
│   │       ├── career.py    # POST /api/v1/career-match (end-to-end)
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
│   │   │   ├── react_controller.py # ReactAgent (Thought→Action→Observation loop)
│   │   │   ├── tools.py            # Default ReAct tools over shared state
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
├── experiments/             # Standalone experiment scripts
│   ├── day1_embedding_demo.py
│   ├── day2_tokenizer_demo.py
│   ├── day3_embedding_demo.py
│   ├── day4_position_encoding_demo.py
│   ├── day5_self_attention_demo.py
│   └── day6_multi_head_attention_demo.py
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

This gives the Week-3 LLM a structured starting point for risk analysis instead of auditing from scratch. The audit is exposed standalone at `POST /api/v1/audit`, attached to every `/match` result as `project_audit`, and rendered as a "Project Risk Audit" section in the generated report.

## Agent

`app/services/agent/` is a **ReAct agent** (`ReactAgent`): instead of routing a task to a single tool, it runs a Reason→Act→Observe loop. At each step the LLM emits a thought and either an action (tool call) or a final answer; the tool runs, its observation is fed back, and the loop continues until the LLM finishes or a step budget is hit.

- `build_default_agent(llm)` wires the services as ReAct tools — `parse_jd`, `parse_resume`, `match`, `rank_projects`, `audit`, `generate_report`.
- Tools operate on a shared `ReactState` (working memory), so the model passes only small inputs and reads concise observations rather than echoing large objects between steps.
- Tool preconditions surface as error observations (e.g. "parse the JD first"), which the agent reasons about and recovers from — genuine self-correction.
- `run(task, state)` returns a `ReactResult` with the final answer, the recorded `steps` (thought/action/observation), and whether it completed within the budget.

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

- **Store:** a curated interview-question / skill-note KB (`data/knowledge/`, ~145 docs across ~30 topics) lives in **PostgreSQL + pgvector**. Entries carry `skill` / `type` plus optional `role` / `difficulty` / `tags` / `answer_outline`, which the loader puts into the `metadata` (`jsonb`) column. `scripts/ingest_kb.py` embeds each document and upserts it into `knowledge_doc` (`vector(384)`, GIN index on `metadata`). (The `fastapi` topic is fully enriched as the reference; other topics can be enriched incrementally with the same fields.)
- **Retriever:** `PgVectorRetriever.search(query, k, filters=...)` implements the same `Retriever` interface but ranks DB-side with pgvector's cosine operator (`<=>`), and supports **metadata-filtered vector search** — scalar equality (`{"difficulty": "mid"}`) and jsonb array overlap (`{"role": ["backend"]}`) applied before ranking. That filtering + persistence is the concrete reason to use pgvector over an in-memory index. Because it's behind the `Retriever` protocol, the in-memory BM25/vector retriever is used for the offline test suite while pgvector backs production — a single integration test exercises the real DB (including a metadata filter) and self-skips when Postgres isn't available.
- **Generation:** `interview_prep.generate_interview_prep(jd, resume, kb_retriever, llm)` retrieves the questions relevant to the JD's skills, then asks the LLM to write a prep guide **grounded strictly on the retrieved questions**, highlighting the candidate's skill gaps. With no LLM it falls back to listing the retrieved questions + gaps. Exposed at `POST /api/v1/interview-prep`.

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

### Running Experiments

```bash
# Day 1: Embedding similarity demo
docker compose exec backend python experiments/day1_embedding_demo.py

# Day 2: Tokenizer & JD Parser demo
docker compose exec backend python experiments/day2_tokenizer_demo.py

# Day 3: Embedding semantic matching demo
docker compose exec backend python experiments/day3_embedding_demo.py

# Day 4: Position encoding demo
docker compose exec backend python experiments/day4_position_encoding_demo.py

# Day 5: Self-attention mechanism demo
docker compose exec backend python experiments/day5_self_attention_demo.py

# Day 6: Multi-head attention demo
docker compose exec backend python experiments/day6_multi_head_attention_demo.py
```

## Development Plan

See the 4-week development schedule for the full roadmap covering:
- Week 1: Project setup + ML/DL basics + Transformer intro
- Week 2: RAG + retrieval + reranking + BERT/GPT
- Week 3: Agent + MCP + fine-tuning
- Week 4: Evaluation + deployment + optimization

## License

MIT
