# CareerAgent

Agentic RAG system for technical job matching, resume analysis, and interview preparation.

## Architecture

```
User Input (JD / Resume)
       ↓
JD Parser / Resume Parser
       ↓
BM25 + Vector Retrieval
       ↓
Reranker
       ↓
Agent Controller
       ↓
MCP Tools
       ↓
GPT / LLM Report Generation
       ↓
Match Analysis + Skill Gap + Learning Plan
```

## Project Structure

```
career-agent-rag/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── api/
│   │   ├── router.py        # Route aggregator
│   │   └── v1/              # API v1 endpoints
│   │       ├── jd.py        # POST /api/v1/jd/parse
│   │       ├── resume.py    # POST /api/v1/resume/parse
│   │       └── match.py     # POST /api/v1/match
│   ├── models/              # Pydantic data schemas
│   │   ├── jd.py
│   │   ├── resume.py
│   │   ├── match.py
│   │   └── audit.py
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
│   │   └── retrieval/            # Retrieval backends (shared interface)
│   │       ├── base.py             # Retriever protocol, tokenizer, corpus builder
│   │       ├── bm25_retriever.py   # BM25Retriever (Okapi BM25, lexical recall)
│   │       ├── vector_retriever.py # VectorRetriever (embedding, semantic recall)
│   │       ├── hybrid_retriever.py # HybridRetriever (RRF / weighted fusion)
│   │       ├── reranker.py         # Reranker + RerankingRetriever (cross-encoder)
│   │       └── factory.py          # build_retriever(method, ...) ablation switch
│   ├── eval/                  # Retrieval evaluation harness
│   │   ├── metrics.py           # recall@k, MRR, nDCG@k
│   │   ├── datasets.py          # fixture loaders (corpus + labeled queries)
│   │   └── runner.py            # evaluate_retriever() → EvalReport
│   ├── core/
│       └── config.py        # pydantic-settings configuration
│   └── Dockerfile               # Backend Docker image
├── tests/                   # Pytest test suite
│   ├── conftest.py          # TestClient fixture
│   ├── test_health.py
│   ├── api/
│   │   └── v1/
│   │       ├── test_jd.py
│   │       ├── test_resume.py
│   │       └── test_match.py
│   ├── fixtures/            # Evaluation dataset
│   │   ├── retrieval_documents.json # Pooled resume corpus (with stable ids)
│   │   ├── relevance_queries.json   # Queries + graded relevance labels
│   │   ├── sample_jobs.json         # Eval jobs (job_id link + requirements)
│   │   └── job_descriptions.json    # Same jobs in JobDescription model shape
│   ├── eval/
│   │   ├── test_metrics.py          # recall@k, MRR, nDCG@k
│   │   ├── test_datasets.py         # fixture loaders
│   │   └── test_runner.py           # evaluate_retriever over fixtures
│   └── services/
│       ├── test_jd_parser.py
│       ├── test_resume_parser.py
│       ├── test_keyword_matcher.py
│       ├── test_vector_matcher.py
│       ├── test_report_generator.py
│       ├── test_match_pipeline.py
│       ├── test_project_auditor.py
│       └── retrieval/
│           ├── test_bm25_retriever.py
│           ├── test_vector_retriever.py
│           ├── test_hybrid_retriever.py
│           ├── test_reranker.py
│           └── test_factory.py
├── frontend/                # Reserved for future frontend
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
| GET | `/health` | Health check |
| POST | `/api/v1/jd/parse` | Parse job description |
| POST | `/api/v1/resume/parse` | Parse resume |
| POST | `/api/v1/match` | Match JD against resume (keyword + vector) |
| POST | `/api/v1/match/report` | Generate matching report |

Swagger docs: `http://localhost:8000/docs`

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

This gives the Week-3 LLM a structured starting point for risk analysis instead of auditing from scratch.

## Evaluation

`app/eval/` measures retrieval quality on a labeled dataset:

- **metrics.py** — `recall_at_k` (was the right doc recalled?), `mrr` (how high is the first hit?), `ndcg_at_k` (graded ranking quality, the one that judges reranking).
- **datasets.py** — loads the fixtures in `tests/fixtures/`: a pooled corpus of resume documents (`retrieval_documents.json`) and queries with graded relevance labels (`relevance_queries.json`). Pooling several resumes into one index gives realistic distractors, which is what makes the metrics meaningful.
- **runner.py** — `evaluate_retriever(retriever, documents, queries, k)` runs each query, maps results back to stable ids, and returns mean recall@k / MRR / nDCG@k in an `EvalReport`.

Swapping the retrieval method through `build_retriever` and re-running the harness is the basis for the Week-4 ablations.

## Development (Docker)

### Prerequisites

- Docker & Docker Compose

### Quick Start

```bash
# Copy environment config
cp .env.example .env

# Build and start the development server
docker compose up --build

# The API is available at http://localhost:8000
```

### Hot Reload

The development server uses `uvicorn --reload` with volume-mounted code. Edit any file on your host — the server restarts automatically.

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

### Running Tests

```bash
docker compose exec backend pytest tests/ -v
```

## Development Plan

See the 4-week development schedule for the full roadmap covering:
- Week 1: Project setup + ML/DL basics + Transformer intro
- Week 2: RAG + retrieval + reranking + BERT/GPT
- Week 3: Agent + MCP + fine-tuning
- Week 4: Evaluation + deployment + optimization

## License

MIT
