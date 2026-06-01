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
│   │   └── match.py
│   ├── services/            # Business logic
│   │   └── embedding.py     # EmbeddingService
│   ├── core/
│       └── config.py        # pydantic-settings configuration
│   └── Dockerfile               # Backend Docker image
├── tests/                   # Pytest test suite
├── frontend/                # Reserved for future frontend
├── experiments/             # Standalone experiment scripts
├── docker-compose.yml       # Multi-service orchestration
└── requirements.txt
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/jd/parse` | Parse job description |
| POST | `/api/v1/resume/parse` | Parse resume |
| POST | `/api/v1/match` | Match JD against resume |

Swagger docs: `http://localhost:8000/docs`

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
docker compose exec backend python experiments/day1_embedding_demo.py
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
