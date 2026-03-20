# GraphMind

GraphMind is a user-centric financial memory assistant.

It stores chat memories in a Neo4j knowledge graph, persists chat/session history in PostgreSQL, and generates grounded answers with citations using an LLM.

## What Is Implemented Today

- Multi-user auth with JWT + bcrypt
- Strict user isolation for graph and chat history
- Graph ingestion pipeline for chat messages and uploaded documents
- Hybrid retrieval (graph + vectorized text memories) with:
  - mode-based query planning
  - adaptive depth and adaptive `top_k`
  - timeline filtering
  - real shortest-path hop distance in query
  - weighted ranking (`graph_distance`, `recency`, `confidence`, `reinforcement`)
- Deferred reinforcement of cited nodes
- Optional background hard-decay worker for persisted confidence
- Mindmap endpoint + frontend graph visualization

## High-Level Architecture

- Frontend: React + TypeScript + Vite
- Backend: FastAPI
- Datastores:
  - PostgreSQL: users, sessions, messages, metadata
  - Neo4j: memory graph (facts, entities, relations)
- LLM: Google Gemini (fallback behavior when unavailable)
- Optional storage: AWS S3 for document uploads (best-effort)

## Current Data/Memory Approach

### Ingestion

1. Ensure/create user node in Neo4j
2. Store raw message node
3. Deduplicate facts by `(user_id, fact_text)`:
   - existing fact -> reinforce
   - new fact -> create and link from message
4. Merge entity nodes with `id + user_id`
5. Create canonical links (for example `User -> MADE_TRANSACTION -> Transaction -> AFFECTS_ASSET -> Asset`)
6. Link evidence facts to structured nodes (`CONFIRMS`, `RELATES_TO`)
7. Run contradiction detection heuristic (skipped for document ingestion routes)

### Retrieval

1. Classify query into one of:
   - `DIRECT_LOOKUP`
   - `AGGREGATION`
   - `RELATIONAL_REASONING`
2. Build adaptive retrieval plan (`depth`, `top_k`) from mode + query length
3. Execute mode-specific Cypher with user scoping and optional timeline filter
4. Compute hop distance in the same query using `shortestPath`
5. Rank by weighted score:

```text
score = 0.4*graph_distance + 0.3*recency + 0.2*confidence + 0.1*reinforcement
```

6. Return ranked nodes + citations, then reinforce cited nodes after answer generation

## API Surface (Current)

### Health

- `GET /health`

### Auth

- `POST /auth/signup`
- `POST /auth/login`
- `GET /auth/me`

### Chat + Sessions

- `POST /chat`
- `GET /sessions`
- `GET /sessions/{session_id}/messages`
- `POST /sessions/{session_id}/archive`
- `DELETE /sessions/{session_id}`

### Memory Graph

- `GET /memory/mindmap`

### Documents

- `POST /documents/upload`
- `POST /documents/ingest`
- `POST /documents/upload-and-ingest`

OpenAPI docs are available at:

- `http://localhost:8001/docs`

## Retrieval Metrics Behavior

`/chat` returns retrieval and generation timings separately:

- `graph_query_ms`
- `vector_search_ms`
- `context_assembly_ms`
- `retrieval_ms` (excludes LLM generation)
- `llm_generation_ms`

## Repository Structure

```text
graphmind/
├── backend/
│   ├── api/
│   │   ├── main.py
│   │   ├── models.py
│   │   ├── models_auth.py
│   │   ├── models_mindmap.py
│   │   └── routes/
│   │       ├── auth.py
│   │       ├── chat.py
│   │       ├── documents.py
│   │       ├── health.py
│   │       └── memory.py
│   ├── config/
│   │   └── settings.py
│   ├── database/
│   │   ├── init.sql
│   │   └── postgres.py
│   ├── services/
│   │   ├── auth/
│   │   ├── database/
│   │   ├── extraction/
│   │   ├── graph/
│   │   │   ├── ingestion.py
│   │   │   ├── memory_decay.py
│   │   │   ├── mindmap_service.py
│   │   │   ├── query_understanding.py
│   │   │   ├── retrieval.py
│   │   │   ├── retrieval_old.py
│   │   │   └── schema.cypher
│   │   ├── llm/
│   │   ├── orchestrator/
│   │   └── storage/
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── contexts/
│   │   ├── lib/
│   │   └── pages/
│   └── package.json
├── docker-compose.yml
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Docker + Docker Compose

### 1) Start Databases

```bash
cd /home/tanmay08/graphmind
docker compose up -d
```

### 2) Backend Setup

```bash
cd /home/tanmay08/graphmind/backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set required values in `.env`:

- `GEMINI_API_KEY`
- `JWT_SECRET_KEY`
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- PostgreSQL values if different from Docker defaults

### 3) Frontend Setup

```bash
cd /home/tanmay08/graphmind/frontend
npm install
```

### 4) Run App

Backend:

```bash
cd /home/tanmay08/graphmind/backend
source venv/bin/activate
uvicorn api.main:app --reload --host 0.0.0.0 --port 8001
```

Frontend:

```bash
cd /home/tanmay08/graphmind/frontend
npm run dev
```

### Access

- Frontend: `http://localhost:5173`
- API docs: `http://localhost:8001/docs`

## Environment Notes

The backend supports optional confidence hard-decay worker through env flags:

- `MEMORY_HARD_DECAY_ENABLED`
- `MEMORY_HARD_DECAY_INTERVAL_SECONDS`
- `MEMORY_HARD_DECAY_BATCH_SIZE`
- `MEMORY_DECAY_HALF_LIFE_DAYS`
- `MEMORY_DECAY_FLOOR`

If hard-decay is disabled (default), retrieval still uses recency + confidence + reinforcement from stored graph properties.

## Current Limitations

- Retrieval is graph-only (vector retrieval is planned but not active)
- Query understanding is deterministic keyword based
- Contradiction detection is heuristic
- Document extraction chunk-level vector indexing is not yet implemented

## License

Internal project / hackathon-style repository. Add a formal license if you plan to open source.
