# QBR Intelligence Pipeline

**PDF → validated business metadata → PostgreSQL → chunks/embeddings → semantic search**

A Python/FastAPI proof of concept for processing **Quarterly Business Review (QBR)** PDF documents. The system extracts and cleans PDF content, uses a structured LLM extraction step to identify business facts, validates those facts against source evidence, persists document records, and optionally indexes the PDF for semantic retrieval using **PostgreSQL + pgvector**.

> **Scope and status:** This guide describes the assembled project developed through **V22**, with the **V27 API organization patch**. These versions were delivered as incremental patches, **not as a verified, single standalone repository**. You must already have the application/domain/adapter modules and migrations installed in your project. V23–V26 (hybrid full-text search, RAG answers, comprehensive evaluation and production hardening) are **planned**, not implemented by the V27 API patch. Commands marked *example* may require adapting to your actual repository configuration.

## 1. What the system does

1. **Upload** a QBR PDF using FastAPI. Validate content type, size and PDF format; store the file and its SHA-256 hash.
2. **Extract and preprocess** page text and structured fragments, preserving document/page provenance. PyMuPDF is the main PDF parser; other parsing fallbacks depend on which earlier adapters are installed.
3. **Generate structured metadata** with the configured extraction pipeline/LLM. Target fields include customer, reporting period, summary, goals, KPIs, achievements, challenges, risks and action items.
4. **Validate** required fields, conflicts and evidence. Flag incomplete or unsupported metadata for review; the LLM does not override deterministic validation.
5. **Persist** extraction status, metadata, validation results, fragments and evidence in PostgreSQL.
6. **Index on demand**: reload the stored PDF, split it into section-aware chunks, generate embeddings, and persist vectors with provenance in pgvector.
7. **Search** stored chunks using vector similarity; return text, source document, page numbers and source IDs.

**Not yet delivered as integrated API features:** PostgreSQL full-text/Hybrid Search, grounded RAG answer generation, and production-grade authorization/job queues. Optional DeepEval/Langfuse adapters exist from earlier work but are not a guarantee that evaluations/traces are enabled in the assembled application.

## 2. Technology stack

| Layer | Technology |
| --- | --- |
| API | FastAPI, Uvicorn, Pydantic |
| Runtime | Python, asynchronous application services, dependency injection |
| PDF | PyMuPDF; optional pdfplumber-based processing where installed |
| LLM metadata | Configured extraction pipeline; Instructor/OpenAI adapters in earlier versions |
| Persistence | PostgreSQL 16, SQLAlchemy 2, asyncpg, Alembic |
| Vector search | pgvector, 1536-dimensional vectors, HNSW cosine index (V21) |
| Embeddings | Configured OpenAI embedder; default `text-embedding-3-small` (1536 dimensions) |
| Testing | pytest, pytest-asyncio, HTTPX; optional evaluation adapters |
| Infrastructure | Docker Compose for PostgreSQL |

## 3. Project structure

The following is the **intended assembled layout**. Exact modules depend on which incremental patches have been applied.

```text
qbr-intelligence/
├── .env                         # local secrets; do not commit
├── docker-compose.yml           # PostgreSQL 16 + pgvector
├── alembic.ini
├── alembic/
│   ├── env.py
│   └── versions/
│       ├── 0001_qbr_extraction.py
│       └── 0002_qbr_pgvector.py
├── src/qbr/
│   ├── api/
│   │   ├── __main__.py         # python -m qbr.api
│   │   ├── main.py             # ASGI application
│   │   ├── routes.py           # one central route registry
│   │   ├── dependencies.py
│   │   ├── endpoints/
│   │   │   ├── health.py
│   │   │   ├── documents.py
│   │   │   └── vectors.py
│   │   ├── contracts/
│   │   │   ├── documents.py
│   │   │   └── vectors.py
│   │   └── qbr_*.py            # compatibility modules
│   ├── application/            # processing, validation, indexing, search
│   ├── domain/                 # domain models, rules and exceptions
│   ├── ports/                  # interfaces for adapters
│   ├── adapters/
│   │   ├── pdf/
│   │   ├── llm/
│   │   ├── persistence/
│   │   ├── storage/
│   │   ├── chunking/
│   │   └── embedding/
│   ├── infrastructure/         # database, configuration, logging
│   └── bootstrap/
│       ├── runtime.py          # lifespan: creates live services
│       └── vector_wiring.py    # V22 vector service composition
├── tests/
│   ├── unit/
│   └── integration/
└── data/uploads/               # local PDF storage
```

The API layer contains HTTP handling only. `application` owns use cases; `ports` define contracts; `adapters` implement external integrations; `bootstrap` constructs services.

## 4. Prerequisites (Ubuntu/Linux)

- Python compatible with the project's installed dependencies (the project has been used with Python 3.14; check third-party package support if installation fails).
- Docker Engine and Docker Compose plugin (`docker compose version`).
- A valid API key and a working extraction pipeline factory if you intend to upload/process PDFs with an LLM.
- An assembled QBR repository containing `src/qbr`, `alembic`, and tests.

Check:

```bash
python3 --version
docker --version
docker compose version
docker ps
```

If `docker ps` says `permission denied` on `/var/run/docker.sock`, fix Docker daemon/socket permissions before continuing. Adding your user to the `docker` group requires a new login session and grants effectively root-equivalent Docker access. Do **not** work around this with `chmod 666 /var/run/docker.sock`.

## 5. Clone/open the project and create a virtual environment

```bash
cd ~/Desktop/task_orit/qbr-intelligence
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
```

Install the **project's own dependency specification** if available:

```bash
# Preferred if pyproject.toml defines installable project/dependencies:
python -m pip install -e .

# Or, if the repository uses requirements.txt instead:
# python -m pip install -r requirements.txt
```

**If the project has neither dependency file**, the following is a *development baseline*, not a complete guaranteed dependency lockfile:

```bash
python -m pip install \
  fastapi 'uvicorn[standard]' python-multipart \
  'pydantic>=2' 'sqlalchemy>=2' asyncpg alembic pgvector \
  pymupdf pdfplumber openai instructor \
  pytest pytest-asyncio httpx
```

Install additional packages required by your selected extraction pipeline and optional evaluation/tracing adapters. Do not assume installing the baseline automatically configures the LLM pipeline.

## 6. Environment variables

Create `.env` in the project root. **Example for local development only** (replace credentials and key):

```dotenv
POSTGRES_DB=qbr_db
POSTGRES_USER=qbr
POSTGRES_PASSWORD=replace-with-a-strong-local-password
POSTGRES_PORT=5432
DATABASE_URL=postgresql+asyncpg://qbr:replace-with-a-strong-local-password@127.0.0.1:5432/qbr_db

OPENAI_API_KEY=replace-with-your-real-key
QBR_PIPELINE_FACTORY=your_project.pipeline_factory:create_pipeline

QBR_UPLOAD_DIR=data/uploads
PDF_MAX_SIZE_MB=20
DATABASE_ECHO=false
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10

QBR_HOST=127.0.0.1
QBR_PORT=8000
QBR_RELOAD=1
```

**Critical:** `QBR_PIPELINE_FACTORY` above is a **placeholder**, not a known implemented module. The existing `qbr.bootstrap.runtime` requires a real importable `module:function` that returns an object exposing `async execute(PDFSource)`. Find your actual factory and replace this value before starting the API. If no factory exists, implement/wire it in your project first; the README alone cannot supply one.

The `DATABASE_URL` password must match `POSTGRES_PASSWORD`. URL-encode special characters in passwords embedded in URLs. Keep `.env` out of Git (`.gitignore`).

Docker Compose automatically reads `.env`, but plain Python/Alembic do **not necessarily** load it. For a trusted local `.env` containing shell-compatible assignments:

```bash
set -a
source .env
set +a
```

Alternatively use your application's existing dotenv/settings loader. Avoid sourcing untrusted files.

## 7. Start PostgreSQL with pgvector

Use the **existing Compose service** and keep the original named volume. For PostgreSQL 16 the service should resemble:

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    restart: unless-stopped
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-qbr_db}
      POSTGRES_USER: ${POSTGRES_USER:-qbr}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?Set POSTGRES_PASSWORD in .env}
    ports:
      - "127.0.0.1:${POSTGRES_PORT:-5432}:5432"
    volumes:
      - qbr_pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $$POSTGRES_USER -d $$POSTGRES_DB"]
      interval: 5s
      timeout: 3s
      retries: 12

volumes:
  qbr_pgdata:
```

Start and check:

```bash
docker compose config
docker compose up -d postgres
docker compose ps
docker compose logs --tail=50 postgres
```

**Existing database?** Before changing the image from `postgres:16-alpine` to `pgvector/pgvector:pg16`, make a backup:

```bash
docker compose exec -T postgres \
  sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' \
  > qbr_backup.dump
ls -lh qbr_backup.dump
```

Retain the existing `qbr_pgdata` volume; do **not** run `docker compose down -v` or `docker volume rm`. Switching from Alpine to a Debian-based image can require collation/locale checks. If the existing server fails to start, preserve the volume and inspect logs rather than initializing a new database. Keep the backup secure because it contains application data.

Enable the extension in the application database:

```bash
docker compose exec postgres sh -c \
  'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "CREATE EXTENSION IF NOT EXISTS vector;"'

docker compose exec postgres sh -c \
  'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT extname, extversion FROM pg_extension WHERE extname = '\''vector'\'';"'
```

If you see `extension "vector" is not available` or missing `vector.control`, the **running PostgreSQL server image** does not have pgvector installed. Installing the Python `pgvector` package alone does not fix that error.

## 8. Run Alembic migrations

Ensure the `.env` values are available to the Alembic process (Section 6), then:

```bash
alembic current
alembic upgrade head
alembic current
alembic history
```

Expected migrations from earlier phases:

- `0001_qbr_extraction.py`: document/extraction persistence.
- `0002_qbr_pgvector.py`: pgvector-backed chunks/embeddings and search index.

The vector migration expects **1536 dimensions**. If you use another embedding model/dimension, update the schema and corresponding embedder/store configuration consistently **before** indexing. Do not arbitrarily modify an already-applied migration in a shared environment.

## 9. Verify bootstrap integration (important)

V27 `qbr.api.main:create_app` registers the document and vector routes **once**, before startup. The lifespan function comes from `qbr.bootstrap.runtime.api_lifespan`.

The earlier V16 runtime initializes only `app.state.document_service`. For V22 vector routes to function, it must also initialize `app.state.index_document` and `app.state.vector_search` **after** creating `document_service` and the existing SQLAlchemy `sessions`, **before** the lifespan `yield`:

```python
from qbr.bootstrap.vector_wiring import build_vector_services

app.state.index_document, app.state.vector_search = build_vector_services(
    document_service=app.state.document_service,
    session_factory=sessions,
)
```

Add this inside the existing lifespan, not in `api/main.py`, and do **not** create a second database engine. On shutdown, dispose of the existing engine and clear app state as appropriate. Do not add another `app.include_router(...)` for vectors: V27 `api/routes.py` already registers them.

If these two services are missing, vector endpoints return **503 `Vector services not configured`**; the health endpoint can still return 200.

## 10. Start FastAPI

From the project root, with the virtual environment active and environment variables loaded:

```bash
PYTHONPATH=src python -m qbr.api
```

Equivalent direct Uvicorn command:

```bash
uvicorn qbr.api.main:app \
  --reload \
  --host 127.0.0.1 \
  --port 8000 \
  --env-file .env
  ```

For a single process without automatic reload:

```bash
QBR_RELOAD=0 PYTHONPATH=src python -m qbr.api
```

URLs:

- Swagger UI: <http://127.0.0.1:8000/docs>
- OpenAPI JSON: <http://127.0.0.1:8000/openapi.json>
- Health: <http://127.0.0.1:8000/health>

Smoke test:

```bash
curl -i http://127.0.0.1:8000/health
```

Expected body: `{"status":"ok"}`. A successful health response does **not** prove that the LLM, pgvector or embeddings are working; verify them separately.

## 11. API reference and end-to-end walkthrough

The routes below are the **implemented V27 route registry** (documents + V22 vector endpoints):

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/health` | API liveness |
| `POST` | `/api/v1/qbr/documents` | Upload/process PDF |
| `GET` | `/api/v1/qbr/documents/{document_id}` | Retrieve metadata and status |
| `GET` | `/api/v1/qbr/documents/{document_id}/validation` | Retrieve validation/evidence |
| `POST` | `/api/v1/qbr/documents/{document_id}/prepare-vectors` | Index an existing PDF |
| `POST` | `/api/v1/qbr/search` | Vector similarity search |

### A. Upload a PDF

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/qbr/documents \
  -F 'file=@tests/fixtures/sample_qbr.pdf;type=application/pdf'
```

Replace `tests/fixtures/sample_qbr.pdf` with a **real PDF path** in your repository; that filename is illustrative, not a guaranteed included fixture. Upload triggers extraction and persistence and requires a working `QBR_PIPELINE_FACTORY` and LLM credentials if your pipeline uses OpenAI.

Example response shape (values illustrative):

```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "COMPLETED",
  "incomplete": false,
  "page_count": 8,
  "fragment_count": 15
}
```

### B. Read the document and its validation

```bash
export DOCUMENT_ID=550e8400-e29b-41d4-a716-446655440000
curl -sS "http://127.0.0.1:8000/api/v1/qbr/documents/$DOCUMENT_ID"
curl -sS "http://127.0.0.1:8000/api/v1/qbr/documents/$DOCUMENT_ID/validation"
```

Use the **actual UUID returned by upload**, not the example UUID.

### C. Generate/store embeddings for that document

```bash
curl -sS -X POST \
  "http://127.0.0.1:8000/api/v1/qbr/documents/$DOCUMENT_ID/prepare-vectors"
```

This is a separate action; PDF upload alone does not automatically index vectors. The default embedding configuration is `text-embedding-3-small` with 1536 dimensions. This action requires a functioning pgvector migration, embedder and V22 bootstrap wiring.

### D. Search indexed QBR content

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/qbr/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"What risks did the customer report?","top_k":5}'
```

Limit search to one document when needed:

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/qbr/search \
  -H 'Content-Type: application/json' \
  -d "{\"query\":\"What KPIs improved?\",\"top_k\":5,\"document_id\":\"$DOCUMENT_ID\"}"
```

Search returns an array of hits with chunk text, `document_id`, `page_numbers`, `source_ids`, model and distance/similarity values. **Similarity is a transformed cosine-distance score, not a calibrated probability of correctness.** This endpoint returns passages, not an LLM-generated RAG answer.

## 12. Tests and quality checks

From the project root:

```bash
source env/bin/activate
set -a; source .env; set +a
export PYTHONPATH=src
```

### Quick checks

```bash
python -m compileall -q src/qbr
python -m pytest -q
```

### Unit tests

```bash
python -m pytest tests/unit/ -vv --tb=short
```

### API route registry (V27)

If you copied the V27 test into `tests/test_api_route_registry.py`:

```bash
python -m pytest tests/test_api_route_registry.py -vv --tb=short
```

### Relevant earlier version tests (only if present)

```bash
python -m pytest tests/unit/test_document_service_v20.py -vv --tb=short
python -m pytest tests/unit/test_pgvector_v21.py -vv --tb=short
python -m pytest tests/unit/test_vectors_api_v22.py -vv --tb=short
```

Some test files may have been copied under different paths during earlier patches. Check what exists with `find tests -type f -name 'test_*.py'` rather than assuming every named file is present.

### Integration tests (require configured external services)

```bash
python -m pytest tests/integration/ -vv --tb=short
```

This command is valid **only if** your repository contains that directory and its integration fixtures are configured. Real database tests should use a dedicated test database, not your development/production database. LLM integration tests may incur API charges; keep them separately selectable.

### Useful diagnostics

```bash
python -m pytest --collect-only -q
alembic current
docker compose ps
docker compose logs --tail=100 postgres
curl -i http://127.0.0.1:8000/health
```

If pytest reports `collected 0 items`, confirm your test file contains functions named `test_*` (and was not accidentally replaced by application/router source code).

> **Verification note:** These commands document how to run the tests; this README does **not** claim that the complete project or all integration tests have been executed successfully in your local environment.

## 13. Common failures and fixes

| Symptom | Likely cause | Next check |
| --- | --- | --- |
| `extension "vector" is not available` | PostgreSQL image lacks server-side pgvector | `docker compose config --images`; use compatible `pgvector/pgvector:pg16` |
| `permission denied` for Docker socket | User/session lacks Docker socket permissions | `id -nG`, `ls -l /var/run/docker.sock`; re-login after group change |
| `DATABASE_URL is required` | `.env` not exported/loaded into Python process | Section 6; inspect environment without printing secrets |
| `QBR_PIPELINE_FACTORY is required` | Missing extraction factory setting | Set a real `module:function` |
| `ModuleNotFoundError: qbr` | `src` not on Python path or package not installed | `export PYTHONPATH=src` or `pip install -e .` |
| Vector API returns 503 | V22 services not placed on `app.state` | Section 9; lifespan wiring |
| `relation ... does not exist` | Alembic migration not applied to selected DB | `alembic current`, `alembic upgrade head` |
| Vector dimension mismatch | Embedder and pgvector schema disagree | Align model, dimensions and migration |
| `401` from LLM provider | Invalid/unavailable API key | Verify configured provider credentials |
| Upload 413/415/422 | Too large, unsupported media or invalid PDF | PDF limit, MIME type, actual file content |
| Search returns no hits | Document not indexed, wrong filter, or no matching chunks | Call prepare-vectors; inspect indexing and DB state |

## 14. Operational and security notes

- Never commit `.env`, API keys, database backups or uploaded customer PDFs.
- Bind development API and PostgreSQL ports to `127.0.0.1` unless access control, TLS and network restrictions are in place.
- Do not expose the current proof of concept publicly without authentication, authorization, tenant isolation and request limits.
- Protect document provenance: returned passages must be traceable to original documents and pages.
- Re-indexing and concurrent indexing need careful coordination; the current V21 vector store replaces vectors per document and is not a complete multi-version/concurrent indexing system.
- Backup PostgreSQL before image/migration changes; never delete the named volume casually.
- Use `--reload` only for local development.

## 15. Development history and what comes next

| Stage | Delivered focus |
| --- | --- |
| V10–V16 | PostgreSQL persistence, API upload/read/validation, runtime composition |
| V17 | Optional evaluation, tracing and embedding adapters |
| V18 | Section-aware chunking |
| V19–V20 | Application use cases and `ProcessDocument`/`DocumentService` wiring |
| V21 | pgvector schema/store and cosine similarity retrieval |
| V22 | Document vector indexing and semantic search APIs |
| V27 | Clean API endpoints/contracts, central route registry and `python -m qbr.api` launcher |
| V23–V26 | **Planned**: hybrid FTS+vector search, evidence-grounded RAG answers, evaluation/observability integration and production hardening |

**Recommended next milestone:** validate the complete PDF → extraction → PostgreSQL → vector indexing → search workflow against a real local PostgreSQL instance. Then implement hybrid retrieval and grounded RAG answering with automated quality gates.