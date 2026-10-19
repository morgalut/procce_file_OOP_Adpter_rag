# QBR Intelligence Pipeline — Development Design v3 (Hybrid PDF Processing + LLM Tools)

**Stack:** Python · FastAPI · PostgreSQL · SOLID/OOP · DeepEval · Langfuse  
**Goal:** A focused working PoC that converts heterogeneous Quarterly Business Review (QBR) PDFs into validated, evidence-grounded structured records. Embeddings and retrieval are downstream extensions.

## 1. Requirements and deliverables

**Input:** QBR PDF. **Output:** customer/business unit, reporting period, executive summary, goals, KPIs/results, achievements, challenges, risks, follow-up actions, and optional owner/deadline. Detect missing, ambiguous, conflicting, or unsupported information; persist the results in PostgreSQL. Deliver runnable code, setup instructions, tests, and an architecture rationale.

**Initial non-goals:** UI, distributed production orchestration, full OCR, and RAG question answering. **Assumptions:** single organization, English sample documents, external configured LLM, PostgreSQL in Docker. Apply access control and redaction to sensitive content.

## 2. Architecture and processing sequence — hybrid design

```text
FastAPI PDF upload
  → file validation + SHA-256 + authorized storage → PostgreSQL: RECEIVED
  → PyMuPDFAdapter (PDFExtractor port): incremental PDFPage(text, tables, page, warnings)
      ↳ selective pdfplumber fallback for difficult tables (planned)
  → PDFProcessingAdapter / DocumentPreprocessor (planned)
      → deterministic text/table normalization; preserve original excerpts and provenance
      → section detection; bounded CleanDocumentFragment objects
  → application/extract_metadata.py (planned orchestrator)
      → ModelRouter (policy-controlled model/provider selection)
      → InstructorMetadataAdapter (MetadataGenerator port)
          → Pydantic structured QBRMetadata fragments
          ↔ optional LLM function calls via controlled PDFTools gateway
              → read_pdf_page / extract_pdf_tables / search_pdf_content
              → PDFExtractor + processed-fragment store (reuse existing adapters)
  → MetadataAccumulator: merge fragments, preserve conflicting alternatives
  → ValidationEngine: deterministic schema, evidence and consistency checks
  → optional DeepEval evaluation + human review when needed
  → PostgreSQL: metadata, evidence, evaluations, status
  → section-aware chunking + metadata overlay: EMBEDDING_PENDING
  → LATER: embedding adapter → pgvector → retrieval → RAG evaluation

Cross-cutting: Langfuse traces, correlation IDs, token/cost accounting, access controls.
```

**Execution modes:** (A) default deterministic extraction followed by LLM structured generation; (B) hybrid mode, where the LLM may request additional authorized read-only PDF tools after receiving clean fragments. The LLM never directly opens arbitrary filesystem paths, selects unapproved models, or overrides validation. The application runs tools, not the model itself.

**State:** `RECEIVED → PARSED → EXTRACTED → VALIDATING → COMPLETED | REVIEW_REQUIRED | FAILED`. Here `PARSED` means PDF text/tables and clean fragments are available; `EXTRACTED` means LLM metadata fragments have been generated. Independent chunk/vector states remain `PREPARED` and `EMBEDDED`. Store stage versions, errors, timestamps and evidence; retry transient errors only. Retain reviewed versions on reprocessing.

**Incremental processing:** pages are read in document order, cleaned and emitted as bounded fragments; no mandatory full-document string is sent to the LLM. Source page and original text are retained for audit. Cleaning must not discard numerical signs, units, dates, table headers or row associations. A warning is not proof of extraction completeness.

## 3. Libraries and trade-offs

| Layer | Selection | Rationale / alternatives |
|---|---|---|
| API | FastAPI | Typed REST and async; Flask/Django unnecessary for PoC |
| PDF | PyMuPDF + selective pdfplumber | Incremental page text/tables; pdfplumber fallback planned, not yet implemented |
| PDF preprocessing | Deterministic adapter + typed fragments | Normalize text and tables without LLM; retain source evidence |
| Agentic tools | Typed PDFTools gateway + bounded executor | Reuse parser for optional read-only LLM function calls |
| Model selection | Application ModelRouter | Explicit policy, provider capabilities, fallback and budgets; LLM does not choose models |
| LLM | Instructor + provider SDK | Typed Pydantic outputs, simple adapter; LangChain/LlamaIndex add abstraction |
| Schema | Pydantic v2 | Strict field validation, not factual correctness |
| Persistence | PostgreSQL + SQLAlchemy 2 + Alembic | Transactions, JSONB, migrations; repository port enables replacement |
| Evaluation | DeepEval | Custom extraction checks and later RAG faithfulness/relevancy; Ragas is separate |
| Observability | Langfuse + OpenTelemetry | Traces, I/O, tokens, latency, cost; LangSmith alternative |
| Vector stage | pgvector + embedding provider port | Reuses Postgres; dimension depends on model |
| Testing | pytest, pytest-asyncio, httpx | Mock ports; real PostgreSQL integration tests |

## 4. Domain contracts and provenance

- `QBRMetadata`: `customer_name: str | None`, `period: str | None`, `executive_summary: str | None`, `goals: list[Finding]`, `kpis: list[KPI]`, `achievements/challenges/risks: list[Finding]`, `actions: list[ActionItem]`.
- `Finding`: text, evidence references, validation status.
- `KPI`: name, current/previous values, unit, period, approximation flag, evidence.
- `ActionItem`: description, `owner: str | None`, `due_date_raw: str | None`, `due_date_iso: date | None`, evidence.
- `Evidence`: 1-based page number, exact source excerpt, optional bounding box, extraction method.

Preserve original and normalized values separately. Missing data is `null` or an empty collection, never invented. Conflicting values are retained as alternatives with a conflict finding.

## 5. SOLID architecture and repository layout

```text
src/qbr/
├── api/routes.py, dependencies.py
├── domain/models.py, validation_rules.py, metadata_accumulator.py
├── domain/clean_fragments.py                         # planned typed cleaned fragments
├── application/process_document.py, extract_metadata.py  # orchestration
├── application/prepare_chunks.py
├── ports/pdf_extractor.py, pdf_processor.py          # processor port planned
├── ports/metadata_generator.py, repository.py
├── ports/pdf_tools.py, model_router.py               # tool/router ports planned
├── ports/evaluator.py, embedder.py, tracer.py
├── adapters/pdf/pymupdf_adapter.py                   # existing parser
├── adapters/pdf/pdf_processing_adapter.py           # next implementation
├── adapters/pdf/pdfplumber_adapter.py                # optional fallback, planned
├── adapters/tools/pdf_tools.py                       # planned controlled gateway
├── adapters/llm/instructor_adapter.py, model_router.py  # planned
├── adapters/persistence/, evaluation/, observability/
└── infrastructure/config.py, database.py
tests/unit/, integration/, evaluation/, fixtures/
alembic/, docker-compose.yml, .env.example, README.md
```

**Existing contracts take precedence over old illustrative signatures.** In particular, the implemented `PDFExtractor.iter_pages(source, options=...)` yields `PDFPage` asynchronously, rather than the earlier draft's `extract(pdf_bytes)` method. The `MetadataGenerator.generate_fragment(request)` protocol specifies the LLM-facing operation; the concrete adapter implements it. `PDFProcessingAdapter` consumes `PDFPage` and emits `CleanDocumentFragment` with page-level provenance. The application service wires these components and does not import PyMuPDF or an LLM provider directly.

**SRP:** separate parsing, preprocessing, tool execution, LLM generation, accumulation and validation. **OCP/LSP:** interchangeable ports with contract tests. **ISP:** narrow protocols. **DIP:** application depends on ports; FastAPI dependency injection selects adapters. Avoid a second PDF parsing implementation inside tools.

## 6. PostgreSQL schema

| Table | Key fields |
|---|---|
| `qbr_documents` | UUID PK, filename, sha256, storage_uri, customer_name, reporting_period, raw_content TEXT, metadata JSONB, status, parser/prompt/schema versions, timestamps |
| `qbr_evidence` | id, document_id FK, field_path, page_number, source_excerpt, bounding_box JSONB, validation_status |
| `qbr_chunks` | id, document_id FK, chunk_index, chunk_version, section, chunk_text, metadata_overlay JSONB, embedding_status, embedding_model, nullable embedding |
| `qbr_evaluations` | id, document_id FK, evaluator_version, dataset_version, scores JSONB, findings JSONB, created_at |
| `llm_usage` | id, document_id FK, trace_id, provider, model, input/output/cached tokens, latency_ms, estimated_cost, created_at |

Use transactions, foreign keys, indexes on customer/period/status, and unique `(document_id, chunk_index, chunk_version)`. SHA-256 supports deduplication under explicit reprocessing rules. Store original PDFs in file/object storage, with URI/hash in PostgreSQL. Enable pgvector later, matching the selected embedding dimension.

## 7. API specification

| Method | Endpoint | Behavior |
|---|---|---|
| POST | `/api/v1/qbr/documents` | Multipart PDF; return ID/status and metadata if completed |
| GET | `/api/v1/qbr/documents/{id}` | Metadata, status, versions |
| GET | `/api/v1/qbr/documents/{id}/validation` | Evidence, conflicts, missing fields, evaluations |
| POST | `/api/v1/qbr/documents/{id}/reprocess` | Re-run with new versions, preserve history |
| POST | `/api/v1/qbr/documents/{id}/prepare-vectors` | Prepare chunks, no embedding required |

Reject invalid MIME/content, oversized or malformed PDFs; enforce page and time limits. Use 400 for invalid uploads, 422 for unprocessable content, 404 for unknown IDs, and suitable 5xx for transient failures. A worker can later run the same application service.

## 8. Validation and review policy

1. **Structural:** Pydantic strict types, constraints, normalized dates.
2. **Deterministic:** evidence existence, KPI units/periods, relevant arithmetic, missing owners/deadlines, duplicate tables, conflicting figures.
3. **Semantic:** DeepEval assesses source-grounding and summary quality.

Accepted results must be source-supported. Ambiguous/unsupported cases become `REVIEW_REQUIRED`. Self-reported LLM confidence is not calibrated evidence.

**Validate the validator:** human-labeled positive/negative examples; inject fabricated owners, altered KPIs, wrong dates, unsupported summaries; measure false positives/negatives, field precision/recall, and reviewer agreement. Version judge prompts/models; retain a holdout set; monitor drift. LLM-as-judge alone cannot approve data.

## 9. Six supplied PDFs as golden cases

| Case | Expected assertions |
|---|---|
| Northwind | Q2 2026; MAU 640 → 875; adoption 61% → 74%; three actions with owners/targets |
| Globex | April–June 2026; adoption 68% summary vs 64% dashboard is a conflict; training action lacks owner |
| Contoso | 318 active users; ERP sync root cause unconfirmed; first two action dates absent |
| BluePeak | Failed executions 1.7% meets objective below 2%; governance approval process missing |
| Apex | Approx. 520 in prose vs 524 in table: preserve approximation; Hospital South risk |
| Vertex | Approx. 70% vs dashboard 67.9%; Poland readiness review owner missing; investigation date TBD |

Create human-reviewed expected JSON with field-level page/excerpt evidence. Do not automatically treat rounding differences as material contradictions.

## 10. Tests and evaluation gates

- **Unit:** PDF layouts/tables/corruption, model malformed output/retries/timeouts, schema nulls/units, conflict logic, chunk enrichment, orchestration transitions.
- **Integration:** FastAPI upload/read, real PostgreSQL transactions/migrations, deduplication and recovery.
- **Contract:** swap database/model adapters without changing use cases.
- **Evaluation:** gold-set exact/normalized matches, evidence coverage, hallucination injection, validator false-positive/false-negative rates.

**DeepEval:** use `FaithfulnessMetric` with source passages as `retrieval_context` for grounded summaries. `AnswerRelevancyMetric` applies primarily to later question answering, not as a replacement for extraction correctness. For later RAG use contextual precision/recall and answer relevancy with appropriate reference answers.

**Proposed initial gates:** 100% structural validity for accepted records; no known invented critical KPIs/owners in gold cases; all injected critical contradictions detected; all tests passing. Calibrate thresholds against held-out labels.

## 11. Model observability and token accounting

Propagate `document_id`, `run_id`, and `trace_id` through parsing, generation, validation, persistence and chunking. Capture provider/model, prompt version, input/output references, provider-reported input/output/cached tokens where available, retries, latency, error category, evaluation results and cost estimate with pricing version. Redact sensitive content and set access/retention rules. Mark estimated token counts as estimates. Use Langfuse through a `Tracer` port to allow LangSmith replacement.

## 12. Vectorization as a subsequent independent stage

After metadata acceptance, split on heading/table boundaries while preserving page, customer, period and section. Start with token-aware 500–1,000 token chunks and modest overlap, then tune experimentally. Do not split KPI labels from values. Save chunks with `EMBEDDING_PENDING`. An embedding adapter later generates vectors and stores model/version in pgvector; retry failed embeddings separately. Retrieval must filter by customer/period and provide citations. Re-embed only when content or embedding model changes.

## 12A. Detailed PDF processing contract and runtime behavior

**Run before any LLM call:**

1. Resolve an authorized `PDFSource` by document ID; check content type/signature, size, encryption and page limits.
2. Iterate with `PDFExtractor.iter_pages` and collect page-local text, tables and extraction warnings. A parser failure is explicit; a blank page is marked for review/OCR fallback rather than silently accepted.
3. Normalize Unicode/whitespace conservatively, remove repeated noise only when safe, preserve original excerpts, table row/column structure, numeric values and units.
4. Identify headings/sections heuristically; split long pages into bounded fragments without breaking KPI label/value pairs or table rows.
5. Emit typed `CleanDocumentFragment` values containing document ID, page number(s), section hint, normalized text, structured tables, original source references, warning codes and preprocessing version. Persist/cache fragments keyed by document hash and parser/preprocessor versions when useful.
6. Send fragments to the metadata-extraction application, which controls LLM invocation and may expose optional read-only tools.

**Do not run during deterministic PDF processing:** LLM prompt construction, provider/model selection, LLM calls, Pydantic metadata generation, semantic judgment, conflict resolution or final business-rule validation. Those belong to subsequent layers.

**Tool-calling boundary:** expose `get_pdf_pages`, `read_pdf_page`, `extract_pdf_tables`, and `search_pdf_content` only through typed tool definitions and a bounded executor. The LLM provides a document ID and constrained parameters; the executor verifies the current user's document access and allowed ranges, then calls the existing extractor or processed-fragment repository. Limit result size, calls per run, time, and repeated requests. Treat tool outputs as untrusted document content, not instructions. Record call inputs/outputs and provenance with redaction. Never accept arbitrary file paths or shell commands from model output.

**Routing policy:** choose model/provider based on configuration, structured-output support, context budget, data policy, cost and availability. Optional retries/fallback must not silently change compliance constraints. Model decisions are made by application code, not by the model.

**Failure policy:** corrupted/encrypted/oversized PDFs fail clearly; extraction gaps and unresolved evidence trigger warnings or `REVIEW_REQUIRED`; no invented fields. Preserve the raw document and extracted evidence for audit. OCR remains out of scope for the initial PoC.

**Test boundaries:** parser tests with real PDFs and tables; preprocessing unit tests for Unicode, whitespace, units, multi-page evidence, tables, fragmentation, blank pages and determinism; tool tests for authorization, page limits, bounded results, repeated calls and injection attempts; LLM adapter tests with fake clients; end-to-end golden QBR cases and evaluation gates.

## 13. Implementation order and definition of done

| Priority | Deliverable |
|---|---|
| P0 | Docker PostgreSQL, Alembic, typed contracts, incremental PDF parser, deterministic PDF preprocessing, model adapter, validation, persistence, API, unit/integration tests |
| P0.5 | Optional PDFTools gateway, controlled LLM tool calling, ModelRouter, contract/security tests |
| P1 | Evidence/conflict reporting, golden JSON, DeepEval regression tests, Langfuse traces/tokens |
| P2 | Versioned chunks, embedding adapter, pgvector, retrieval/RAG evaluation |

**Definition of done:** `docker compose up`; apply migrations; POST each of six PDFs; GET persisted metadata/evidence; verify explicit conflicts/missing fields; run `pytest` and documented DeepEval golden tests; inspect token/trace records; prepare chunks without embeddings. Include `.env.example`, README commands and assumptions.

---

**Source basis:** Soulmynd.ai Forward Deployed AI Engineer home assignment, six supplied QBR reports, and accompanying technical proposal. Library choices, architecture and release gates are proposed design decisions, not requirements mandated by the assignment.