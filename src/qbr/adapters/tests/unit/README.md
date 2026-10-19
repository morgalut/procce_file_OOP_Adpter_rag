# QBR V18 — Chunker port + deterministic adapter

Only `ports/chunker.py` among the eight supplied ports was empty. The other seven contain usable interface contracts and are **not replaced** by this package.

Merge `src/qbr/` into your existing tree, and copy `tests/unit/test_section_chunker.py` into your tests. The adapter is intentionally not wired into vector persistence; this requires inspecting your existing `application/prepare_chunks.py` and storage schema.

```bash
PYTHONPATH=src python -m pytest tests/unit/test_section_chunker.py -vv
```

**Behavior:** per-source section/page boundaries, stable chunk indices, source IDs, configurable overlap, and configurable token-count callback. The default regex counter is approximate, not model-accurate. Text whitespace is normalized when chunked; preserve original excerpts separately for audit. Very long unbroken tokens are rejected. This adapter does **not** guarantee that table rows or KPI/value pairs remain intact when a single source exceeds the budget; a future table-aware splitter should receive typed tables and source spans.
