"""HTTP contract tests for QBR document endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI

from qbr.api.qbr_documents import create_qbr_documents_router
from qbr.application.document_service import DocumentService
from qbr.adapters.storage.local_pdf_storage import LocalPDFStorage
from qbr.ports.extraction_store import ExtractionRecord


class InMemoryStore:
    def __init__(self):
        self.records = {}

    async def save(self, record):
        self.records[record.document_id] = record

    async def get(self, document_id):
        return self.records.get(document_id)


@dataclass
class FakeResult:
    document_id: str


class FakePipeline:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    async def execute(self, source):
        self.calls.append(source)

        assert source.path.read_bytes().startswith(b"%PDF-")

        if self.fail:
            raise ValueError("Malformed PDF")

        return FakeResult(source.document_id)


class FakePersistence:
    def __init__(self, store):
        self.store = store

    async def execute(
        self,
        result,
        *,
        filename,
        sha256,
        storage_uri=None,
    ):
        record = ExtractionRecord(
            document_id=result.document_id,
            filename=filename,
            sha256=sha256,
            storage_uri=storage_uri,
            status="review_required",
            incomplete=True,
            page_count=1,
            fragment_count=2,
            metadata={"customer_name": "Northwind"},
            validation={"status": "failed"},
            fragments=(
                {"failure_code": "missing_evidence"},
            ),
            evidence=(
                {
                    "source_id": "p1-t0-r1-c1",
                    "source_excerpt": "875",
                },
            ),
        )

        await self.store.save(record)
        return record


def build(tmp_path, fail=False, max_pdf_bytes=1000):
    store = InMemoryStore()
    pipeline = FakePipeline(fail=fail)

    service = DocumentService(
        pipeline=pipeline,
        persistence=FakePersistence(store),
        store=store,
        storage=LocalPDFStorage(tmp_path / "storage"),
        max_pdf_bytes=max_pdf_bytes,
    )

    app = FastAPI()
    app.include_router(
        create_qbr_documents_router(service=service)
    )

    return app, store, pipeline


@pytest.mark.asyncio
async def test_upload_persist_read(tmp_path):
    app, store, pipeline = build(tmp_path)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/v1/qbr/documents",
            files={
                "file": (
                    "northwind.pdf",
                    b"%PDF-1.7\ncontent",
                    "application/pdf",
                )
            },
        )

        assert response.status_code == 201, response.text

        doc_id = response.json()["document_id"]
        UUID(doc_id)

        assert len(pipeline.calls) == 1
        assert Path(store.records[doc_id].storage_uri).exists()

        document = await client.get(
            f"/api/v1/qbr/documents/{doc_id}"
        )

        assert document.status_code == 200
        assert (
            document.json()["metadata"]["customer_name"]
            == "Northwind"
        )

        validation = await client.get(
            f"/api/v1/qbr/documents/{doc_id}/validation"
        )

        assert validation.status_code == 200
        assert (
            validation.json()["evidence"][0]["source_id"]
            == "p1-t0-r1-c1"
        )


@pytest.mark.asyncio
async def test_invalid_pdf_and_size_limit(tmp_path):
    app, store, pipeline = build(
        tmp_path,
        max_pdf_bytes=16,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:

        invalid = await client.post(
            "/api/v1/qbr/documents",
            files={
                "file": (
                    "a.pdf",
                    b"not pdf",
                    "application/pdf",
                )
            },
        )

        assert invalid.status_code == 422

        oversized = await client.post(
            "/api/v1/qbr/documents",
            files={
                "file": (
                    "a.pdf",
                    b"%PDF-" + b"a" * 30,
                    "application/pdf",
                )
            },
        )

        assert oversized.status_code == 413

        unsupported = await client.post(
            "/api/v1/qbr/documents",
            files={
                "file": (
                    "a.txt",
                    b"%PDF-1.7",
                    "text/plain",
                )
            },
        )

        assert unsupported.status_code == 415

    assert not store.records
    assert not pipeline.calls
    assert not list((tmp_path / "storage").iterdir())


@pytest.mark.asyncio
async def test_failed_pipeline_removes_file(tmp_path):
    app, store, _ = build(
        tmp_path,
        fail=True,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:

        with pytest.raises(
            ValueError,
            match="Malformed PDF",
        ):
            await client.post(
                "/api/v1/qbr/documents",
                files={
                    "file": (
                        "a.pdf",
                        b"%PDF-1.7",
                        "application/pdf",
                    )
                },
            )

    assert not store.records
    assert not list((tmp_path / "storage").iterdir())


@pytest.mark.asyncio
async def test_unknown_id(tmp_path):
    app, _, _ = build(tmp_path)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:

        invalid = await client.get(
            "/api/v1/qbr/documents/not-a-uuid"
        )

        assert invalid.status_code == 422

        missing = await client.get(
            f"/api/v1/qbr/documents/{uuid4()}"
        )

        assert missing.status_code == 404