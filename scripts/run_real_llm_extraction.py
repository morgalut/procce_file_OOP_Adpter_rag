"""Run the existing QBR pipeline against a real PDF with a real OpenAI LLM.

From the project root:
    pip install python-dotenv
    # Use the existing LLM_*, PDF_* settings in project-root .env
    python scripts/run_real_llm_extraction.py

This is an opt-in integration runner, not a pytest unit test. It makes paid
external API requests and sends the extracted PDF content to the provider.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_project_env() -> None:
    """Load project-root .env without overriding existing process variables."""
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("Install python-dotenv: pip install python-dotenv") from exc
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


DEFAULT_PDF = Path("tests/fixtures/01_Northwind_Technologies_Q2_2026.pdf")
DEFAULT_OUTPUT = Path("tests/output/northwind_real_llm_metadata.json")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract evidence-grounded QBR metadata with a real LLM")
    parser.add_argument("--pdf", type=Path, default=Path(os.getenv("QBR_PDF_PATH", str(DEFAULT_PDF))), help="Existing local PDF file")
    parser.add_argument("--output", type=Path, default=Path(os.getenv("QBR_OUTPUT_PATH", str(DEFAULT_OUTPUT))), help="JSON output path")
    parser.add_argument("--document-id", default=os.getenv("QBR_DOCUMENT_ID", "northwind-q2-2026"))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "openai"))
    parser.add_argument("--timeout", type=float, default=_env_float("LLM_TIMEOUT_SECONDS", 60.0))
    parser.add_argument("--max-pages", type=int, default=_env_int("PDF_MAX_PAGES", 100))
    parser.add_argument("--max-size-mb", type=int, default=_env_int("PDF_MAX_SIZE_MB", 20))
    parser.add_argument("--max-fragments", type=int, default=_env_int("QBR_MAX_FRAGMENTS", 200))
    parser.add_argument("--max-input-chars", type=int, default=_env_int("QBR_MAX_INPUT_CHARS", 18000))
    parser.add_argument("--retries", type=int, default=_env_int("LLM_MAX_RETRIES", 2))
    args = parser.parse_args()
    if args.provider.lower() != "openai":
        parser.error("This runner currently supports only LLM_PROVIDER=openai")
    if args.timeout <= 0 or args.max_pages < 1 or args.max_size_mb < 1 or args.max_fragments < 1:
        parser.error("timeout, max-pages, max-size-mb, and max-fragments must be positive")
    if not args.pdf.is_absolute():
        args.pdf = PROJECT_ROOT / args.pdf
    if not args.output.is_absolute():
        args.output = PROJECT_ROOT / args.output
    return args


def _jsonable(value: Any) -> Any:
    """Convert Pydantic models and ordinary dataclasses to JSON-safe values."""
    from dataclasses import asdict, is_dataclass
    from enum import Enum

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


async def _run(args: argparse.Namespace) -> int:
    if not args.pdf.is_file():
        print(f"ERROR: PDF does not exist: {args.pdf}", file=sys.stderr)
        return 2
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        print("ERROR: Set LLM_API_KEY in project-root .env or your environment.", file=sys.stderr)
        return 2

    try:
        import instructor
        from openai import AsyncOpenAI
        from qbr.adapters.llm.instructor_adapter import InstructorMetadataAdapter, LLMExtractionSettings
        from qbr.adapters.pdf.pdf_processing_adapter import PDFProcessingAdapter
        from qbr.adapters.pdf.pymupdf_adapter import PyMuPDFAdapter, PyMuPDFLimits
        from qbr.application.extract_metadata import ExtractMetadata, ExtractionLimits
        from qbr.domain.validation_rules import ValidationEngine
        from qbr.ports.pdf_extractor import PDFSource
    except ImportError as exc:
        print(f"ERROR: Missing dependency or project import: {exc}", file=sys.stderr)
        return 2

    client = AsyncOpenAI(api_key=api_key, timeout=args.timeout, max_retries=args.retries)
    try:
        patched = instructor.from_openai(client)
        generator = InstructorMetadataAdapter(
            completions=patched.chat.completions,
            settings=LLMExtractionSettings(
                model=args.model,
                max_retries=args.retries,
                max_input_chars=args.max_input_chars,
            ),
        )
        pipeline = ExtractMetadata(
            extractor=PyMuPDFAdapter(
                limits=PyMuPDFLimits(
                    max_file_size_mb=args.max_size_mb,
                    max_pages=args.max_pages,
                )
            ),
            processor=PDFProcessingAdapter(),
            generator=generator,
            validator=ValidationEngine(),
            limits=ExtractionLimits(
                max_pages=args.max_pages,
                max_fragments=args.max_fragments,
            ),
        )
        result = await pipeline.execute(
            PDFSource(document_id=args.document_id, path=args.pdf.resolve())
        )
    except Exception as exc:
        # The orchestrator retains stage context and chains the underlying cause.
        print(f"ERROR: Pipeline failed: {exc}", file=sys.stderr)
        if exc.__cause__ is not None:
            print(f"CAUSE: {type(exc.__cause__).__name__}: {exc.__cause__}", file=sys.stderr)
        return 1
    finally:
        await client.close()

    payload = {
        "document_id": result.document_id,
        "source_pdf": str(args.pdf),
        "extraction_mode": "real_llm",
        "provider": args.provider.lower(),
        "model": args.model,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "page_count": result.page_count,
        "fragment_count": result.fragment_count,
        "metadata": _jsonable(result.metadata),
        "validation": _jsonable(result.validation),
        "conflicts": _jsonable(result.accumulation.conflicts),
        "fragment_outcomes": _jsonable(result.fragments),
        "notes": [
            "LLM-generated metadata requires review of evidence and validation issues.",
            "This runner does not perform persistence, tool calling, or token usage accounting.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Output: {args.output.resolve()}")
    print(f"Pages: {result.page_count}; fragments: {result.fragment_count}")
    print(f"Validation status: {result.validation.status}")
    print(f"Customer: {result.metadata.customer_name}")
    print(f"Period: {result.metadata.reporting_period}")
    print(f"KPIs: {len(result.metadata.kpis)}; next steps: {len(result.metadata.next_steps)}")
    return 0


def main() -> int:
    try:
        _load_project_env()
        args = _arguments()
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
