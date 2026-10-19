"""Additive RAG endpoint: keeps /search as retrieval-only."""
from __future__ import annotations
import os
from typing import Any
from fastapi import APIRouter, Depends, HTTPException
from openai import AsyncOpenAI, OpenAIError
from qbr.api.dependencies import get_vector_search
from qbr.api.contracts.search_answer import AnswerRequest, AnswerResponse
from qbr.application.answer_search import AnswerSearch

router = APIRouter(tags=['QBR search answers'])

@router.post('/api/v1/qbr/search/answer', response_model=AnswerResponse)
async def search_answer(
    payload: AnswerRequest,
    search: Any = Depends(get_vector_search),
) -> AnswerResponse:
    api_key = os.getenv('OPENAI_API_KEY') or os.getenv('LLM_API_KEY')
    if not api_key:
        raise HTTPException(status_code=503, detail='LLM API key is not configured')
    client = AsyncOpenAI(api_key=api_key, timeout=60.0, max_retries=2)
    try:
        service = AnswerSearch(search, client, model=os.getenv('QBR_ANSWER_MODEL') or os.getenv('LLM_MODEL') or 'gpt-4o-mini')
        return await service.execute(
            payload.query, top_k=payload.top_k,
            document_id=str(payload.document_id) if payload.document_id else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail='Answer provider failed') from exc
    finally:
        await client.close()
