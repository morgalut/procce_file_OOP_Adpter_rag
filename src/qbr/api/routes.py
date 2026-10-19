"""Central route registry. Register every endpoint exactly once."""
from fastapi import APIRouter
from qbr.api.endpoints.documents import router as documents_router
from qbr.api.endpoints.health import router as health_router
from qbr.api.endpoints.vectors import create_qbr_vectors_runtime_router

def build_api_router() -> APIRouter:
    router = APIRouter()
    router.include_router(health_router)
    router.include_router(documents_router)
    router.include_router(create_qbr_vectors_runtime_router())
    return router
