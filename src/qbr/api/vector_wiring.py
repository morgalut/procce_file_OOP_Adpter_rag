"""Backward-compatible composition facade; prefer qbr.bootstrap.vector_wiring."""
from qbr.bootstrap.vector_wiring import build_vector_services, build_vectors_router
__all__ = ['build_vector_services', 'build_vectors_router']
