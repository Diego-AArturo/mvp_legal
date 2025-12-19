"""
Routers de acceso y recuperacion de datos.

Este modulo contiene routers para operaciones de datos:
- RAG (Retrieval-Augmented Generation)
"""

from .rag_router import router as rag_router

__all__ = [
    "rag_router",
]
