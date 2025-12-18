"""
Routers de acceso y recuperación de datos.

Este módulo contiene routers para operaciones de datos:
- RAG (Retrieval-Augmented Generation)
- Embeddings vectoriales
- Normografía (grafo legal)
"""

from .embedding_router import router as embedding_router
from .normography_router import router as normography_router
from .rag_router import router as rag_router

__all__ = [
    "embedding_router",
    "normography_router",
    "rag_router",
]
