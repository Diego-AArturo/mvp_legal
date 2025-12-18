"""
Servicios de generación de embeddings.

Exporta servicios para generar embeddings semánticos usando sentence-transformers
con gestión del ciclo de vida del modelo.
"""

from app.services.embeddings.embedding_service import (
    EmbeddingService,
    cleanup_embedding_service,
    get_embedding_service,
)

__all__ = [
    "EmbeddingService",
    "cleanup_embedding_service",
    "get_embedding_service",
]
