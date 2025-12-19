"""
Servicios de Petition API - Organizados por dominio funcional.

Este módulo proporciona una interfaz centralizada a todos los servicios
de Petition API, organizados en grupos lógicos para mejor mantenibilidad
y descubribilidad.

Arquitectura:
- auth: Autenticación, autorización y gestión de usuarios
- documents: Procesamiento, conversión y extracción de documentos
- legal: Análisis legal, búsqueda y generación de argumentos
- normography: Gestión de normografía legal (Neo4j y PostgreSQL)
- tutelas: Procesamiento y enriquecimiento específico de tutelas
- embeddings: Generación de embeddings vectoriales
- ai: Servicios AI/ML (Ollama, salida estructurada, guardrails)
- postgres: Servicios de persistencia PostgreSQL
- retrieval: Recuperación de contexto desde PgVector y Neo4j
- export: Servicios de exportación de documentos
- ml: Pipelines de machine learning
- core: Servicios transversales (jobs, layers)
"""

# Re-export commonly used services for backward compatibility
from app.services.embeddings.embedding_service import (
    EmbeddingService,
    get_embedding_service,
)

__all__ = [
    "EmbeddingService",
    "get_embedding_service",
]
