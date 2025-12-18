"""
Módulo de configuración de la aplicación.

Centraliza la exportación de todas las clases de configuración organizadas
por dominio: base de datos, LLM, modelos (embeddings y generación) y almacenamiento.
"""

from .databases import (
    DatabaseSettings,
    Neo4jSettings,
    PostgreSQLSettings,
    get_database_settings,
    get_neo4j_settings,
    get_postgresql_settings,
)
from .llm import (
    LLMProviderSettings,
    LLMSettings,
    OllamaSettings,
    get_llm_provider_settings,
    get_llm_settings,
    get_ollama_settings,
)
from .models import (
    EmbeddingSettings,
    GenerationSettings,
    ModelsSettings,
    get_embedding_settings,
    get_generation_settings,
    get_models_settings,
)
from .settings import Settings
from .storage import (
    MinIOSettings,
    StorageSettings,
    get_minio_settings,
    get_storage_settings,
)

__all__ = [
    "Settings",
    # Configuración de bases de datos
    "DatabaseSettings",
    "Neo4jSettings",
    "PostgreSQLSettings",
    "get_database_settings",
    "get_neo4j_settings",
    "get_postgresql_settings",
    # Configuración de LLM
    "LLMSettings",
    "LLMProviderSettings",
    "OllamaSettings",
    "get_llm_settings",
    "get_llm_provider_settings",
    "get_ollama_settings",
    # Configuración de modelos (embeddings y generación)
    "ModelsSettings",
    "EmbeddingSettings",
    "GenerationSettings",
    "get_models_settings",
    "get_embedding_settings",
    "get_generation_settings",
    # Configuración de almacenamiento
    "StorageSettings",
    "MinIOSettings",
    "get_storage_settings",
    "get_minio_settings",
]
