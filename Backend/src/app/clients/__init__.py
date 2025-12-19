"""
Módulo de clientes para servicios externos.

Exporta todos los clientes y funciones de gestión del ciclo de vida para
servicios externos: Ollama, Neo4j y PostgreSQL.
"""

from .neo4j_client import get_neo4j_driver, get_neo4j_session, lifespan_neo4j
from .sql_pgvector_client import (
    get_database_session,
    get_db,
    lifespan_pgvector,
    session_scope,
)
from .ollama_client import OllamaClient

__all__ = [
    # Cliente Ollama
    "OllamaClient",
    # Cliente Neo4j
    "get_neo4j_driver",
    "get_neo4j_session",
    "lifespan_neo4j",
    # Cliente PostgreSQL
    "get_database_session",
    "get_db",
    "lifespan_pgvector",
    "session_scope",
]
