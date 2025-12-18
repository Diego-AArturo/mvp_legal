"""
Módulo principal de routers de la API v1.

Organiza e incluye todos los routers de la aplicación agrupados por dominio:
- core: Funcionalidad básica del sistema (health, auth)
- tutelas: Gestión de tutelas (CRUD y generación de respuestas)
- conversations: Conversaciones y chat (mensajes, drafts, streaming)
- data: Datos y normografía (embeddings, RAG, normography)
"""

from fastapi import APIRouter

from app.extensions import get_logger

from .conversations import (
    chat_router,
    downloads_router,
    drafts_router,
    messages_router,
)
from .core import agent_logs_router, auth_router, audit_logs_router, health_router
from .data import embedding_router, normography_router, rag_router
from .tutelas import tutelas_router

router = APIRouter()
logger = get_logger(__name__)

# Incluir todos los routers organizados por dominio
# Core: funcionalidad básica
router.include_router(health_router)
router.include_router(auth_router, tags=["authentication"])
router.include_router(audit_logs_router)
router.include_router(agent_logs_router)

# Tutelas: gestión de tutelas y generación de respuestas
router.include_router(tutelas_router)
# Nota: Los endpoints de generación están en chat_router.py

# Conversations: conversaciones y chat
router.include_router(chat_router)
router.include_router(downloads_router)
router.include_router(messages_router)
router.include_router(drafts_router)

# Data: datos y normografía
router.include_router(rag_router)
router.include_router(embedding_router)
router.include_router(normography_router)

logger.info("Todos los routers cargados desde carpetas organizadas")
