"""
Modulo principal de routers de la API v1.

Organiza e incluye los routers disponibles agrupados por dominio:
- core: Funcionalidad basica del sistema (health)
- conversations: Conversaciones y chat (streaming)
- data: Datos y normografia (RAG)
- tutelas: Generacion de conversation_id para tutelas externas
"""

from fastapi import APIRouter

from app.extensions import get_logger

from .conversations import chat_router
from .core import health_router
from .data import rag_router
from .tutelas import tutelas_router

router = APIRouter()
logger = get_logger(__name__)

# Incluir todos los routers organizados por dominio
# Core: funcionalidad basica
router.include_router(health_router)

# Conversations: conversaciones y chat
router.include_router(chat_router)

# Data: datos y normografia
router.include_router(rag_router)

router.include_router(tutelas_router)

logger.info("Routers basicos cargados")
