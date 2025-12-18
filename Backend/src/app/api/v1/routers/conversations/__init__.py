"""
Routers de conversaciones.

Este módulo contiene routers para gestión de conversaciones:
- chat_router: Endpoints de chat con streaming
- downloads_router: Descargas de mensajes de chat (PDF, DOCX, ODT)
- messages_router: Recuperación de mensajes y borradores
- drafts_router: Versionado y descargas de borradores
"""

from .chat_router import router as chat_router
from .downloads_router import router as downloads_router
from .drafts_router import router as drafts_router
from .messages_router import router as messages_router

__all__ = ["chat_router", "downloads_router", "messages_router", "drafts_router"]
