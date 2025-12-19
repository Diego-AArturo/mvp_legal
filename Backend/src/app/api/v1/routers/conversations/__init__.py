"""
Routers de conversaciones.

Este modulo contiene routers para gestion de conversaciones:
- chat_router: Endpoints de chat con streaming
"""

from .chat_router import router as chat_router

__all__ = ["chat_router"]
