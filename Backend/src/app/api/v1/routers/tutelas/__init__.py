"""
Routers de tutelas.

Este módulo contiene routers para gestión de tutelas:
- tutelas_router: Operaciones CRUD para tutelas

Nota: Los endpoints de generación fueron movidos a chat_router.py
"""

from .tutelas_router import router as tutelas_router

__all__ = ["tutelas_router"]
