"""
Routers de funcionalidad principal del sistema.

Este modulo contiene routers para operaciones principales del sistema:
- Health checks y estado del sistema
"""

from .health_router import router as health_router

__all__ = ["health_router"]
