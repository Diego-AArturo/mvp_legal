"""
Routers de funcionalidad principal del sistema.

Este módulo contiene routers para operaciones principales del sistema:
- Health checks y estado del sistema
- Autenticación y autorización
- Logs de auditoría y agentes
"""

from .agent_logs_router import router as agent_logs_router
from .audit_logs_router import router as audit_logs_router
from .auth_router import router as auth_router
from .health_router import router as health_router

__all__ = ["health_router", "auth_router", "audit_logs_router", "agent_logs_router"]
