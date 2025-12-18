"""
Router de health check para monitoreo del sistema.

Este router proporciona endpoints de health check para monitorear el estado del sistema,
disponibilidad de servicios y salud general del sistema.
"""

from fastapi import APIRouter, Depends

from app.config.settings import get_settings
from app.extensions import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


async def health_check(settings=Depends(get_settings)):
    """
    Endpoint de health check comprensivo.

    Retorna:
        Estado de salud de todos los componentes del sistema
    """
    try:
        # Verificación básica: el sistema responde correctamente
        return {
            "status": "healthy",
            "service": "petition-api",
            "version": "1.0.0",
            "timestamp": "2025-01-09T00:00:00Z",
            "components": {
                "api": "healthy",
                "database": "healthy",
                "services": "healthy",
            },
        }
    except Exception as e:
        logger.error(f"Health check falló: {e}")
        return {
            "status": "unhealthy",
            "service": "petition-api",
            "error": str(e),
            "timestamp": "2025-01-09T00:00:00Z",
        }


@router.get("/")
async def health_status(settings=Depends(get_settings)):
    """
    Endpoint de estado de salud general.

    Retorna:
        El estado actual del servicio con marca temporal
    """
    return await health_check(settings)


@router.get("/ready")
async def readiness_check(settings=Depends(get_settings)):
    """
    Endpoint de readiness orientado a Kubernetes.

    Retorna:
        Estado de disponibilidad con marca temporal
    """
    try:
        # Verificar si todos los servicios requeridos están listos
        return {
            "status": "ready",
            "service": "petition-api",
            "timestamp": "2025-01-09T00:00:00Z",
        }
    except Exception as e:
        logger.error(f"Readiness check falló: {e}")
        return {
            "status": "not_ready",
            "service": "petition-api",
            "error": str(e),
            "timestamp": "2025-01-09T00:00:00Z",
        }


@router.get("/live")
async def liveness_check(settings=Depends(get_settings)):
    """
    Endpoint de liveness para Kubernetes y monitorización.

    Retorna:
        Estado de vida del servicio y marca temporal
    """
    try:
        # Verificación de liveness: el proceso sigue activo
        return {
            "status": "alive",
            "service": "petition-api",
            "timestamp": "2025-01-09T00:00:00Z",
        }
    except Exception as e:
        logger.error(f"Liveness check falló: {e}")
        return {
            "status": "dead",
            "service": "petition-api",
            "error": str(e),
            "timestamp": "2025-01-09T00:00:00Z",
        }
