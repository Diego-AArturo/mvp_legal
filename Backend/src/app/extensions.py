"""
Módulo de extensiones y utilidades compartidas de la aplicación.

Proporciona configuración centralizada de logging y funciones de utilidad
para obtener loggers configurados en toda la aplicación.
"""

import logging
import logging.config
from typing import Any

# Configuración de logging para toda la aplicación
# Define formatters, handlers y niveles de log por módulo
LOGGING_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "detailed",
            "level": "DEBUG",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "app": {
            "level": "DEBUG",
            "handlers": ["console"],
            "propagate": False,
        },
        "app.agents": {
            "level": "DEBUG",
            "handlers": ["console"],
            "propagate": False,
        },
        "app.api": {
            "level": "DEBUG",
            "handlers": ["console"],
            "propagate": False,
        },
        "app.graphs": {
            "level": "DEBUG",
            "handlers": ["console"],
            "propagate": False,
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "DEBUG",
    },
}


def configure_logging() -> None:
    """
    Configura el sistema de logging de la aplicación.

    Aplica la configuración definida en LOGGING_CONFIG a todos los loggers
    de la aplicación. Debe llamarse al inicio de la aplicación.
    """
    logging.config.dictConfig(LOGGING_CONFIG)


def get_logger(name: str = "app") -> logging.Logger:
    """
    Obtiene un logger configurado para el módulo especificado.

    Argumentos:
        name: Nombre del logger (normalmente __name__ del módulo).
              Por defecto usa "app" para el logger raíz.

    Retorna:
        Logger configurado según LOGGING_CONFIG con formatters y handlers
        apropiados para el módulo.
    """
    return logging.getLogger(name)
