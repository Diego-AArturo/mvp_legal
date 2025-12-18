"""
Módulo principal de la aplicación Petition API.

Exporta las clases y funciones principales necesarias para configurar
y utilizar la aplicación desde otros módulos.
"""

from .config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
