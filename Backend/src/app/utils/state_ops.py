"""
Utilidades para manipulación de estado en LangGraph.

Proporciona funciones auxiliares para convertir diferentes tipos de objetos
de estado (dicts, modelos Pydantic) a diccionarios estándar de Python.
"""

from typing import Any, Dict


def as_dict(state: Any) -> Dict[str, Any]:
    """
    Convierte un objeto de estado a diccionario.

    Soporta múltiples formatos de entrada:
    - Diccionarios Python nativos (retorna sin cambios)
    - Modelos Pydantic v2 (usa model_dump())
    - Modelos Pydantic v1 (usa dict())
    - Objetos con atributos (usa __dict__)

    Argumentos:
        state: Objeto de estado a convertir. Puede ser dict, modelo Pydantic
               o cualquier objeto con atributos.

    Retorna:
        Diccionario con los datos del estado. Si state ya es un dict,
        se retorna sin modificaciones.
    """
    if isinstance(state, dict):
        return state
    try:
        # Pydantic v2
        return state.model_dump()  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        # Pydantic v1
        return state.dict()  # type: ignore[attr-defined]
    except Exception:
        pass
    return dict(getattr(state, "__dict__", {}) or {})
