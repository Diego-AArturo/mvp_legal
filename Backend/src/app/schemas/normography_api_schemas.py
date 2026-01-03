"""
Schemas de API de normografía.

Define modelos de datos para nodos y relaciones del grafo legal,
incluyendo detalles completos y respuestas estructuradas del grafo.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel


class LawNodeDetail(BaseModel):
    """
    Detalle de nodo de ley en el grafo de normografía.

    Representa un nodo individual del grafo legal con toda su información
    incluyendo texto, metadata, temática y propiedades.
    """

    node_uid: UUID
    element_id: Optional[str] = None
    label: str
    graph_id: Optional[str] = None
    nombre: Optional[str] = None
    texto: Optional[str] = None
    numero_original: Optional[str] = None
    tipo: Optional[str] = None
    source: Optional[str] = None
    processing_status: Optional[str] = None
    tematica: Optional[str] = None
    resumen_tematica: Optional[str] = None
    enabled: bool
    created_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    properties: Optional[Dict[str, Any]] = None


class LawEdgeDetail(BaseModel):
    """
    Detalle de relación (edge) en el grafo de normografía.

    Representa una relación entre nodos del grafo legal con tipo de relación
    y propiedades asociadas.
    """

    edge_id: int
    rel_type: str
    start_node: UUID
    end_node: UUID
    enabled: bool
    created_at: Optional[datetime] = None
    properties: Optional[Dict[str, Any]] = None


class LawGraphResponse(BaseModel):
    """
    Respuesta completa del grafo legal.

    Contiene todos los nodos y relaciones del grafo junto con metadata
    adicional para la respuesta de la API.
    """

    nodes: List[LawNodeDetail]
    edges: List[LawEdgeDetail]
    meta: Optional[Dict[str, Any]] = None


class LawUpdateRequest(BaseModel):
    """
    Request para actualizar una ley existente.
    
    Todos los campos son opcionales - solo se actualizarán los campos proporcionados.
    """
    nombre: Optional[str] = None
    texto: Optional[str] = None
    numero_original: Optional[str] = None
    tipo: Optional[str] = None
    source: Optional[str] = None
    tematica: Optional[str] = None
    resumen_tematica: Optional[str] = None
    enabled: Optional[bool] = None
    properties: Optional[Dict[str, Any]] = None


class LawUpdateResponse(BaseModel):
    """
    Respuesta de actualización de ley.
    """
    success: bool
    node_uid: Optional[str] = None
    element_id: Optional[str] = None
    message: Optional[str] = None
    error_message: Optional[str] = None


# Reconstruir modelos para asegurar que todas las referencias forward están resueltas
LawGraphResponse.model_rebuild()
