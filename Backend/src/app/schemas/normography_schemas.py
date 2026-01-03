# -*- coding: utf-8 -*-
"""
Esquemas Pydantic para el servicio de normografía.

Estos esquemas definen las estructuras de datos para el procesamiento
de documentos normativos y su carga/actualización en Neo4j.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BulkLoadRequest(BaseModel):
    """Request de carga masiva de normografía."""

    file_path: str = Field(..., description="Ruta al archivo Markdown a procesar")
    skip_normalization: bool = Field(
        default=False,
        description="Si es True, omite la normalización del archivo Markdown (útil cuando el archivo ya tiene la jerarquía correcta)",
    )


class DocumentProcessRequest(BaseModel):
    """Request de procesamiento de documento normativo."""

    file_path: str = Field(..., description="Ruta al archivo PDF/DOCX/DOC a procesar")
    section_path: Optional[List[str]] = Field(
        default=None, description="Ruta jerárquica de la sección a extraer"
    )


class DocumentProcessResponse(BaseModel):
    """Response de procesamiento de documento normativo."""

    success: bool = Field(description="Indica si el procesamiento fue exitoso")
    section_content: Optional[str] = Field(
        default=None, description="Contenido de la sección extraída en Markdown"
    )
    processing_stats: Optional[Dict[str, Any]] = Field(
        default=None, description="Estadísticas del procesamiento"
    )
    error_message: Optional[str] = Field(
        default=None, description="Mensaje de error si hubo falla"
    )


class NormographyUpdateRequest(BaseModel):
    """Request de actualización de normografía."""

    category: str = Field(..., description="Categoría de la normografía")
    law_name: str = Field(..., description="Nombre de la ley")
    path: List[str] = Field(..., description="Ruta jerárquica de la sección")
    markdown_content: str = Field(..., description="Contenido Markdown de la sección")


class NormographyUpdateResponse(BaseModel):
    """Response de actualización de normografía."""

    success: bool = Field(description="Indica si la actualización fue exitosa")
    nodes_created: int = Field(default=0, description="Número de nodos creados")
    relations_created: int = Field(default=0, description="Número de relaciones creadas")
    error_message: Optional[str] = Field(default=None, description="Mensaje de error si hubo falla")


class BulkLoadResponse(BaseModel):
    """Response de carga masiva de normografía."""

    success: bool = Field(description="Indica si la carga fue exitosa")
    nodes_processed: int = Field(default=0, description="Número de nodos procesados")
    relations_created: int = Field(default=0, description="Número de relaciones creadas")
    processing_time: float = Field(default=0.0, description="Tiempo de procesamiento en segundos")
    error_message: Optional[str] = Field(default=None, description="Mensaje de error si hubo falla")
    normalized_file_path: Optional[str] = Field(default=None, description="Ruta del archivo normalizado generado")


__all__ = [
    "BulkLoadRequest",
    "BulkLoadResponse",
    "DocumentProcessRequest",
    "DocumentProcessResponse",
    "NormographyUpdateRequest",
    "NormographyUpdateResponse",
]
