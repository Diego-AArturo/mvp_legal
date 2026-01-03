"""
Router de Normografía para gestión del grafo de leyes.

Proporciona endpoints para consultar y gestionar el grafo de normografía legal
almacenado en Neo4j, incluyendo nodos y relaciones entre leyes, decretos y artículos.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from neo4j import AsyncDriver
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.clients.neo4j_client import get_neo4j_driver
from app.clients.sql_pgvector_client import session_scope
from app.config.databases import get_neo4j_settings
from app.config.settings import get_settings
from app.extensions import get_logger
from app.schemas.normography_api_schemas import LawGraphResponse, LawUpdateRequest, LawUpdateResponse
from app.services.embeddings.embedding_service import get_embedding_service
from app.services.normography.normography_admin_service import NormographyAdminService
from app.services.retrieval.neo4j_retriever_service import Neo4jRetrieverService

logger = get_logger(__name__)
router = APIRouter(prefix="/normography", tags=["normography"])


@router.get("/laws/graph", response_model=LawGraphResponse)
async def get_law_graph(
    limit: int = 500,
    offset: int = 0,
    edge_limit: int = 5000,
    edge_offset: int = 0,
    include_disabled: bool = True,
) -> LawGraphResponse:
    """
    Obtiene el grafo completo de leyes con nodos y relaciones.

    Argumentos:
        limit: Límite de nodos a retornar
        offset: Offset para paginación de nodos
        edge_limit: Límite de relaciones a retornar
        edge_offset: Offset para paginación de relaciones
        include_disabled: Incluir nodos deshabilitados

    Retorna:
        LawGraphResponse con nodos, relaciones y metadata

    Lanza:
        HTTPException: Si hay un error al obtener el grafo
    """
    try:
        logger.info(f"Obteniendo grafo de leyes con limit={limit}, offset={offset}, " f"edge_limit={edge_limit}, include_disabled={include_disabled}")

        nodes = NormographyAdminService.list_law_nodes(
            limit=limit,
            offset=offset,
            include_disabled=include_disabled,
        )
        logger.info(f"Obtenidos {len(nodes)} nodos")

        # Limpiar nombres de caracteres especiales
        nodes = _clean_node_names(nodes)

        node_ids = {str(row["node_uid"]) for row in nodes}
        logger.info(f"IDs de nodos: {len(node_ids)}")

        edges_raw = NormographyAdminService.list_law_edges(
            limit=edge_limit,
            offset=edge_offset,
            include_disabled=include_disabled,
        )
        logger.info(f"Obtenidas {len(edges_raw)} relaciones sin filtrar")

        edges = [edge for edge in edges_raw if str(edge["start_node"]) in node_ids and str(edge["end_node"]) in node_ids]
        logger.info(f"Filtradas a {len(edges)} relaciones")

        meta = {
            "count_nodes": len(nodes),
            "count_edges": len(edges),
            "include_disabled": include_disabled,
            "limits": {"nodes": limit, "edges": edge_limit},
        }
        return LawGraphResponse(nodes=nodes, edges=edges, meta=meta)
    except Exception as e:
        logger.error(f"Error al obtener grafo de leyes: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error al obtener grafo de leyes: {str(e)}")


def _clean_node_names(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Limpia caracteres especiales de los nombres de nodos.

    Remueve caracteres especiales pero mantiene espacios, letras, números y algunos
    símbolos básicos para preservar la legibilidad del texto legal.

    Argumentos:
        nodes: Lista de nodos a limpiar

    Retorna:
        Lista de nodos con nombres limpios
    """
    import re

    def clean_text(text: Optional[str]) -> Optional[str]:
        if not text:
            return text

        # Remover caracteres especiales pero mantener espacios, letras, números y algunos símbolos básicos
        cleaned = re.sub(r"[^\w\s\-\(\)\.,°/ñÑáéíóúÁÉÍÓÚüÜ]", "", text, flags=re.UNICODE)

        # Limpiar espacios múltiples
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

    cleaned_nodes = []
    for node in nodes:
        cleaned_node = node.copy()

        # Limpiar campos de texto principales
        text_fields = [
            "nombre",
            "texto",
            "numero_original",
            "tipo",
            "source",
            "tematica",
            "resumen_tematica",
        ]

        for field in text_fields:
            if field in cleaned_node and cleaned_node[field]:
                cleaned_node[field] = clean_text(cleaned_node[field])

        cleaned_nodes.append(cleaned_node)

    return cleaned_nodes


@router.patch("/laws/{node_uid}", response_model=LawUpdateResponse)
async def update_law(
    node_uid: str,
    update_request: LawUpdateRequest,
    neo4j_driver: AsyncDriver = Depends(get_neo4j_driver),
) -> LawUpdateResponse:
    """
    Actualiza una ley existente en Neo4j y PostgreSQL.

    Este endpoint:
    1. Obtiene el element_id de PostgreSQL usando node_uid
    2. Actualiza el nodo en Neo4j usando element_id
    3. Actualiza el nodo en PostgreSQL usando node_uid
    4. Retorna el resultado de la operación

    Argumentos:
        node_uid: UUID del nodo en PostgreSQL
        update_request: Datos a actualizar (campos opcionales)
        neo4j_driver: Driver de Neo4j (inyectado)

    Retorna:
        LawUpdateResponse con el resultado de la actualización

    Lanza:
        HTTPException: Si el nodo no existe o hay un error en la actualización
    """
    try:
        # Validar que el node_uid es un UUID válido
        try:
            uuid_obj = UUID(node_uid)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"node_uid inválido: {node_uid}")

        # 1. Obtener el element_id desde PostgreSQL
        logger.info(f"Obteniendo element_id para node_uid: {node_uid}")
        with session_scope() as session:
            result = session.execute(
                text("SELECT element_id, label FROM normografia.node WHERE node_uid = :node_uid"),
                {"node_uid": str(uuid_obj)},
            ).fetchone()

            if not result:
                raise HTTPException(status_code=404, detail=f"Ley con node_uid {node_uid} no encontrada")

            element_id = result[0]
            label = result[1]

            if not element_id:
                raise HTTPException(
                    status_code=400,
                    detail=f"El nodo {node_uid} no tiene element_id asociado. No se puede actualizar en Neo4j.",
                )

        logger.info(f"Element_id encontrado: {element_id}")

        # 2. Preparar datos para actualización
        updates_dict: Dict[str, Any] = {}
        if update_request.nombre is not None:
            updates_dict["nombre"] = update_request.nombre
        if update_request.texto is not None:
            updates_dict["texto"] = update_request.texto
        if update_request.numero_original is not None:
            updates_dict["numero_original"] = update_request.numero_original
        if update_request.tipo is not None:
            updates_dict["tipo"] = update_request.tipo
        if update_request.source is not None:
            updates_dict["source"] = update_request.source
        if update_request.tematica is not None:
            updates_dict["tematica"] = update_request.tematica
        if update_request.resumen_tematica is not None:
            updates_dict["resumen_tematica"] = update_request.resumen_tematica
        if update_request.enabled is not None:
            updates_dict["enabled"] = update_request.enabled
        if update_request.properties is not None:
            updates_dict["properties"] = update_request.properties

        if not updates_dict:
            return LawUpdateResponse(
                success=True,
                node_uid=node_uid,
                element_id=element_id,
                message="No hay campos para actualizar",
            )

        # 3. Actualizar en Neo4j
        logger.info(f"Actualizando nodo en Neo4j con element_id: {element_id}")
        try:
            neo4j_settings = get_neo4j_settings()
            async with neo4j_driver.session(database=neo4j_settings.neo4j_database) as session:
                # Construir query de actualización
                set_parts = []
                params: Dict[str, Any] = {"element_id": element_id}

                for key, value in updates_dict.items():
                    if key == "properties":
                        # Las propiedades se actualizan como un objeto completo
                        set_parts.append(f"n.{key} = ${key}")
                        params[key] = value
                    elif key == "enabled":
                        # Actualizar enabled como propiedad del nodo en Neo4j
                        # Esto permitirá que el GraphScraperWorker lo lea correctamente
                        set_parts.append(f"n.enabled = ${key}")
                        params[key] = value
                    else:
                        set_parts.append(f"n.{key} = ${key}")
                        params[key] = value

                if set_parts:
                    update_query = f"""
                    MATCH (n)
                    WHERE elementId(n) = $element_id
                    SET {', '.join(set_parts)}
                    RETURN elementId(n) as element_id, n.nombre as nombre
                    """
                    result = await session.run(update_query, params)
                    record = await result.single()
                    if not record:
                        raise HTTPException(status_code=404, detail=f"Nodo no encontrado en Neo4j con element_id: {element_id}")

                    logger.info(f"Nodo actualizado en Neo4j: {record['nombre']}")

        except Exception as e:
            logger.error(f"Error actualizando en Neo4j: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error actualizando en Neo4j: {str(e)}")

        # 4. Actualizar en PostgreSQL
        logger.info(f"Actualizando nodo en PostgreSQL con node_uid: {node_uid}")
        try:
            NormographyAdminService.update_node(str(uuid_obj), updates_dict)
            logger.info("Nodo actualizado en PostgreSQL exitosamente")
        except Exception as e:
            logger.error(f"Error actualizando en PostgreSQL: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error actualizando en PostgreSQL: {str(e)}")

        return LawUpdateResponse(
            success=True,
            node_uid=node_uid,
            element_id=element_id,
            message="Ley actualizada exitosamente en Neo4j y PostgreSQL",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error inesperado actualizando ley: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error inesperado: {str(e)}")


