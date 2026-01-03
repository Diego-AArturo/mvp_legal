"""
Servicio Administrativo de Normografía.

Proporciona operaciones de lectura/escritura en las tablas de normografía y
sincronización con Neo4j para gestión administrativa del grafo legal.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.clients.sql_pgvector_client import session_scope
from app.extensions import get_logger

logger = get_logger(__name__)


class NormographyAdminService:
    """
    Servicio para leer/escribir tablas de normografía y reflejar en Neo4j.

    Proporciona operaciones administrativas para gestionar nodos y relaciones
    del grafo de normografía tanto en PostgreSQL como en Neo4j.
    """

    @staticmethod
    def _serialize_jsonb(value: Any) -> Optional[str]:
        """
        Serializa un valor a JSON para campos jsonb de PostgreSQL.

        Argumentos:
            value: Valor a serializar (dict, list, string, None)

        Retorna:
            String JSON o None si el valor está vacío

        Lanza:
            ValueError: Si el valor no es serializable a JSON
        """
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            logger.error(
                "Valor JSON inválido para campo jsonb",
                extra={"error": str(exc), "value": value},
            )
            raise

    # --- LECTURAS ---
    @staticmethod
    def list_nodes(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Lista nodos de normografía con paginación.

        Argumentos:
            limit: Límite de resultados por página
            offset: Offset para paginación

        Retorna:
            Lista de diccionarios con datos de nodos
        """
        with session_scope() as session:
            rows = (
                session.execute(
                    text(
                        """
                    SELECT node_uid, element_id, label, graph_id, nombre, tipo, enabled, created_at, last_updated
                    FROM normografia.node
                    ORDER BY created_at NULLS LAST, node_uid
                    LIMIT :limit OFFSET :offset
                    """
                    ),
                    {"limit": limit, "offset": offset},
                )
                .mappings()
                .all()
            )
            return [dict(r) for r in rows]

    @staticmethod
    def list_edges(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Lista relaciones (edges) de normografía con paginación.

        Argumentos:
            limit: Límite de resultados por página
            offset: Offset para paginación

        Retorna:
            Lista de diccionarios con datos de relaciones
        """
        with session_scope() as session:
            rows = (
                session.execute(
                    text(
                        """
                    SELECT edge_id, rel_type::text AS rel_type, start_node, end_node, enabled, created_at
                    FROM normografia.edge
                    ORDER BY edge_id
                    LIMIT :limit OFFSET :offset
                    """
                    ),
                    {"limit": limit, "offset": offset},
                )
                .mappings()
                .all()
            )
            return [dict(r) for r in rows]

    @staticmethod
    def list_law_nodes(limit: int = 500, offset: int = 0, include_disabled: bool = True) -> List[Dict[str, Any]]:
        """
        Retorna nodos de normografía legal principales (excluye fragmentos y sub-elementos).
        Incluye solo tipos principales: Ley, Decreto, DecretoLey, Resolución, Sentencia, Circular, Categoria.

        Argumentos:
            limit: Límite de resultados por página
            offset: Offset para paginación
            include_disabled: Incluir nodos deshabilitados

        Retorna:
            Lista de diccionarios con datos de nodos de normografía legal
        """
        # Filtrar solo tipos principales de normografía legal (excluir fragmentos y sub-elementos)
        # Incluir solo: Ley, Decreto, DecretoLey, Resolución, Sentencia, Circular, Categoria
        conditions = [
            "label IN ('Ley', 'Decreto', 'DecretoLey', 'Resolución', 'Normografia', 'Sentencia', 'Circular', 'Categoria')"
        ]
        if not include_disabled:
            conditions.append("enabled = true")
        where_clause = " AND ".join(conditions)
        query = f"""
            SELECT node_uid, element_id, label, graph_id, nombre, texto, numero_original, tipo,
                   source, processing_status, tematica, resumen_tematica, enabled, created_at, last_updated, properties
            FROM normografia.node
            WHERE {where_clause}
            ORDER BY COALESCE(nombre, graph_id) NULLS LAST, created_at DESC NULLS LAST
            LIMIT :limit OFFSET :offset
        """
        with session_scope() as session:
            rows = session.execute(text(query), {"limit": limit, "offset": offset}).mappings().all()
            return [dict(r) for r in rows]

    @staticmethod
    def list_law_edges(limit: int = 5000, offset: int = 0, include_disabled: bool = True) -> List[Dict[str, Any]]:
        """
        Retorna relaciones entre nodos de normografía legal principales.
        Incluye solo tipos principales: Ley, Decreto, DecretoLey, Resolución, Sentencia, Circular, Categoria.

        Argumentos:
            limit: Límite de resultados por página
            offset: Offset para paginación
            include_disabled: Incluir relaciones deshabilitadas

        Retorna:
            Lista de diccionarios con datos de relaciones entre nodos de normografía legal
        """
        # Filtrar relaciones solo entre tipos principales de normografía legal
        normography_labels = (
            "'Ley', 'Decreto', 'DecretoLey', 'Resolución', 'Normografia', 'Sentencia', 'Circular', 'Categoria'"
        )
        conditions = [
            f"s.label IN ({normography_labels})",
            f"t.label IN ({normography_labels})"
        ]
        if not include_disabled:
            conditions.extend(["e.enabled = true", "s.enabled = true", "t.enabled = true"])
        where_clause = " AND ".join(conditions)
        query = f"""
            SELECT e.edge_id, e.rel_type::text AS rel_type, e.start_node, e.end_node, e.enabled, e.created_at, e.properties
            FROM normografia.edge e
            JOIN normografia.node s ON s.node_uid = e.start_node
            JOIN normografia.node t ON t.node_uid = e.end_node
            WHERE {where_clause}
            ORDER BY e.created_at DESC NULLS LAST, e.edge_id DESC
            LIMIT :limit OFFSET :offset
        """
        with session_scope() as session:
            rows = session.execute(text(query), {"limit": limit, "offset": offset}).mappings().all()
            return [dict(r) for r in rows]

    @staticmethod
    def list_edges_by_source_node(source_node_uid: str, limit: int = 1000, offset: int = 0, include_disabled: bool = True) -> List[Dict[str, Any]]:
        """Return edges where the given node is the source (start_node)."""
        conditions = ["e.start_node = :source_node_uid"]
        if not include_disabled:
            conditions.append("e.enabled = true")
        where_clause = " AND ".join(conditions)
        query = f"""
            SELECT e.edge_id, e.rel_type::text AS rel_type, e.start_node, e.end_node, e.enabled, e.created_at, e.properties
            FROM normografia.edge e
            WHERE {where_clause}
            ORDER BY e.created_at DESC NULLS LAST, e.edge_id DESC
            LIMIT :limit OFFSET :offset
        """
        logger.info(f"Executing query for source_node_uid: {source_node_uid}")
        logger.info(f"Query: {query}")
        with session_scope() as session:
            rows = session.execute(text(query), {"source_node_uid": source_node_uid, "limit": limit, "offset": offset}).mappings().all()
            logger.info(f"Query returned {len(rows)} rows")
            return [dict(r) for r in rows]

    # --- WRITES ---
    @staticmethod
    def create_node(payload: Dict[str, Any]) -> str:
        """Insert node in Postgres and create it in Neo4j."""
        with session_scope() as session:
            node_uid = NormographyAdminService._insert_node(session, payload)
        # Mirror to Neo4j
        try:
            pass

            # Use app-level driver via dependency in router layer; here we assume router calls mirror funcs
        except Exception:  # no-op in service; router will call mirror
            pass
        return node_uid

    @staticmethod
    def update_node(node_uid: str, updates: Dict[str, Any]) -> None:
        with session_scope() as session:
            set_parts = []
            params: Dict[str, Any] = {"node_uid": node_uid}
            for key in [
                "label",
                "nombre",
                "texto",
                "numero_original",
                "tipo",
                "source",
                "processing_status",
                "hash",
                "tematica",
                "resumen_tematica",
                "embedding",
                "properties",
                "enabled",
            ]:
                if key in updates and updates[key] is not None:
                    if key == "embedding":
                        set_parts.append("embedding = CAST(:embedding AS vector)")
                        params["embedding"] = updates[key]
                    elif key == "properties":
                        set_parts.append("properties = CAST(:properties AS jsonb)")
                        params["properties"] = NormographyAdminService._serialize_jsonb(updates[key])
                    else:
                        set_parts.append(f"{key} = :{key}")
                        params[key] = updates[key]

            if set_parts:
                session.execute(
                    text(f"UPDATE normografia.node SET {', '.join(set_parts)}, last_updated = now() WHERE node_uid = :node_uid"),
                    params,
                )

    @staticmethod
    def soft_delete_node(node_uid: str) -> None:
        with session_scope() as session:
            session.execute(
                text("UPDATE normografia.node SET enabled = false, last_updated = now() WHERE node_uid = :node_uid"),
                {"node_uid": node_uid},
            )

    @staticmethod
    def create_edge(payload: Dict[str, Any]) -> int:
        with session_scope() as session:
            row = session.execute(
                text(
                    """
                    INSERT INTO normografia.edge (rel_type, start_node, end_node, properties, enabled)
                    VALUES (CAST(:rel_type AS public.normografia_rel_type), :start_node, :end_node, CAST(:properties AS jsonb), :enabled)
                    RETURNING edge_id
                    """
                ),
                {
                    "rel_type": payload["rel_type"],
                    "start_node": payload["start_node_uid"],
                    "end_node": payload["end_node_uid"],
                    "properties": NormographyAdminService._serialize_jsonb(payload.get("properties")),
                    "enabled": payload.get("enabled", True),
                },
            ).scalar()
            return int(row)

    @staticmethod
    def update_edge(edge_id: int, updates: Dict[str, Any]) -> None:
        with session_scope() as session:
            set_parts = []
            params: Dict[str, Any] = {"edge_id": edge_id}
            if "rel_type" in updates and updates["rel_type"]:
                set_parts.append("rel_type = CAST(:rel_type AS public.normografia_rel_type)")
                params["rel_type"] = updates["rel_type"]
            if "start_node_uid" in updates and updates["start_node_uid"]:
                set_parts.append("start_node = :start_node")
                params["start_node"] = updates["start_node_uid"]
            if "end_node_uid" in updates and updates["end_node_uid"]:
                set_parts.append("end_node = :end_node")
                params["end_node"] = updates["end_node_uid"]
            if "properties" in updates and updates["properties"] is not None:
                set_parts.append("properties = CAST(:properties AS jsonb)")
                params["properties"] = NormographyAdminService._serialize_jsonb(updates["properties"])
            if "enabled" in updates and updates["enabled"] is not None:
                set_parts.append("enabled = :enabled")
                params["enabled"] = updates["enabled"]
            if set_parts:
                session.execute(
                    text(f"UPDATE normografia.edge SET {', '.join(set_parts)} WHERE edge_id = :edge_id"),
                    params,
                )

    @staticmethod
    def soft_delete_edge(edge_id: int) -> None:
        with session_scope() as session:
            session.execute(
                text("UPDATE normografia.edge SET enabled = false WHERE edge_id = :edge_id"),
                {"edge_id": edge_id},
            )

    # --- helpers ---
    @staticmethod
    def _insert_node(session: Session, p: Dict[str, Any]) -> str:
        row = session.execute(
            text(
                """
                INSERT INTO normografia.node (
                    element_id, label, graph_id, nombre, texto, numero_original, tipo,
                    source, processing_status, total_chunks, hash, tematica, resumen_tematica,
                    embedding, properties, enabled
                ) VALUES (
                    NULL, :label, COALESCE(:graph_id, gen_random_uuid()::text), :nombre, :texto, :numero_original, :tipo,
                    :source, :processing_status, :total_chunks, :hash, :tematica, :resumen_tematica,
                    CAST(:embedding AS vector), CAST(:properties AS jsonb), :enabled
                )
                ON CONFLICT (graph_id) DO UPDATE SET
                    label = EXCLUDED.label,
                    nombre = EXCLUDED.nombre,
                    texto = EXCLUDED.texto,
                    numero_original = EXCLUDED.numero_original,
                    tipo = EXCLUDED.tipo,
                    source = EXCLUDED.source,
                    processing_status = EXCLUDED.processing_status,
                    total_chunks = EXCLUDED.total_chunks,
                    hash = EXCLUDED.hash,
                    tematica = EXCLUDED.tematica,
                    resumen_tematica = EXCLUDED.resumen_tematica,
                    embedding = EXCLUDED.embedding,
                    properties = EXCLUDED.properties,
                    enabled = true,
                    last_updated = now()
                RETURNING node_uid
                """
            ),
            {
                "label": p["label"],
                "graph_id": p.get("graph_id"),
                "nombre": p.get("nombre"),
                "texto": p.get("texto"),
                "numero_original": p.get("numero_original"),
                "tipo": p.get("tipo"),
                "source": p.get("source"),
                "processing_status": p.get("processing_status"),
                "total_chunks": p.get("total_chunks"),
                "hash": p.get("hash"),
                "tematica": p.get("tematica"),
                "resumen_tematica": p.get("resumen_tematica"),
                "embedding": p.get("embedding"),
                "properties": NormographyAdminService._serialize_jsonb(p.get("properties")),
                "enabled": p.get("enabled", True),
            },
        ).scalar()
        return str(row)
