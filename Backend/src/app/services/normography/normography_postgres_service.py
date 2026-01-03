"""
Servicio de Normografía para PostgreSQL.

Proporciona sincronización bidireccional entre Neo4j y PostgreSQL para la estructura
de normografía, incluyendo creación de esquemas, tipos y operaciones de upsert en lotes.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, time
from typing import Any, Dict, List, Optional

try:
    # Import opcional; el código maneja la ausencia gracefully
    from neo4j.time import Date as NeoDate
    from neo4j.time import DateTime as NeoDateTime  # type: ignore
    from neo4j.time import Time as NeoTime
except Exception:  # pragma: no cover - environment without neo4j module
    NeoDateTime = tuple()  # type: ignore
    NeoDate = tuple()  # type: ignore
    NeoTime = tuple()  # type: ignore

import json

from sqlalchemy import text

from app.clients.sql_pgvector_client import session_scope
from app.extensions import get_logger

logger = get_logger(__name__)


# --- Bootstrap DDL ---------------------------------------------------------


DDL_ENABLE_PGCRYPTO = "CREATE EXTENSION IF NOT EXISTS pgcrypto"  # gen_random_uuid()
DDL_ENABLE_PG_TRGM = "CREATE EXTENSION IF NOT EXISTS pg_trgm"  # trigram text search
DDL_ENABLE_VECTOR = "CREATE EXTENSION IF NOT EXISTS vector"  # pgvector


DDL_CREATE_SCHEMA_AND_TYPES = """
CREATE SCHEMA IF NOT EXISTS normografia;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'normografia_rel_type') THEN
    CREATE TYPE public.normografia_rel_type AS ENUM
      ('CONTAINS','MENTIONS','DESCRIBES','REFERS_TO','PERTENECE_A');
  END IF;
END $$;
"""


# Note: we add element_id for idempotent mapping using Neo4j elementId(n)
DDL_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS normografia.node (
  node_uid        uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  element_id      text UNIQUE,                    -- Neo4j elementId(n)
  label           text NOT NULL,                  -- e.g. 'Ley','Seccion','Articulo','Chunk','Documento', ...
  graph_id        text UNIQUE NOT NULL,           -- Neo4j logical id if present; synthetic fallback otherwise
  nombre          text,
  texto           text,
  numero_original text,
  tipo            text,
  source          text,
  processing_status text,
  total_chunks    integer,
  hash            text,
  tematica        text,
  resumen_tematica text,
  embedding       vector(384),                    -- adjust to your embedding dimension
  created_at      timestamptz DEFAULT now(),
  last_updated    timestamptz,
  enabled         boolean NOT NULL DEFAULT true,
  properties      jsonb,
  parent          uuid NULL REFERENCES normografia.node(node_uid)
);

CREATE INDEX IF NOT EXISTS ix_normografia_node_label ON normografia.node (label);
CREATE INDEX IF NOT EXISTS ix_normografia_node_enabled ON normografia.node (enabled);
CREATE INDEX IF NOT EXISTS ix_normografia_node_nombre_trgm ON normografia.node USING gin (nombre gin_trgm_ops);

-- Optional ANN index (requires ANALYZE and pgvector available)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes WHERE schemaname = 'normografia' AND indexname = 'ix_normografia_node_embedding_ivfflat_cosine'
  ) THEN
    EXECUTE 'CREATE INDEX ix_normografia_node_embedding_ivfflat_cosine ON normografia.node USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)';
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS normografia.edge (
  edge_id     bigserial PRIMARY KEY,
  rel_type    public.normografia_rel_type NOT NULL,
  start_node  uuid NOT NULL REFERENCES normografia.node(node_uid) ON DELETE CASCADE,
  end_node    uuid NOT NULL REFERENCES normografia.node(node_uid) ON DELETE CASCADE,
  properties  jsonb,
  created_at  timestamptz DEFAULT now(),
  enabled     boolean NOT NULL DEFAULT true
);

CREATE INDEX IF NOT EXISTS ix_normografia_edge_start ON normografia.edge (start_node);
CREATE INDEX IF NOT EXISTS ix_normografia_edge_end   ON normografia.edge (end_node);
CREATE INDEX IF NOT EXISTS ix_normografia_edge_enabled ON normografia.edge (enabled);
"""


DDL_CREATE_VIEWS = """
CREATE OR REPLACE VIEW normografia.vw_normas AS
SELECT *
FROM normografia.node
WHERE label IN ('Ley','Decreto','DecretoLey','Resolución','Sentencia','Circular','Titulo','Capitulo','Libro','Parte','Seccion','Articulo','Paragrafo');

CREATE OR REPLACE VIEW normografia.vw_enabled_edges AS
SELECT e.*
FROM normografia.edge e
JOIN normografia.node s ON s.node_uid = e.start_node
JOIN normografia.node t ON t.node_uid = e.end_node
WHERE e.enabled AND s.enabled AND t.enabled;

CREATE OR REPLACE VIEW normografia.vw_hijos AS
SELECT c.*, e.rel_type, p.node_uid AS parent_uid
FROM normografia.node c
JOIN normografia.edge e
  ON e.end_node = c.node_uid AND e.rel_type = 'PERTENECE_A' AND e.enabled
JOIN normografia.node p
  ON p.node_uid = e.start_node
WHERE c.enabled AND p.enabled;
"""


def ensure_normografia_schema() -> None:
    """
    Crea extensiones, esquema, tipos, tablas, índices y vistas requeridas.

    Seguro de ejecutar múltiples veces (idempotente). Intenta habilitar extensiones
    individualmente pero no falla si los permisos son insuficientes.
    """
    with session_scope() as session:
        logger.info("Asegurando esquema y extensiones de normografía en PostgreSQL...")
        # Intentar habilitar extensiones individualmente pero no fallar si
        # los permisos son insuficientes
        for ddl, name in (
            (DDL_ENABLE_PGCRYPTO, "pgcrypto"),
            (DDL_ENABLE_PG_TRGM, "pg_trgm"),
            (DDL_ENABLE_VECTOR, "vector"),
        ):
            try:
                session.execute(text(ddl))
            except Exception as e:
                logger.warning("Extensión %s no habilitada (continuando): %s", name, str(e))

        # Crear esquema, tipos, tablas y vistas (idempotente)
        session.execute(text(DDL_CREATE_SCHEMA_AND_TYPES))
        session.execute(text(DDL_CREATE_TABLES))
        session.execute(text(DDL_CREATE_VIEWS))
        logger.info("Esquema de normografía está listo")


# --- Upsert helpers --------------------------------------------------------


def _vector_literal(embedding: Optional[List[float]]) -> Optional[List[float]]:
    """
    Retorna lista de embedding para pgvector o None.

    Argumentos:
        embedding: Lista de floats o None

    Retorna:
        Lista de floats o None si embedding está vacío
    """
    if not embedding:
        return None
    try:
        # Retornar la lista directamente - SQLAlchemy manejará la conversión
        return [float(x) for x in embedding]
    except Exception:
        return None


def _coerce_value_for_sql(value: Any) -> Any:
    """
    Coacciona tipos temporales de Neo4j y estructuras anidadas a valores adaptables a SQL.

    Conversiones:
    - neo4j.time.DateTime → python datetime (preferido) o string ISO como fallback
    - neo4j.time.Date → python date o string ISO
    - neo4j.time.Time → python time o string ISO
    - lists/dicts → recursivo

    Argumentos:
        value: Valor a coaccionar

    Retorna:
        Valor adaptado para SQL
    """
    # Neo4j-temporal handling
    try:
        if isinstance(value, NeoDateTime):  # type: ignore[arg-type]
            # Prefer native python datetime if available
            if hasattr(value, "to_native"):
                return value.to_native()  # type: ignore[no-any-return]
            if hasattr(value, "iso_format"):
                return value.iso_format()  # type: ignore[no-any-return]
            return str(value)
        if isinstance(value, NeoDate):  # type: ignore[arg-type]
            if hasattr(value, "to_native"):
                return value.to_native()
            if hasattr(value, "iso_format"):
                return value.iso_format()
            return str(value)
        if isinstance(value, NeoTime):  # type: ignore[arg-type]
            if hasattr(value, "to_native"):
                return value.to_native()
            if hasattr(value, "iso_format"):
                return value.iso_format()
            return str(value)
    except Exception:
        # Fall through to generic handling
        pass

    # Python native datetimes are acceptable as-is
    if isinstance(value, (datetime, date, time)):
        return value

    if isinstance(value, list):
        return [_coerce_value_for_sql(v) for v in value]
    if isinstance(value, dict):
        return {k: _coerce_value_for_sql(v) for k, v in value.items()}
    return value


def _json_default(obj: Any) -> Any:
    """
    Serializador por defecto para json.dumps: datetime/date/time → strings ISO.

    Argumentos:
        obj: Objeto a serializar

    Retorna:
        String ISO para fechas/times, string genérico para otros tipos
    """
    if isinstance(obj, (datetime, date, time)):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    # Fallback: convertir tipos desconocidos a string
    return str(obj)


def _sanitize_string_for_sql(value: str) -> str:
    """
    Remueve caracteres de control problemáticos (null bytes) no permitidos por Postgres text/jsonb.

    Argumentos:
        value: String a sanitizar

    Retorna:
        String sanitizado sin bytes nulos
    """
    try:
        # Remover NULs; mantener tabs/newlines si están presentes
        return value.replace("\x00", "").replace("\u0000", "")
    except Exception:
        return value


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize values for safe JSONB ingestion (remove NULs in strings)."""
    if isinstance(obj, str):
        return _sanitize_string_for_sql(obj)
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    return obj


def upsert_nodes_by_element_batch(nodes: List[Dict[str, Any]]) -> int:
    """Upsert a batch of nodes using element_id for idempotency.

    Each node dict should contain keys: element_id (str), labels (List[str]), props (Dict[str, Any]).
    We'll derive graph_id = props.get('id') or synthetic f"{label}:{nombre}:{tipo}".
    """
    if not nodes:
        return 0

    inserted = 0
    with session_scope() as session:
        for node in nodes:
            element_id: Optional[str] = node.get("element_id")
            if not element_id:
                continue
            labels: List[str] = node.get("labels") or []
            props: Dict[str, Any] = dict(node.get("props") or {})

            label = labels[0] if labels else "Unknown"
            nombre = _coerce_value_for_sql(props.get("nombre"))
            # Some nodes (e.g., Chunk) use 'text' property; map to 'texto' column
            texto_prop = props.get("texto") if props.get("texto") is not None else props.get("text")
            if isinstance(texto_prop, str):
                texto_prop = _sanitize_string_for_sql(texto_prop)
            texto = _coerce_value_for_sql(texto_prop)
            numero_original = _coerce_value_for_sql(props.get("numero_original"))
            tipo = _coerce_value_for_sql(props.get("tipo"))
            source = _coerce_value_for_sql(props.get("source"))
            processing_status = _coerce_value_for_sql(props.get("processing_status"))
            total_chunks = _coerce_value_for_sql(props.get("total_chunks"))
            hash_val = _coerce_value_for_sql(props.get("hash"))
            tematica = _coerce_value_for_sql(props.get("tematica"))
            resumen_tematica = _coerce_value_for_sql(props.get("resumen_tematica"))
            created_at = _coerce_value_for_sql(props.get("created_at"))
            last_updated = _coerce_value_for_sql(props.get("last_updated"))
            embedding_list = props.get("embedding")

            # Remove large/duplicated fields from properties
            for k in ("embedding", "texto", "text"):
                if k in props:
                    props.pop(k, None)

            # Compute graph_id
            graph_id = props.get("id") or f"{label}:{nombre or ''}:{tipo or ''}"

            embedding_literal = _vector_literal(embedding_list)

            # Prepare JSON-safe properties (datetimes → ISO strings, nested ok)
            safe_properties = _sanitize_for_json(props) if props else None

            # Use explicit CAST for pgvector to avoid parser issues with ::
            embedding_sql = "CAST(:embedding AS vector)" if embedding_literal is not None else "NULL"

            # Leer enabled desde props de Neo4j si existe
            # Si no existe en Neo4j, preservar el valor existente en PostgreSQL (no sobrescribir)
            enabled_from_neo4j = props.get("enabled")
            if enabled_from_neo4j is None:
                # Si no existe en props de Neo4j, usar NULL para preservar el valor existente en PostgreSQL
                enabled_value = None
            else:
                # Convertir a boolean si viene como string o número
                enabled_value = bool(enabled_from_neo4j) if not isinstance(enabled_from_neo4j, bool) else enabled_from_neo4j

            stmt = text(
                """
                INSERT INTO normografia.node (
                    element_id, label, graph_id, nombre, texto, numero_original, tipo,
                    source, processing_status, total_chunks, hash, tematica, resumen_tematica,
                    embedding, created_at, last_updated, enabled, properties
                ) VALUES (
                    :element_id, :label, :graph_id, :nombre, :texto, :numero_original, :tipo,
                    :source, :processing_status, :total_chunks, :hash, :tematica, :resumen_tematica,
                    {embedding_sql}, :created_at, :last_updated, COALESCE(:enabled, true), CAST(:properties AS jsonb)
                )
                ON CONFLICT (element_id) DO UPDATE SET
                    label = EXCLUDED.label,
                    graph_id = EXCLUDED.graph_id,
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
                    last_updated = now(),
                    enabled = CASE 
                        WHEN EXCLUDED.enabled IS NOT NULL THEN EXCLUDED.enabled
                        ELSE normografia.node.enabled
                    END,
                    properties = EXCLUDED.properties
                RETURNING node_uid
                """.format(
                    embedding_sql=embedding_sql
                )
            )

            params = {
                "element_id": element_id,
                "label": label,
                "graph_id": graph_id,
                "nombre": nombre,
                "texto": texto,
                "numero_original": numero_original,
                "tipo": tipo,
                "source": source,
                "processing_status": processing_status,
                "total_chunks": total_chunks,
                "hash": hash_val,
                "tematica": tematica,
                "resumen_tematica": resumen_tematica,
                "embedding": embedding_literal,
                "created_at": created_at,
                "last_updated": last_updated,
                "enabled": enabled_value,
                "properties": None if safe_properties is None else json.dumps(safe_properties, default=_json_default),
            }

            session.execute(stmt, params)
            inserted += 1

    return inserted


def upsert_edges_by_element_batch(relationships: List[Dict[str, Any]]) -> int:
    """Upsert relationships by resolving start/end via element_id.

    Each item requires keys: type (str), start.element_id (str), end.element_id (str), props (dict)
    """
    if not relationships:
        return 0

    inserted = 0
    with session_scope() as session:
        for rel in relationships:
            rel_type: Optional[str] = rel.get("type")
            if not rel_type:
                continue
            start_id: str = rel.get("start", {}).get("internal_id")
            end_id: str = rel.get("end", {}).get("internal_id")
            props: Dict[str, Any] = dict(rel.get("properties") or {})

            # Sanitizar propiedades: convertir DateTime de Neo4j a valores SQL-compatibles
            # y luego a JSON string para JSONB en PostgreSQL
            if props:
                # Primero coaccionar valores temporales de Neo4j a tipos Python nativos
                coerced_props = {k: _coerce_value_for_sql(v) for k, v in props.items()}
                # Luego sanitizar strings (remover NULs) y convertir a JSON string
                safe_properties = _sanitize_for_json(coerced_props)
                properties_json = json.dumps(safe_properties, default=_json_default)
            else:
                properties_json = None

            stmt = text(
                """
                INSERT INTO normografia.edge (rel_type, start_node, end_node, properties, enabled)
                SELECT CAST(:rel_type AS public.normografia_rel_type), s.node_uid, e.node_uid, CAST(:properties AS jsonb), true
                FROM normografia.node s
                JOIN normografia.node e ON e.element_id = :end_element_id
                WHERE s.element_id = :start_element_id
                ON CONFLICT (rel_type, start_node, end_node) DO UPDATE SET
                  enabled = true,
                  properties = EXCLUDED.properties
                RETURNING edge_id
                """
            )

            result = session.execute(
                stmt,
                {
                    "rel_type": rel_type,
                    "start_element_id": start_id,
                    "end_element_id": end_id,
                    "properties": properties_json,
                },
            )
            # Solo incrementar si realmente se insertó/actualizó una relación
            row = result.fetchone()
            if row:
                inserted += 1
            else:
                # Log si no se pudo insertar (nodos no encontrados)
                logger.warning(
                    f"No se pudo insertar relación {rel_type}: nodos no encontrados",
                    extra={
                        "rel_type": rel_type,
                        "start_element_id": start_id,
                        "end_element_id": end_id,
                    },
                )

    return inserted


# Convenience async wrappers to avoid blocking event loop in workers
async def async_ensure_normografia_schema() -> None:
    await asyncio.to_thread(ensure_normografia_schema)


async def async_upsert_nodes_by_element_batch(nodes: List[Dict[str, Any]]) -> int:
    return await asyncio.to_thread(upsert_nodes_by_element_batch, nodes)


async def async_upsert_edges_by_element_batch(rels: List[Dict[str, Any]]) -> int:
    return await asyncio.to_thread(upsert_edges_by_element_batch, rels)
