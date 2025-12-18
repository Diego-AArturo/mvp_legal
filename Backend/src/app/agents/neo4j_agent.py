"""
Agente Neo4j con recuperación real usando Neo4jRetrieverService.

Restricción actual: todas las búsquedas se realizan solo sobre nodos de
normografía (índice `norma_embeddings`), sin consultar chunks ni documentos.

Mantiene compatibilidad con el orquestador: lee `neo4j_request` del estado,
ejecuta búsqueda semántica y devuelve deltas (`graph_results`, `neo4j_status`).
"""

import time
import uuid
from typing import Any, Dict, Optional

from neo4j import AsyncDriver

from app.config.settings import Settings
from app.extensions import get_logger
from app.schemas.graph_search_schemas import GraphSearchResponse
from app.services.embeddings.embedding_service import EmbeddingService
from app.services.retrieval.neo4j_retriever_service import Neo4jRetrieverService

logger = get_logger(__name__)


class Neo4jAgent:
    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService,
        neo4j_driver: Optional[AsyncDriver] = None,
    ) -> None:
        self.settings = settings
        self.embedding_service = embedding_service
        self.driver: Optional[AsyncDriver] = neo4j_driver
        self.database = getattr(settings.databases.neo4j, "neo4j_database", None)
        self.retriever: Optional[Neo4jRetrieverService] = None
        if self.driver:
            self.retriever = Neo4jRetrieverService(self.driver, settings, embedding_service)

    def set_driver(self, driver: AsyncDriver) -> None:
        if not self.driver:
            self.driver = driver
            self.retriever = Neo4jRetrieverService(driver, self.settings, self.embedding_service)

    async def run(self, state: Any) -> Dict[str, Any]:
        t0 = time.time()
        trace_id = self._get_state_value(state, "workflow_metadata", {}).get("trace_id") or str(uuid.uuid4())[:8]
        req = self._get_state_value(state, "neo4j_request", {}) or {}
        user_query_fallback = (self._get_state_value(state, "user_query", "") or "").strip()
        query = (req.get("query") or user_query_fallback or "").strip()

        try:
            limit = int(req.get("top_k") or self._get_nested_value(state, "config.neo4j_top_k", 10))
        except Exception:
            limit = 10

        min_score = req.get("similarity_threshold")
        if min_score is None:
            min_score = getattr(self.settings.databases.neo4j, "neo4j_vector_similarity_threshold", 0.5)
        try:
            min_score = float(min_score)
        except Exception:
            min_score = 0.5

        if not query:
            empty = GraphSearchResponse(search_strategy="error", metadata={"error": "empty_query"})
            return self._finish(state, empty, False, "invalid_input", "Consulta vacía", trace_id, t0)

        if not self.driver or not self.retriever:
            empty = GraphSearchResponse(search_strategy="error", metadata={"error": "no_driver"})
            return self._finish(state, empty, False, "no_driver", "Driver Neo4j no inicializado", trace_id, t0)

        try:
            await self.driver.verify_connectivity()
        except Exception as conn_err:
            empty = GraphSearchResponse(search_strategy="error", metadata={"error": str(conn_err)})
            return self._finish(state, empty, False, "connectivity_error", f"Error de conectividad: {conn_err}", trace_id, t0)

        try:
            result = await self.retriever.semantic_search(query, limit=limit, similarity_threshold=min_score)
            code = "ok" if (result.total_found or 0) > 0 else "no_results"
            msg = "Búsqueda completada" if code == "ok" else "Sin resultados"
            return self._finish(state, result, True, code, msg, trace_id, t0)
        except Exception as e:
            error_resp = GraphSearchResponse(search_strategy="error", metadata={"error": str(e)})
            return self._finish(state, error_resp, False, "semantic_search_error", str(e), trace_id, t0)

    # Helpers ----------------------------------------------------------
    def _get_state_value(self, state: Any, key: str, default: Any = None) -> Any:
        if isinstance(state, dict):
            return state.get(key, default)
        return getattr(state, key, default)

    def _get_nested_value(self, state: Any, path: str, default: Any = None) -> Any:
        try:
            if isinstance(state, dict):
                cur = state
                for k in path.split("."):
                    if not isinstance(cur, dict):
                        return default
                    cur = cur.get(k)
                    if cur is None:
                        return default
                return cur
            cur = state
            for k in path.split("."):
                cur = getattr(cur, k, None)
                if cur is None:
                    return default
            return cur
        except Exception:
            return default

    def _finish(
        self,
        state: Dict[str, Any],
        result: GraphSearchResponse,
        success: bool,
        code: str,
        message: str,
        trace_id: str,
        t0: float,
    ) -> Dict[str, Any]:
        elapsed_ms = int((time.time() - t0) * 1000)
        neo4j_status = {
            "success": success,
            "code": code,
            "message": message,
            "trace_id": trace_id,
            "elapsed_ms": elapsed_ms,
            "total_found": int(getattr(result, "total_found", 0) or 0),
            "strategy": getattr(result, "search_strategy", None),
            "scope": "normografia_only",
        }
        workflow_metadata = dict(self._get_state_value(state, "workflow_metadata", {}) or {})
        workflow_metadata["neo4j_agent_response"] = result.model_dump() if hasattr(result, "model_dump") else {}

        return {
            "graph_results": result,
            "neo4j_status": neo4j_status,
            "neo4j_request": None,
            "workflow_metadata": workflow_metadata,
        }
