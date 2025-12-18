"""
Agente PgVector con persistencia y búsqueda reales usando PgVectorRetriever.

Lee `pgvector_request` desde el estado y soporta:
- persist: persiste un mensaje en la conversación.
- persist+retrieve / retrieve: persiste (si aplica) y devuelve contexto + resultados semánticos.
"""

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional

from app.config.settings import Settings
from app.extensions import get_logger
from app.schemas.agent_schemas import VectorSearchResult
from app.services.embeddings.embedding_service import EmbeddingService
from app.services.retrieval.pgvector_retriever_service import PgVectorRetriever

logger = get_logger(__name__)


class PgVectorAgent:
    def __init__(self, settings: Settings, embedding_service: EmbeddingService):
        self.settings = settings
        self.embedding_service = embedding_service
        self.retriever = PgVectorRetriever(settings, embedding_service)
        self.recent_messages_limit = settings.databases.postgresql.pgvector_recent_messages_limit
        self.semantic_search_limit = settings.databases.postgresql.pgvector_semantic_limit
        self.minio_service: Optional[Any] = None

    def set_minio_client(self, minio_client: Any) -> None:
        self.minio_service = minio_client

    def init_minio(self, minio_service: Any) -> None:
        self.minio_service = minio_service

    async def run(self, state: Any) -> Dict[str, Any]:
        sd: Dict[str, Any] = state if isinstance(state, dict) else getattr(state, "__dict__", {}) or {}
        t0 = time.time()
        trace_id = (sd.get("workflow_metadata") or {}).get("trace_id") or str(uuid.uuid4())[:8]

        req = sd.get("pgvector_request") or {}
        action = (req.get("action") or sd.get("action") or "").strip().lower() or "retrieve"
        conversation_id = self._normalize_conversation_id(sd.get("conversation_id")) or str(uuid.uuid4())
        user_query = req.get("user_query") or sd.get("user_query") or ""
        message_payload = req.get("message") or sd.get("message")

        persist_status = {}
        retrieval_status = {}
        vector_results: List[Dict[str, Any]] = []
        conversation_context: Dict[str, Any] = {}

        try:
            if action in {"persist", "persist+retrieve"} and message_payload:
                persist_status = await self._persist_message(conversation_id, message_payload)

            if action in {"retrieve", "persist+retrieve"}:
                conversation_context = await self.retriever.get_conversation_context(
                    conversation_id=conversation_id,
                    user_query=user_query,
                    recent_messages_limit=self.recent_messages_limit,
                )
                # Búsqueda semántica en documentos (mensajes) adicionales
                docs = await self.retriever.semantic_search_documents(
                    query=user_query,
                    limit=self.semantic_search_limit,
                    similarity_threshold=self.settings.databases.postgresql.pgvector_similarity_threshold,
                )
                vector_results = [r.model_dump() for r in docs]

                retrieval_status = {
                    "success": True,
                    "results_count": len(vector_results),
                    "strategy": conversation_context.get("context_strategy"),
                    "current_message_id": conversation_context.get("current_message_id"),
                }

            elapsed = time.time() - t0
            return {
                "conversation_id": conversation_id,
                "vector_results": vector_results,
                "conversation_context": conversation_context,
                "pgvector_retrieval_status": retrieval_status or None,
                "pgvector_write_status": persist_status or sd.get("pgvector_write_status"),
                "pgvector_request": None,
                "workflow_metadata": {
                    **(sd.get("workflow_metadata") or {}),
                    "pgvector_retrieval_status": retrieval_status or None,
                    "pgvector_write_status": persist_status or sd.get("pgvector_write_status"),
                    "pgvector_agent_response": {
                        "success": True,
                        "elapsed": elapsed,
                        "action": action,
                    },
                },
            }
        except asyncio.TimeoutError as te:
            logger.error("PgVectorAgent timeout", extra={"trace_id": trace_id, "error": str(te)})
            return {
                "conversation_id": conversation_id,
                "vector_results": [],
                "conversation_context": {},
                "pgvector_retrieval_status": {"success": False, "error": "timeout"},
            }
        except Exception as e:
            logger.exception("PgVectorAgent error", extra={"trace_id": trace_id})
            return {
                "conversation_id": conversation_id,
                "vector_results": [],
                "conversation_context": {},
                "pgvector_retrieval_status": {"success": False, "error": str(e)},
            }

    async def _persist_message(self, conversation_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        role = (message.get("role") or message.get("sender") or "user").lower()
        content = (message.get("content") or message.get("text") or "").strip()
        if not content:
            return {"success": False, "error": "empty_content"}
        # asegurar conversación existe
        exists = await self.retriever._check_conversation_exists(conversation_id)
        if not exists:
            await self.retriever._create_conversation(conversation_id)
        msg_id = await self.retriever._add_message_to_conversation(conversation_id, content, role=role)
        return {
            "success": True,
            "message_id": msg_id,
            "sender": role,
            "kind": message.get("kind"),
            "created_at": time.time(),
        }

    def _normalize_conversation_id(self, conversation_id: Optional[str]) -> Optional[str]:
        try:
            cid = str(conversation_id or "").strip()
            return cid or None
        except Exception:
            return None

