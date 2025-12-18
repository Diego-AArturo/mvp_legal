"""
Servicio de recuperación PgVector para contexto de conversación y búsqueda semántica.

Proporciona capacidades de gestión de conversaciones y búsqueda semántica usando
PostgreSQL con extensión pgvector para similitud vectorial.
"""

import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.clients.sql_pgvector_client import session_scope
from app.config.settings import Settings
from app.extensions import get_logger
from app.schemas.agent_schemas import VectorSearchResult
from app.services.embeddings.embedding_service import EmbeddingService

logger = get_logger(__name__)


class PgVectorRetriever:
    """
    Retriever PgVector para contexto de conversación y búsqueda semántica.

    Proporciona capacidades de gestión de conversaciones y búsqueda semántica
    usando PostgreSQL con extensión pgvector para similitud vectorial.
    """

    def __init__(self, settings: Settings, embedding_service: EmbeddingService):
        """
        Inicializa el retriever PgVector.

        Argumentos:
            settings: Configuración de la aplicación
            embedding_service: Servicio de embeddings para generar vectores
        """
        self.settings = settings
        self.embedding_service = embedding_service
        # Usar límite semántico configurado desde PostgreSQL settings
        try:
            self.max_conversation_messages = (
                self.settings.databases.postgresql.pgvector_semantic_limit
            )
            self.similarity_threshold = (
                self.settings.databases.postgresql.pgvector_similarity_threshold
            )
        except Exception:
            self.max_conversation_messages = 10
            self.similarity_threshold = 0.7

    async def get_conversation_context(
        self,
        conversation_id: str,
        user_query: str,
        recent_messages_limit: int = 3,
    ) -> Dict[str, Any]:
        """
        Obtiene el contexto de conversación incluyendo mensajes recientes y búsqueda semántica.

        Argumentos:
            conversation_id: Identificador único de la conversación
            user_query: Consulta actual del usuario
            recent_messages_limit: Número de mensajes recientes a incluir

        Retorna:
            Diccionario con contexto de conversación y resultados de búsqueda
        """
        # Verificar si la conversación existe, crear si no existe
        conversation_exists = await self._check_conversation_exists(conversation_id)

        if not conversation_exists:
            await self._create_conversation(conversation_id)
            logger.info(f"Nueva conversación creada: {conversation_id}")

        # Agregar mensaje actual del usuario a la conversación
        new_message_id = await self._add_message_to_conversation(
            conversation_id, user_query, "user"
        )

        # Obtener mensajes recientes de la conversación
        recent_messages = await self._get_recent_messages(
            conversation_id, recent_messages_limit
        )

        # Verificar si se necesita búsqueda semántica (más que el límite de mensajes)
        total_messages = await self._get_conversation_message_count(conversation_id)

        semantic_results = []
        if total_messages > recent_messages_limit:
            # Realizar búsqueda semántica en mensajes más antiguos
            semantic_results = await self._semantic_search_conversation(
                conversation_id, user_query, exclude_recent=recent_messages_limit
            )

        return {
            "conversation_id": conversation_id,
            "recent_messages": recent_messages,
            "semantic_results": semantic_results,
            "total_messages": total_messages,
            "context_strategy": "hybrid" if semantic_results else "recent_only",
            "current_message_id": new_message_id,
        }

    async def semantic_search_documents(
        self, query: str, limit: int = 10, similarity_threshold: float = 0.7
    ) -> List[VectorSearchResult]:
        """
        Realiza búsqueda semántica sobre embeddings de documentos.

        Argumentos:
            query: Texto de consulta de búsqueda
            limit: Número máximo de resultados
            similarity_threshold: Puntuación mínima de similitud

        Retorna:
            Lista de objetos VectorSearchResult
        """
        query = (query or "").strip()
        if not query:
            return []

        results: List[VectorSearchResult] = []

        try:
            query_embedding = await self._generate_message_embedding(query)

            with session_scope() as session:
                raw_results = await self._execute_document_search(
                    session,
                    query_embedding,
                    query,
                    limit,
                    similarity_threshold,
                )

            results = self._convert_to_vector_results(raw_results)

        except Exception as e:
            logger.error(f"Error in document semantic search: {e}")

        return results

    async def _execute_conversation_check(
        self, session: Session, conversation_id: str
    ) -> Optional[int]:
        """
        Return 1 if conversation exists, otherwise None.
        """
        stmt = text("SELECT 1 FROM app_conversations WHERE id = :cid LIMIT 1")
        result = session.execute(stmt, {"cid": conversation_id}).scalar()
        return result

    async def _check_conversation_exists(self, conversation_id: str) -> bool:
        """
        Check if a conversation with the given ID exists.

        Argumentos:
            conversation_id: Unique conversation identifier

        Retorna:
            True if conversation exists, False otherwise
        """
        try:
            with session_scope() as session:
                result = await self._execute_conversation_check(
                    session, conversation_id
                )
                return result is not None
        except Exception as e:
            logger.error(f"Error checking conversation existence: {e}")
            return False

    async def _create_conversation(self, conversation_id: str) -> None:
        """
        Create a new conversation record.

        Argumentos:
            conversation_id: Unique conversation identifier
        """
        try:
            with session_scope() as session:
                await self._execute_conversation_creation(session, conversation_id)
                logger.info(f"Created conversation: {conversation_id}")
        except Exception as e:
            logger.error(f"Error creating conversation: {e}")
            raise

    async def _add_message_to_conversation(
        self, conversation_id: str, content: str, role: str = "user"
    ) -> str:
        """
        Add a message to an existing conversation.

        Argumentos:
            conversation_id: Conversation identifier
            content: Message content
            role: Message role (user, assistant, system)

        Retorna:
            Message ID of the created message
        """
        message_id = str(uuid4())

        try:
            with session_scope() as session:
                # Placeholder for future embedding generation
                embedding = await self._generate_message_embedding(content)

                await self._execute_message_insertion(
                    session,
                    message_id,
                    conversation_id,
                    content,
                    role,
                    embedding,
                )

                await self._update_conversation_timestamp(session, conversation_id)

                logger.debug(f"Added message to conversation {conversation_id}")

        except Exception as e:
            logger.error(f"Error adding message to conversation: {e}")
            raise

        return message_id

    async def _get_recent_messages(
        self, conversation_id: str, limit: int
    ) -> List[Dict[str, Any]]:
        """
        Get the most recent messages from a conversation.

        Argumentos:
            conversation_id: Conversation identifier
            limit: Maximum number of messages to retrieve

        Retorna:
            List of message dictionaries with metadata
        """
        try:
            with session_scope() as session:
                messages = await self._execute_recent_messages_query(
                    session, conversation_id, limit
                )
                return self._format_messages_for_context(messages)

        except Exception as e:
            logger.error(f"Error getting recent messages: {e}")
            return []

    async def _get_conversation_message_count(self, conversation_id: str) -> int:
        """
        Get total number of messages in a conversation.

        Argumentos:
            conversation_id: Conversation identifier

        Retorna:
            Total message count
        """

        try:
            with session_scope() as session:
                stmt = text(
                    "SELECT COUNT(*) FROM messages WHERE conversation_id = :cid"
                )
                count = session.execute(stmt, {"cid": conversation_id}).scalar()
                return int(count or 0)
        except Exception as e:
            logger.error(f"Error counting conversation messages: {e}")
            return 0

    async def _semantic_search_conversation(
        self, conversation_id: str, query: str, exclude_recent: int = 3
    ) -> List[VectorSearchResult]:
        """
        Perform semantic search on older conversation messages.

        Argumentos:
            conversation_id: Conversation identifier
            query: Search query
            exclude_recent: Number of recent messages to exclude

        Retorna:
            List of relevant messages as VectorSearchResult
        """
        query = (query or "").strip()
        if not query:
            return []

        try:
            query_embedding = await self._generate_message_embedding(query)

            with session_scope() as session:
                recent_ids = await self._get_recent_message_ids(
                    session, conversation_id, exclude_recent
                )

                vector_candidates: List[Dict[str, Any]] = []
                if query_embedding and any(x != 0.0 for x in query_embedding):
                    vector_candidates = await self._execute_conversation_vector_search(
                        session,
                        conversation_id,
                        query_embedding,
                        recent_ids,
                        self.max_conversation_messages,
                        self.similarity_threshold,
                    )

                if vector_candidates:
                    return self._convert_messages_to_vector_results(vector_candidates)

                fallback_results = await self._execute_conversation_semantic_search(
                    session, conversation_id, query, recent_ids
                )
                return self._convert_messages_to_vector_results(fallback_results)

        except Exception as e:
            logger.error(f"Error in conversation semantic search: {e}")
            return []

    async def _generate_message_embedding(self, content: str) -> List[float]:
        """
        Generate vector embedding for message content.

        Uses the configured embedding service to generate semantic embeddings
        for the given message content.

        Argumentos:
            content: Message content to embed

        Retorna:
            Vector embedding as list of floats

        Note:
            The paraphrase-multilingual-MiniLM-L12-v2 model generates 384-dimensional
            embeddings, which is perfect for pgvector storage and semantic search.
        """
        try:
            # Generate embedding using the embedding service
            embedding = await self.embedding_service.generate_embedding(
                content, normalize=True
            )

            if not embedding:
                logger.error(
                    f"Empty embedding generated for message content: {content[:50]}..."
                )
                # Return zero vector as fallback
                return [0.0] * self.embedding_service.get_embedding_dimension()

            logger.debug(
                f"Generated embedding of dimension {len(embedding)} for message"
            )
            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding for message content: {e}")
            # Return zero vector as fallback
            dimension = self.embedding_service.get_embedding_dimension()
            logger.warning(f"Using zero embedding fallback with dimension {dimension}")
            return [0.0] * dimension

    async def _execute_document_search(
        self,
        session: Session,
        query_embedding: List[float],
        query_text: str,
        limit: int,
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """
        Execute document semantic search query using pgvector.

        Argumentos:
            session: Database session
            query: Search query
            limit: Result limit
            threshold: Similarity threshold

        Retorna:
            Raw search results
        """
        try:
            if query_embedding and any(x != 0.0 for x in query_embedding):
                vector_query = text(
                    """
                    SELECT id,
                           conversation_id,
                           sender,
                           role,
                           metadata,
                           content,
                           created_at,
                           (1 - (embedding <=> CAST(:query_embedding AS vector))) AS rank
                    FROM messages
                    WHERE content IS NOT NULL
                      AND length(btrim(content)) > 0
                      AND embedding IS NOT NULL
                      AND (role IN ('user'::public.chat_role, 'assistant'::public.chat_role)
                           OR (role IS NULL AND sender IN ('user', 'assistant')))
                      AND (1 - (embedding <=> CAST(:query_embedding AS vector))) >= :thr
                    ORDER BY embedding <=> CAST(:query_embedding AS vector)
                    LIMIT :lim
                    """
                )
                embedding_str = f"[{','.join(map(str, query_embedding))}]"
                rows = (
                    session.execute(
                        vector_query,
                        {
                            "query_embedding": embedding_str,
                            "thr": threshold,
                            "lim": limit,
                        },
                    )
                    .mappings()
                    .all()
                )
                if rows:
                    logger.debug(f"Vector search found {len(rows)} results")
                    return [dict(r) for r in rows]

        except Exception as e:
            logger.warning(f"Vector search failed, falling back to text search: {e}")

        # Fallback to full-text search on conversations table
        ts_query = text(
            "\n".join(
                [
                    "SELECT id, conversation_id, sender, role, metadata, content, created_at,",
                    "       ts_rank_cd(to_tsvector('spanish', coalesce(content,'')),",
                    "                  plainto_tsquery('spanish', :q)) AS rank",
                    "FROM messages",
                    "WHERE content IS NOT NULL",
                    "  AND length(btrim(content)) > 0",
                    "  AND sender IN ('user','assistant')",
                    "  AND to_tsvector('spanish', coalesce(content,'')) @@",
                    "      plainto_tsquery('spanish', :q)",
                    "  AND ts_rank_cd(to_tsvector('spanish', coalesce(content,'')),",
                    "                   plainto_tsquery('spanish', :q)) >= :thr",
                    "ORDER BY rank DESC, created_at DESC",
                    "LIMIT :lim",
                ]
            )
        )
        rows = (
            session.execute(
                ts_query, {"q": query_text, "thr": threshold, "lim": limit}
            )
            .mappings()
            .all()
        )
        logger.debug(f"Text search found {len(rows)} results")
        return [dict(r) for r in rows]

    def _convert_to_vector_results(
        self, raw_results: List[Dict[str, Any]]
    ) -> List[VectorSearchResult]:
        """
        Convert raw database results to VectorSearchResult objects.

        Argumentos:
            raw_results: Raw database query results

        Retorna:
            List of structured VectorSearchResult objects
        """
        results: List[VectorSearchResult] = []
        for row in raw_results:
            score = float(row.get("rank", 0.0))
            if score < 0.0:
                score = 0.0
            if score > 1.0:
                score = 1.0
            row_metadata = row.get("metadata") or {}
            if isinstance(row_metadata, str):
                try:
                    row_metadata = json.loads(row_metadata)
                except Exception:
                    row_metadata = {}
            metadata_payload: Dict[str, Any] = {
                "source": row_metadata.get("source") or "document",
                "sender": row.get("sender"),
                "role": str(row.get("role")) if row.get("role") else None,
                "conversation_id": (
                    str(row.get("conversation_id")) if row.get("conversation_id") else None
                ),
                "created_at": str(row.get("created_at")),
            }
            if row_metadata:
                metadata_payload["document_metadata"] = row_metadata
            results.append(
                VectorSearchResult(
                    content=row.get("content", ""),
                    similarity_score=score,
                    metadata=metadata_payload,
                    source_id=str(row.get("id")),
                )
            )
        return results

    def _format_messages_for_context(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Format messages for conversation context.

        Argumentos:
            messages: Raw message data from database

        Retorna:
            Formatted messages for context use
        """
        # Order chronologically (oldest first)
        messages_sorted = sorted(messages, key=lambda m: m.get("created_at") or "")
        formatted: List[Dict[str, Any]] = []
        for m in messages_sorted:
            formatted.append(
                {
                    "id": str(m.get("id")),
                    "role": m.get("sender") or "unknown",
                    "content": m.get("content"),
                    "model": m.get("model"),
                    "metadata": m.get("metadata") or {},
                    "created_at": str(m.get("created_at")),
                }
            )
        return formatted

    def _convert_messages_to_vector_results(
        self, messages: List[Dict[str, Any]]
    ) -> List[VectorSearchResult]:
        """
        Convert conversation messages to VectorSearchResult format.

        Argumentos:
            messages: Message data with similarity scores

        Retorna:
            List of VectorSearchResult objects
        """
        results: List[VectorSearchResult] = []
        for row in messages:
            score = float(row.get("rank", 0.0))
            if score < 0.0:
                score = 0.0
            if score > 1.0:
                score = 1.0
            results.append(
                VectorSearchResult(
                    content=row.get("content", ""),
                    similarity_score=score,
                    metadata={
                        "sender": row.get("sender"),
                        "created_at": str(row.get("created_at")),
                    },
                    source_id=str(row.get("id")),
                )
            )
        return results

    async def _execute_conversation_creation(
        self, session: Session, conversation_id: str
    ) -> None:
        """
        Create a conversation using an existing user as owner.
        Uses the first available user to satisfy NOT NULL constraint.
        """
        # Pick any existing user to satisfy NOT NULL constraint
        user_stmt = text("SELECT id, email, role FROM app_users ORDER BY created_at ASC LIMIT 1")
        user_result = session.execute(user_stmt).first()
        if not user_result:
            raise ValueError(
                "Cannot create conversation: no users found to assign as owner"
            )
        
        user_id, user_email, user_role = user_result

        insert_stmt = text(
            """
            INSERT INTO app_conversations (
                id, user_id, title, status, started_at, updated_at, last_message_at
            ) VALUES (
                :id, :user_id, :title, :status, now(), now(), NULL
            )
            ON CONFLICT (id) DO NOTHING
            """
        )
        session.execute(
            insert_stmt,
            {
                "id": conversation_id,
                "user_id": user_id,
                "title": "Nueva conversación",
                "status": "open",
            },
        )
        
    async def _execute_message_insertion(
        self,
        session: Session,
        message_id: str,
        conversation_id: str,
        content: str,
        role: str,
        embedding: List[float],
    ) -> None:
        """
        Insert a message into messages table.
        """
        if embedding:
            embedding_value = f"[{','.join(map(str, embedding))}]"
        else:
            zeros = ["0"] * self.embedding_service.get_embedding_dimension()
            embedding_value = f"[{','.join(zeros)}]"

        insert_stmt = text(
            """
            INSERT INTO messages (
                id, conversation_id, sender, role, content, model, minio_key,
                minio_bucket, metadata, embedding, created_at, updated_at
            ) VALUES (
                :id, :conversation_id, :sender, CAST(:role AS public.chat_role), :content, :model,
                NULL, NULL, '{}'::jsonb, CAST(:embedding AS vector), now(), now()
            )
            """
        )
        session.execute(
            insert_stmt,
            {
                "id": message_id,
                "conversation_id": conversation_id,
                "sender": role,
                "role": role,
                "content": content,
                "model": None,
                "embedding": embedding_value,
            },
        )

    async def _update_conversation_timestamp(
        self, session: Session, conversation_id: str
    ) -> None:
        # Update conversation timestamps
        update_stmt = text(
            """
            UPDATE app_conversations
            SET last_message_at = now(), updated_at = now()
            WHERE id = :cid
            """
        )
        session.execute(update_stmt, {"cid": conversation_id})

    async def _execute_recent_messages_query(
        self, session: Session, conversation_id: str, limit: int
    ) -> List[Dict[str, Any]]:
        stmt = text(
            """
            SELECT id as id, sender, role, content, metadata, created_at
            FROM messages
            WHERE conversation_id = :cid
              AND content IS NOT NULL
              AND length(btrim(content)) > 0
              AND (role IN ('user'::public.chat_role, 'assistant'::public.chat_role) OR (role IS NULL AND sender IN ('user', 'assistant')))
            ORDER BY created_at DESC
            LIMIT :lim
            """
        )
        rows = (
            session.execute(stmt, {"cid": conversation_id, "lim": limit})
            .mappings()
            .all()
        )
        return [dict(r) for r in rows]

    async def _get_recent_message_ids(
        self, session: Session, conversation_id: str, limit: int
    ) -> List[str]:
        if limit <= 0:
            return []
        stmt = text(
            """
            SELECT id
            FROM messages
            WHERE conversation_id = :cid
              AND content IS NOT NULL
              AND length(btrim(content)) > 0
              AND (role IN ('user'::public.chat_role, 'assistant'::public.chat_role) OR (role IS NULL AND sender IN ('user', 'assistant')))
            ORDER BY created_at DESC
            LIMIT :lim
            """
        )
        rows = (
            session.execute(stmt, {"cid": conversation_id, "lim": limit})
            .scalars()
            .all()
        )
        return [str(r) for r in rows]

    async def _execute_conversation_vector_search(
        self,
        session: Session,
        conversation_id: str,
        query_embedding: List[float],
        recent_ids: List[str],
        limit: int,
        threshold: float,
    ) -> List[Dict[str, Any]]:
        base_sql = [
            "SELECT id as id,",
            "       COALESCE(role::text, sender) as sender,",
            "       role,",
            "       content,",
            "       metadata,",
            "       created_at,",
            "       (1 - (embedding <=> CAST(:query_embedding AS vector))) AS rank",
            "FROM messages",
            "WHERE conversation_id = CAST(:cid AS uuid)",
            "  AND embedding IS NOT NULL",
            "  AND content IS NOT NULL",
            "  AND length(btrim(content)) > 0",
            "  AND (role IN ('user'::public.chat_role, 'assistant'::public.chat_role)",
            "       OR (role IS NULL AND sender IN ('user', 'assistant')))",
            "  AND (1 - (embedding <=> CAST(:query_embedding AS vector))) >= :thr",
        ]

        params_values: Dict[str, Any] = {
            "cid": conversation_id,
            "query_embedding": f"[{','.join(map(str, query_embedding))}]",
            "thr": threshold,
            "lim": limit,
        }

        if recent_ids:
            from uuid import UUID

            valid_ids: List[str] = []
            for recent_id in recent_ids:
                try:
                    UUID(recent_id)
                    valid_ids.append(recent_id)
                except ValueError:
                    logger.warning(f"Invalid UUID in recent_ids for vector search: {recent_id}")

            if valid_ids:
                placeholders = ", ".join(
                    [f"CAST(:vec_id_{i} AS uuid)" for i in range(len(valid_ids))]
                )
                base_sql.append(f"  AND id NOT IN ({placeholders})")
                for i, valid_id in enumerate(valid_ids):
                    params_values[f"vec_id_{i}"] = valid_id

        base_sql.append("ORDER BY embedding <=> CAST(:query_embedding AS vector)")
        base_sql.append("LIMIT :lim")
        stmt = text("\n".join(base_sql))

        rows = session.execute(stmt, params_values).mappings().all()
        return [dict(r) for r in rows]

    async def _execute_conversation_semantic_search(
        self,
        session: Session,
        conversation_id: str,
        query: str,
        recent_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Fallback semantic search using PostgreSQL full-text search on content,
        excluding the most recent message IDs.
        """
        base_sql = "\n".join(
            [
                "SELECT id as id, COALESCE(role::text, sender) as sender, role, content, created_at,",
                "       ts_rank_cd(to_tsvector('spanish', coalesce(content,'')),",
                "                  plainto_tsquery('spanish', :q)) AS rank",
                "FROM messages",
                "WHERE conversation_id = :cid",
                "  AND content IS NOT NULL",
                "  AND length(btrim(content)) > 0",
                "  AND (role IN ('user'::public.chat_role, 'assistant'::public.chat_role) OR (role IS NULL AND sender IN ('user', 'assistant')))",
                "  AND to_tsvector('spanish', coalesce(content,'')) @@ ",
                "      plainto_tsquery('spanish', :q)",
                "  AND ts_rank_cd(to_tsvector('spanish', coalesce(content,'')),",
                "                   plainto_tsquery('spanish', :q)) >= :thr",
            ]
        )

        params_values: Dict[str, Any] = {
            "cid": conversation_id,
            "q": query,
            "thr": self.similarity_threshold,
            "lim": self.max_conversation_messages,
        }

        if recent_ids:
            # Validar y filtrar UUIDs válidos
            from uuid import UUID

            valid_ids = []
            for recent_id in recent_ids:
                try:
                    UUID(recent_id)  # Validar que es un UUID válido
                    valid_ids.append(recent_id)
                except ValueError:
                    logger.warning(f"Invalid UUID in recent_ids: {recent_id}")

            if valid_ids:
                # Usar CAST() para conversión segura
                placeholders = ", ".join(
                    [f"CAST(:id_{i} AS uuid)" for i in range(len(valid_ids))]
                )
                base_sql += f" AND id NOT IN ({placeholders})"

                # Agregar parámetros individuales
                for i, valid_id in enumerate(valid_ids):
                    params_values[f"id_{i}"] = valid_id

        base_sql += " ORDER BY rank DESC, created_at DESC LIMIT :lim"
        stmt = text(base_sql)

        rows = session.execute(stmt, params_values).mappings().all()
        return [dict(r) for r in rows]
