"""
Router de chat con streaming para interacciones conversacionales.

Proporciona endpoints para conversaciones en tiempo real con streaming SSE,
gestión de contexto y generación de respuestas usando el sistema multi-agente.
"""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import text
from sqlalchemy.orm import Session


from app.clients.neo4j_client import get_neo4j_driver
from app.config.databases import get_postgres_session
from app.extensions import get_logger


logger = get_logger(__name__)
router = APIRouter(prefix="/generation", tags=["chat"])


def safe_json_serialize(obj: Any) -> Any:
    """
    Serializa objetos de forma segura a formato compatible con JSON.

    Maneja objetos complejos como dicts, listas, modelos Pydantic y objetos
    con atributos, convirtiéndolos recursivamente a tipos JSON básicos.

    Argumentos:
        obj: Objeto a serializar

    Retorna:
        Objeto serializado compatible con JSON
    """
    if isinstance(obj, (dict, list, tuple, set)):
        if isinstance(obj, dict):
            return {k: safe_json_serialize(v) for k, v in obj.items()}
        return [safe_json_serialize(item) for item in obj]
    if hasattr(obj, "__dict__"):
        return {k: safe_json_serialize(v) for k, v in obj.__dict__.items()}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


def _convert_graph_results_safely(graph_results: Any) -> Dict[str, Any]:
    """
    Convierte graph_results (ahora GraphSearchResponse) a diccionario de forma segura.

    Maneja diferentes tipos de entrada y los convierte a un formato de diccionario
    consistente para uso en el contexto de generación.

    Argumentos:
        graph_results: Resultados de búsqueda en grafo (GraphSearchResponse o dict)

    Retorna:
        Diccionario con resultados de búsqueda en formato consistente
    """
    if not graph_results:
        logger.critical("_convert_graph_results_safely: graph_results es None/vacío")
        return {}

    logger.debug(f"_convert_graph_results_safely: graph_results tipo: {type(graph_results)}")

    if isinstance(graph_results, dict):
        # Ya es un diccionario
        return graph_results

    elif hasattr(graph_results, "results") and hasattr(graph_results, "total_found"):
        # Es un objeto GraphSearchResponse - acceso directo sin pérdida de datos
        # Acceder directamente a los atributos (SIN usar model_dump para evitar pérdida)
        results = graph_results.results if hasattr(graph_results, "results") else []
        if results and hasattr(results[0], "model_dump"):
            converted_results = [result.model_dump() for result in results]
        else:
            converted_results = results

        graph_data = {
            "results": converted_results,
            "total_found": graph_results.total_found,
            "search_strategy": getattr(graph_results, "search_strategy", "unknown"),
            "metadata": getattr(graph_results, "metadata", {}),
            "execution_time": getattr(graph_results, "execution_time", 0.0),
        }
        return graph_data

    else:
        # Fallback para tipos desconocidos
        try:
            if hasattr(graph_results, "model_dump"):
                return graph_results.model_dump()
            else:
                return dict(graph_results)
        except Exception as e:
            logger.error(f"No se pudo convertir graph_results tipo {type(graph_results)}: {e}")
            return {}


def _ensure_conversation(
    db: Session,
    conversation_id: str,
    *,
    title: str,
    metadata: Dict[str, Any],
) -> None:
    exists = db.execute(
        text("SELECT 1 FROM app_conversations WHERE id = :cid"),
        {"cid": conversation_id},
    ).scalar()
    if exists:
        return
    insert_stmt = text(
        """
        INSERT INTO app_conversations (
            id, user_id, title, status, started_at, updated_at, metadata
        )
        VALUES (:id, NULL, :title, 'open', now(), now(), CAST(:metadata AS jsonb))
        ON CONFLICT (id) DO NOTHING
        """
    )
    db.execute(
        insert_stmt,
        {
            "id": conversation_id,
            "title": title,
            "metadata": json.dumps(metadata or {}),
        },
    )
    db.commit()


# =========================
# Request/Response Models
# =========================


class TutelaRequest(BaseModel):
    """Flujo inicial: ejecución completa del grafo para un documento de tutela (sin SSE)."""

    tutela_text: str = Field(..., min_length=1, description="Texto completo de la tutela para responder")
    first_interaction: bool = Field(True, description="Debe ser True para el flujo inicial de tutela")
    conversation_id: Optional[str] = Field(
        None,
        description="ID de conversación opcional; se generará si no se proporciona",
    )


class TutelaResponse(BaseModel):
    """Salida que resume los resultados completos del grafo para responder tutela."""

    status: str
    conversation_id: str
    final_response: Dict[str, Any]
    pgvector_inserted: bool
    pertinence_valid: bool
    neo4j_context: Optional[Dict[str, Any]] = None
    execution_time: float


class ChatStreamRequest(BaseModel):
    """Flujo de chat SSE: actualizaciones incrementales del orquestador para un mensaje de usuario."""

    message: str = Field(..., min_length=1, max_length=1000, description="Mensaje del usuario")
    conversation_id: str = Field(..., description="ID de conversación existente")
    mode: str = Field(
        "chat",
        description="Modo de interacción elegido por el usuario: 'chat' (conversar) o 'edit' (editar borrador).",
    )

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, v: Any) -> str:
        if v is None:
            return "chat"
        mode = str(v).strip().lower()
        if mode == "edith":
            mode = "edit"
        if mode not in {"chat", "edit"}:
            raise ValueError("mode must be 'chat' or 'edit'")
        return mode


# =======================================
# 1) Flujo Inicial de Tutela (SIN SSE)
# =======================================


@router.post("/tutela", response_model=TutelaResponse)
async def run_tutela_full_graph(
    request: TutelaRequest,
    neo4j_driver=Depends(get_neo4j_driver),
    db: Session = Depends(get_postgres_session),
):
    """
    Ejecuta el flujo completo de generación de respuesta para una tutela inicial usando LangGraph.
    
    Este endpoint procesa el texto completo de una tutela y genera una respuesta oficial
    utilizando el sistema multi-agente con búsqueda en Neo4j y PgVector.
    
    **Características**:
    - Ejecuta el grafo completo sin streaming (SSE)
    - Genera embeddings y realiza búsquedas semánticas
    - Valida pertinencia mediante el orquestador
    
    **Argumentos**:
    - request: Datos de la tutela incluyendo texto completo y conversation_id opcional
    - neo4j_driver: Driver de Neo4j (inyectado)
    
    **Retorna**:
    - TutelaResponse con respuesta generada, metadata y contexto de Neo4j
    
    **Códigos de Estado**:
    - 200: Generación exitosa
    - 400: first_interaction debe ser True para este flujo
    - 500: Error en el workflow de generación
    """
    start_time = time.time()
    logger.info("=== TUTELA ENDPOINT CALLED ===")
    # logger.info(f"Request: {request}")

    # Enforce first interaction is True for this flow
    if request.first_interaction is not True:
        raise HTTPException(
            status_code=400,
            detail="For the initial tutela flow, 'first_interaction' must be True.",
        )

    try:
        # Lazy import to avoid circular deps
        # --- Normalize conversation_id at the API boundary
        import uuid as _uuid

        from app.graphs.generation_graph import make_graph

        def _normalize_cid(raw: Optional[str]) -> tuple[str, Optional[str]]:
            """Map any string to a UUID. Keep external id for UI/debug."""
            if not raw:
                return str(_uuid.uuid4()), None
            try:
                _uuid.UUID(str(raw))
                return str(raw), None  # already a UUID
            except Exception:
                # stable UUID from human-readable id
                return str(_uuid.uuid5(_uuid.NAMESPACE_URL, str(raw))), str(raw)

        normalized_cid, external_cid = _normalize_cid(request.conversation_id)
        workflow_id = str(uuid.uuid4())

        logger.info(f"Starting tutela flow with conversation_id: {normalized_cid}")

        _ensure_conversation(
            db,
            normalized_cid,
            title="Tutela (inicial)",
            metadata={
                "source": "generation_tutela",
                "external_conversation_id": external_cid,
            },
        )

        # Build initial state for the graph (use normalized UUID)
        # Sin heurísticas: directivas explícitas para tutela
        initial_state: Dict[str, Any] = {
            "workflow_id": workflow_id,
            "conversation_id": normalized_cid,
            "external_conversation_id": external_cid,  # optional, for UI/logs
            # Usa el texto real de la tutela como consulta base
            "user_query": (
                "Redacta un borrador de respuesta oficial a la accion de tutela "
                "con base en el documento adjunto. Responde en formato legal claro "
                "y sin comentarios adicionales."
            ),
            "timestamp": time.time(),
            "flow_mode": "tutela_init",
            "is_first_interaction": True,
            "generation_request": {
                "operation": "compose",
                "goal": "draft_official_response",
                "base_text": request.tutela_text,
                # En el arranque queremos persistir y recuperar contexto conversacional
                "plan": {"use_neo4j_context": True, "use_conversation_memory": True},
                "constraints": {},
                # Más espacio de salida + auto-continue para evitar cortes por longitud
                "params": {
                    "max_tokens": 10000,
                    "temperature": 0.15,
                    "allow_auto_continue": True,
                    "auto_continue_max_steps": 2,
                },
            },
            # Timeout mayor para generaciones largas del borrador inicial
            "config": {"timeouts": {"generation": 420000}},
            # Inicialización explícita de campos de estado para evitar None
            "pgvector_write_status": {},
            "pgvector_retrieval_status": {},
            "neo4j_status": {},
            "generation_status": {},
        }

        # Execute the graph to completion (no SSE)
        print("Creating graph and executing workflow")
        logger.info("Creating graph and executing workflow")
        logger.info(f" Router: Neo4j driver available: {neo4j_driver is not None}")
        if neo4j_driver:
            logger.info(f" Router: Driver type: {type(neo4j_driver).__name__}")
        logger.info("Creating graph and executing workflow")
        # Log seguro: evita acceder claves inexistentes y no imprime payloads grandes
        logger.info(
            "Initial state: %s",
            {
                "workflow_id": initial_state.get("workflow_id"),
                "conversation_id": initial_state.get("conversation_id"),
                "external_conversation_id": initial_state.get("external_conversation_id"),
                "flow_mode": initial_state.get("flow_mode"),
                "goal": (initial_state.get("generation_request") or {}).get("goal"),
            },
        )

        logger.info(" Router: About to call make_graph...")
        graph = make_graph(neo4j_driver=neo4j_driver)
        # Configure the graph execution with required checkpointer parameters and recursion limit
        # Raise recursion limit temporarily to aid debugging (will not loop once id is valid)
        config = {
            "configurable": {"thread_id": normalized_cid},
            "recursion_limit": 30,  # Increased temporarily for debugging
        }
        logger.info("About to invoke graph")

        final_state: Dict[str, Any] = await graph.ainvoke(initial_state, config=config)  # type: ignore[attr-defined]
        

        #  VALIDACIÓN: Verificar que final_state no sea None
        if final_state is None:
            error_msg = "Graph execution returned None in non-SSE path"
            logger.error(error_msg, extra={"conversation_id": normalized_cid})
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "graph_execution_failed",
                    "message": error_msg,
                    "conversation_id": normalized_cid,
                },
            )

        logger.info(f"Graph execution completed. Final state keys: {list(final_state.keys())}")
        logger.info(f"Final state: {final_state}")
        print(f"Final state keys: {list(final_state.keys())}")

        # Extract fields with robust fallbacks
        final_response_raw = final_state.get("final_response", "")
        if isinstance(final_response_raw, dict):
            final_response = final_response_raw
        elif isinstance(final_response_raw, str):
            final_response = {"content": final_response_raw}
        else:
            final_response = {"content": ""}

        final_conversation_id = final_state.get("conversation_id", normalized_cid)

        # PgVector insertion status (try various known keys)
        pgv_status = final_state.get("pgvector_write_status") or {}
        if not pgv_status:
            pgv_status = (final_state.get("workflow_metadata") or {}).get("pgvector_write_status") or {}
        pgvector_inserted = bool(pgv_status.get("success") or final_state.get("pgvector_write_ok") or final_state.get("pgvector_inserted"))

        print("=== PGVECTOR INSERTION STATUS ANALYSIS ===")
        print(f"pgvector_write_status: {pgv_status}")
        print(f"pgvector_write_ok: {final_state.get('pgvector_write_ok')}")
        print(f"pgvector_inserted: {final_state.get('pgvector_inserted')}")
        print(f"Final pgvector_inserted: {pgvector_inserted}")

        # Orchestrator pertinence validation
        orch_res = final_state.get("orchestrator_result") or final_state.get("orchestrator_decision") or {}
        pertinence_valid = bool(orch_res.get("is_pertinent") or orch_res.get("pertinence_valid") or orch_res.get("validated_pertinence"))

        # Neo4j context (prefer 'graph_results'; fallback to 'legal_context')
        neo4j_context_obj = final_state.get("graph_results") or final_state.get("legal_context")
        neo4j_context = _convert_graph_results_safely(neo4j_context_obj) if neo4j_context_obj else None

        execution_time = time.time() - start_time
        status = "success" if final_response.get("content") else "partial_success"

        logger.info(f"Constructing response: status={status}, conversation_id={final_conversation_id}, pgvector_inserted={pgvector_inserted}")

        return TutelaResponse(
            status=status,
            conversation_id=final_conversation_id,
            final_response=final_response,
            pgvector_inserted=pgvector_inserted,
            pertinence_valid=pertinence_valid,
            neo4j_context=neo4j_context,
            execution_time=execution_time,
        )

    except Exception as e:
        logger.exception("Tutela flow failed")
        error_info = _classify_generation_error(e)
        raise HTTPException(
            status_code=error_info.get("status_code", 500),
            detail={
                "error": error_info.get("code", "tutela_workflow_failed"),
                "message": error_info.get("message"),
                "detail": error_info.get("detail"),
                "retryable": error_info.get("retryable", True),
                "conversation_id": request.conversation_id,
            },
        )


# =======================================
# SSE Chat Flow
# =======================================
@router.post("/chat/stream")
async def chat_stream(
    request: ChatStreamRequest,
    neo4j_driver=Depends(get_neo4j_driver),
    db: Session = Depends(get_postgres_session),
):
    """
    Chat con streaming SSE (Server-Sent Events) para interacciones conversacionales.
    
    Este endpoint procesa un mensaje del usuario dentro de una conversación existente
    y emite actualizaciones en tiempo real del workflow de generación usando LangGraph.
    El modo (chat vs edit) lo decide explícitamente el usuario vía `mode` en el request,
    para que el front pueda alternarlo con un botón.
    
    **Eventos SSE emitidos**:
    - `start`: Inicio del workflow
    - `orchestrator`: Decisión del orquestador (goal, operation, context_sources)
    - `node`: Actualización de nodos (pgvector, neo4j, generation)
    - `complete`: Finalización con respuesta completa y metadata
    - `warn`: Advertencias durante la ejecución
    - `error`: Errores durante la ejecución
    
    **Argumentos**:
    - request: Mensaje del usuario y conversation_id
    - neo4j_driver: Driver de Neo4j (inyectado)
    
    **Retorna**:
    - Streaming SSE con eventos en tiempo real
    
    **Códigos de Estado**:
    - 200: Streaming iniciado
    - 400: Datos inválidos
    - 500: Error en el workflow
    
    **Notas**:
    - `mode=edit` fuerza operación de edición (no hay clasificación automática)
    - Se crea una nueva versión de borrador si se genera contenido
    """

    start_time = time.time()

    async def event_stream():
        # Local fallback to avoid UnboundLocalError in except
        safe_conversation_id = request.conversation_id
        try:
            # 1) Normaliza conversation_id (mismo helper que en /tutela)
            import uuid as _uuid

            def _normalize_cid(raw: Optional[str]) -> tuple[str, Optional[str]]:
                if not raw:
                    return str(_uuid.uuid4()), None
                try:
                    _uuid.UUID(str(raw))
                    return str(raw), None
                except Exception:
                    return str(_uuid.uuid5(_uuid.NAMESPACE_URL, str(raw))), str(raw)

            logger.info(f" Chat stream: Neo4j driver available: {neo4j_driver is not None}")
            if neo4j_driver:
                logger.info(f" Chat stream: Driver type: {type(neo4j_driver).__name__}")

            normalized_cid, external_cid = _normalize_cid(request.conversation_id)
            safe_conversation_id = normalized_cid

            _ensure_conversation(
                db,
                normalized_cid,
                title="Chat",
                metadata={
                    "source": "generation_chat_stream",
                    "external_conversation_id": external_cid,
                },
            )

            #  USAR EL GRAFO CORRECTAMENTE
            from app.graphs.generation_graph import make_graph

            graph = make_graph(neo4j_driver=neo4j_driver)

            
            draft_info = {
                "has_draft_history": False,
                "latest_draft": None,
                "draft_count": 0,
                "latest_draft_id": None,
                "conversation_metrics": {},
            }

            # Proporcionar contexto REAL para el LLM (p.ej. borradores existentes)
            # La decisión de modo (chat vs edit) viene del request (no del orquestador).
            user_mode = request.mode
            generation_request = {
                "operation": user_mode,
                "goal": ("edit_draft" if user_mode == "edit" else "direct_answer"),
                "plan": {
                    "use_conversation_memory": True,
                    "use_neo4j_context": user_mode == "edit",
                },
                "params": {
                    "max_tokens": 8000 if user_mode == "edit" else 6000,
                    "temperature": 0.15 if user_mode == "edit" else 0.2,
                },
            }

            initial_state = {
                "workflow_id": str(uuid.uuid4()),
                "conversation_id": normalized_cid,
                "external_conversation_id": external_cid,
                "user_query": request.message,
                "timestamp": time.time(),
                "flow_mode": "chat_edit",
                "is_first_interaction": False,
                # Especificar generation_request explícito: el usuario elige el modo.
                "generation_request": generation_request,
                # Proporcionar contexto para la clasificación
                "conversation_context": {
                    "has_draft_history": draft_info["has_draft_history"],
                    "latest_draft": draft_info["latest_draft"],
                    "draft_count": draft_info["draft_count"],
                    "latest_draft_id": draft_info["latest_draft_id"],
                    "conversation_metrics": draft_info.get("conversation_metrics", {}),
                },
                # Inicialización de campos de estado
                "pgvector_write_status": {},
                "pgvector_retrieval_status": {},
                "neo4j_status": {},
                "generation_status": {},
                "control": {"attempts": {}},  # Para el grafo
            }

            logger.info(
                "ROUTER: Iniciando flujo chat_edit con modo explícito",
                extra={
                    "conversation_id": normalized_cid,
                    "external_conversation_id": external_cid,
                    "message_preview": request.message[:100] + "..." if len(request.message) > 100 else request.message,
                    "message_length": len(request.message),
                    "flow_mode": "chat_edit",
                    "has_generation_request": "generation_request" in initial_state,
                    "user_mode": user_mode,
                    "conversation_context": initial_state.get("conversation_context", {}),
                },
            )

            # start event
            start_data = {
                "workflow_id": initial_state["workflow_id"],
                "conversation_id": normalized_cid,
            }
            yield "event: start\n" + f"data: {json.dumps(start_data)}\n\n"

            logger.info(
                "ROUTER: Enviando estado inicial al orchestrator",
                extra={
                    "conversation_id": normalized_cid,
                    "workflow_id": initial_state["workflow_id"],
                    "state_keys": list(initial_state.keys()),
                    "orchestrator_will_classify": False,
                    "user_mode": user_mode,
                },
            )

            # Configurar el grafo
            config = {
                "configurable": {"thread_id": normalized_cid},
                "recursion_limit": 30,
            }

            #  USAR STREAMING DEL GRAFO
            try:
                # Helper para detectar fuentes usadas
                def _detect_sources(st: Dict[str, Any]) -> tuple[bool, bool]:
                    gr = st.get("graph_results")
                    vr = st.get("vector_results") or []
                    uses_neo = False
                    if gr is not None:
                        if hasattr(gr, "total_found"):
                            try:
                                uses_neo = (getattr(gr, "total_found") or 0) > 0
                            except Exception:
                                uses_neo = False
                        elif isinstance(gr, dict):
                            uses_neo = bool(gr.get("total_found", 0) > 0 or gr.get("entities") or gr.get("results") or gr.get("relationships") or gr.get("insights"))
                    uses_vec = bool(vr and len(vr) > 0)
                    return uses_neo, uses_vec

                #  EJECUTAR EL GRAFO CON STREAMING
                current_node = None
                final_state = None
                graph_execution_error = None

                print("=" * 80)
                logger.info(f"Iniciando stream del grafo - conversation_id: {normalized_cid}, workflow_id: {initial_state['workflow_id']}")
                print("=" * 80)

                # Usar astream para obtener estados intermedios
                try:
                    async for state_update in graph.astream(initial_state, config=config):
                        print(f" state_update recibido: {list(state_update.keys())}")
                        # state_update es un dict con el nodo como clave y el estado como valor
                        for node_name, node_state in state_update.items():
                            print(f" Procesando nodo: {node_name}")
                            if node_name != current_node:
                                # Nuevo nodo iniciado
                                if current_node:
                                    # Completar nodo anterior
                                    node_data = {
                                        "node": current_node,
                                        "status": "completed",
                                        "timestamp": time.time(),
                                    }
                                    yield f"event: node\ndata: {json.dumps(node_data)}\n\n"

                                # Iniciar nuevo nodo
                                current_node = node_name
                                node_display_name = {
                                    "orchestrator_plan": "orchestrator",
                                    "pgvector": "pgvector",
                                    "neo4j": "neo4j",
                                    "generation": "generation",
                                    "finalize": "finalize",
                                }.get(node_name, node_name)

                                node_data = {
                                    "node": current_node,
                                    "status": "completed",
                                    "timestamp": time.time(),
                                }
                                yield f"event: node\ndata: {json.dumps(node_data)}\n\n"

                            # Actualizar estado final
                            #  VALIDACIÓN: Verificar que node_state no sea None
                            if node_state is None:
                                print(f" ALERTA: Nodo '{node_name}' devolvió None state")
                                logger.error(
                                    f"Node '{node_name}' returned None state",
                                    extra={
                                        "node_name": node_name,
                                        "conversation_id": normalized_cid,
                                    },
                                )
                            else:
                                print(f" Nodo '{node_name}' devolvió state válido")
                                if isinstance(node_state, dict):
                                    print(f"   Keys en node_state: {list(node_state.keys())}")
                                final_state = node_state
                except Exception as stream_error:
                    graph_execution_error = stream_error
                    print("=" * 80)
                    print(f" ERROR EN STREAM: {stream_error}")
                    print(f"   Tipo: {type(stream_error).__name__}")
                    print(f"   Nodo actual: {current_node}")
                    print("=" * 80)
                    logger.error(
                        f"Error during graph streaming: {stream_error}",
                        extra={
                            "error_type": type(stream_error).__name__,
                            "current_node": current_node,
                            "conversation_id": normalized_cid,
                        },
                    )
                    warn_evt = _build_error_payload(stream_error, normalized_cid)
                    warn_evt["warning"] = "graph_stream_failed"
                    warn_evt["phase"] = current_node or "unknown"
                    yield "event: warn\n" + f"data: {json.dumps(warn_evt)}\n\n"
                    # Don't re-raise here, let the validation below handle it

                print("=" * 80)
                print(" STREAM DEL GRAFO COMPLETADO")
                print(f"   final_state es None: {final_state is None}")
                print(f"   Último nodo: {current_node}")
                if final_state and isinstance(final_state, dict):
                    print(f"   Keys en final_state: {list(final_state.keys())}")
                print("=" * 80)

                # Completar último nodo
                if current_node:
                    node_display_name = {
                        "orchestrator_plan": "orchestrator",
                        "pgvector": "pgvector",
                        "neo4j": "neo4j",
                        "generation": "generation",
                        "finalize": "finalize",
                    }.get(current_node, current_node)

                    node_data = {
                        "node": node_display_name,
                        "status": "started",
                        "timestamp": time.time(),
                    }
                    yield f"event: node\ndata: {json.dumps(node_data)}\n\n"

                # Si no obtuvimos estado final del streaming, ejecutar una vez más
                if not final_state:
                    final_state = await graph.ainvoke(initial_state, config=config)

                #  VALIDACIÓN CRÍTICA: Verificar que final_state no sea None
                if final_state is None:
                    error_details = {
                        "conversation_id": normalized_cid,
                        "workflow_id": initial_state["workflow_id"],
                        "last_node": current_node,
                    }
                    if graph_execution_error:
                        error_msg = f"Graph execution failed: {graph_execution_error}"
                        error_details["streaming_error"] = str(graph_execution_error)
                        error_details["error_type"] = type(graph_execution_error).__name__
                    else:
                        error_msg = "Graph execution returned None - no state yielded by any node"

                    logger.error(error_msg, extra=error_details)
                    raise RuntimeError(error_msg)

                # Log del estado final
                #  SEGURIDAD: Verificar que final_state no sea None antes de acceder
                if final_state is not None:
                    logger.info(
                        "Chat stream: Estado final después de la ejecución del grafo",
                        extra={
                            "final_state_keys": list(final_state.keys()),
                            "conversation_id": normalized_cid,
                            "workflow_id": initial_state["workflow_id"],
                        },
                    )
                else:
                    logger.error(
                        "Chat stream: final_state es None después de la ejecución del grafo",
                        extra={
                            "conversation_id": normalized_cid,
                            "workflow_id": initial_state["workflow_id"],
                            "last_node": current_node,
                        },
                    )

                # Extraer metadatos del estado final
                #  SEGURIDAD: Proteger contra generation_metadata siendo None
                if final_state:
                    generation_metadata_raw = final_state.get("generation_metadata", {})
                    # Asegurar que generation_metadata siempre sea un dict, nunca None
                    if isinstance(generation_metadata_raw, dict):
                        generation_metadata = generation_metadata_raw.copy()
                    else:
                        generation_metadata = {}
                else:
                    generation_metadata = {}
                orch_decision = final_state.get("orchestrator_decision", {}) if final_state else {}

                # Detectar fuentes utilizadas
                uses_neo4j, uses_pgvector = _detect_sources(final_state) if final_state else (False, False)

                # Actualizar metadata con información de fuentes
                if uses_neo4j:
                    generation_metadata["uses_neo4j"] = True
                    if "context_sources" not in generation_metadata:
                        generation_metadata["context_sources"] = []
                    if "neo4j" not in generation_metadata["context_sources"]:
                        generation_metadata["context_sources"].append("neo4j")

                if uses_pgvector:
                    generation_metadata["uses_pgvector"] = True
                    if "context_sources" not in generation_metadata:
                        generation_metadata["context_sources"] = []
                    if "pgvector" not in generation_metadata["context_sources"]:
                        generation_metadata["context_sources"].append("pgvector")

                # Emitir evento de orquestador completado
                #  MEJORA: Obtener goal y operation desde FUENTES PRIMARIAS con fallbacks robustos
                # Orden de prioridad:
                # 1. generation_request (fuente primaria - establecido por orchestrator)
                # 2. generation_metadata (puede tener valores preservados)
                # 3. Fallback por defecto

                gen_request = final_state.get("generation_request") if final_state else {}

                # Obtener goal con prioridad en generation_request
                actual_goal = None
                if gen_request and isinstance(gen_request, dict):
                    actual_goal = gen_request.get("goal")

                # Fallback a generation_metadata si no está en generation_request
                if not actual_goal and generation_metadata and isinstance(generation_metadata, dict):
                    actual_goal = generation_metadata.get("goal")

                # Último fallback
                if not actual_goal:
                    actual_goal = "direct_answer"

                # Obtener operation con la misma lógica de prioridad
                actual_operation = None
                if gen_request and isinstance(gen_request, dict):
                    actual_operation = gen_request.get("operation")

                # Fallback a generation_metadata
                if not actual_operation and generation_metadata and isinstance(generation_metadata, dict):
                    actual_operation = generation_metadata.get("operation")

                # Último fallback
                if not actual_operation:
                    actual_operation = "chat"

                #  PROTECCIÓN: Obtener goal de generation_request de forma segura
                print("=" * 80)
                print(" DEBUG: Obteniendo goal para logging")
                gen_req_for_log = final_state.get("generation_request") if final_state else None
                print(f"   gen_req_for_log tipo: {type(gen_req_for_log).__name__}")
                print(f"   gen_req_for_log es None: {gen_req_for_log is None}")
                goal_from_request = None
                if gen_req_for_log and isinstance(gen_req_for_log, dict):
                    goal_from_request = gen_req_for_log.get("goal")
                    print(f"   goal_from_request: {goal_from_request}")
                else:
                    print("   gen_req_for_log no es un dict válido, usando None")
                print("=" * 80)

                logger.info(
                    "Chat stream: Determined goal for complete event",
                    extra={
                        "goal_from_metadata": generation_metadata.get("goal") if generation_metadata else None,
                        "goal_from_request": goal_from_request,
                        "actual_goal_used": actual_goal,
                    },
                )

                orchestrator_payload = {
                    "node": "orchestrator",
                    "status": "completed",
                    "orchestrator_decision": safe_json_serialize(orch_decision),
                    "goal": actual_goal,  #  Usar el goal real desde fuente primaria
                    "flow_mode": "chat_edit",
                    "operation": actual_operation,  #  Usar operation desde fuente primaria
                    "generation_metadata": {
                        "uses_neo4j": generation_metadata.get("uses_neo4j", False) if generation_metadata else False,
                        "uses_pgvector": generation_metadata.get("uses_pgvector", False) if generation_metadata else False,
                        "context_sources": generation_metadata.get("context_sources", []) if generation_metadata else [],
                        "language": generation_metadata.get("language", "es") if generation_metadata else "es",
                        "operation": actual_operation,  #  Usar operation desde fuente primaria
                    },
                    "timestamp": time.time(),
                }

                yield ("event: orchestrator\n" + f"data: {json.dumps(orchestrator_payload)}\n\n")

                # Emit pgvector completion event if it was used
                #  SEGURIDAD: Verificar final_state antes de acceder
                if final_state and final_state.get("vector_results"):
                    yield ("event: node\n" + f"data: {json.dumps({'node': 'pgvector', 'status': 'completed', 'timestamp': time.time()})}\n\n")

                # Emit neo4j completion event with detailed results
                #  SEGURIDAD: Verificar final_state antes de acceder
                graph_results = final_state.get("graph_results") if final_state else None

                # DIAGNÓSTICO CRÍTICO: Verificar el objeto graph_results antes de construir el evento neo4j
                logger.critical(" CONSTRUYENDO EVENTO NEO4J")
                if graph_results:
                    logger.critical(f" graph_results está presente para evento neo4j, tipo: {type(graph_results)}")
                    if hasattr(graph_results, "total_found"):
                        logger.critical(f" graph_results.total_found: {graph_results.total_found}")
                        logger.critical(f" graph_results.has_results: {graph_results.has_results}")
                    elif isinstance(graph_results, dict):
                        logger.critical(f" graph_results keys: {list(graph_results.keys())}")
                        if "total_found" in graph_results:
                            logger.critical(f" graph_results['total_found']: {graph_results['total_found']}")
                else:
                    logger.critical(" graph_results NO está presente para evento neo4j")

                if graph_results:
                    # Extract key information from graph results
                    neo4j_payload = {
                        "node": "neo4j",
                        "status": "completed",
                        "timestamp": time.time(),
                    }

                    # Add graph analysis results if available
                    # SOLUCION: Usar función helper para conversion segura sin pérdida de datos
                    graph_data = _convert_graph_results_safely(graph_results)

                    # DIAGNÓSTICO: Log de estructura de graph_results
                    logger.info(f" Neo4j graph_results structure: {type(graph_results)}")
                    if hasattr(graph_results, "model_dump"):
                        logger.info(f" Neo4j graph_results keys: {list(graph_results.model_dump().keys())}")
                    elif isinstance(graph_results, dict):
                        logger.info(f" Neo4j graph_results keys: {list(graph_results.keys())}")

                    # Adaptación para GraphSearchResponse
                    if hasattr(graph_results, "total_found") and hasattr(graph_results, "results"):
                        # Es un objeto GraphSearchResponse
                        results = graph_results.results if hasattr(graph_results, "results") else []
                        total_found = graph_results.total_found if hasattr(graph_results, "total_found") else 0
                        search_strategy = graph_results.search_strategy if hasattr(graph_results, "search_strategy") else "unknown"

                        logger.critical(f" GraphSearchResponse detectado: total_found={total_found}, results_count={len(results)}")

                        # Extraer entidades de los resultados
                        entities = []
                        for result in results:
                            if hasattr(result, "model_dump"):
                                result_dict = result.model_dump()
                            else:
                                result_dict = result if isinstance(result, dict) else {}

                            # Añadir como entidad
                            entity = {
                                "id": result_dict.get("id", ""),
                                "tipo": result_dict.get("tipo", ""),
                                "nombre": result_dict.get("nombre", ""),
                            }
                            entities.append(entity)

                        logger.critical(f" Entidades extraídas: {entities}")

                        neo4j_payload["graph_analysis"] = {
                            "entities_count": total_found,
                            "entities_preview": entities[:5],  # Primeras 5 entidades
                            "insights_count": 0,  # No disponible en GraphSearchResponse
                            "insights_preview": [],
                            "relationships_count": 0,  # No disponible en GraphSearchResponse
                            "has_legal_precedents": False,  # No disponible en GraphSearchResponse
                            "search_strategy": search_strategy,
                        }

                        logger.critical(f" graph_analysis construido: {neo4j_payload['graph_analysis']}")

                    elif graph_data:
                        # Formato antiguo con entities, insights, relationships
                        entities = graph_data.get("entities", [])
                        insights = graph_data.get("insights", [])
                        relationships = graph_data.get("relationships", [])

                        neo4j_payload["graph_analysis"] = {
                            "entities_count": len(entities),
                            "entities_preview": [
                                {
                                    "id": entity.get("id", ""),
                                    "labels": entity.get("labels", []),
                                    "properties": {
                                        "name": entity.get("properties", {}).get("name", ""),
                                        "title": entity.get("properties", {}).get("title", ""),
                                    },
                                }
                                for entity in entities[:5]  # First 5 entities
                            ],
                            "insights_count": len(insights),
                            "insights_preview": insights[:3] if isinstance(insights, list) else [],
                            "relationships_count": len(relationships),
                            "has_legal_precedents": bool(graph_data.get("precedents")),
                            "search_strategy": graph_data.get("search_strategy", "unknown"),
                        }

                    yield ("event: node\n" + f"data: {json.dumps(safe_json_serialize(neo4j_payload))}\n\n")

                # Emit generation completion event with metadata
                #  SEGURIDAD: Verificar final_state antes de acceder a keys()
                if final_state is not None:
                    print(f" final_state: {final_state.keys()}")
                else:
                    print(" final_state: None (no state available)")

                if final_state and final_state.get("final_response"):
                    #  SEGURIDAD: Proteger contra generation_metadata siendo None
                    generation_metadata_raw = final_state.get("generation_metadata", {})
                    # Asegurar que generation_metadata siempre sea un dict, nunca None
                    if isinstance(generation_metadata_raw, dict):
                        generation_metadata = generation_metadata_raw.copy()
                    else:
                        generation_metadata = {}

                    uses_neo4j, uses_pgvector = _detect_sources(final_state)

                    # Actualizar metadata con la información de Neo4j
                    if uses_neo4j:
                        generation_metadata["uses_neo4j"] = True
                        if "context_sources" not in generation_metadata:
                            generation_metadata["context_sources"] = []
                        if "neo4j" not in generation_metadata["context_sources"]:
                            generation_metadata["context_sources"].append("neo4j")

                    # Verificar si hay resultados de PgVector para actualizar uses_pgvector
                    if uses_pgvector:
                        generation_metadata["uses_pgvector"] = True
                        if "context_sources" not in generation_metadata:
                            generation_metadata["context_sources"] = []
                        if "pgvector" not in generation_metadata["context_sources"]:
                            generation_metadata["context_sources"].append("pgvector")

                    # Actualizar context_sources_count
                    generation_metadata["context_sources_count"] = len(generation_metadata.get("context_sources", []))

                    # Emitir detalles de generación si están disponibles
                    if generation_metadata:
                        generation_payload = {
                            "node": "generation",
                            "status": "completed",
                            "generation_details": {
                                "goal": actual_goal,  #  Usar goal desde fuente primaria
                                "mode": generation_metadata.get("mode", "chat_edit"),
                                "operation": actual_operation,  #  Usar operation desde fuente primaria
                                "response_length": generation_metadata.get("response_length", 0),
                                "language": generation_metadata.get("language", "es"),
                                "uses_neo4j": generation_metadata.get("uses_neo4j", False),
                                "uses_pgvector": generation_metadata.get("uses_pgvector", False),
                                "context_sources": generation_metadata.get("context_sources", []),
                                "context_sources_count": generation_metadata.get("context_sources_count", 0),
                                "sections_present": generation_metadata.get("sections_present", []),
                                "sections_missing": generation_metadata.get("sections_missing", []),
                                "citations_count": generation_metadata.get("citations_count", 0),
                                "elapsed_s": generation_metadata.get("elapsed_s", 0),
                            },
                            "timestamp": time.time(),
                        }

                        yield ("event: node\n" + f"data: {json.dumps(generation_payload)}\n\n")
                        print(f" generation_payload: {generation_payload}")

                # Extraer respuesta final
                print("=" * 80)
                print(" CONSTRUYENDO final_response")
                final_response_raw = final_state.get("final_response", "") if final_state else ""
                print(f"   final_response_raw tipo: {type(final_response_raw).__name__}")
                print(f"   final_response_raw es None: {final_response_raw is None}")
                if isinstance(final_response_raw, dict):
                    print(f"   Es dict con keys: {list(final_response_raw.keys())}")
                    final_response = final_response_raw
                elif isinstance(final_response_raw, str):
                    print(f"   Es string de longitud: {len(final_response_raw)}")
                    final_response = {"content": final_response_raw}
                else:
                    print("   Tipo no reconocido, usando fallback")
                    final_response = {"content": ""}
                print(f"   final_response FINAL: {type(final_response).__name__}")
                print(f"   final_response es None: {final_response is None}")
                if final_response and isinstance(final_response, dict):
                    print(f"   final_response keys: {list(final_response.keys())}")
                print("=" * 80)

                safe_conversation_id = final_state.get("conversation_id", request.conversation_id) if final_state else request.conversation_id

                logger.info(f"Chat stream: Finalizando con uses_neo4j={uses_neo4j}, uses_pgvector={uses_pgvector}")

                # Extraer información de la versión creada si existe
                #  SEGURIDAD: Verificar final_state antes de acceder
                pgvector_write_status = final_state.get("pgvector_write_status") if final_state else {}
                if not pgvector_write_status:
                    pgvector_write_status = {}

                #  DIAGNÓSTICO DETALLADO
                logger.critical(
                    " GENERATION ROUTER - Extrayendo draft_version info",
                    extra={
                        "has_pgvector_write_status": bool(pgvector_write_status),
                        "pgvector_write_status_keys": list(pgvector_write_status.keys()) if pgvector_write_status else [],
                        "pgvector_write_status": pgvector_write_status,
                        "has_success": pgvector_write_status.get("success") if pgvector_write_status else None,
                        "has_version": pgvector_write_status.get("version") if pgvector_write_status else None,
                        "final_state_keys": list(final_state.keys()) if final_state else [],
                    },
                )

                draft_version_info = {}
                
                # Determinar el kind del mensaje para el render_mode
                message_kind = pgvector_write_status.get("kind", "chat_response")
                
                # render_mode indica al frontend cómo mostrar la respuesta:
                # - "draft_update": Actualizar el panel del borrador (initial_draft, revision)
                # - "chat_message": Mostrar como mensaje de chat (chat_response)
                is_draft_update = message_kind in ("initial_draft", "revision")
                render_mode = "draft_update" if is_draft_update else "chat_message"
                
                print("=" * 60)
                print("🎨 CHAT_ROUTER: DETERMINANDO RENDER_MODE PARA FRONTEND")
                print(f"   pgvector_write_status.kind: {message_kind}")
                print(f"   is_draft_update: {is_draft_update}")
                print(f"   render_mode: {render_mode}")
                if render_mode == "draft_update":
                    print("   ✅ FRONTEND: Debe actualizar panel del BORRADOR")
                else:
                    print("   💬 FRONTEND: Debe mostrar como mensaje de CHAT")
                print("=" * 60)
                
                if pgvector_write_status.get("success") and pgvector_write_status.get("version"):
                    draft_version_info = {
                        "version": pgvector_write_status.get("version"),
                        "message_id": pgvector_write_status.get("message_id"),
                        "created_at": pgvector_write_status.get("created_at"),
                        "kind": message_kind,
                    }
                    logger.info(
                        "Chat stream: Including draft version info in complete event",
                        extra={
                            "version": draft_version_info.get("version"),
                            "message_id": draft_version_info.get("message_id"),
                            "kind": message_kind,
                            "render_mode": render_mode,
                        },
                    )
                    logger.critical(
                        " GENERATION ROUTER - draft_version_info construido exitosamente",
                        extra={"draft_version_info": draft_version_info, "render_mode": render_mode},
                    )
                else:
                    logger.critical(
                        " GENERATION ROUTER - NO SE CONSTRUYÓ draft_version_info (puede ser chat_response)",
                        extra={
                            "reason": "Missing success or version in pgvector_write_status",
                            "message_kind": message_kind,
                            "render_mode": render_mode,
                            "pgvector_write_status": pgvector_write_status,
                        },
                    )

                # Emitir evento de completado
                #  Usar el mismo actual_goal determinado anteriormente
                
                # Validación segura para el status
                has_content = False
                if final_response:
                    print("   final_response NO es None")
                    if isinstance(final_response, dict):
                        print("   final_response es dict")
                        has_content = bool(final_response.get("content"))
                        print(f"   has_content: {has_content}")
                    else:
                        print(f"   final_response NO es dict, es: {type(final_response).__name__}")
                else:
                    print("   final_response ES None")

                payload = {
                    "status": "success" if has_content else "partial_success",
                    "conversation_id": safe_conversation_id,
                    "final_response": final_response if final_response else {"content": ""},
                    "execution_time": time.time() - start_time,
                    # render_mode indica al frontend cómo mostrar la respuesta:
                    # - "draft_update": Actualizar el panel del borrador (initial_draft, revision)
                    # - "chat_message": Mostrar como mensaje de chat (chat_response)
                    "render_mode": render_mode,
                    "message_kind": message_kind,  # El kind del mensaje persistido
                    "workflow_summary": {
                        "goal": actual_goal,  #  Usar el goal real desde fuente primaria
                        "flow_mode": "chat_edit",
                        "operation": actual_operation,  #  Usar operation desde fuente primaria
                        "context_sources_used": generation_metadata.get("context_sources", []) if generation_metadata else [],
                        "uses_neo4j": generation_metadata.get("uses_neo4j", False) if generation_metadata else False,
                        "uses_pgvector": generation_metadata.get("uses_pgvector", False) if generation_metadata else False,
                        "response_type": "draft" if is_draft_update else "conversational",
                    },
                }

                # Incluir información de la versión del draft si existe (solo para draft_update)
                if draft_version_info and is_draft_update:
                    payload["draft_version"] = draft_version_info

                print("   Payload construido exitosamente")
                print(f"   Keys en payload: {list(payload.keys())}")
                print("=" * 80)

                logger.critical(
                    " GENERATION ROUTER - Evento complete final",
                    extra={
                        "goal_in_payload": actual_goal,
                        "has_draft_version": bool(draft_version_info),
                        "draft_version": draft_version_info if draft_version_info else None,
                    },
                )

                if payload.get('render_mode') == 'draft_update':
                    print("   FRONTEND DEBE: Actualizar panel del BORRADOR")
                    if payload.get('draft_version'):
                        print(f"      version: {payload['draft_version'].get('version')}")
                else:
                    print("   FRONTEND DEBE: Mostrar como CHAT (no tocar borrador)")
                print("=" * 60)

                yield "event: complete\n" + f"data: {json.dumps(payload)}\n\n"

                
            except Exception as stream_err:
                
                import traceback

                print("   Traceback:")
                traceback.print_exc()
                print("=" * 80)

                logger.error(f"Error en ejecución del grafo: {stream_err}")
                warn_evt = _build_error_payload(stream_err, safe_conversation_id)
                warn_evt["warning"] = "graph_execution_error"
                warn_evt["phase"] = "complete_event"
                yield "event: warn\n" + f"data: {json.dumps(warn_evt)}\n\n"

        except Exception as e:
            
            import traceback

            print("   Traceback:")
            traceback.print_exc()
            print("=" * 80)

            err = _build_error_payload(e, safe_conversation_id)
            err["error"] = err.get("error", "chat_stream_failed")
            err["execution_time"] = time.time() - start_time
            yield "event: error\n" + f"data: {json.dumps(err)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# =========================
# Helper Functions
# =========================


def _classify_generation_error(error: Exception | str | None) -> Dict[str, Any]:
    """Mapea errores de generación comunes a códigos y mensajes amigables."""

    message = str(error) if error else "Error desconocido"
    lowered = message.lower()

    classification: Dict[str, Any] = {
        "code": "unknown_generation_error",
        "title": "Error durante la generación",
        "message": "No pudimos generar la respuesta. Intenta nuevamente.",
        "detail": message,
        "retryable": True,
        "status_code": 500,
    }

    def set_info(code: str, title: str, user_message: str, *, retryable: bool, status_code: int, help_text: Optional[str] = None) -> None:
        classification.update(
            {
                "code": code,
                "title": title,
                "message": user_message,
                "retryable": retryable,
                "status_code": status_code,
            }
        )
        if help_text:
            classification["help"] = help_text

    if isinstance(error, asyncio.TimeoutError) or "timeout" in lowered or "timed out" in lowered:
        set_info(
            "llm_timeout",
            "Tiempo de espera excedido",
            "El modelo tardó demasiado en responder.",
            retryable=True,
            status_code=504,
            help_text="Vuelve a intentar en unos segundos.",
        )
        return classification

    if "maximum context length" in lowered or ("context length" in lowered and "token" in lowered):
        set_info(
            "llm_context_limit",
            "Documento demasiado grande",
            "El texto enviado supera el límite del modelo.",
            retryable=False,
            status_code=400,
            help_text="Reduce el texto o divide la solicitud en partes más pequeñas.",
        )
        return classification

    if "rate limit" in lowered or "too many requests" in lowered or "server error 429" in lowered:
        set_info(
            "llm_rate_limited",
            "Modelo saturado",
            "El modelo recibió demasiadas solicitudes.",
            retryable=True,
            status_code=429,
            help_text="Espera unos segundos e inténtalo de nuevo.",
        )
        return classification

    connection_markers = (
        "connection refused",
        "failed to establish a new connection",
        "connection reset",
        "connection aborted",
        "not available",
    )
    if any(marker in lowered for marker in connection_markers):
        set_info(
            "llm_unavailable",
            "Modelo no disponible",
            "No pudimos conectarnos con el servicio de generación.",
            retryable=True,
            status_code=503,
            help_text="Verifica la conexión e inténtalo más tarde.",
        )
        return classification

    if "server error 500" in lowered or "server error 503" in lowered or "internal server error" in lowered:
        set_info(
            "llm_server_error",
            "Falla del modelo",
            "El proveedor del modelo devolvió un error interno.",
            retryable=True,
            status_code=502,
            help_text="Intenta nuevamente en unos minutos.",
        )
        return classification

    if "badrequesterror" in lowered or "server error 400" in lowered:
        set_info(
            "llm_bad_request",
            "Solicitud no válida",
            "El modelo rechazó la solicitud enviada.",
            retryable=False,
            status_code=400,
        )
        return classification

    return classification


def _build_error_payload(error: Exception | str | None, conversation_id: Optional[str] = None) -> Dict[str, Any]:
    """Construye un payload serializable para eventos SSE/HTTP de error."""

    info = _classify_generation_error(error)
    payload: Dict[str, Any] = {
        "error": info["code"],
        "title": info["title"],
        "message": info["message"],
        "detail": info.get("detail"),
        "retryable": info.get("retryable", True),
    }
    if conversation_id:
        payload["conversation_id"] = conversation_id
    if info.get("help"):
        payload["help"] = info["help"]
    if info.get("status_code"):
        payload["status_code"] = info["status_code"]
    return payload
