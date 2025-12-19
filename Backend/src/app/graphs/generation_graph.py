"""
Grafo de Generación LangGraph - Workflow Centrado en Orquestador.

Topología Hub-and-Spoke (Planificar + Supervisar + Finalizar):
- El orquestador planifica, supervisa reintentos por spoke y valida/finaliza.
- Los spokes (PgVector / Neo4j / Generation) no conocen el enrutamiento.
- Contadores de intentos en state["control"]["attempts"] evitan bucles infinitos.
"""

import asyncio
import time
from typing import Any, Dict, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RunnableConfig

from app.agents.generation_agent import GenerationAgent
from app.agents.neo4j_agent import Neo4jAgent
from app.agents.orchestrator_agent import OrchestratorAgent
from app.agents.pgvector_agent import PgVectorAgent

from app.config.settings import get_settings
from app.extensions import get_logger
from app.services.embeddings.embedding_service import get_embedding_service

logger = get_logger(__name__)

# ---- Constantes de nombres de nodos -------------------------------------------

# Nodos existentes (sin cambios)
NODE_ORCH_PLAN = "orchestrator_plan"
NODE_PGV = "pgvector"
NODE_SUP_PGV = "supervisor_pgvector"
NODE_NEO4J = "neo4j"
NODE_SUP_NEO4J = "supervisor_neo4j"
NODE_GEN = "generation"
NODE_FINAL = "finalize"

# Etiquetas de transición
# Transiciones existentes (sin cambios)
TR_TO_GEN = "generation"
TR_TO_PGV = "pgvector"
TR_TO_NEO4J = "neo4j"
TR_TO_FINAL = "finalize"
TR_RETRY = "retry"
TR_NEXT = "next"

# ---- Hard-stop de intentos ----------------------------------------------------

GEN_MAX_ATTEMPTS = 1  # generación: una sola pasada
PGV_MAX_ATTEMPTS = 1  # pgvector: un intento
NEO_MAX_ATTEMPTS = 1  # neo4j: un intento

# ---- Helpers de acceso seguro a estado ---------------------------------------


def is_mapping(obj: Any) -> bool:
    """
    Verifica si un objeto es un mapping (dict).

    Argumentos:
        obj: Objeto a verificar

    Retorna:
        True si es un dict, False en caso contrario
    """
    return isinstance(obj, dict)


def getf(state: Any, key: str, default: Any = None) -> Any:
    """
    Obtiene un valor del estado de forma segura.

    Soporta dicts y objetos con atributos.

    Argumentos:
        state: Estado a consultar
        key: Clave a obtener
        default: Valor por defecto si no se encuentra

    Retorna:
        Valor obtenido o default
    """
    if is_mapping(state):
        return state.get(key, default)
    try:
        return getattr(state, key)
    except Exception:
        return default


def setf(state: Any, key: str, value: Any) -> Any:
    """
    Establece un valor en el estado de forma segura.

    Soporta dicts y objetos Pydantic, mutando o retornando copia según el tipo.

    Argumentos:
        state: Estado a modificar
        key: Clave a establecer
        value: Valor a asignar

    Retorna:
        Estado modificado (puede ser una copia en caso de Pydantic)
    """
    if is_mapping(state):
        state[key] = value
        return state
    try:
        return state.model_copy(update={key: value}, deep=True)  # pydantic v2
    except Exception:
        try:
            setattr(state, key, value)
        except Exception:
            pass
        return state


def updatef(state: Any, patch: Dict[str, Any]) -> Any:
    """
    Actualiza el estado con un diccionario de cambios.

    Soporta dicts y objetos Pydantic, aplicando múltiples cambios a la vez.

    IMPORTANTE:
    - Filtra valores None del patch para evitar sobrescribir campos existentes
    - Hace merge profundo de diccionarios anidados específicos (generation_metadata, workflow_metadata, etc.)
      para preservar campos que no están en el patch

    Argumentos:
        state: Estado a actualizar
        patch: Diccionario con cambios a aplicar

    Retorna:
        Estado actualizado (puede ser una copia en caso de Pydantic)
    """
    if not patch:
        return state

    # Filtrar valores None del patch para preservar campos existentes
    # Solo actualizar campos que tienen valores explícitos (no None)
    filtered_patch = {k: v for k, v in patch.items() if v is not None}

    # Si después de filtrar no queda nada, retornar el estado original
    if not filtered_patch:
        return state

    if is_mapping(state):
        # MEJORA: Hacer merge profundo para diccionarios de metadata específicos
        # Esto previene que se sobrescriban campos como 'operation', 'goal', etc.
        metadata_keys = {'generation_metadata', 'workflow_metadata', 'orchestrator_decision', 'generation_status'}

        for key, value in filtered_patch.items():
            if key in metadata_keys and isinstance(value, dict):
                # Merge profundo: preservar campos existentes que no están en el patch
                existing = state.get(key, {}) or {}
                if isinstance(existing, dict):
                    # Crear nuevo dict con merge: existing primero, luego value sobrescribe
                    merged = {**existing, **value}
                    state[key] = merged
                    logger.debug(
                        f"updatef: Deep merge applied for '{key}'",
                        extra={
                            "key": key,
                            "existing_keys": list(existing.keys()) if existing else [],
                            "patch_keys": list(value.keys()),
                            "merged_keys": list(merged.keys()),
                        }
                    )
                else:
                    # Si existing no es dict, usar el nuevo valor directamente
                    state[key] = value
            else:
                # Para otros campos, actualización directa (comportamiento original)
                state[key] = value
        return state
    try:
        return state.model_copy(update=filtered_patch, deep=True)
    except Exception:
        for k, v in filtered_patch.items():
            try:
                setattr(state, k, v)
            except Exception:
                pass
        return state


# ---- Otros helpers ------------------------------------------------------------


def _get_resource(config: Optional[RunnableConfig], name: str, agent_default: Any = None) -> Any:
    if not config:
        return agent_default
    try:
        cfg = config.get("configurable", {})  # type: ignore[attr-defined]
        res = cfg.get("resources", {})
        return res.get(name, agent_default)
    except Exception:
        try:
            cfg = getattr(config, "configurable", {})  # type: ignore[attr-defined]
            res = getattr(cfg, "resources", {})
            return getattr(res, name, agent_default)
        except Exception:
            return agent_default


def _has_meaningful_neo4j_results(graph_results: Any) -> bool:
    if graph_results is None:
        return False
    if hasattr(graph_results, "total_found"):
        try:
            return (graph_results.total_found or 0) > 0
        except Exception:
            return False
    if isinstance(graph_results, dict):
        return bool(graph_results.get("total_found", 0) > 0 or graph_results.get("entities") or graph_results.get("relationships") or graph_results.get("insights") or graph_results.get("results"))
    return False


def _vector_present(vector_results: Any) -> bool:
    try:
        return bool(vector_results and len(vector_results) > 0)
    except Exception:
        return False


def _normalize_orch_flags(decision: Any) -> Dict[str, bool]:
    def _get(attr: str, default: bool = False) -> bool:
        try:
            return bool(getattr(decision, attr))
        except Exception:
            try:
                return bool(decision.get(attr, default))  # type: ignore[attr-defined]
            except Exception:
                return default

    return {
        "direct_generation": _get("direct_generation", False),
        "use_vector_search": _get("use_vector_search", False),
        "use_graph_analysis": _get("use_graph_analysis", False),
    }


def _control_map(state: Any) -> Dict[str, Any]:
    ctrl = getf(state, "control", {}) or {}
    if not isinstance(ctrl, dict):
        ctrl = dict(ctrl)
    return ctrl


def _should_retry_from_state(state: Any, agent_name: str) -> bool:
    ctrl = _control_map(state)
    try:
        return bool(ctrl.get("retry") and ctrl.get("retry_agent") == agent_name)
    except Exception:
        return False


def _retry_sleep_ms(state: Any) -> int:
    ctrl = _control_map(state)
    try:
        return int(ctrl.get("retry_sleep_ms") or 0)
    except Exception:
        return 0


def _generation_success(state: Any) -> bool:
    gs = getf(state, "generation_status", None)
    try:
        if isinstance(gs, dict):
            return bool(gs.get("success", False))
        return bool(getattr(gs, "success", False))
    except Exception:
        return False


def _control_attempts_map(state: Any) -> Dict[str, int]:
    ctrl = _control_map(state)
    attempts = ctrl.get("attempts") or {}
    if not isinstance(attempts, dict):
        attempts = dict(attempts)
    out: Dict[str, int] = {}
    for k, v in attempts.items():
        try:
            out[k] = int(v)
        except Exception:
            out[k] = 0
    return out


def _get_attempts(state: Any, agent_name: str) -> int:
    attempts_map = _control_attempts_map(state)
    if agent_name in attempts_map and attempts_map[agent_name] > 0:
        return attempts_map[agent_name]

    # Fallback por status
    status = {}
    present_flag = False
    try:
        if agent_name == "pgvector":
            status = getf(state, "pgvector_retrieval_status", None) or getf(state, "pgvector_write_status", None) or {}
            present_flag = bool(getf(state, "pgvector_retrieval_status", None) or getf(state, "pgvector_write_status", None))
        elif agent_name == "neo4j":
            status = getf(state, "neo4j_status", None) or {}
            present_flag = bool(getf(state, "neo4j_status", None))
        elif agent_name == "generation":
            status = getf(state, "generation_status", None) or {}
            present_flag = bool(getf(state, "generation_status", None))
    except Exception:
        status = {}

    try:
        att = int(status.get("attempts") or status.get("attempt") or 0)
    except Exception:
        att = 0
    if att == 0 and present_flag:
        att = 1
    return att


def _bump_attempt_counter(state: Any, agent_name: str) -> Any:
    ctrl = _control_map(state)
    attempts_map = _control_attempts_map(state)
    prev = attempts_map.get(agent_name, 0)
    attempts_map[agent_name] = prev + 1
    ctrl["attempts"] = attempts_map
    return setf(state, "control", ctrl)


def _attempt_emergency_persistence(state: Any) -> None:
    """
    Intenta guardar el estado parcial cuando hay un bloqueo de seguridad.

    Guarda información crítica en logs para debugging y recuperación.
    """
    try:
        conversation_id = getf(state, "conversation_id", "unknown")
        trace_id = getf(state, "workflow_metadata", {}).get("trace_id", "unknown")
        final_response = getf(state, "final_response", None)

        emergency_data = {
            "conversation_id": conversation_id,
            "trace_id": trace_id,
            "timestamp": time.time(),
            "final_response_available": bool(final_response),
            "final_response_length": len(str(final_response)) if final_response else 0,
            "workflow_metadata": getf(state, "workflow_metadata", {}),
            "generation_status": getf(state, "generation_status", {}),
            "pgvector_write_status": getf(state, "pgvector_write_status", {}),
        }

        logger.critical(
            "Emergency persistence: workflow terminated due to security block",
            extra={
                "emergency_data": emergency_data,
                "conversation_id": conversation_id,
                "trace_id": trace_id,
            },
        )

        # Si hay una respuesta final, intentar guardarla en un archivo de emergencia
        if final_response:
            try:
                import json
                import os
                from datetime import datetime

                emergency_dir = "/tmp/petition_api_emergency"
                os.makedirs(emergency_dir, exist_ok=True)

                emergency_file = os.path.join(emergency_dir, f"emergency_{conversation_id}_{trace_id}_{datetime.now().isoformat()}.json")

                with open(emergency_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {"conversation_id": conversation_id, "trace_id": trace_id, "final_response": final_response, "timestamp": datetime.now().isoformat(), "metadata": emergency_data},
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )

                logger.info(f"Emergency response saved to: {emergency_file}")

            except Exception as e:
                logger.error(f"Failed to save emergency response: {e}")

    except Exception as e:
        logger.error(f"Emergency persistence failed: {e}")


# ---- Graph factory ------------------------------------------------------------


def make_graph(
    neo4j_driver=None,
    *,
    checkpointer: Optional[Any] = None,
) -> CompiledStateGraph:
    settings = get_settings()
    embedding_service = get_embedding_service()
    # Use centralized embedding service for knowledge extraction
    
    # Agents existentes (sin cambios)
    pgvector_agent = PgVectorAgent(settings, embedding_service)
    

    # Mantén la misma firma de constructor usada en el router (/chat/stream)
    neo4j_agent = Neo4jAgent(settings, embedding_service,  neo4j_driver=neo4j_driver)
    generation_agent = GenerationAgent(settings, neo4j_agent=neo4j_agent)
    orchestrator = OrchestratorAgent(
        settings=settings,
        embedding_service=embedding_service,
        pgvector_agent=pgvector_agent,
        neo4j_agent=neo4j_agent,
        generation_agent=generation_agent,
    )

    # ---- Node wrappers --------------------------------------------------------

    #  NUEVO: Security Agent node (primer filtro de entrada)
    async def _security_agent_node(state: Any, config: Optional[RunnableConfig] = None) -> Any:
        """Nodo de validación de seguridad - primer filtro de entrada."""
        conversation_id = getf(state, "conversation_id", "unknown")

        logger.debug(
            "Nodo de seguridad iniciado",
            extra={
                "conversation_id": conversation_id,
                "state_type": str(type(state)),
                "has_pre_detected_role": getf(state, "pre_detected_role") is not None,
            },
        )

        if isinstance(state, dict):
            if "pre_detected_role" in state:
                print(f"[SECURITY_NODE] state['pre_detected_role'] = {state['pre_detected_role']}")
        else:
            print(f"[SECURITY_NODE] State is NOT a dict, it's: {type(state).__name__}")
            print(f"[SECURITY_NODE] State has pre_detected_role attr: {hasattr(state, 'pre_detected_role')}")
            if hasattr(state, "pre_detected_role"):
                print(f"[SECURITY_NODE] state.pre_detected_role = {state.pre_detected_role}")

        # Usar getf para acceso seguro
        pre_detected_role_via_getf = getf(state, "pre_detected_role", "NOT_FOUND")
        print(f"[SECURITY_NODE] getf(state, 'pre_detected_role') = {pre_detected_role_via_getf}")

        conversation_id = getf(state, "conversation_id", "unknown")
        trace_id = getf(state, "workflow_metadata", {}).get("trace_id", "unknown")

        #  PRESERVAR pre_detected_role desde el inicio
        pre_detected_role = getf(state, "pre_detected_role", None)
        print(f"[SECURITY_NODE] Received pre_detected_role: {pre_detected_role}")

        logger.info(
            "Security agent node started",
            extra={
                "conversation_id": conversation_id,
                "trace_id": trace_id,
                "state_type": type(state).__name__,
                "flow_mode": getf(state, "flow_mode", ""),
                "operation": (getf(state, "generation_request", {}) or {}).get("operation", ""),
                "pre_detected_role": pre_detected_role,
            },
        )

        try:
            # Ejecutar SecurityAgent para todos los flujos.
            # El agente está diseñado para manejar casos sin user_query (como tutela_init) y permitirles pasar.
            
            # CRÍTICO: Preservar campos críticos del estado que SecurityAgent no retorna
            # El SecurityAgent solo retorna campos relacionados con seguridad, no otros campos del estado
            generation_request = getf(state, "generation_request", None)
            pgvector_request = getf(state, "pgvector_request", None)
            neo4j_request = getf(state, "neo4j_request", None)
            vector_results = getf(state, "vector_results", None)
            graph_results = getf(state, "graph_results", None)
            conversation_context = getf(state, "conversation_context", None)
            
            # Preservar generation_request si no está en deltas
            

            # Fusionar deltas con el estado existente usando el helper updatef
            updated_state = updatef(state)

            #  VERIFICAR que pre_detected_role se preservó
            final_pre_detected_role = getf(updated_state, "pre_detected_role", None)
            print(f"[SECURITY_NODE] Final pre_detected_role in updated_state: {final_pre_detected_role}")

            # VERIFICAR que generation_request se preservó
            final_generation_request = getf(updated_state, "generation_request", None)
            if final_generation_request:
                final_operation = final_generation_request.get("operation") if isinstance(final_generation_request, dict) else None
                final_goal = final_generation_request.get("goal") if isinstance(final_generation_request, dict) else None
                logger.info(
                    "Security agent node: generation_request preserved",
                    extra={
                        "conversation_id": conversation_id,
                        "trace_id": trace_id,
                        "operation": final_operation,
                        "goal": final_goal,
                    },
                )
            else:
                logger.warning(
                    "Security agent node: generation_request is None after update",
                    extra={
                        "conversation_id": conversation_id,
                        "trace_id": trace_id,
                        "had_generation_request_before": bool(generation_request),
                    },
                )

            logger.info(
                "Security agent node completed",
                extra={
                    "conversation_id": conversation_id,
                    "trace_id": trace_id,
                    "final_pre_detected_role": final_pre_detected_role,
                    "has_generation_request": bool(final_generation_request),
                    "generation_request_operation": final_generation_request.get("operation") if isinstance(final_generation_request, dict) else None,
                    "generation_request_goal": final_generation_request.get("goal") if isinstance(final_generation_request, dict) else None,
                },
            )

            return updated_state

        except Exception as e:
            logger.error(
                "Security agent node failed",
                extra={
                    "conversation_id": conversation_id,
                    "trace_id": trace_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            # En caso de error en seguridad, ser conservador y bloquear
            emergency_block_state = updatef(
                state,
                {
                    "should_block": True,
                    "final_response": ("Error interno de validación de seguridad. " "Por favor, contacta con soporte técnico."),
                    "security_validation": {
                        "timestamp": time.time(),
                        "trace_id": trace_id,
                        "is_threat": True,
                        "threat_level": "critical",
                        "reason": f"SecurityAgent error: {str(e)}",
                        "validation_applied": False,
                    },
                },
            )
            return emergency_block_state

    async def _orchestrator_plan_node(state: Any, config: Optional[RunnableConfig] = None) -> Any:
        conversation_id = getf(state, "conversation_id", "unknown")

        # PRESERVAR conversation_context antes de la planificación
        original_conversation_context = getf(state, "conversation_context", {}) or {}

        #  PRESERVAR pre_detected_role desde el worker
        pre_detected_role = getf(state, "pre_detected_role", None)
        logger.debug(f"Orchestrator plan: pre_detected_role recibido: {pre_detected_role}", extra={"conversation_id": conversation_id})

        logger.info(
            "Orchestrator planning started",
            extra={
                "conversation_id": conversation_id,
                "state_type": str(type(state)),
                "has_conversation_context": bool(original_conversation_context),
                "context_keys": list(original_conversation_context.keys()) if original_conversation_context else [],
                "has_draft_history": original_conversation_context.get("has_draft_history", False) if original_conversation_context else False,
                "context_content": original_conversation_context,
                "pre_detected_role": pre_detected_role,
            },
        )

        deltas = await orchestrator.run(state)

        #  CRÍTICO: Preservar conversation_context si no está en deltas
        if "conversation_context" not in deltas and original_conversation_context:
            deltas["conversation_context"] = original_conversation_context
            logger.info(
                "Orchestrator: Preserving original conversation_context",
                extra={
                    "conversation_id": conversation_id,
                    "preserved_context_keys": list(original_conversation_context.keys()),
                    "has_draft_history": original_conversation_context.get("has_draft_history", False),
                },
            )

        #  CRÍTICO: Preservar pre_detected_role si no está en deltas
        if "pre_detected_role" not in deltas and pre_detected_role:
            deltas["pre_detected_role"] = pre_detected_role
            print(f"[ORCH_PLAN] Re-adding pre_detected_role to deltas: {pre_detected_role}")
            logger.info(
                "Orchestrator: Preserving pre_detected_role from worker",
                extra={
                    "conversation_id": conversation_id,
                    "pre_detected_role": pre_detected_role,
                },
            )

        # Usar updatef() para mergear deltas con el estado Pydantic actual
        updated_state = updatef(state, deltas)

        #  VERIFICAR que el contexto se preservó correctamente
        final_conversation_context = getf(updated_state, "conversation_context", {}) or {}

        #  VERIFICAR que pre_detected_role se preservó
        final_pre_detected_role = getf(updated_state, "pre_detected_role", None)

        logger.info(
            "Orchestrator planning completed",
            extra={
                "conversation_id": conversation_id,
                "has_orchestrator_decision": hasattr(updated_state, "orchestrator_decision"),
                "delta_keys": list(deltas.keys()) if deltas else None,
                "final_context_keys": list(final_conversation_context.keys()) if final_conversation_context else [],
                "context_preserved": bool(final_conversation_context.get("has_draft_history", False)),
                "final_pre_detected_role": final_pre_detected_role,
            },
        )

        return updated_state

    async def _pgvector_node(state: Any, config: Optional[RunnableConfig] = None) -> Any:
        # LOGGING: Estado inicial del nodo
        conversation_id = getf(state, "conversation_id", "unknown")
        trace_id = getf(state, "workflow_metadata", {}).get("trace_id", "unknown")

        logger.info(
            "PgVector node ENTRY",
            extra={
                "conversation_id": conversation_id,
                "trace_id": trace_id,
                "has_pgvector_request": bool(getf(state, "pgvector_request")),
                "pgvector_request": getf(state, "pgvector_request"),
                "is_first_interaction": getf(state, "is_first_interaction"),
                "flow_mode": getf(state, "flow_mode"),
            },
        )
        # Handle pgvector_request directive from chat_stream endpoint
        req = (getf(state, "pgvector_request") or {}).copy()

        logger.info(
            " PgVector node: request analysis",
            extra={
                "conversation_id": conversation_id,
                "trace_id": trace_id,
                "req_exists": bool(req),
                "req_action": req.get("action") if req else None,
                "req_strategy": req.get("strategy") if req else None,
            },
        )

        if req:
            #  Si es solo persistencia, usar run() directamente
            action = req.get("action", "")
            if action == "persist":
                logger.info(
                    " PgVector node: persist-only request, using run() method",
                    extra={
                        "conversation_id": conversation_id,
                        "trace_id": trace_id,
                        "action": action,
                    },
                )
                result = await pgvector_agent.run(state)
                logger.info(
                    " PgVector node: persist-only COMPLETED",
                    extra={
                        "conversation_id": conversation_id,
                        "trace_id": trace_id,
                        "result_keys": list(result.keys()) if isinstance(result, dict) else "not_dict",
                        "pgvector_request_cleared": result.get("pgvector_request") if isinstance(result, dict) else "unknown",
                    },
                )
                return result

            # Use explicit pgvector_request directive para retrieve/persist+retrieve
            strategy = req.get("strategy")
            if not strategy:
                # Por defecto, en chat queremos mirar semántico + recientes
                strategy = "semantic_and_recent" if not getf(state, "is_first_interaction") else "recent_only"

                logger.info(
                    "PgVector node: usando lógica de recuperación personalizada",
                    extra={
                        "conversation_id": conversation_id,
                        "trace_id": trace_id,
                        "action": action,
                        "strategy": strategy,
                        "query_length": len(req.get("query") or getf(state, "user_query", "")),
                    },
                )

            try:
                # Usar _do_retrieve que es el método disponible
                conversation_context, document_results = await pgvector_agent._do_retrieve(
                    conversation_id=getf(state, "conversation_id", ""),
                    user_query=req.get("query") or getf(state, "user_query", ""),
                    include_kinds=req.get(
                        "include_kinds",
                        ["assistant", "user", "initial_petition", "draft_version", "tutela_chunk"],
                    ),
                )

                # Combinar resultados como hace el método run()
                combined_results = await pgvector_agent._combine_search_results(conversation_context, document_results, req.get("query") or getf(state, "user_query", ""))

                # Update state with results
                state = setf(state, "vector_results", [r.model_dump() for r in combined_results])
                state = setf(state, "conversation_context", conversation_context)

                #  CRÍTICO: Limpiar pgvector_request para evitar bucles infinitos
                state = setf(state, "pgvector_request", None)

                logger.info(
                    "PgVector node: recuperación personalizada completada",
                    extra={
                        "conversation_id": conversation_id,
                        "trace_id": trace_id,
                        "results_count": len(combined_results),
                        "document_results_count": len(document_results),
                        "pgvector_request_cleared": True,
                    },
                )

                # (opcional) guarda metadatos para debug
                wm = getf(state, "workflow_metadata", {}) or {}
                wm["pgvector_agent_response"] = {
                    "success": True,
                    "metadata": {
                        "strategy": strategy,
                        "results_count": len(combined_results),
                        "document_results_count": len(document_results),
                    },
                }
                state = setf(state, "workflow_metadata", wm)

                # PRESERVAR conversation_context del router y agregar tutela si viene del contexto vectorial
                try:
                    ctx = getf(state, "conversation_context", {}) or {}
                    if not isinstance(ctx, dict):
                        ctx = dict(ctx)

                    # Buscar initial_petition en los resultados combinados
                    items = combined_results or []
                    initial = None
                    for it in items:
                        # Manejar tanto objetos VectorSearchResult como dicts
                        if hasattr(it, 'metadata'):
                            kind = it.metadata.get("kind") or ""
                        elif isinstance(it, dict):
                            kind = it.get("metadata", {}).get("kind") or it.get("kind") or ""
                        else:
                            continue
                        if kind.lower() == "initial_petition":
                            initial = it
                            break

                    # Solo agregar initial_petition si no existe en el contexto
                    if initial:
                        # Manejar tanto objetos VectorSearchResult como dicts
                        if hasattr(initial, 'content'):
                            initial_content = initial.content
                        elif isinstance(initial, dict):
                            initial_content = initial.get("content") or ""
                        else:
                            initial_content = ""
                        
                        if initial_content and not ctx.get("initial_petition"):
                            ctx["initial_petition"] = initial_content
                        logger.debug("Agregando initial_petition al contexto existente")

                    # SIEMPRE actualizar el estado con el contexto preservado
                    state = setf(state, "conversation_context", ctx)

                except Exception as e:
                    logger.warning(f"Error preservando contexto: {e}")
                    # En caso de error, al menos preservar el contexto original
                    try:
                        original_ctx = getf(state, "conversation_context", {}) or {}
                        if original_ctx:
                            state = setf(state, "conversation_context", original_ctx)
                    except Exception:
                        pass

                return state

            except Exception as e:
                logger.error(
                    " PgVector node: custom retrieve FAILED, falling back to run()",
                    extra={
                        "conversation_id": conversation_id,
                        "trace_id": trace_id,
                        "error": str(e),
                    },
                )
                # Fallback to original behavior
                result = await pgvector_agent.run(state)
                logger.info(
                    " PgVector node: fallback run() COMPLETED",
                    extra={
                        "conversation_id": conversation_id,
                        "trace_id": trace_id,
                        "result_keys": list(result.keys()) if isinstance(result, dict) else "not_dict",
                    },
                )
                return result
        else:
            # No explicit directive, use original behavior
            logger.info(
                "PgVector node: sin request, usando método run()",
                extra={
                    "conversation_id": conversation_id,
                    "trace_id": trace_id,
                },
            )
            result = await pgvector_agent.run(state)
            logger.info(
                "PgVector node: run() sin request completado",
                extra={
                    "conversation_id": conversation_id,
                    "trace_id": trace_id,
                    "result_keys": list(result.keys()) if isinstance(result, dict) else "not_dict",
                },
            )
            return result

    async def _neo4j_node(state: Any, config: Optional[RunnableConfig] = None) -> Any:
        try:
            runtime_driver = _get_resource(config, "neo4j_driver", None)

            if runtime_driver is not None and hasattr(neo4j_agent, "set_driver"):
                neo4j_agent.set_driver(runtime_driver)
        except Exception:
            pass

        # El orchestrator debería haber creado neo4j_request
        state_with_request = state

        try:
            result_state = await neo4j_agent.run(state_with_request)

            return result_state
        except Exception as neo4j_error:
            import traceback

            logger.error(f"NEO4J_AGENT ERROR: {str(neo4j_error)} - {type(neo4j_error).__name__}")
            logger.debug(f"TRACEBACK: {traceback.format_exc()}")

            # En lugar de fallar, devolver un estado con error pero continuar
            error_state = dict(state) if isinstance(state, dict) else state
            error_state = setf(
                error_state,
                "neo4j_status",
                {
                    "success": False,
                    "error": str(neo4j_error),
                    "error_type": type(neo4j_error).__name__,
                },
            )
            error_state = setf(error_state, "graph_results", None)

            logger.info("Returning error state to continue workflow")
            return error_state

    async def _generation_node(state: Any, config: Optional[RunnableConfig] = None) -> Any:
        # Verificar generation_request
        gen_req = getf(state, "generation_request", {}) or {}
        flow_mode = getf(state, "flow_mode", "MISSING")
        pre_detected_role = getf(state, "pre_detected_role", None)

        logger.debug(f"generation_request.operation: '{gen_req.get('operation', 'MISSING')}'")
        logger.debug(f"generation_request.goal: '{gen_req.get('goal', 'MISSING')}'")
        logger.debug(f"flow_mode: '{flow_mode}'")
        logger.debug(f"pre_detected_role: '{pre_detected_role}'")

        try:
            result_state = await generation_agent.run(state)
        except Exception as gen_error:
            logger.error(f"GENERATION_AGENT ERROR: {str(gen_error)} - {type(gen_error).__name__}")
            raise

        final_response = getf(result_state, "final_response")
        if final_response:
            logger.info(f"FINAL_RESPONSE LENGTH: {len(str(final_response))}")
            logger.debug(f"FINAL_RESPONSE PREVIEW: {str(final_response)[:200]}...")
        else:
            logger.warning("NO FINAL_RESPONSE GENERATED!")

        # NUEVA LÓGICA: Persistir respuesta final directamente aquí
        conversation_id = getf(state, "conversation_id", None)
        logger.info("_generation_node: INICIANDO PERSISTENCIA")
        logger.debug(f"_generation_node: final_response = {bool(final_response)}")
        logger.debug(f"_generation_node: conversation_id = {conversation_id}")
        logger.debug(f"_generation_node: final_response_length = {len(str(final_response)) if final_response else 0}")

        logger.info(
            "_generation_node: Verificando condiciones de persistencia",
            extra={
                "has_final_response": bool(final_response),
                "has_conversation_id": bool(conversation_id),
                "final_response_length": len(str(final_response)) if final_response else 0,
                "conversation_id": conversation_id,
            },
        )

        if final_response and conversation_id:
            logger.info("_generation_node: CONDICIONES CUMPLIDAS - INICIANDO PERSISTENCIA")
            try:
                # Verificar si ya se persistió para evitar duplicados
                workflow_metadata = getf(result_state, "workflow_metadata", {}) or {}
                already_persisted = workflow_metadata.get("response_persisted", False)
                logger.debug(f"_generation_node: already_persisted = {already_persisted}")

                if not already_persisted:
                    logger.info(
                        "_generation_node: Persisting final response directly",
                        extra={
                            "conversation_id": conversation_id,
                            "trace_id": workflow_metadata.get("trace_id", "unknown"),
                            "response_length": len(str(final_response)),
                        },
                    )

                    # PASO 1: Persistir la tutela inicial si no existe
                    user_query = getf(result_state, "user_query", None)

                    #  Obtener flow_mode con fallback desde múltiples fuentes
                    workflow_metadata = getf(result_state, "workflow_metadata", {}) or {}
                    generation_metadata = getf(result_state, "generation_metadata", {}) or {}
                    flow_mode = getf(result_state, "flow_mode") or workflow_metadata.get("flow_mode") or generation_metadata.get("mode") or "tutela_init"  # Fallback final

                    logger.info(
                        " _generation_node: flow_mode detectado",
                        extra={
                            "flow_mode": flow_mode,
                            "source": (
                                "state"
                                if getf(result_state, "flow_mode")
                                else ("workflow_metadata" if workflow_metadata.get("flow_mode") else ("generation_metadata" if generation_metadata.get("mode") else "fallback"))
                            ),
                            "conversation_id": conversation_id,
                        },
                    )

                    if user_query and flow_mode == "tutela_init":
                        try:
                            logger.info(
                                " _generation_node: Persisting initial tutela",
                                extra={
                                    "conversation_id": conversation_id,
                                    "trace_id": workflow_metadata.get("trace_id", "unknown"),
                                    "tutela_length": len(str(user_query)),
                                },
                            )

                            tutela_message = {
                                "role": "user",
                                "kind": "initial_petition",
                                "content": str(user_query),
                                "metadata": {
                                    "source": "tutela_init",
                                    "flow_mode": flow_mode,
                                    "trace_id": workflow_metadata.get("trace_id"),
                                },
                            }

                            tutela_result = await pgvector_agent._persist_from_state(conversation_id, tutela_message)

                            if tutela_result.get("success") or tutela_result.get("ok"):
                                logger.info(
                                    " _generation_node: Initial tutela persisted successfully",
                                    extra={
                                        "conversation_id": conversation_id,
                                        "tutela_message_id": tutela_result.get("message_id"),
                                        "trace_id": workflow_metadata.get("trace_id", "unknown"),
                                    },
                                )
                                workflow_metadata["tutela_message_id"] = tutela_result.get("message_id")

                        except Exception as tutela_error:
                            logger.warning(
                                " _generation_node: Failed to persist initial tutela, continuing with response",
                                extra={
                                    "conversation_id": conversation_id,
                                    "error": str(tutela_error),
                                    "trace_id": workflow_metadata.get("trace_id", "unknown"),
                                },
                            )

                    # PASO 2: Persistir la respuesta del asistente
                    # IMPORTANTE: Usar el kind del persistence_request del generation_agent si está disponible
                    # Esto preserva el kind correcto (revision/chat_response) basado en la operación real
                    persistence_request = getf(result_state, "persistence_request", {}) or {}
                    generation_metadata = getf(result_state, "generation_metadata", {}) or {}
                    operation = generation_metadata.get("operation") or gen_req.get("operation") or "chat"
                    goal = generation_metadata.get("goal") or gen_req.get("goal") or "direct_answer"
                    
                    # Determinar kind: usar del persistence_request si existe, sino determinar basado en operation/goal
                    if persistence_request.get("kind"):
                        persist_kind = persistence_request.get("kind")
                    elif flow_mode == "tutela_init":
                        persist_kind = "initial_draft"
                    elif operation == "edit" and goal == "edit_draft":
                        # Si es edit y tiene goal edit_draft, debería ser revision (a menos que sea fallback)
                        # El persistence_request ya tiene la lógica de fallback, así que si no está, asumimos revision
                        persist_kind = "revision"
                    elif operation == "chat" or goal == "direct_answer":
                        persist_kind = "chat_response"
                    else:
                        # Fallback: usar el mismo criterio que antes
                        persist_kind = "initial_draft" if flow_mode == "tutela_init" else "chat_response"
                    
                    persist_message = {
                        "role": "assistant",
                        "kind": persist_kind,  # ✅ Usar kind correcto basado en operation/goal
                        "content": str(final_response),
                        "metadata": {
                            "source": "generation_agent",
                            "flow_mode": flow_mode,  #  Usar variable local para preservar flow_mode correcto
                            "trace_id": workflow_metadata.get("trace_id"),
                            "generation_metadata": getf(result_state, "generation_metadata", {}),
                        },
                    }
                    
                    logger.info(
                        " _generation_node: Kind determinado para persistencia",
                        extra={
                            "conversation_id": conversation_id,
                            "persist_kind": persist_kind,
                            "operation": operation,
                            "goal": goal,
                            "flow_mode": flow_mode,
                            "from_persistence_request": bool(persistence_request.get("kind")),
                        },
                    )

                    # Usar el PgVectorAgent directamente para persistir
                    persist_result = await pgvector_agent._persist_from_state(conversation_id, persist_message)

                    if persist_result.get("success") or persist_result.get("ok"):
                        logger.info(
                            " _generation_node: Response persisted successfully",
                            extra={
                                "conversation_id": conversation_id,
                                "message_id": persist_result.get("message_id"),
                                "trace_id": workflow_metadata.get("trace_id", "unknown"),
                            },
                        )

                        # Actualizar workflow_metadata para marcar como persistido
                        workflow_metadata["response_persisted"] = True
                        workflow_metadata["persist_message_id"] = persist_result.get("message_id")
                        workflow_metadata["persist_status"] = "success"

                        # Actualizar pgvector_write_status para compatibilidad
                        persist_status = {
                            "success": True,
                            "message_id": persist_result.get("message_id"),
                            "sender": persist_result.get("sender"),
                            "kind": persist_result.get("kind"),
                            "version": persist_result.get("version"),
                            "created_at": persist_result.get("created_at"),
                        }

                        result_state = setf(result_state, "pgvector_write_status", persist_status)
                        result_state = setf(result_state, "workflow_metadata", workflow_metadata)

                    else:
                        logger.warning(
                            " _generation_node: Failed to persist response",
                            extra={
                                "conversation_id": conversation_id,
                                "persist_result": persist_result,
                                "trace_id": workflow_metadata.get("trace_id", "unknown"),
                            },
                        )
                        workflow_metadata["persist_status"] = "failed"
                        workflow_metadata["persist_error"] = persist_result.get("error", "Unknown error")
                        result_state = setf(result_state, "workflow_metadata", workflow_metadata)

            except Exception as e:
                logger.error(
                    " _generation_node: Error during persistence",
                    extra={
                        "conversation_id": conversation_id,
                        "error": str(e),
                        "trace_id": workflow_metadata.get("trace_id", "unknown"),
                    },
                )
                workflow_metadata = getf(result_state, "workflow_metadata", {}) or {}
                workflow_metadata["persist_status"] = "error"
                workflow_metadata["persist_error"] = str(e)
                result_state = setf(result_state, "workflow_metadata", workflow_metadata)
        else:
            logger.info("_generation_node: CONDICIONES NO CUMPLIDAS - SALTANDO PERSISTENCIA")
            logger.debug(f"_generation_node: final_response = {bool(final_response)}")
            logger.debug(f"_generation_node: conversation_id = {bool(conversation_id)}")
        logger.info("_generation_node: PERSISTENCIA COMPLETADA")
        return result_state

    async def _supervisor_pgvector_node(state: Any, config: Optional[RunnableConfig] = None) -> Any:
        state = await orchestrator.run_supervisor(state, agent_name="pgvector")
        state = _bump_attempt_counter(state, "pgvector")
        if _should_retry_from_state(state, "pgvector"):
            # cortar si ya alcanzó el tope local
            if _get_attempts(state, "pgvector") >= PGV_MAX_ATTEMPTS:
                logger.warning("PGVector max attempts reached; ignoring retry and returning to orchestrator.")
                # opcional: limpiar bandera retry para que el orquestador lo vea claro
                ctrl = _control_map(state)
                ctrl["retry"] = False
                state = setf(state, "control", ctrl)
                return state
            sleep_ms = _retry_sleep_ms(state)
            if sleep_ms > 0:
                await asyncio.sleep(sleep_ms / 1000)
        return state

    async def _supervisor_neo4j_node(state: Any, config: Optional[RunnableConfig] = None) -> Any:
        state = await orchestrator.run_supervisor(state, agent_name="neo4j")
        state = _bump_attempt_counter(state, "neo4j")
        if _should_retry_from_state(state, "neo4j"):
            if _get_attempts(state, "neo4j") >= NEO_MAX_ATTEMPTS:
                logger.warning("Neo4j max attempts reached; ignoring retry and returning to orchestrator.")
                ctrl = _control_map(state)
                ctrl["retry"] = False
                state = setf(state, "control", ctrl)
                return state
            sleep_ms = _retry_sleep_ms(state)
            if sleep_ms > 0:
                await asyncio.sleep(sleep_ms / 1000)
        return state

    async def _finalize_node(state: Any, config: Optional[RunnableConfig] = None) -> Any:
        logger.info(" _finalize_node: INICIANDO")

        # 1. Ejecutar la finalización normal del orchestrator (obtener solo el delta)
        orchestrator_delta = await orchestrator.run_finalize(state)
        # 2. Fusionar el delta con el estado existente en lugar de sobrescribir
        for key, value in orchestrator_delta.items():
            state = setf(state, key, value)

        state = _bump_attempt_counter(state, "generation")

        # Obtener final_response para verificaciones posteriores
        final_response = getf(state, "final_response", None)

        # 2. NUEVA LÓGICA: Persistir respuesta final directamente aquí
        conversation_id = getf(state, "conversation_id", None)

        logger.info(
            " _finalize_node: Verificando condiciones de persistencia",
            extra={
                "has_final_response": bool(final_response),
                "has_conversation_id": bool(conversation_id),
                "final_response_length": len(str(final_response)) if final_response else 0,
                "conversation_id": conversation_id,
            },
        )

        if final_response and conversation_id:
            try:
                # Verificar si ya se persistió para evitar duplicados
                workflow_metadata = getf(state, "workflow_metadata", {}) or {}
                pgvector_status = getf(state, "pgvector_write_status", {}) or {}
                already_persisted = bool(
                    workflow_metadata.get("response_persisted")
                    or pgvector_status.get("success")
                    or pgvector_status.get("version")
                )

                if not already_persisted:
                    logger.info(
                        " Finalize: Persisting final response directly",
                        extra={
                            "conversation_id": conversation_id,
                            "trace_id": workflow_metadata.get("trace_id", "unknown"),
                            "response_length": len(str(final_response)),
                        },
                    )

                    # PASO 1: Persistir la tutela inicial si no existe
                    user_query = getf(state, "user_query", None)

                    #  Obtener flow_mode con fallback desde múltiples fuentes
                    workflow_metadata = getf(state, "workflow_metadata", {}) or {}
                    generation_metadata = getf(state, "generation_metadata", {}) or {}
                    flow_mode = getf(state, "flow_mode") or workflow_metadata.get("flow_mode") or generation_metadata.get("mode") or "tutela_init"  # Fallback final

                    logger.info(
                        " FINALIZE: flow_mode detectado",
                        extra={
                            "flow_mode": flow_mode,
                            "source": (
                                "state"
                                if getf(state, "flow_mode")
                                else ("workflow_metadata" if workflow_metadata.get("flow_mode") else ("generation_metadata" if generation_metadata.get("mode") else "fallback"))
                            ),
                            "conversation_id": conversation_id,
                        },
                    )

                    if user_query and flow_mode == "tutela_init":
                        try:
                            # Verificar si ya existe la tutela inicial
                            conversation_context = getf(state, "conversation_context", {})
                            has_initial_petition = conversation_context.get("initial_petition") is not None

                            if not has_initial_petition:
                                logger.info(
                                    " Finalize: Persisting initial tutela",
                                    extra={
                                        "conversation_id": conversation_id,
                                        "trace_id": workflow_metadata.get("trace_id", "unknown"),
                                        "tutela_length": len(str(user_query)),
                                    },
                                )

                                tutela_message = {
                                    "role": "user",
                                    "kind": "initial_petition",
                                    "content": str(user_query),
                                    "metadata": {
                                        "source": "tutela_init",
                                        "flow_mode": flow_mode,
                                        "trace_id": workflow_metadata.get("trace_id"),
                                    },
                                }

                                tutela_result = await pgvector_agent._persist_from_state(conversation_id, tutela_message)

                                if tutela_result.get("success") or tutela_result.get("ok"):
                                    logger.info(
                                        " Finalize: Initial tutela persisted successfully",
                                        extra={
                                            "conversation_id": conversation_id,
                                            "tutela_message_id": tutela_result.get("message_id"),
                                            "trace_id": workflow_metadata.get("trace_id", "unknown"),
                                        },
                                    )
                                    workflow_metadata["tutela_message_id"] = tutela_result.get("message_id")

                        except Exception as tutela_error:
                            logger.warning(
                                " Finalize: Failed to persist initial tutela, continuing with response",
                                extra={
                                    "conversation_id": conversation_id,
                                    "error": str(tutela_error),
                                    "trace_id": workflow_metadata.get("trace_id", "unknown"),
                                },
                            )

                    # PASO 2: Persistir la respuesta del asistente
                    # IMPORTANTE: Usar el kind del persistence_request del generation_agent si está disponible
                    persistence_request = getf(state, "persistence_request", {}) or {}
                    generation_metadata = getf(state, "generation_metadata", {}) or {}
                    gen_req = getf(state, "generation_request", {}) or {}
                    operation = generation_metadata.get("operation") or gen_req.get("operation") or "chat"
                    goal = generation_metadata.get("goal") or gen_req.get("goal") or "direct_answer"
                    
                    # Determinar kind: usar del persistence_request si existe, sino determinar basado en operation/goal
                    if persistence_request.get("kind"):
                        persist_kind = persistence_request.get("kind")
                    elif flow_mode == "tutela_init":
                        persist_kind = "initial_draft"
                    elif operation == "edit" and goal == "edit_draft":
                        persist_kind = "revision"
                    elif operation == "chat" or goal == "direct_answer":
                        persist_kind = "chat_response"
                    else:
                        persist_kind = "initial_draft" if flow_mode == "tutela_init" else "chat_response"
                    
                    persist_message = {
                        "role": "assistant",
                        "kind": persist_kind,  # ✅ Usar kind correcto basado en operation/goal
                        "content": str(final_response),
                        "metadata": {
                            "source": "generation_agent",
                            "flow_mode": flow_mode,  #  Usar variable local para preservar flow_mode correcto
                            "trace_id": workflow_metadata.get("trace_id"),
                            "generation_metadata": getf(state, "generation_metadata", {}),
                        },
                    }
                    
                    logger.info(
                        " Finalize: Kind determinado para persistencia",
                        extra={
                            "conversation_id": conversation_id,
                            "persist_kind": persist_kind,
                            "operation": operation,
                            "goal": goal,
                            "flow_mode": flow_mode,
                            "from_persistence_request": bool(persistence_request.get("kind")),
                        },
                    )

                    # Usar el PgVectorAgent directamente para persistir
                    persist_result = await pgvector_agent._persist_from_state(conversation_id, persist_message)

                    if persist_result.get("success") or persist_result.get("ok"):
                        logger.info(
                            " Finalize: Response persisted successfully",
                            extra={
                                "conversation_id": conversation_id,
                                "message_id": persist_result.get("message_id"),
                                "trace_id": workflow_metadata.get("trace_id", "unknown"),
                            },
                        )

                        # Actualizar workflow_metadata para marcar como persistido
                        workflow_metadata["response_persisted"] = True
                        workflow_metadata["persist_message_id"] = persist_result.get("message_id")
                        workflow_metadata["persist_status"] = "success"

                        #  CHUNKING: Si es tutela_init, chunkear la tutela ahora para futuras ediciones
                        if flow_mode == "tutela_init" and user_query:
                            try:
                                logger.info(
                                    " FINALIZE: Iniciando chunking de tutela para RAG",
                                    extra={
                                        "conversation_id": conversation_id,
                                        "trace_id": workflow_metadata.get("trace_id", "unknown"),
                                    },
                                )

                                
                                

                                
                            except Exception as chunking_error:
                                logger.error(
                                    " FINALIZE: Error en chunking de tutela",
                                    extra={
                                        "conversation_id": conversation_id,
                                        "error": str(chunking_error),
                                        "trace_id": workflow_metadata.get("trace_id", "unknown"),
                                    },
                                    exc_info=True,
                                )
                                # No fallar el flujo completo por error de chunking
                                workflow_metadata["tutela_chunked"] = False
                                workflow_metadata["chunking_error"] = str(chunking_error)
                        # Actualizar pgvector_write_status para compatibilidad
                        persist_status = {
                            "success": True,
                            "message_id": persist_result.get("message_id"),
                            "sender": persist_result.get("sender"),
                            "kind": persist_result.get("kind"),
                            "version": persist_result.get("version"),
                            "created_at": persist_result.get("created_at"),
                        }

                        state = setf(state, "pgvector_write_status", persist_status)
                        state = setf(state, "workflow_metadata", workflow_metadata)

                    else:
                        logger.error(
                            " Finalize: Failed to persist response",
                            extra={
                                "conversation_id": conversation_id,
                                "persist_result": persist_result,
                                "trace_id": workflow_metadata.get("trace_id", "unknown"),
                            },
                        )

                        # Marcar error pero no fallar el flujo
                        workflow_metadata["persist_status"] = "failed"
                        workflow_metadata["persist_error"] = persist_result.get("error", "unknown")
                        state = setf(state, "workflow_metadata", workflow_metadata)

                else:
                    logger.debug(
                        " Finalize: Response already persisted, skipping",
                        extra={
                            "conversation_id": conversation_id,
                            "trace_id": workflow_metadata.get("trace_id", "unknown"),
                        },
                    )

            except Exception as e:
                logger.error(
                    " Finalize: Exception during persistence",
                    extra={
                        "conversation_id": conversation_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "trace_id": getf(state, "workflow_metadata", {}).get("trace_id", "unknown"),
                    },
                )

                # Marcar error en metadata pero continuar
                workflow_metadata = getf(state, "workflow_metadata", {}) or {}
                workflow_metadata["persist_status"] = "error"
                workflow_metadata["persist_error"] = str(e)
                state = setf(state, "workflow_metadata", workflow_metadata)

        # 3. Continuar con la lógica de retry si es necesario
        if _should_retry_from_state(state, "generation"):
            sleep_ms = _retry_sleep_ms(state)
            if sleep_ms > 0:
                await asyncio.sleep(sleep_ms / 1000)
        logger.info(" _finalize_node: COMPLETADO")
        return state

    # ---- Routers --------------------------------------------------------------

    #  NUEVO: Router desde SecurityAgent
    def _route_from_security(state: Any) -> str:
        """Decide si continuar al orchestrator o bloquear por seguridad."""
        conversation_id = getf(state, "conversation_id", "unknown")
        trace_id = getf(state, "workflow_metadata", {}).get("trace_id", "unknown")
        should_block = getf(state, "should_block", False)
        security_validation = getf(state, "security_validation", {})

        if should_block:
            logger.warning(
                "Security router: BLOCKING request - terminating workflow",
                extra={
                    "conversation_id": conversation_id,
                    "trace_id": trace_id,
                    "security_reason": security_validation.get("reason", "unknown"),
                    "threat_level": security_validation.get("threat_level", "unknown"),
                    "patterns": security_validation.get("matched_patterns", []),
                },
            )
            return END

        logger.info(
            "Security router: Request approved - continuing to orchestrator",
            extra={
                "conversation_id": conversation_id,
                "trace_id": trace_id,
                "security_validated": True,
            },
        )

        return NODE_ORCH_PLAN  # Continuar al orchestrator normal

    def _route_from_orchestrator(state: Any) -> str:
        conversation_id = getf(state, "conversation_id", "unknown")
        trace_id = getf(state, "workflow_metadata", {}).get("trace_id", "unknown")

        # Obtener decisión del orchestrator
        orchestrator_decision = getf(state, "orchestrator_decision", None)

        if orchestrator_decision:
            logger.info(
                " Orchestrator decision received",
                extra={
                    "conversation_id": conversation_id,
                    "use_vector_search": orchestrator_decision.get("use_vector_search", False),
                    "use_graph_analysis": orchestrator_decision.get("use_graph_analysis", False),
                    "direct_generation": orchestrator_decision.get("direct_generation", False),
                },
            )
        else:
            logger.warning(" No orchestrator decision found - using fallback routing", extra={"conversation_id": conversation_id})

        #  SAFEGUARD: Usar un contador simple basado en el número de agentes ejecutados
        # Como no podemos mutar state en routers, usamos la suma de attempts como proxy
        pgv_attempts = _get_attempts(state, "pgvector")
        neo_attempts = _get_attempts(state, "neo4j")
        gen_attempts = _get_attempts(state, "generation")

        # Aproximación del número de iteraciones basado en attempts
        estimated_iterations = pgv_attempts + neo_attempts + gen_attempts

        # También contar si hay status de agentes (indica que se han ejecutado)
        status_count = 0
        if getf(state, "pgvector_retrieval_status") or getf(state, "pgvector_write_status"):
            status_count += 1
        if getf(state, "neo4j_status"):
            status_count += 1
        if getf(state, "generation_status"):
            status_count += 1

        # Usar el mayor de los dos como estimación de iteraciones
        global_iterations = max(estimated_iterations, status_count)
        logger.warning(
            " ROUTING from orchestrator - ITERATION CHECK",
            extra={
                "conversation_id": conversation_id,
                "trace_id": trace_id,
                "global_iterations": global_iterations,
                "max_safe_iterations": 25,  # Límite de seguridad
            },
        )

        #  EMERGENCY STOP: Si hay demasiadas iteraciones, forzar END
        if global_iterations >= 5:  # Reducir para debugging
            logger.critical(
                " EMERGENCY STOP: Too many iterations detected - forcing END",
                extra={
                    "conversation_id": conversation_id,
                    "trace_id": trace_id,
                    "global_iterations": global_iterations,
                    "emergency_stop": True,
                },
            )
            return "END"

        decision = getf(state, "orchestrator_decision", None)
        flags = _normalize_orch_flags(decision or {})

        #  FALLBACK: Si todas las flags están en False, configurar defaults para tutela
        if not any([flags["direct_generation"], flags["use_vector_search"], flags["use_graph_analysis"]]):

            #  IMPORTANTE: Si ya tenemos contexto suficiente, ir directo a generation
            vector_results = getf(state, "vector_results", []) or []
            graph_results = getf(state, "graph_results", None)
            has_vec = _vector_present(vector_results)
            has_graph = _has_meaningful_neo4j_results(graph_results)

            if has_vec and has_graph:
                logger.debug(f"CONTEXT ALREADY AVAILABLE - Setting direct_generation=True (has_vec={has_vec}, has_graph={has_graph})")
                flags["direct_generation"] = True
                #  Limpiar otras flags para evitar conflictos
                flags["use_graph_analysis"] = False
                flags["use_vector_search"] = False
            else:
                #  COMENTADO: Lógica de detección de tutela movida al orchestrator para evitar sobrescritura de flags
                # El orchestrator ahora maneja toda la detección y planificación de manera centralizada

                # CÓDIGO ORIGINAL COMENTADO (mantener para referencia):
                # # Para flujo de tutela, por defecto necesitamos contexto
                # flow_mode = getf(state, "flow_mode", "")
                # generation_request = getf(state, "generation_request", {}) or {}
                #
                # print(f" FLOW DETECTION DEBUG:")
                # print(f"   flow_mode: '{flow_mode}'")
                # print(f"   generation_request: {generation_request}")
                # print(f"   generation_request.operation: '{generation_request.get('operation', '')}'")
                # print(f"   tutela in flow_mode: {'tutela' in str(flow_mode).lower()}")
                # print(f"   tutela in operation: {'tutela' in str(generation_request.get('operation', '')).lower()}")
                #
                # #  MEJORAR DETECCIÓN: Buscar más indicadores de tutela
                # is_tutela_flow = (
                #     "tutela" in str(flow_mode).lower() or
                #     "tutela" in str(generation_request.get("operation", "")).lower() or
                #     "tutela" in str(generation_request.get("goal", "")).lower() or
                #     generation_request.get("plan", {}).get("use_neo4j_context", False) or
                #     generation_request.get("plan", {}).get("use_conversation_memory", False)
                # )
                #
                # print(f"   FINAL is_tutela_flow: {is_tutela_flow}")
                #
                # if is_tutela_flow:
                #     print(f" TUTELA FLOW DETECTED - Setting use_graph_analysis=True, use_vector_search=True")
                #     flags["direct_generation"] = False
                #     flags["use_graph_analysis"] = True
                #     flags["use_vector_search"] = True
                # else:
                #     print(f" NON-TUTELA FLOW - Setting direct_generation=True")
                #     flags["direct_generation"] = True
                #     flags["use_graph_analysis"] = False
                #     flags["use_vector_search"] = False
                #     print(f" FLAGS AFTER NON-TUTELA: direct_gen={flags['direct_generation']}, use_vec={flags['use_vector_search']}, use_graph={flags['use_graph_analysis']}")

                logger.debug("GRAFO: Respetando flags del orchestrator - no sobrescribiendo decisiones")

        logger.info(
            " ROUTING from orchestrator",
            extra={
                "conversation_id": conversation_id,
                "trace_id": trace_id,
                "global_iterations": global_iterations,
                "orchestrator_decision": decision,
                "flags": flags,
                "has_pgvector_request": bool(getf(state, "pgvector_request")),
                "pgvector_request": getf(state, "pgvector_request"),
                "has_final_response": bool(getf(state, "final_response")),
            },
        )

        orchestrator_result = getf(state, "orchestrator_result", {})
        if orchestrator_result.get("security_blocked"):
            logger.warning(" ROUTING: Security blocked detected -> END")
            #  Intentar guardar estado parcial antes de terminar
            _attempt_emergency_persistence(state)
            return "END"

        #  TEMPORALMENTE DESHABILITADO: Persistencia que causa bucle infinito
        #  Priorizar persistencia si hay pgvector_request pendiente
        # pgvector_request = getf(state, "pgvector_request", None)
        # if pgvector_request and pgvector_request.get("action") == "persist":
        #     logger.info(
        #         " ROUTING: persist request -> pgvector",
        #         extra={
        #             "conversation_id": conversation_id,
        #             "trace_id": trace_id,
        #             "pgvector_request": pgvector_request,
        #         }
        #     )
        #     return TR_TO_PGV

        final_response = getf(state, "final_response", None)
        if final_response:
            logger.info(
                " ROUTING: final_response present -> finalize",
                extra={
                    "conversation_id": conversation_id,
                    "trace_id": trace_id,
                    "global_iterations": global_iterations,
                    "final_response_length": len(str(final_response)) if final_response else 0,
                },
            )
            return TR_TO_FINAL

        # Stop when we already have final draft and there are no pending requests
        # Normaliza pgvector_write_status a dict
        _pgws = getf(state, "pgvector_write_status", None) or {}
        try:
            if not isinstance(_pgws, dict):
                _pgws = dict(_pgws)
        except Exception:
            _pgws = {}
        pg_ok = bool(not _pgws or _pgws.get("success") or _pgws.get("ok"))

        pending_requests = {
            "pgvector_request": getf(state, "pgvector_request", None),
            "neo4j_request": getf(state, "neo4j_request", None),
            "generation_request": getf(state, "generation_request", None),
        }
        no_triggers = not any(pending_requests.values())

        logger.info(
            " ROUTING: Checking stop conditions",
            extra={
                "conversation_id": conversation_id,
                "trace_id": trace_id,
                "global_iterations": global_iterations,
                "has_final_response": bool(final_response),
                "pg_ok": pg_ok,
                "no_triggers": no_triggers,
                "pgvector_write_status": _pgws,
                "pending_requests": {k: bool(v) for k, v in pending_requests.items()},
                "stop_condition_met": bool(final_response and pg_ok and no_triggers),
            },
        )

        if final_response and pg_ok and no_triggers:
            logger.info(
                " ROUTING: STOP CONDITION MET -> END",
                extra={
                    "conversation_id": conversation_id,
                    "trace_id": trace_id,
                    "global_iterations": global_iterations,
                    "reason": "final_response_and_no_pending_requests",
                },
            )
            return "END"

        vector_results = getf(state, "vector_results", []) or []
        graph_results = getf(state, "graph_results", None)
        has_vec = _vector_present(vector_results)
        has_graph = _has_meaningful_neo4j_results(graph_results)

        pgv_attempts = _get_attempts(state, "pgvector")
        neo_attempts = _get_attempts(state, "neo4j")

        logger.info(
            " ROUTING: Analyzing routing conditions",
            extra={
                "conversation_id": conversation_id,
                "trace_id": trace_id,
                "global_iterations": global_iterations,
                "direct_generation": flags["direct_generation"],
                "use_vector_search": flags["use_vector_search"],
                "use_graph_analysis": flags["use_graph_analysis"],
                "vec_present": has_vec,
                "graph_present": has_graph,
                "pgv_attempts": pgv_attempts,
                "neo_attempts": neo_attempts,
                "vector_results_count": len(vector_results),
                "has_graph_results": bool(graph_results),
            },
        )

        if flags["direct_generation"]:
            logger.info(
                " ROUTING: direct_generation -> generation",
                extra={
                    "conversation_id": conversation_id,
                    "trace_id": trace_id,
                    "global_iterations": global_iterations,
                    "routing_reason": "direct_generation_flag",
                },
            )
            return TR_TO_GEN

        #  MEJORAR: Considerar errores de pgvector en el routing
        pgvector_failed = False
        pgvector_status = getf(state, "pgvector_retrieval_status", {}) or {}
        if isinstance(pgvector_status, dict):
            pgvector_failed = pgvector_status.get("success") is False or pgvector_status.get("error") or pgvector_status.get("both_methods_failed")

        if flags["use_vector_search"] and not has_vec and pgv_attempts == 0 and not pgvector_failed:
            logger.info(
                " Routing to PgVector for context retrieval",
                extra={"conversation_id": conversation_id},
            )
            logger.info(
                " ROUTING: need vector search -> pgvector",
                extra={
                    "conversation_id": conversation_id,
                    "trace_id": trace_id,
                    "global_iterations": global_iterations,
                    "routing_reason": "need_vector_search",
                    "has_vec": has_vec,
                    "pgv_attempts": pgv_attempts,
                    "pgvector_failed": pgvector_failed,
                },
            )
            return TR_TO_PGV
        elif flags["use_vector_search"] and pgvector_failed:
            logger.warning(
                " ROUTING: pgvector failed, skipping vector search -> continue to generation",
                extra={
                    "conversation_id": conversation_id,
                    "trace_id": trace_id,
                    "global_iterations": global_iterations,
                    "routing_reason": "pgvector_failed_skip",
                    "pgvector_status": pgvector_status,
                },
            )

        #  MEJORAR: Considerar errores de neo4j en el routing
        neo4j_failed = False
        neo4j_status = getf(state, "neo4j_status", {}) or {}
        if isinstance(neo4j_status, dict):
            neo4j_failed = neo4j_status.get("success") is False or neo4j_status.get("error")

        if flags["use_graph_analysis"] and not has_graph and neo_attempts == 0 and not neo4j_failed:
            logger.info(
                " Routing to Neo4j for legal context",
                extra={"conversation_id": conversation_id},
            )
            logger.info(
                " ROUTING: need graph analysis -> neo4j",
                extra={
                    "conversation_id": conversation_id,
                    "trace_id": trace_id,
                    "global_iterations": global_iterations,
                    "routing_reason": "need_graph_analysis",
                    "has_graph": has_graph,
                    "neo_attempts": neo_attempts,
                },
            )
            return TR_TO_NEO4J
        elif flags["use_graph_analysis"] and neo4j_failed:
            logger.warning(
                " ROUTING: neo4j failed, skipping graph search -> continue to generation",
                extra={
                    "conversation_id": conversation_id,
                    "trace_id": trace_id,
                    "global_iterations": global_iterations,
                    "routing_reason": "neo4j_failed_skip",
                    "neo4j_status": neo4j_status,
                },
            )

        #  CONDICIÓN ESPECIAL: Si tenemos vector results y Neo4j falló, ir a generation
        if has_vec and (neo4j_failed or neo_attempts > 0):
            logger.info(
                " ROUTING: has vector context, neo4j failed/attempted -> generation",
                extra={
                    "conversation_id": conversation_id,
                    "trace_id": trace_id,
                    "global_iterations": global_iterations,
                    "routing_reason": "vector_context_available_neo4j_failed",
                    "has_vec": has_vec,
                    "neo4j_failed": neo4j_failed,
                    "neo_attempts": neo_attempts,
                },
            )
            return TR_TO_GEN

        logger.warning(f"DEFAULT FALLBACK TRIGGERED - ITERATION {global_iterations} - {conversation_id}")
        logger.debug(f"FLAGS AFTER CORRECTION: direct_gen={flags['direct_generation']}, use_vec={flags['use_vector_search']}, use_graph={flags['use_graph_analysis']}")
        logger.debug(f"STATE: has_vec={has_vec}, has_graph={has_graph}, pgv_attempts={pgv_attempts}, neo_attempts={neo_attempts}")
        logger.debug(f"FLOW_MODE: {getf(state, 'flow_mode', 'unknown')}")
        logger.debug(f"GENERATION_REQUEST: {getf(state, 'generation_request', {})}")

        logger.error(  # Changed to ERROR to ensure visibility
            " ROUTING: DEFAULT FALLBACK -> generation (POTENTIAL LOOP RISK)",
            extra={
                "conversation_id": conversation_id,
                "trace_id": trace_id,
                "global_iterations": global_iterations,
                "routing_reason": "default_fallback",
                "flags": flags,
                "has_vec": has_vec,
                "has_graph": has_graph,
                "pgv_attempts": pgv_attempts,
                "neo_attempts": neo_attempts,
                "warning": "This might cause infinite loop if generation doesn't produce final_response",
            },
        )
        return TR_TO_GEN

    def _route_after_supervisor_pgvector(state: Any) -> str:
        # si alcanzó el tope, no reintenta aunque el supervisor pida retry
        if _get_attempts(state, "pgvector") >= PGV_MAX_ATTEMPTS:
            return TR_NEXT
        return TR_RETRY if _should_retry_from_state(state, "pgvector") else TR_NEXT

    def _route_after_supervisor_neo4j(state: Any) -> str:
        if _get_attempts(state, "neo4j") >= NEO_MAX_ATTEMPTS:
            return TR_NEXT
        return TR_RETRY if _should_retry_from_state(state, "neo4j") else TR_NEXT

    def _route_after_finalize(state: Any) -> str:
        """Router después de finalize - ahora va a OutputGuardrail en lugar de END."""
        conversation_id = getf(state, "conversation_id", "unknown")
        trace_id = getf(state, "workflow_metadata", {}).get("trace_id", "unknown")

        #  TEMPORALMENTE DESHABILITADO: Persistencia que causa bucle infinito
        #  Priorizar persistencia pendiente antes de terminar
        # pgvector_request = getf(state, "pgvector_request", None)
        # if pgvector_request and pgvector_request.get("action") == "persist":
        #     logger.debug("Finalize router: persist request pending -> orchestrator")
        #     return "orchestrator"

        logger.debug("_route_after_finalize: Skipping persistence step to avoid infinite loop")

        if _generation_success(state):
            logger.debug(
                "_route_after_finalize: Generation success -> END",
                extra={
                    "conversation_id": conversation_id,
                    "trace_id": trace_id,
                },
            )
            return TR_NEXT

        gen_attempts = _get_attempts(state, "generation")
        logger.info("Finalize router: generation attempts so far = %s", gen_attempts)

        if gen_attempts >= GEN_MAX_ATTEMPTS:
            logger.warning(
                "Hard-stop: GEN_MAX_ATTEMPTS alcanzado (%s). No se reintenta Generation -> END",
                gen_attempts,
                extra={
                    "conversation_id": conversation_id,
                    "trace_id": trace_id,
                },
            )
            return TR_NEXT

        # Si necesita retry, volver a generation, si no, terminar
        if _should_retry_from_state(state, "generation"):
            logger.debug("_route_after_finalize: Retry needed -> generation")
            return TR_RETRY
        logger.debug("_route_after_finalize: No retry needed -> END")
        return TR_NEXT

    # ---- Build graph ----------------------------------------------------------

    workflow = StateGraph(dict)

    #  NUEVOS: Nodos de seguridad
    workflow.add_node("security_agent", _security_agent_node)

    # Nodos existentes (sin cambios)
    workflow.add_node(NODE_ORCH_PLAN, _orchestrator_plan_node)
    workflow.add_node(NODE_PGV, _pgvector_node)
    workflow.add_node(NODE_SUP_PGV, _supervisor_pgvector_node)
    workflow.add_node(NODE_NEO4J, _neo4j_node)
    workflow.add_node(NODE_SUP_NEO4J, _supervisor_neo4j_node)
    workflow.add_node(NODE_GEN, _generation_node)
    workflow.add_node(NODE_FINAL, _finalize_node)

    #  NUEVO: El flujo comienza en SecurityAgent (era NODE_ORCH_PLAN)
    workflow.set_entry_point("security_agent")

    #  NUEVO: Routing desde SecurityAgent
    workflow.add_conditional_edges(
        "security_agent",
        _route_from_security,
        {
            NODE_ORCH_PLAN: NODE_ORCH_PLAN,  # Continuar al orchestrator
            END: END,  # Bloquear por seguridad
        },
    )

    # Routing desde orchestrator (sin cambios)
    workflow.add_conditional_edges(
        NODE_ORCH_PLAN,
        _route_from_orchestrator,
        {
            TR_TO_GEN: NODE_GEN,
            TR_TO_PGV: NODE_PGV,
            TR_TO_NEO4J: NODE_NEO4J,
            TR_TO_FINAL: NODE_FINAL,
            "END": END,
        },
    )

    workflow.add_edge(NODE_PGV, NODE_SUP_PGV)
    workflow.add_edge(NODE_NEO4J, NODE_SUP_NEO4J)
    workflow.add_edge(NODE_GEN, NODE_FINAL)

    workflow.add_conditional_edges(
        NODE_SUP_PGV,
        _route_after_supervisor_pgvector,
        {TR_RETRY: NODE_PGV, TR_NEXT: NODE_ORCH_PLAN},
    )
    workflow.add_conditional_edges(
        NODE_SUP_NEO4J,
        _route_after_supervisor_neo4j,
        {TR_RETRY: NODE_NEO4J, TR_NEXT: NODE_ORCH_PLAN},
    )

    workflow.add_conditional_edges(
        NODE_FINAL,
        _route_after_finalize,
        {
            TR_RETRY: NODE_GEN,
            TR_NEXT: END,
            "orchestrator": NODE_ORCH_PLAN,  # Nueva ruta para persistencia
        },
    )

    compiled = workflow.compile(checkpointer=checkpointer or MemorySaver())
    logger.info(
        " Workflow compiled SUCCESSFULLY with Security Layers",
        extra={
            "nodes": [
                "security_agent",  #  NUEVO: Primer nodo (entrada)
                NODE_ORCH_PLAN,
                NODE_PGV,
                NODE_SUP_PGV,
                NODE_NEO4J,
                NODE_SUP_NEO4J,
                NODE_GEN,
                NODE_FINAL,
            ],
            "entry": "security_agent",  #  NUEVO: Entrada por SecurityAgent
            "security_enabled": True,
            "gen_max_attempts": GEN_MAX_ATTEMPTS,
            "pgv_max_attempts": PGV_MAX_ATTEMPTS,
            "neo_max_attempts": NEO_MAX_ATTEMPTS,
            "routing_edges": {
                "security_agent": ["orchestrator_plan", "END"],  #  NUEVO
                "orchestrator_plan": ["generation", "pgvector", "neo4j", "finalize", "END"],
                "pgvector": ["supervisor_pgvector"],
                "neo4j": ["supervisor_neo4j"],
                "generation": ["finalize"],
                "supervisor_pgvector": ["pgvector", "orchestrator_plan"],
                "supervisor_neo4j": ["neo4j", "orchestrator_plan"],
                "finalize": ["generation", "END", "orchestrator_plan"],  #  ACTUALIZADO
            },
        },
    )

    logger.info(" Graph factory: returning compiled workflow")
    return compiled
