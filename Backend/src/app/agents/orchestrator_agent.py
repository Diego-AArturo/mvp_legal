"""
Agente Orquestador (Planificar + Supervisar + Finalizar).

- Planifica: decide spokes (PgVector / Neo4j / Generation).
- Supervisa: evalúa resultados por spoke y gestiona reintentos con backoff.
- Finaliza: valida la respuesta y puede reintentar Generation.

Notas de diseño:
- State-agnostic: soporta dict o PetitionState (Pydantic).
- Los métodos devuelven SIEMPRE deltas (no mutan ni devuelven el estado completo).
- orchestrator_decision es dict plano (no Pydantic) para evitar fricciones de serialización.
"""

from __future__ import annotations

import asyncio
import json as _json
import random
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from app.config.settings import Settings
from app.extensions import get_logger
from app.services.ai.ollama_service import OllamaService

from app.services.embeddings.embedding_service import EmbeddingService
logger = get_logger(__name__)


class _AgentLogServiceStub:
    @staticmethod
    def log_agent_start(*args: Any, **kwargs: Any) -> None:
        return None

    @staticmethod
    def log_agent_complete(*args: Any, **kwargs: Any) -> None:
        return None

    @staticmethod
    def log_agent_error(*args: Any, **kwargs: Any) -> None:
        return None


class _SecurityServiceStub:
    def __init__(self) -> None:
        self.settings = type(
            "SettingsStub",
            (object,),
            {
                "get_timeout_config": lambda self: {
                    "orchestrator": 8000,
                    "pgvector": 8000,
                    "neo4j": 8000,
                    "generation": 8000,
                }
            },
        )()
        self.dos_settings = type(
            "DosStub",
            (object,),
            {
                "max_cpu_usage_percent": 95,
                "max_memory_usage_percent": 95,
            },
        )()

    async def check_internal_operation_allowed(self, trace_id: str) -> tuple[bool, Optional[str]]:
        return True, None

    async def check_request_allowed(self, trace_id: str, user_query: str, client_ip: str) -> tuple[bool, Optional[str]]:
        return True, None

    async def start_request_tracking(self, trace_id: str, user_query: str, timeout_ms: int) -> None:
        return None

    async def end_request_tracking(self, trace_id: str, status: str) -> None:
        return None

    async def check_circuit_breaker_recovery(self, svc: str) -> bool:
        return True

    async def monitor_resources(self):
        return type("ResStub", (object,), {"cpu_percent": 0, "memory_percent": 0})()

    async def update_circuit_breaker(self, svc: str, healthy: bool) -> None:
        return None


AgentLogService = _AgentLogServiceStub()
SecurityService = _SecurityServiceStub

# -------------------------- Helpers: state-agnostic --------------------------


def _state_as_dict(state: Any) -> Dict[str, Any]:
    """
    Convierte un estado a diccionario independientemente de su tipo.

    Soporta dicts nativos, modelos Pydantic v1/v2 y objetos con atributos.

    Argumentos:
        state: Estado a convertir (dict, Pydantic model, o objeto con atributos)

    Retorna:
        Diccionario con el estado convertido
    """
    if isinstance(state, dict):
        return state
    # Pydantic v2
    try:
        return state.model_dump()  # type: ignore[attr-defined]
    except Exception:
        pass
    # Pydantic v1
    try:
        return state.dict()  # type: ignore[attr-defined]
    except Exception:
        pass
    # Fallback
    try:
        return dict(getattr(state, "__dict__", {}))
    except Exception:
        return {}


def _sget(state: Any, key: str, default: Any = None) -> Any:
    """
    Obtiene un valor del estado de forma segura.

    Argumentos:
        state: Estado (dict o objeto)
        key: Clave a obtener
        default: Valor por defecto si no se encuentra

    Retorna:
        Valor obtenido o default
    """
    d = _state_as_dict(state)
    return d.get(key, default)


def _get_in_dict(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Obtiene un valor de un diccionario usando una ruta con puntos.

    Argumentos:
        d: Diccionario a consultar
        path: Ruta con puntos (ej: "control.attempts")
        default: Valor por defecto si no se encuentra

    Retorna:
        Valor obtenido o default
    """
    cur = d or {}
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# -------------------------- Orchestrator --------------------------


def _should_stop(state: dict) -> bool:
    """
    Determina si el flujo debe detenerse.

    Se detiene cuando ya hay una respuesta final y no hay solicitudes
    pendientes de agentes.

    Argumentos:
        state: Estado actual del flujo

    Retorna:
        True si el flujo debe detenerse, False en caso contrario
    """
    if not state.get("final_response"):
        return False

    # No pending triggers
    pending = any(
        [
            state.get("pgvector_request"),
            state.get("neo4j_request"),
            state.get("generation_request"),
        ]
    )

    # Accept "skip" as OK as well (e.g., text-only init without MinIO document)
    pg_ok = state.get("pgvector_write_status") in (None, "success", "skip")

    return (not pending) and pg_ok


class OrchestratorAgent:
    """
    Agente Orquestador centrado en planificación, supervisión y validación final.

    Coordina el flujo de trabajo multi-agente tomando decisiones de routing,
    supervisando ejecución de agentes especializados y validando resultados finales.
    """

    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService,
        knowledge_extractor: Any = None,
        neo4j_agent=None,
        pgvector_agent=None,
        generation_agent=None,
    ):
        self.settings = settings
        self.embedding_service = embedding_service
        self.knowledge_extractor = knowledge_extractor

        self._neo4j_driver = getattr(neo4j_agent, "driver", None) if neo4j_agent else None

        self.security_service = SecurityService()
        self.ollama_service = OllamaService(settings)

        # Cache para embeddings de patrones (optimización)
        self._pattern_embeddings_cache: Dict[str, List[float]] = {}

        self.default_retry_cfg = {
            "pgvector": {
                "max_retries": 2,
                "base_sleep_ms": 400,
                "max_sleep_ms": 4000,
                "exp_base": 2.0,
                "jitter": True,
            },
            "neo4j": {
                "max_retries": 2,
                "base_sleep_ms": 400,
                "max_sleep_ms": 4000,
                "exp_base": 2.0,
                "jitter": True,
            },
            "generation": {
                "max_retries": 1,
                "base_sleep_ms": 600,
                "max_sleep_ms": 6000,
                "exp_base": 2.0,
                "jitter": True,
            },
        }

        self.UNAMBIGUOUS_EDIT_VERBS = [
            "modifica",
            "cambia",
            "agrega",
            "elimina",
            "quita",
            "corrige",
            "inserta",
            "añade",
            "reemplaza",
            "actualiza",
        ]

        # --- Fusión de patrones existentes y propuestos ---

        # Patrones de EDICIÓN: Combinamos los conceptuales con tus ejemplos concretos.
        self.edit_patterns = [
            # Conceptuales (capturan la intención general)
            "instrucción directa para modificar el contenido de un documento",
            "comando para cambiar, añadir o quitar una sección del borrador",
            "solicitud explícita de corrección o reescritura de un párrafo",
            "orden para insertar información nueva en el texto existente",
            # Tus ejemplos (aportan especificidad)
            "modifica la sección de hechos",
            "cambia la sección del borrador con estos datos",
            "agrega la siguiente jurisprudencia",
            "corrige el nombre del accionado",
            "elimina esta parte del documento",
            "reescribe la sección con mejor redacción",
            "ajusta el tono para que sea más persuasivo",
            "resume la parte de los fundamentos de derecho",
            "eso está incorrecto, el artículo correcto es el 23",  # Implica una corrección directa
        ]

        # Patrones de CHAT: Combinamos los conceptuales con tus ejemplos.
        self.chat_patterns = [
            # Conceptuales
            "pregunta sobre normativa o jurisprudencia aplicable",
            "solicitud de explicación sobre un concepto legal",
            "búsqueda de información general relacionada con el caso",
            "conversación casual, saludo o agradecimiento",
            # Preguntas directas (interrogativas)
            "¿qué dice la ley sobre este tema?",
            "explícame los procedimientos legales",
            "dame información sobre jurisprudencia",
            "¿cuáles son los requisitos para?",
            "búscame sentencias relacionadas",
            "muchas gracias por la ayuda",
            # NUEVOS: Patrones de pregunta más específicos
            "¿qué es una tutela?",
            "¿cómo funciona este proceso?",
            "¿qué significa esto?",
            "¿me puedes explicar?",
            "¿qué opinas de esto?",
            "¿cuál es tu opinión?",
            "tengo una pregunta",
            "necesito saber",
            "¿podrías aclararme?",
            "dime más sobre",
            "información sobre",
            "cuéntame sobre",
            "¿por qué?",
            "¿para qué sirve?",
            "¿es correcto que?",
            "¿cómo se hace?",
            "¿qué debería hacer?",
            "quiero entender",
            "no entiendo",
            "¿qué quiere decir?",
            "hola",
            "gracias",
            "perfecto",
            "ok",
            "entendido",
            "de acuerdo",
        ]

        # NUEVO: Patrones de AMBIGÜEDAD para capturar "falsos positivos"
        # Estos patrones se clasifican como CHAT cuando son los más similares
        self.ambiguous_patterns = [
            "pregunta acerca del contenido del borrador sin pedir un cambio",
            "petición de opinión o justificación sobre una parte del texto",
            "feedback general sobre la calidad del documento",
            "¿por qué incluiste la sentencia T-025 en el borrador?",
            "explícame mejor la parte de los hechos que escribiste",
            "¿estás seguro de que esa es la ley correcta?",
            "esa respuesta no me sirve, genérala de nuevo con otro enfoque",  # Es más una 'regeneración' que una 'edición' específica
            # NUEVOS: Preguntas sobre el contenido que NO piden modificación
            "¿qué pusiste en el borrador?",
            "¿qué dice el documento?",
            "¿qué contiene la respuesta?",
            "¿cómo quedó?",
            "muéstrame lo que escribiste",
            "¿está bien así?",
            "¿está correcto?",
            "¿qué te parece?",
            "léeme el borrador",
            "¿qué incluiste?",
        ]

        logger.info("OrchestratorAgent initialized.")

    # ==========================================================================
    # PLAN
    # ==========================================================================

    async def run(self, state: Any) -> Dict[str, Any]:
        """Fase PLAN: calcula decisiones y directivas. Retorna deltas."""
        t0 = time.time()
        trace_id = str(uuid.uuid4())[:8]
        sd = _state_as_dict(state)

        #  Log agent start
        log_id = AgentLogService.log_agent_start(
            conversation_id=sd.get("conversation_id"),
            trace_id=trace_id,
            agent_name="orchestrator",
            operation="run",
            agent_phase="plan",
            input_summary={
                "flow_mode": sd.get("flow_mode"),
                "is_first_interaction": sd.get("is_first_interaction"),
                "has_generation_request": bool(sd.get("generation_request")),
                "user_query_length": len(sd.get("user_query", "")),
            },
            metadata={
                "state_keys": list(sd.keys()),
                "client_ip": sd.get("client_ip"),
            },
        )

        # log seguro
        logger.info(
            "Orchestrator PLAN started",
            extra={
                "trace_id": trace_id,
                "state_keys": list(sd.keys()),
                "log_id": log_id,
            },
        )

        # --- Early stop (in case we arrive here with a draft already present)
        if _should_stop(sd):
            # Ensure minimal orchestrator_result for the test endpoint
            sd.setdefault(
                "orchestrator_result",
                {
                    "agent": "orchestrator",
                    "trace_id": trace_id,
                    "routing_decision": {"reason": "final_response already present"},
                    "executed_agents": sd.get("executed_agents", []),
                    "results_summary": {"final_response": True},
                    "generator_dispatch": {"ok": True},
                    "workflow_strategy": sd.get("workflow_strategy", "direct_generate"),
                    "success": True,
                },
            )
            return sd

        if not self._validate_state_shape(sd):
            return self._error_update(
                trace_id=trace_id,
                message="Missing keys: user_query, conversation_id",
                reason="invalid_state",
            )

        user_query: str = (sd.get("user_query") or "").strip()
        conversation_id: str = sd.get("conversation_id", "")
        client_ip: str = sd.get("client_ip", "unknown")

        # Determinar si es una operación interna del grafo
        is_internal_operation = self._is_internal_graph_operation(sd)

        logger.info(
            "Orchestrator security check",
            extra={
                "trace_id": trace_id,
                "is_internal_operation": is_internal_operation,
                "conversation_id": conversation_id,
                "has_pgvector_request": bool(sd.get("pgvector_request")),
                "has_final_response": bool(sd.get("final_response")),
                "is_first_interaction": sd.get("is_first_interaction", True),
            },
        )

        # Puertas de seguridad - más inteligentes
        if is_internal_operation:
            # Para operaciones internas, verificación ligera
            (
                allowed,
                reason,
            ) = await self.security_service.check_internal_operation_allowed(trace_id)
            if not allowed:
                logger.warning(
                    "Internal operation blocked",
                    extra={"trace_id": trace_id, "reason": reason},
                )
                return self._blocked_update(trace_id, reason)
        else:
            # Para requests externas, verificación completa o bypass explícito
            if (sd.get("config") or {}).get("security", {}).get("allow_recursion_like"):
                allowed, reason = (True, None)
            else:
                allowed, reason = await self.security_service.check_request_allowed(trace_id, user_query, client_ip)
            if not allowed:
                logger.warning(
                    "External request blocked",
                    extra={"trace_id": trace_id, "reason": reason},
                )
                return self._blocked_update(trace_id, reason)

        timeout_cfg = self.security_service.settings.get_timeout_config()
        orch_timeout_ms = timeout_cfg["orchestrator"]
        await self.security_service.start_request_tracking(trace_id, user_query, orch_timeout_ms)

        try:
            res = await asyncio.wait_for(
                self._compute_plan(state=sd, trace_id=trace_id, timeout_cfg=timeout_cfg),
                timeout=orch_timeout_ms / 1000,
            )
            await self.security_service.end_request_tracking(trace_id, "completed")

            # Telemetría mínima
            orr = res.setdefault("orchestrator_result", {})
            orr.update(
                {
                    "agent": "orchestrator",
                    "trace_id": trace_id,
                    "elapsed_ms": int((time.time() - t0) * 1000),
                    "phase": "plan",
                    "success": True,
                }
            )
            res["orchestrator_success"] = True

            #  Log agent completion
            AgentLogService.log_agent_complete(
                log_id,
                output_summary={
                    "orchestrator_success": True,
                    "use_vector_search": (res.get("orchestrator_decision") or {}).get("use_vector_search"),
                    "use_graph_analysis": (res.get("orchestrator_decision") or {}).get("use_graph_analysis"),
                    "direct_generation": (res.get("orchestrator_decision") or {}).get("direct_generation"),
                    "flow_mode": res.get("flow_mode"),
                    "goal": res.get("goal"),
                    "has_final_response": bool(res.get("final_response")),
                },
                metadata_update={
                    "elapsed_ms": orr.get("elapsed_ms"),
                    "phase": "plan",
                },
            )

            # --- After executing agents, reevaluate stop condition
            # IMPORTANTE: No limpiar triggers en PLAN. Cada agente limpia su trigger al terminar.

            if _should_stop(res):
                res.setdefault(
                    "orchestrator_result",
                    {
                        "agent": "orchestrator",
                        "trace_id": trace_id,
                        "routing_decision": {"reason": "stop condition met (final_response)"},
                        "executed_agents": res.get("executed_agents", []),
                        "results_summary": {"final_response": True},
                        "generator_dispatch": {"ok": True},
                        "workflow_strategy": res.get("workflow_strategy", "direct_generate"),
                        "success": True,
                    },
                )
                return res

            return res

        except asyncio.TimeoutError as timeout_err:
            await self.security_service.end_request_tracking(trace_id, "timeout")
            logger.error(
                "Orchestrator PLAN timeout",
                extra={"trace_id": trace_id, "timeout_ms": orch_timeout_ms},
            )

            #  Log timeout
            AgentLogService.log_agent_error(
                log_id,
                error=timeout_err,
                error_type="TimeoutError",
                metadata_update={"timeout_ms": orch_timeout_ms},
            )

            return self._timeout_update(trace_id, orch_timeout_ms)
        except Exception as e:
            logger.exception("Orchestrator PLAN error", extra={"trace_id": trace_id})
            await self.security_service.end_request_tracking(trace_id, "error")

            #  Log error
            AgentLogService.log_agent_error(log_id, error=e, metadata_update={"phase": "plan"})

            return self._error_update(trace_id=trace_id, message=str(e))

    async def _compute_plan(self, *, state: Dict[str, Any], trace_id: str, timeout_cfg: Dict[str, int]) -> Dict[str, Any]:
        user_query = (state.get("user_query") or "").strip()
        ctx = state.get("conversation_context") or {}

        gen_req = state.get("generation_request") or {}
        flow_mode = state.get("flow_mode")  # no derivar
        goal = gen_req.get("goal")
        operation = gen_req.get("operation")

        # En flujo `chat_edit`, el modo (chat vs edit) debe venir explícitamente del cliente.
        # Esto evita que el sistema tome la decisión usando heurísticas/clasificación.
        if flow_mode == "chat_edit" and not operation:
            return self._error_update(
                trace_id=trace_id,
                message="Missing generation_request.operation for chat_edit flow (expected 'chat' or 'edit')",
                reason="missing_operation",
            )

        # Permitir defaults razonables de goal si viene operation explícita.
        if operation and not goal:
            op = str(operation).lower()
            if op == "edit":
                goal = "edit_draft"
            elif op == "compose":
                goal = "draft_official_response"
            else:
                goal = "direct_answer"
            gen_req = {**gen_req, "operation": op, "goal": goal}
            operation = op

        # Validación final
        if not (flow_mode and goal and operation):
            return self._error_update(
                trace_id=trace_id,
                message="Missing explicit flow_mode/goal/operation in generation_request",
                reason="invalid_directives",
            )
        use_vector = bool((gen_req.get("plan") or {}).get("use_conversation_memory"))
        use_graph = bool((gen_req.get("plan") or {}).get("use_neo4j_context"))
        direct_gen = not (use_vector or use_graph)

        # DETECCIÓN ROBUSTA DE TUTELA - Sobrescribir flags si es necesario
        is_tutela_flow = self._detect_tutela_flow(flow_mode, gen_req, state)
        if is_tutela_flow:
            logger.info(
                "TUTELA FLOW DETECTED - Overriding flags for comprehensive analysis",
                extra={
                    "trace_id": trace_id,
                    "original_use_vector": use_vector,
                    "original_use_graph": use_graph,
                    "original_direct_gen": direct_gen,
                },
            )
            use_vector = True
            use_graph = True
            direct_gen = False

        # Desactivar Neo4j si no hay driver
        logger.info(
            "Orchestrator: Checking Neo4j driver availability",
            extra={
                "trace_id": trace_id,
                "neo4j_driver_present": self._neo4j_driver is not None,
                "neo4j_driver_type": type(self._neo4j_driver).__name__ if self._neo4j_driver else "None",
            },
        )

        if not self._neo4j_driver:
            use_graph = False
            logger.warning(
                "Neo4j driver not present; disabling graph context",
                extra={"trace_id": trace_id},
            )
        else:
            logger.info(
                "Orchestrator: Neo4j driver available, enabling graph context",
                extra={"trace_id": trace_id},
            )

        orch_decision = {
            "use_vector_search": use_vector,
            "use_graph_analysis": use_graph,
            "direct_generation": direct_gen,
            "reasoning": "explicit_directives",
            "confidence": 1.0,
        }

        # 4) Semilla de metadatos para que el SSE lo muestre correcto
        gen_meta = (state.get("generation_metadata") or {}).copy()
        gen_meta.update({
            "goal": goal,
            "mode": flow_mode,
            "operation": operation,  # CRÍTICO: incluir operation para mantener consistencia
            "language": "es"
        })

        wf_meta = (state.get("workflow_metadata") or {}).copy()
        wf_meta.update(
            {
                "trace_id": trace_id,
                "current_trace_id": trace_id,
                "flow_mode": flow_mode,
                "policy_source": "explicit_directives",
                "neo4j_driver_present": bool(self._neo4j_driver),
                "context_sources_planned": [s for s, need in (("pgvector", use_vector), ("neo4j", use_graph)) if need],
            }
        )

        # Signal that text-only init is allowed (no external document required)
        if flow_mode == "tutela_init":
            has_inline_text = bool(_get_in_dict(state, "conversation_context.initial_petition") or state.get("user_query"))
            wf_meta.setdefault("policy_flags", {})
            wf_meta["policy_flags"].update(
                {
                    "allow_text_only_init": has_inline_text,
                    "require_source_document": False,
                }
            )

        orchestrator_evt = {
            "phase": "plan",
            "planned_sources": wf_meta["context_sources_planned"],
            "flow_mode": flow_mode,
            "policy_source": wf_meta.get("policy_source"),
            "rationale": f"Explicit directives: {operation} -> {goal}",
            "trace_id": trace_id,
            "ts": time.time(),
        }

        #  Establecer fecha/hora actual en formato ISO con zona horaria de Colombia
        try:
            from datetime import datetime

            import pytz

            bogota_tz = pytz.timezone("America/Bogota")
            current_datetime = datetime.now(bogota_tz).isoformat()
            logger.info("Orchestrator: Estableciendo contexto temporal", extra={"trace_id": trace_id, "current_datetime": current_datetime})
        except Exception as e:
            logger.warning("Orchestrator: Error al establecer contexto temporal, usando fallback", extra={"trace_id": trace_id, "error": str(e)})
            from datetime import datetime

            current_datetime = datetime.now().isoformat()

        # 5) Órdenes para los spokes
        pgvector_req = None
        if use_vector:
            pgvector_req = {
                "action": "persist+retrieve",
                "conversation_id": state.get("conversation_id", ""),
                "strategy": "recent_only" if flow_mode == "tutela_init" else "semantic_and_recent",
                "k_recent": 12,
                "semantic_top_k": 8,
                "include_kinds": [
                    "initial_petition",
                    "draft_version",
                    "assistant",
                    "user",
                    "tutela_chunk",  #  AGREGADO: Incluir chunks de tutela para RAG
                ],
                "must_include_kinds": ["initial_petition"],
                "max_tokens_context": 10000,
                "query": user_query,
                "user_query": user_query,
                "message": {
                    "role": "user",
                    "kind": ("initial_petition" if flow_mode == "tutela_init" else "edit_request"),
                    "content": user_query,
                    "metadata": {"source": flow_mode},
                },
                "top_k": 0 if flow_mode == "tutela_init" else 8,
                #  RAG: Configuración para recuperar chunks relevantes de tutela
                "retrieve_tutela_chunks": flow_mode != "tutela_init",  # Activar en chat_edit
                "chunks_top_k": 5,  # Número de chunks relevantes a recuperar
                "chunks_include_critical": True,  # Incluir chunks críticos siempre
            }

        neo4j_req = None
        if use_graph:
            neo4j_req = self._build_neo4j_request(flow_mode, state)

        #  CORRECCIÓN: No sobrescribir gen_req si ya fue configurado por clasificación inteligente
        # Solo usar el del state si gen_req no existe (para compatibilidad con endpoints que lo proveen)
        if not gen_req:
            gen_req = state.get("generation_request")
            logger.info(
                "ORCHESTRATOR: Usando generation_request del state (endpoint externo)",
                extra={
                    "trace_id": trace_id,
                    "gen_req_source": "state",
                    "gen_req_keys": list((gen_req or {}).keys()),
                },
            )
        else:
            logger.info(
                "ORCHESTRATOR: Preservando generation_request de clasificación inteligente",
                extra={
                    "trace_id": trace_id,
                    "gen_req_source": "intelligent_classification",
                    "operation": gen_req.get("operation"),
                    "goal": gen_req.get("goal"),
                },
            )

        # Avoid infinite persistence retries on first interaction
        if state.get("is_first_interaction") and not state.get("persist_attempted"):
            if pgvector_req:
                state["persist_attempted"] = True

        # Include timeout configuration in state for downstream agents
        config_delta = (state.get("config") or {}).copy()
        config_delta["timeouts"] = timeout_cfg

        # DEBUG: Log de la decisión del orchestrator antes de retornar
        logger.info(
            "ORCHESTRATOR DECISION GENERATED",
            extra={
                "trace_id": trace_id,
                # "orchestrator_decision": orch_decision,  # Comentado: info extensa
                "use_vector": use_vector,
                "use_graph": use_graph,
                "direct_gen": direct_gen,
                "is_tutela_flow": is_tutela_flow,
                "flow_mode": flow_mode,
                # "generation_request_plan": (gen_req or {}).get("plan", {})  # Comentado: info extensa
            },
        )

        # CORRECCIÓN: Retornar estado completo, no solo deltas
        # Crear deltas con los nuevos campos
        #  ASEGURAR que gen_req nunca sea None
        gen_req = gen_req or {}

        deltas = {
            "orchestrator_decision": orch_decision,
            "pgvector_request": pgvector_req,
            "neo4j_request": neo4j_req,
            "generation_request": gen_req,
            "workflow_metadata": wf_meta,
            "generation_metadata": gen_meta,
            "goal": goal,
            "flow_mode": flow_mode,
            "orchestrator_event": orchestrator_evt,
            "config": config_delta,
            "current_datetime": current_datetime,  #  Contexto temporal para todos los agentes
        }

        logger.info(
            "Orchestrator decision generated",
            extra={
                "trace_id": trace_id,
                "use_vector_search": use_vector,
                "use_graph_analysis": use_graph,
                "direct_generation": direct_gen,
                "flow_mode": flow_mode,
            },
        )

        return deltas

    # ==========================================================================
    # SUPERVISE
    # ==========================================================================

    async def run_supervisor(self, state: Any, *, agent_name: str) -> Dict[str, Any]:
        """Evalúa el resultado del spoke y decide retry. Retorna deltas en control."""
        sd = _state_as_dict(state)
        trace_id = _get_in_dict(sd, "workflow_metadata.trace_id") or str(uuid.uuid4())[:8]
        control = (sd.get("control") or {}).copy()
        attempts = control.get("attempts") or {}
        attempt = int(attempts.get(agent_name, 0))
        cfg = self._get_retry_policy(sd, agent_name)

        #  Log agent start
        log_id = AgentLogService.log_agent_start(
            conversation_id=sd.get("conversation_id"),
            trace_id=trace_id,
            agent_name="orchestrator",
            operation="run_supervisor",
            agent_phase="supervise",
            input_summary={
                "supervised_agent": agent_name,
                "attempt": attempt,
                "max_retries": cfg.get("max_retries"),
            },
            metadata={
                "retry_config": cfg,
            },
        )

        try:
            if agent_name == "pgvector":
                decision, reason = self._decide_retry_pgvector(sd, attempt, cfg)
            elif agent_name == "neo4j":
                decision, reason = self._decide_retry_neo4j(sd, attempt, cfg)
            else:
                # Generation se supervisa en finalize
                return self._merge_control_delta(
                    sd,
                    {
                        "retry": False,
                        "retry_agent": None,
                        "retry_reason": "not_applicable",
                        "retry_sleep_ms": 0,
                        "next_attempt": attempt,
                    },
                    phase=f"supervisor_{agent_name}",
                    trace_id=trace_id,
                )

            if decision:
                if attempt >= cfg["max_retries"]:
                    return self._merge_control_delta(
                        sd,
                        {
                            "retry": False,
                            "retry_agent": None,
                            "retry_reason": "retry_budget_exhausted",
                            "retry_sleep_ms": 0,
                            "next_attempt": attempt,
                        },
                        phase=f"supervisor_{agent_name}",
                        trace_id=trace_id,
                    )
                next_attempt = attempt + 1
                sleep_ms = self._compute_backoff_ms(next_attempt, cfg)
                return self._merge_control_delta(
                    sd,
                    {
                        "retry": True,
                        "retry_agent": agent_name,
                        "retry_reason": reason,
                        "retry_sleep_ms": sleep_ms,
                        "next_attempt": next_attempt,
                    },
                    phase=f"supervisor_{agent_name}",
                    trace_id=trace_id,
                )

            #  Log completion
            AgentLogService.log_agent_complete(
                log_id,
                output_summary={
                    "decision": "no_retry",
                    "reason": "ok",
                    "agent_name": agent_name,
                },
            )

            return self._merge_control_delta(
                sd,
                {
                    "retry": False,
                    "retry_agent": None,
                    "retry_reason": "ok",
                    "retry_sleep_ms": 0,
                    "next_attempt": attempt,
                },
                phase=f"supervisor_{agent_name}",
                trace_id=trace_id,
            )

        except Exception as e:
            #  Log error
            AgentLogService.log_agent_error(log_id, error=e)

            # Falla el supervisor: por seguridad avanzar sin retry
            return self._merge_control_delta(
                sd,
                {
                    "retry": False,
                    "retry_agent": None,
                    "retry_reason": f"supervisor_exception:{e}",
                    "retry_sleep_ms": 0,
                    "next_attempt": attempt,
                },
                phase=f"supervisor_{agent_name}",
                trace_id=trace_id,
            )

    # ==========================================================================
    # FINALIZE
    # ==========================================================================

    async def run_finalize(self, state: Any) -> Dict[str, Any]:
        """Valida la respuesta final y decide si reintentar Generation. Retorna deltas."""
        sd = _state_as_dict(state)
        trace_id = _get_in_dict(sd, "workflow_metadata.trace_id") or str(uuid.uuid4())[:8]

        final_response = sd.get("final_response")
        plan = (sd.get("generation_request") or {}).get("plan") or {}

        valid, issues = self._validate_final_response(final_response, plan)

        control = (sd.get("control") or {}).copy()
        attempts = control.get("attempts") or {}
        attempt = int(attempts.get("generation", 0))
        cfg = self._get_retry_policy(sd, "generation")

        #  Log agent start
        log_id = AgentLogService.log_agent_start(
            conversation_id=sd.get("conversation_id"),
            trace_id=trace_id,
            agent_name="orchestrator",
            operation="run_finalize",
            agent_phase="finalize",
            input_summary={
                "final_response_valid": valid,
                "validation_issues": issues if not valid else None,
                "attempt": attempt,
                "max_retries": cfg.get("max_retries"),
            },
        )

        if valid:
            # Crear request de persistencia para la respuesta generada
            persistence_request = self._build_persistence_request(sd)

            delta = self._merge_control_delta(
                sd,
                {
                    "retry": False,
                    "retry_agent": None,
                    "retry_reason": "final_valid",
                    "retry_sleep_ms": 0,
                    "next_attempt": attempt,
                    "final_valid": True,
                },
                phase="finalize",
                trace_id=trace_id,
            )

            # Agregar request de persistencia si es necesario
            if persistence_request:
                delta["pgvector_request"] = persistence_request
                logger.info(
                    "Finalize: creando request de persistencia",
                    extra={
                        "trace_id": trace_id,
                        "conversation_id": sd.get("conversation_id"),
                        "persistence_action": persistence_request.get("action"),
                    },
                )

            #  Log completion
            AgentLogService.log_agent_complete(
                log_id,
                output_summary={
                    "final_valid": True,
                    "has_persistence_request": bool(persistence_request),
                },
            )

            return delta

        # No válido
        if attempt >= cfg["max_retries"]:
            #  Log completion (retry budget exhausted)
            AgentLogService.log_agent_complete(
                log_id,
                output_summary={
                    "final_valid": False,
                    "retry_decision": "exhausted",
                    "validation_issues": issues,
                },
            )

            return self._merge_control_delta(
                sd,
                {
                    "retry": False,
                    "retry_agent": None,
                    "retry_reason": f"final_invalid_but_retry_budget_exhausted:{issues}",
                    "retry_sleep_ms": 0,
                    "next_attempt": attempt,
                    "final_valid": False,
                },
                phase="finalize",
                trace_id=trace_id,
            )

        next_attempt = attempt + 1
        sleep_ms = self._compute_backoff_ms(next_attempt, cfg)

        # Ajustes para el reintento (delta)
        adjusted_req = self._tune_generation_request_for_retry(sd, issues) or (sd.get("generation_request") or {})

        #  Log completion (will retry)
        AgentLogService.log_agent_complete(
            log_id,
            output_summary={
                "final_valid": False,
                "retry_decision": "will_retry",
                "retry_agent": "generation",
                "next_attempt": next_attempt,
                "sleep_ms": sleep_ms,
            },
        )

        return {
            **self._merge_control_delta(
                sd,
                {
                    "retry": True,
                    "retry_agent": "generation",
                    "retry_reason": f"final_invalid:{issues}",
                    "retry_sleep_ms": sleep_ms,
                    "next_attempt": next_attempt,
                    "final_valid": False,
                },
                phase="finalize",
                trace_id=trace_id,
            ),
            "generation_request": adjusted_req,
        }

    # ==========================================================================
    # Build directives (PLAN)
    # ==========================================================================

    def _build_pgvector_request(self, flow_mode: str, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        conv_id = state.get("conversation_id", "")
        top_k = (state.get("config") or {}).get("pgvector_top_k", 8)

        if flow_mode == "tutela_init":
            initial_petition = (state.get("conversation_context") or {}).get("initial_petition") or state.get("user_query", "")
            return {
                "action": "persist+retrieve",
                "conversation_id": conv_id,
                "message": {
                    "role": "user",
                    "kind": "initial_petition",
                    "content": initial_petition,
                    "metadata": {"source": "tutela_init"},
                },
                "top_k": 0,
            }

        #  RAG: Para chat_edit, también solicitar retrieval de chunks relevantes
        return {
            "action": "persist+retrieve",
            "conversation_id": conv_id,
            "user_query": state.get("user_query", ""),
            "message": {
                "role": "user",
                "kind": "edit_request",
                "content": state.get("user_query", ""),
                "metadata": {"source": "chat_edit"},
            },
            "top_k": top_k,
            # Nueva configuración para RAG con chunks
            "retrieve_tutela_chunks": True,  # Activar recuperación de chunks
            "chunks_top_k": 5,  # Número de chunks relevantes a recuperar
            "chunks_include_critical": True,  # Incluir chunks críticos siempre
        }

    def _build_neo4j_request(self, flow_mode: str, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # SOLUCIÓN: Crear request incluso sin driver, el neo4j_agent manejará la falta de driver
        # if not self._neo4j_driver:
        #     return None
        top_k = (state.get("config") or {}).get("neo4j_top_k", 10)
        threshold = (state.get("config") or {}).get("neo4j_similarity_threshold", None)
        query = state.get("user_query", "")

        neo4j_req = {"query": query, "top_k": top_k, "similarity_threshold": threshold}

        logger.info(
            "OrchestratorAgent: Created neo4j_request",
            extra={
                # "neo4j_request": neo4j_req,  # Comentado: info extensa
                "has_driver": bool(self._neo4j_driver),
                "flow_mode": flow_mode,
                "top_k": top_k,
            },
        )
        return neo4j_req

    def _detect_tutela_flow(self, flow_mode: str, gen_req: Dict[str, Any], state: Dict[str, Any]) -> bool:
        """
        Detecta si el flujo actual es de tutela y requiere análisis completo.

        Criterios:
        1. flow_mode contiene 'tutela'
        2. operation contiene 'tutela'
        3. goal indica análisis legal
        4. Presencia de tutela_reference_id
        5. Plan ya solicita contexto Neo4j o conversacional
        """
        # Criterio 1: flow_mode
        if "tutela" in str(flow_mode).lower():
            return True

        # Criterio 2: operation
        operation = str(gen_req.get("operation", "")).lower()
        if "tutela" in operation:
            return True

        # Criterio 3: goal indica análisis legal
        goal = str(gen_req.get("goal", "")).lower()
        legal_goals = ["legal_analysis", "draft_response", "analyze_petition"]
        if any(legal_goal in goal for legal_goal in legal_goals):
            return True

        # Criterio 4: Presencia de tutela_reference_id
        if state.get("tutela_reference_id"):
            return True

        # Criterio 5: Plan ya solicita contexto (indica análisis complejo)
        plan = gen_req.get("plan", {})
        if plan.get("use_neo4j_context") or plan.get("use_conversation_memory"):
            return True

        return False

    def _build_persistence_request(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Construye request de persistencia para la respuesta generada."""
        final_response = state.get("final_response")
        if not final_response:
            return None

        conv_id = state.get("conversation_id", "")
        flow_mode = state.get("flow_mode", "")

        # Verificar si ya se persistió la respuesta
        workflow_metadata = state.get("workflow_metadata", {})
        already_persisted = workflow_metadata.get("response_persisted", False)

        if already_persisted:
            logger.debug("Response already persisted, skipping persistence request")
            return None

        # También verificar pgvector_write_status exitoso
        pgvector_write_status = state.get("pgvector_write_status") or {}
        if isinstance(pgvector_write_status, dict) and pgvector_write_status.get("success") and pgvector_write_status.get("sender") == "assistant":
            logger.debug("Assistant response already persisted via pgvector_write_status")
            return None

        # Construir mensaje de respuesta para persistir
        response_content = final_response
        if isinstance(final_response, dict):
            response_content = final_response.get("content") or final_response.get("text") or str(final_response)

        # CORRECCIÓN: Obtener el kind del persistence_request del generation_agent
        gen_persistence_req = state.get("persistence_request") or {}
        persistence_kind = gen_persistence_req.get("kind", "revision")

        # Validar que sea un kind válido, si no, usar revision por defecto
        valid_kinds = {"chat_response", "initial_draft", "revision"}
        if persistence_kind not in valid_kinds:
            persistence_kind = "revision"

        return {
            "action": "persist",
            "conversation_id": conv_id,
            "message": {
                "role": "assistant",
                "kind": persistence_kind,  # ← AHORA USA EL KIND CORRECTO
                "content": response_content,
                "metadata": {
                    "source": f"finalize_{flow_mode}",
                    "flow_mode": flow_mode,
                    "generation_metadata": state.get("generation_metadata", {}),
                },
            },
        }

    def _build_generation_request(self, flow_mode: str, state: Dict[str, Any]) -> Dict[str, Any]:
        mode = "tutela_init" if flow_mode == "tutela_init" else "chat_edit"
        goal = "draft_official_response" if flow_mode == "tutela_init" else "edit_draft"
        
        # Determinar la operación según el flujo y el objetivo
        if flow_mode == "tutela_init":
            operation = "compose"
        elif goal == "edit_draft":
            operation = "edit"
        else:
            operation = "chat"
        
        plan = self._generation_plan(flow_mode=flow_mode, state=state)
        constraints = self._generation_constraints(flow_mode=flow_mode)

        req: Dict[str, Any] = {
            "goal": goal,
            "mode": mode,
            "operation": operation,  # Campo crítico para mantener consistencia en metadata
            "plan": plan,
            "constraints": constraints,
            "current_message_id": (state.get("conversation_context") or {}).get("current_message_id"),
        }
        if mode == "chat_edit":
            req["edit_directives"] = self._extract_delta_instructions(state.get("user_query", ""))
        return req

    # ==========================================================================
    # Routing meta (LLM/rules)
    # ==========================================================================

    async def _routing_meta(self, *, user_query: str, policy_mode: str, route_timeout_ms: int, trace_id: str) -> Dict[str, Any]:
        start = time.time()
        meta: Dict[str, Any] = {
            "needs_pgvector_ctx": False,
            "needs_neo4j_ctx": False,
            "policy_source": "rules",
            "rationale": "Fallback rule-based routing",
            "elapsed_ms": 0,
        }

        if policy_mode != "llm":
            meta["rationale"] = "Policy set to rules"
            meta["elapsed_ms"] = int((time.time() - start) * 1000)
            return meta

        svc = "ollama_routing"
        if not await self.security_service.check_circuit_breaker_recovery(svc):
            meta["rationale"] = "Circuit open; rules fallback"
            meta["elapsed_ms"] = int((time.time() - start) * 1000)
            return meta

        res = await self.security_service.monitor_resources()
        if res.cpu_percent > self.security_service.dos_settings.max_cpu_usage_percent or res.memory_percent > self.security_service.dos_settings.max_memory_usage_percent:
            await self.security_service.update_circuit_breaker(svc, False)
            meta["rationale"] = "High resource usage; rules fallback"
            meta["elapsed_ms"] = int((time.time() - start) * 1000)
            return meta

        system_prompt = """Eres un agente de enrutamiento maestro. Su único trabajo es analizar la consulta del usuario y decidir qué contexto de datos se necesita para responderlo.
            No intente responder a la consulta usted mismo.

            Tienes dos fuentes de contexto disponibles:

            1. ** Contexto PGVector (memoria de conversación) **: Esta fuente de contexto contiene la historia de nuestra conversación actual.
            Úselo solo cuando la consulta del usuario se refiera directamente a lo que se dijo anteriormente.
            2. ** contexto neo4j (conocimiento legal) **: Esta fuente de contexto contiene todo nuestro conocimiento legal (normas, tutelas, documentos).
            Use esto para cualquier consulta que solicite información legal.

            Basado en la consulta, responda solo con un objeto JSON válido.
            No agregue ningún texto introductorio.

            El objeto JSON debe contener estas claves:
            "Needs_pgvector_ctx": boolean
            "Needs_neo4j_ctx": boolean
            "rationale": una explicación breve y clara para su decisión.

            ---
            ** Ejemplos: **

            ** Consulta de usuario: ** "¿Qué es una tutela?"
            ** Tu respuesta JSON: **
            {
            "Needs_pgvector_ctx": False,
            "Needs_neo4j_ctx": True,
            "rationale": "El usuario solicita una definición legal externa. Esto requiere un contexto neo4j".
            }

            ** Consulta de usuario: ** "¿Puedes Recordarme la Pregunta que te hice Antes?"
            ** Tu respuesta JSON: **
            {
            "Needs_pgvector_ctx": True,
            "Needs_neo4j_ctx": False,
            "rationale": "El usuario pregunta sobre nuestro historial de conversación. Esto requiere un contexto de PGVector ".
            }

            ** Consulta de usuario: ** "Basado en Las Leyes Que Mencionaste, ¿Como se relaciona eso con la tutela?"
            ** Tu respuesta JSON: **
            {
            "Needs_pgvector_ctx": True,
            "Needs_neo4j_ctx": True,
            "rationale": "La consulta combina una referencia a la conversación pasada ('mi problema inicio') con información legal ('Leyes que mencionaste'). Necesita ambos contextos".
            }

            ** Consulta de usuario: ** "Modifica el texto de la respuesta borrador para que se ajuste a la respuesta real de la tutela"
            ** Tu respuesta JSON: **
            {
            "Needs_pgvector_ctx": True,
            "Needs_neo4j_ctx": True,
            "rationale": "La consulta es una peticion de modificacion del texto de la respuesta borrador con base en la informacion legal".
            }

            ** Consulta de usuario: ** "Modifica el texto de la respuesta borrador agregando una seccion de conclusiones"
            ** Tu respuesta JSON: **
            {
            "Needs_pgvector_ctx": True,
            "Needs_neo4j_ctx": False,
            "rationale": "La consulta es una peticion de modificacion del texto de la respuesta borrador que no necesita de contexto legal".
            }

            ** Consulta de usuario: ** "¿Cuál es la Capital de Colombia?"
            ** Tu respuesta JSON: **
            {
            "Needs_pgvector_ctx": False,
            "Needs_neo4j_ctx": False,
            "rationale": "La consulta es una pregunta de conocimiento general y no requiere memoria de conversación o contexto legal especializado".
            } """
        msg = [{"role": "user", "content": f"{system_prompt}\n\nQuery: {user_query}"}]

        try:
            raw = await asyncio.wait_for(
                self.ollama_service.generate_text(messages=msg, max_tokens=10000, temperature=0.1),
                timeout=route_timeout_ms / 1000,
            )
            json_start, json_end = raw.find("{"), raw.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                parsed = _json.loads(raw[json_start:json_end])
                meta["needs_pgvector_ctx"] = bool(parsed.get("needs_pgvector_ctx", False))
                meta["needs_neo4j_ctx"] = bool(parsed.get("needs_neo4j_ctx", False))
                meta["rationale"] = parsed.get("rationale", "LLM-based routing decision")
                meta["policy_source"] = "llm_gemma_ollama"
                await self.security_service.update_circuit_breaker(svc, True)
            else:
                await self.security_service.update_circuit_breaker(svc, False)
                meta["rationale"] = "LLM returned no JSON; rules fallback"
        except Exception as e:
            await self.security_service.update_circuit_breaker(svc, False)
            meta["rationale"] = f"LLM routing failed: {e}"

        meta["elapsed_ms"] = int((time.time() - start) * 1000)
        return meta

    # ==========================================================================
    # Flow helpers
    # ==========================================================================

    async def _classify_intent_with_embeddings(self, text: str, conversation_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clasificación basada 100% en LLM - SIN HEURÍSTICAS.

        El LLM decide si el mensaje del usuario es:
        - EDIT: Instrucción para modificar el borrador existente
        - CHAT: Pregunta o conversación general

        El LLM tiene contexto de si existe un borrador previo.
        """
        logger.info(
            "CLASSIFICATION: Iniciando clasificación con LLM (sin heurísticas)",
            extra={
                "text_length": len(text),
            },
        )

        has_previous_draft = bool(conversation_context.get("latest_draft") or conversation_context.get("current_draft_id") or conversation_context.get("has_draft_history", False))

        logger.info(
            "CLASSIFICATION: Evaluación de contexto",
            extra={
                "has_previous_draft": has_previous_draft,
                "latest_draft": bool(conversation_context.get("latest_draft")),
                "current_draft_id": bool(conversation_context.get("current_draft_id")),
                "has_draft_history": conversation_context.get("has_draft_history", False),
            },
        )

        # ==============================================================================
        # CLASIFICACIÓN CON LLM (SIN HEURÍSTICAS)
        # ==============================================================================
        try:
            result = await self._classify_with_llm(text, has_previous_draft)
            logger.info("CLASSIFICATION: LLM completó clasificación", extra={"result": result})
            return result
        except Exception as e:
            logger.error(
                "CLASSIFICATION: Error en clasificación con LLM",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            # Fallback seguro: si hay borrador previo → EDIT, sino → CHAT
            if has_previous_draft:
                return {
                    "operation": "edit",
                    "goal": "edit_draft",
                    "confidence": 0.5,
                    "reasoning": "Fallback por error: existe borrador previo",
                }
            else:
                return {
                    "operation": "chat",
                    "goal": "direct_answer",
                    "confidence": 0.5,
                    "reasoning": "Fallback por error: no existe borrador previo",
                }

    async def _get_pattern_embeddings(self, patterns: List[str]) -> List[List[float]]:
        """
        Obtener embeddings con caché para patrones.

        Utiliza un caché interno para evitar regenerar embeddings de los mismos
        patrones múltiples veces, mejorando significativamente el rendimiento.
        """
        embeddings = []
        for pattern in patterns:
            if pattern not in self._pattern_embeddings_cache:
                try:
                    self._pattern_embeddings_cache[pattern] = await self.embedding_service.generate_embedding(pattern)
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for pattern '{pattern}': {e}")
                    # Usar embedding vacío como fallback con dimensión correcta
                    embedding_dim = getattr(self.embedding_service, "get_embedding_dimension", lambda: 384)()
                    self._pattern_embeddings_cache[pattern] = [0.0] * embedding_dim
            embeddings.append(self._pattern_embeddings_cache[pattern])
        return embeddings

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calcular similitud coseno entre dos vectores."""
        try:
            import numpy as np

            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)
            dot_product = np.dot(vec1_np, vec2_np)
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot_product / (norm1 * norm2))
        except Exception as e:
            logger.warning(f"Cosine similarity calculation failed: {e}")
            return 0.0

    async def _classify_with_llm(self, text: str, has_previous_draft: bool) -> Dict[str, Any]:
        """
        Clasificación con LLM - SIN HEURÍSTICAS.

        Request ultra compacto al LLM para decidir si el mensaje es EDIT o CHAT.
        """
        system_prompt = """Eres un clasificador de intenciones. Responde SOLO con un JSON válido.

Tu tarea: Clasificar el mensaje del usuario en una de dos categorías:
- EDIT: El usuario quiere MODIFICAR un documento/borrador existente
- CHAT: El usuario hace una PREGUNTA o tiene una conversación general

Contexto: {context}

Responde EXACTAMENTE en este formato JSON (sin markdown, sin explicaciones):
{{"operation": "edit", "goal": "edit_draft", "confidence": 0.95, "reasoning": "razón breve"}}
o
{{"operation": "chat", "goal": "direct_answer", "confidence": 0.95, "reasoning": "razón breve"}}"""

        context_info = "El usuario tiene un borrador previo en la conversación." if has_previous_draft else "No hay borrador previo en la conversación."

        user_prompt = f"""Mensaje del usuario: "{text}"

Clasifica como EDIT o CHAT."""

        messages = [
            {"role": "system", "content": system_prompt.format(context=context_info)},
            {"role": "user", "content": user_prompt}
        ]

        try:
            # Llamada al LLM con parámetros mínimos
            raw_response = await self.ollama_service.generate_text(
                messages=messages,
                max_tokens=150,  # Suficiente para el JSON
                temperature=0.0,  # Determinista
                extra_args={}
            )

            logger.info(
                "CLASSIFICATION: LLM response raw",
                extra={"raw_response": raw_response[:200]}
            )

            # Parsear JSON de la respuesta
            import json
            import re

            # Limpiar la respuesta (por si tiene markdown)
            cleaned = raw_response.strip()
            # Buscar el JSON en la respuesta
            json_match = re.search(r'\{[^}]+\}', cleaned)
            if json_match:
                result = json.loads(json_match.group(0))

                # Validar campos requeridos
                if "operation" in result and "goal" in result:
                    logger.info(
                        "CLASSIFICATION: LLM clasificó correctamente",
                        extra={"result": result}
                    )
                    return result

            # Si no se pudo parsear, fallback
            raise ValueError("No se pudo parsear JSON de respuesta LLM")

        except Exception as e:
            logger.error(
                "CLASSIFICATION: Error en _classify_with_llm",
                extra={"error": str(e), "text_preview": text[:100]}
            )
            raise

    def _classify_intent_fallback(self, text: str) -> Dict[str, Any]:
        """Fallback con keywords si embeddings fallan."""
        t = text.lower()
        edit_indicators = [
            "modifica",
            "cambia",
            "edita",
            "corrige",
            "elimina",
            "quita",
            "agrega",
            "añade",
            "reescribe",
            "reformula",
            "mejora",
            "actualiza",
            "según lo anterior",
            "del documento",
            "de la respuesta",
            "del borrador",
        ]

        is_edit = any(keyword in t for keyword in edit_indicators)

        return {
            "operation": "edit" if is_edit else "chat",
            "goal": "edit_draft" if is_edit else "direct_answer",
            "confidence": 0.7,
            "reasoning": "Keyword fallback classification",
        }

    async def _classify_intent(self, text: str) -> str:
        """
        Método legacy mantenido por compatibilidad.
        """
        t = text.lower()
        edit = [
            "edita",
            "mejora",
            "corrige",
            "reformula",
            "reescribe",
            "redacta",
            "más formal",
            "mas formal",
            "menos formal",
            "formaliza",
        ]
        cite = ["jurisprudencia", "precedentes", "sentencia", "c-", "t-", "su-"]
        if any(k in t for k in edit):
            return "EDIT_DRAFT"
        if any(k in t for k in cite):
            return "CITE_PRECEDENTS"
        return "ANSWER_PLAIN"

    def _derive_flow_mode(self, state: Dict[str, Any]) -> str:
        if state.get("flow_mode"):
            return state["flow_mode"]
        if state.get("is_first_interaction"):
            return "tutela_init"
        return "chat_edit"

    def _generation_plan(self, *, flow_mode: str, state: Dict[str, Any]) -> Dict[str, Any]:
        # Flags del workflow para relajar validación en texto-solo
        wf_meta = state.get("workflow_metadata") or {}
        policy_flags = wf_meta.get("policy_flags") or {}
        allow_text_only_init = bool(policy_flags.get("allow_text_only_init"))

        if flow_mode == "tutela_init":
            if allow_text_only_init:
                # Modo inicial con solo texto: no exigir estructura/citas
                return {
                    "language": "es",
                    "tone": "formal_institucional",
                    "structure": [],  # ← sin chequeo de secciones
                    "cite_legal_sources": False,  # ← sin chequeo de citas
                    "use_neo4j_context": True,
                    "use_conversation_memory": False,
                }
            # Modo estricto (cuando sí haya doc/contexto robusto)
            return {
                "language": "es",
                "tone": "formal_institucional",
                "structure": [
                    "encabezado_oficial",
                    "hechos_relevantes",
                    "competencia",
                    "problema_juridico",
                    "consideraciones",
                    "jurisprudencia_citada",
                    "resuelve",
                    "firma_y_notificacion",
                ],
                "cite_legal_sources": True,
                "use_neo4j_context": True,
                "use_conversation_memory": False,
            }
        return {
            "language": "es",
            "tone": "formal_institucional",
            "structure": "preserve_existing_sections",
            "edit_policy": "minimal_diff",
            "cite_legal_sources": True,
            "use_neo4j_context": True,
            "use_conversation_memory": True,
        }

    def _generation_constraints(self, *, flow_mode: str) -> Dict[str, Any]:
        base = {
            "max_tokens": 12000,
            "validate_sections": True,
            "avoid_hallucinations": True,
            "cite_format": "footnote_like",
        }
        if flow_mode == "tutela_init":
            # Allow initial flow to work with inline text only (no external document).
            base.update(
                {
                    "require_all_sections": True,
                    "allow_text_only_init": True,
                    "require_source_document": False,
                }
            )
        else:
            base.update({"require_all_sections": False, "limit_changes_to_sections": True})
        return base

    def _extract_delta_instructions(self, user_query: str) -> Dict[str, Any]:
        return {"raw": user_query}

    def _validate_state_shape(self, sd: Dict[str, Any]) -> bool:
        return bool(sd.get("user_query") and sd.get("conversation_id"))

    # ==========================================================================
    # Spoke retry decisions
    # ==========================================================================

    def _decide_retry_pgvector(self, state: Dict[str, Any], attempt: int, cfg: Dict[str, Any]) -> Tuple[bool, str]:
        write_status = state.get("pgvector_write_status") or {}

        if write_status:
            ok = bool(write_status.get("success") or write_status.get("ok"))
            if not ok:
                code = write_status.get("code") or "persist_error"
                return True, f"pgvector_persist_failed:{code}"

        # No reintentar por "no results"; solo por fallo explícito.
        # (need_results variable removed as part of heuristics elimination)

        return False, "ok"

    def _decide_retry_neo4j(self, state: Dict[str, Any], attempt: int, cfg: Dict[str, Any]) -> Tuple[bool, str]:
        neo_status = state.get("neo4j_status") or {}
        graph_results = state.get("graph_results")

        if neo_status:
            ok = bool(neo_status.get("success") or neo_status.get("ok"))
            if not ok:
                code = neo_status.get("code") or "neo4j_error"
                return True, f"neo4j_failed:{code}"

        # No reintentos por ausencia de hallazgos si no hubo error.
        if not self._has_graph_results(graph_results):
            return False, "ok"

        return False, "ok"

    # ==========================================================================
    # Final validation
    # ==========================================================================

    def _validate_final_response(self, final_response: Any, plan: Dict[str, Any]) -> Tuple[bool, str]:
        # Validación mínima, sin heurística de contenido:
        if not final_response or not isinstance(final_response, str):
            return False, "missing_or_nonstring"
        if len(final_response.strip()) == 0:
            return False, "empty"
        return True, "ok"

    def _tune_generation_request_for_retry(self, state: Dict[str, Any], issues: str) -> Dict[str, Any]:
        req = (state.get("generation_request") or {}).copy()
        plan = (req.get("plan") or {}).copy()
        constraints = (req.get("constraints") or {}).copy()

        if issues.startswith("missing_sections"):
            constraints["validate_sections"] = True
            constraints["require_all_sections"] = True
        elif issues == "missing_citations":
            plan["cite_legal_sources"] = True
        elif issues == "too_short":
            constraints["max_tokens"] = max(int(constraints.get("max_tokens", 10000)), 10000)
        elif issues == "contains_placeholder":
            plan["disallow_placeholders"] = True

        req["plan"] = plan
        req["constraints"] = constraints
        return req

    # ==========================================================================
    # Retry policy + control
    # ==========================================================================

    def _get_retry_policy(self, state: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
        cfg = (state.get("config") or {}).get("retries") or {}
        agent_cfg = cfg.get(agent_name) or {}
        base = self.default_retry_cfg.get(agent_name, {})
        return {
            "max_retries": int(agent_cfg.get("max_retries", base.get("max_retries", 1))),
            "base_sleep_ms": int(agent_cfg.get("base_sleep_ms", base.get("base_sleep_ms", 400))),
            "max_sleep_ms": int(agent_cfg.get("max_sleep_ms", base.get("max_sleep_ms", 6000))),
            "exp_base": float(agent_cfg.get("exp_base", base.get("exp_base", 2.0))),
            "jitter": bool(agent_cfg.get("jitter", base.get("jitter", True))),
        }

    def _compute_backoff_ms(self, attempt: int, cfg: Dict[str, Any]) -> int:
        base = max(1, int(cfg.get("base_sleep_ms", 400)))
        exp_base = max(1.1, float(cfg.get("exp_base", 2.0)))
        max_sleep = max(base, int(cfg.get("max_sleep_ms", 6000)))
        raw = int(base * (exp_base ** max(0, attempt - 1)))
        sleep_ms = min(raw, max_sleep)
        if cfg.get("jitter", True):
            jitter = int(sleep_ms * 0.2)
            sleep_ms = max(0, sleep_ms + random.randint(-jitter, jitter))
        return sleep_ms

    def _merge_control_delta(
        self,
        state: Dict[str, Any],
        control_delta: Dict[str, Any],
        *,
        phase: str,
        trace_id: str,
    ) -> Dict[str, Any]:
        """Fusiona control actual con delta y devuelve SOLO deltas (no clona el estado)."""
        control = (state.get("control") or {}).copy()
        attempts = control.get("attempts") or {}
        agent = control_delta.get("retry_agent")
        next_attempt = control_delta.get("next_attempt")
        if agent and next_attempt is not None:
            attempts[agent] = int(next_attempt)
        control["attempts"] = attempts

        for k in (
            "retry",
            "retry_agent",
            "retry_reason",
            "retry_sleep_ms",
            "final_valid",
        ):
            if k in control_delta:
                control[k] = control_delta[k]

        event = {
            "phase": phase,
            "trace_id": trace_id,
            "ts": time.time(),
            "retry": control.get("retry"),
            "retry_agent": control.get("retry_agent"),
            "retry_sleep_ms": control.get("retry_sleep_ms"),
            "next_attempts": control.get("attempts"),
            "reason": control.get("retry_reason"),
        }

        logger.info(
            "Orchestrator control updated",
            extra={
                "trace_id": trace_id,
                "phase": phase,
                "control": {
                    k: control.get(k)
                    for k in (
                        "retry",
                        "retry_agent",
                        "retry_sleep_ms",
                        "retry_reason",
                        "attempts",
                    )
                },
            },
        )
        return {"control": control, "orchestrator_event": event}

    def _has_graph_results(self, graph_results: Any) -> bool:
        if graph_results is None:
            return False
        if hasattr(graph_results, "total_found"):
            try:
                return (getattr(graph_results, "total_found") or 0) > 0
            except Exception:
                return False
        if isinstance(graph_results, dict):
            return bool(graph_results.get("total_found", 0) > 0 or graph_results.get("entities") or graph_results.get("relationships") or graph_results.get("insights") or graph_results.get("results"))
        return False

    # ==========================================================================
    # Error factories (dict plano)
    # ==========================================================================

    def _blocked_update(self, trace_id: str, reason: str) -> Dict[str, Any]:
        return {
            "orchestrator_decision": {
                "use_vector_search": False,
                "use_graph_analysis": False,
                "direct_generation": False,
                "reasoning": f"Security block: {reason}",
                "confidence": 0.0,
            },
            "workflow_metadata": {
                "security_block": {
                    "trace_id": trace_id,
                    "reason": reason,
                    "timestamp": time.time(),
                }
            },
            "orchestrator_result": {
                "agent": "orchestrator",
                "trace_id": trace_id,
                "success": False,
                "security_blocked": True,
                "phase": "plan",
            },
            "orchestrator_success": False,
            "final_response": f"Request blocked for security reasons: {reason}",
        }

    def _timeout_update(self, trace_id: str, timeout_ms: int) -> Dict[str, Any]:
        return {
            "orchestrator_decision": {
                "use_vector_search": False,
                "use_graph_analysis": False,
                "direct_generation": False,
                "reasoning": f"Request timeout after {timeout_ms}ms",
                "confidence": 0.0,
            },
            "workflow_metadata": {
                "timeout_error": {
                    "trace_id": trace_id,
                    "timeout_ms": timeout_ms,
                    "timestamp": time.time(),
                }
            },
            "orchestrator_result": {
                "agent": "orchestrator",
                "trace_id": trace_id,
                "success": False,
                "timeout": True,
                "phase": "plan",
            },
            "orchestrator_success": False,
            "final_response": f"Request timed out after {timeout_ms}ms.",
        }

    def _error_update(self, *, trace_id: str, message: str, reason: str = "error") -> Dict[str, Any]:
        return {
            "orchestrator_decision": {
                "use_vector_search": False,
                "use_graph_analysis": False,
                "direct_generation": False,
                "reasoning": f"{reason}: {message}",
                "confidence": 0.0,
            },
            "workflow_metadata": {
                "error": {
                    "trace_id": trace_id,
                    "message": message,
                    "timestamp": time.time(),
                }
            },
            "orchestrator_result": {
                "agent": "orchestrator",
                "trace_id": trace_id,
                "success": False,
                "error": message,
                "phase": "plan",
            },
            "orchestrator_success": False,
            "final_response": f"Request failed: {message}",
        }

    def _is_internal_graph_operation(self, state: Dict[str, Any]) -> bool:
        """
        Determina si esta es una operación interna del grafo vs una request externa.

        Criterios:
        - Tiene workflow_metadata con trace_id existente (ya está en progreso)
        - Tiene pgvector_request o neo4j_request (operación de retrieval/persistencia)
        - Tiene final_response (operación de finalización)
        - No es la primera interacción con requests pendientes
        """
        workflow_metadata = state.get("workflow_metadata", {})

        # Si ya tiene trace_id en metadata, es una operación en progreso
        if workflow_metadata.get("trace_id"):
            return True

        # Si tiene requests específicos de agentes, es operación interna
        if any(
            [
                state.get("pgvector_request"),
                state.get("neo4j_request"),
                state.get("generation_request"),
            ]
        ):
            return True

        # Si ya tiene respuesta final, es operación de finalización
        if state.get("final_response"):
            return True

        # Si no es la primera interacción, es continuación del flujo
        if not state.get("is_first_interaction", True):
            return True

        return False
