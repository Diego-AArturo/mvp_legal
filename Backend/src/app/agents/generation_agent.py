"""GGeneración minimal para MVP (compose, edit, chat) 
- Usa OllamaService para generar texto según la operation solicitada.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from app.config.settings import Settings
from app.extensions import get_logger
from app.services.ai.ollama_service import OllamaService

logger = get_logger(__name__)


class GenerationAgent:
    """Agente de generación minimalista para compose / edit / chat."""

    def __init__(self, settings: Settings, neo4j_agent: Any = None) -> None:
        self.settings = settings
        self.ollama_service = OllamaService(settings)
        self.neo4j_agent = neo4j_agent  # compatibilidad DI

        self.default_max_tokens = settings.models.generation.generation_max_response_length
        self.default_temperature = 0.1
        self.max_payload_chars = getattr(settings.models.generation, "max_payload_chars", 200_000)
        self.section_char_limit = getattr(settings.models.generation, "section_char_limit", 80_000)

    # ------------------------------------------------------------------ #
    # Entry point                                                        #
    # ------------------------------------------------------------------ #

    async def run(self, state: Any) -> Dict[str, Any]:
        sd = self._to_state_dict(state)
        trace_id = (sd.get("workflow_metadata") or {}).get("trace_id") or f"gen-{int(time.time())}"
        gen_req: Dict[str, Any] = sd.get("generation_request") or {}
        operation = str(gen_req.get("operation") or "chat").strip().lower()
        goal = gen_req.get("goal") or self._default_goal(operation)
        user_query = sd.get("user_query", "") or ""
        conversation_id = sd.get("conversation_id", "")

        params = gen_req.get("params") or {}
        max_tokens = int(params.get("max_tokens", self.default_max_tokens))
        temperature = float(params.get("temperature", self.default_temperature))
        extra_args = {k: v for k, v in params.items() if k not in {"max_tokens", "temperature"}}
        

        # Base text y contexto
        conversation_context = sd.get("conversation_context") or {}
        base_text = gen_req.get("base_text") or self._resolve_base_text(conversation_context)
        graph_results = sd.get("graph_results")
        vector_results = sd.get("vector_results")

        # Construir mensajes para el LLM (sin alterar el contenido de los prompts definidos)
        system_directive = self._system_directive(operation=operation, goal=goal)
        system_msg = {"role": "system", "content": system_directive}

        if operation == "compose":
            if not base_text:
                raise ValueError("compose operation requires 'base_text'")
            user_msg = {
                "role": "user",
                "content": (
                    f"[DOC BASE]\n{self._trim_if_needed(base_text, self.section_char_limit)}\n"
                    f"[INSTRUCCIONES]\n{user_query}\n"
                    f"[CONTEXTO LEGAL]\n{self._format_graph_results_for_llm(graph_results, operation='compose')}"
                ),
            }
        elif operation == "edit":
            base = base_text or ""
            user_msg = {
                "role": "user",
                "content": (
                    f"[DOC BASE]\n{self._trim_if_needed(base, self.section_char_limit)}\n"
                    f"[INSTRUCCIONES]\n{user_query}\n"
                    f"[CONTEXTO LEGAL]\n{self._format_graph_results_for_llm(graph_results, operation='edit')}"
                ),
            }
        else:  # chat
            context_bundle = {
                "graph_results": graph_results,
                "vector_results": vector_results,
                "conversation_context": conversation_context,
            }
            user_msg = {"role": "user", "content": self._user_payload(user_query=user_query, context_bundle=context_bundle)}

        messages = [system_msg, user_msg]

        gen_timeout_ms = int((sd.get("config") or {}).get("timeouts", {}).get("generation", 95000))

        logger.info(
            "GenerationAgent: starting LLM call",
            extra={
                "trace_id": trace_id,
                "operation": operation,
                "goal": goal,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )

        try:
            final_response = await asyncio.wait_for(
                self._call_llm(messages, max_tokens, temperature, extra_args),
                timeout=gen_timeout_ms / 1000.0,
            )
            attempts_prev = sd.get("generation_attempts", 0) or 0
            persistence_kind = self._persistence_kind(operation)

            return {
                "final_response": final_response,
                "generation_metadata": {
                    "mode": sd.get("flow_mode", "chat_edit"),
                    "goal": goal,
                    "operation": operation,
                    "response_length": len(final_response or ""),
                    "language": "es",
                },
                "generation_status": {"success": True, "operation": operation, "goal": goal},
                "generation_attempts": attempts_prev + 1,
                "persistence_request": {
                    "action": "persist",
                    "conversation_id": conversation_id,
                    "message": {
                        "role": "assistant",
                        "kind": persistence_kind,
                        "content": final_response,
                        "metadata": {"source": operation},
                    },
                },
            }
        except asyncio.TimeoutError:
            fb = "No fue posible generar la respuesta dentro del tiempo configurado."
            attempts_prev = sd.get("generation_attempts", 0) or 0
            return {
                "final_response": fb,
                "generation_status": {"success": False, "error": "timeout", "operation": operation, "goal": goal},
                "generation_metadata": {"mode": sd.get("flow_mode", "chat_edit"), "goal": goal, "operation": operation},
                "generation_attempts": attempts_prev + 1,
                "persistence_request": None,
            }
        except Exception as e:
            fb = f"Error al generar la respuesta: {e}"
            attempts_prev = sd.get("generation_attempts", 0) or 0
            return {
                "final_response": fb,
                "generation_status": {"success": False, "error": str(e), "operation": operation, "goal": goal},
                "generation_metadata": {"mode": sd.get("flow_mode", "chat_edit"), "goal": goal, "operation": operation},
                "generation_attempts": attempts_prev + 1,
                "persistence_request": None,
            }

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _default_goal(self, operation: str) -> str:
        if operation == "compose":
            return "draft_official_response"
        if operation == "edit":
            return "edit_draft"
        return "direct_answer"

    def _persistence_kind(self, operation: str) -> str:
        if operation == "compose":
            return "initial_draft"
        if operation == "edit":
            return "revision"
        return "chat_response"

    def _resolve_base_text(self, ctx: Dict[str, Any]) -> str:
        if not isinstance(ctx, dict):
            return ""
        for key in ("latest_draft", "current_draft", "initial_petition"):
            val = ctx.get(key)
            if isinstance(val, dict):
                text = val.get("content") or val.get("texto") or ""
            else:
                text = val or ""
            if text:
                return str(text)
        return ""

    def _to_state_dict(self, state: Any) -> Dict[str, Any]:
        if state is None:
            return {}
        if isinstance(state, Mapping):
            return dict(state)
        if hasattr(state, "model_dump"):
            try:
                return state.model_dump()  # type: ignore[attr-defined]
            except Exception:
                pass
        if hasattr(state, "dict"):
            try:
                return state.dict()  # type: ignore[attr-defined]
            except Exception:
                pass
        if is_dataclass(state):
            return asdict(state)
        return getattr(state, "__dict__", {}) or {}

    def _trim_if_needed(self, text: Optional[str], limit: int) -> str:
        if not isinstance(text, str):
            return "" if text is None else str(text)
        if limit and len(text) > limit:
            return text[: max(0, limit - 1000)] + "\n\n[... contenido truncado por tamaño ...]"
        return text

    def _stringify_dates(self, value: Any) -> Any:
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, dict):
            return {k: self._stringify_dates(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._stringify_dates(item) for item in value]
        return value

    def _safe_json_dumps(self, obj: Any, per_section_limit: Optional[int] = None) -> str:
        try:
            if hasattr(obj, "model_dump"):
                obj = obj.model_dump()
            elif hasattr(obj, "dict"):
                obj = obj.dict()
            raw = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.warning("Falló serialización JSON: %s", e)
            raw = str(obj)
        return self._trim_if_needed(raw, per_section_limit or self.section_char_limit)

    def _format_graph_results_for_llm(
        self,
        graph_results: Any,
        max_results: int = 5,
        max_text_length: int = 500,
        operation: str = "compose",
    ) -> str:
        if not graph_results:
            return ""

        if hasattr(graph_results, "model_dump"):
            graph_dict = graph_results.model_dump()
        elif hasattr(graph_results, "dict"):
            graph_dict = graph_results.dict()
        elif isinstance(graph_results, dict):
            graph_dict = graph_results
        else:
            return ""

        if operation == "compose":
            max_results = min(max_results, 8)
            max_text_length = 800
        elif operation == "edit":
            max_results = min(max_results, 5)
            max_text_length = 600
        else:
            max_results = min(max_results, 3)
            max_text_length = 400

        results = graph_dict.get("results", []) or []
        if hasattr(results, "__iter__") and not isinstance(results, list):
            results = list(results)

        def _format_item(item: Any) -> str:
            try:
                if hasattr(item, "model_dump"):
                    item = item.model_dump()
                elif hasattr(item, "__dict__"):
                    item = item.__dict__
                if not isinstance(item, dict):
                    return ""
                nombre = item.get("nombre") or item.get("documento") or item.get("id", "")
                texto = item.get("texto_relevante") or item.get("texto") or ""
                texto = self._trim_if_needed(str(texto), max_text_length)
                score = item.get("score")
                conceptos = item.get("conceptos_clave") or []
                normas = item.get("normas_citadas") or []
                return (
                    f"- {nombre} (score: {round(score, 3) if isinstance(score, (int, float)) else score})\n"
                    f"  Conceptos: {', '.join(map(str, conceptos[:5]))}\n"
                    f"  Normas: {', '.join(map(str, normas[:5]))}\n"
                    f"  Texto: {texto}"
                )
            except Exception:
                return ""

        formatted = [_format_item(item) for item in results[:max_results]]
        return "\n".join([f"[RESULTADOS NEO4J - Estrategia {graph_dict.get('search_strategy', 'desconocida')}]:"] + formatted)

    async def _call_llm(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        extra_args: Dict[str, Any],
    ) -> str:
        raw = await self.ollama_service.generate_text(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_args=extra_args,
        )
        return raw.strip() if isinstance(raw, str) else str(raw)

    # ------------------------------------------------------------------ #
    # Prompts                                                           #
    # ------------------------------------------------------------------ #

    def _system_directive(self, *, operation: str, goal: str) -> str:
        """
        System directive general y reusable.
        Sirve para generar, editar y chatear con soporte RAG.
        """
        

        if operation == "edit":
            return self._prompt_edit_generic()

        if operation == "compose":
            base = (
                "Eres un asistente profesional experto en derecho administrativo.\n"
                "Tu tarea es generar un borrador claro, preciso y bien estructurado a partir del documento base.\n"
                "Usa el contexto proporcionado solo si es relevante.\n"
                "No inventes datos. Si la información es insuficiente, indícalo claramente.\n"
                "Devuelve SOLO el documento final, sin comentarios ni análisis.\n"
                "Escribe en español neutro, formal pero comprensible."
            )
            normalized_goal = str(goal or "").strip().lower()
            if normalized_goal == "draft_official_response":
                base += (
                    """**TAREA DE ROL: ACCIONADO**

                    **Misión:** Redactar la defensa principal de la entidad.
                    **Argumento Clave:** Demostrar la **debida diligencia** de la Registraduría (que ya estamos actuando o que actuamos correctamente según la ley).

                    ---
                    **INSTRUCCIÓN:** Redacta la contestación usando **EXACTAMENTE** la siguiente plantilla.
                    No añadas secciones. No cambies los títulos. Rellena los marcadores `[MARCADOR]` con los datos proporcionados.
                    Sigue la `REGLA DE RELLENO` para el argumento (Sección IV) y la `REGLA DE PETICIÓN` para la petición (Sección VI).

                    ---------
                    ---------
                    [Ciudad y Fecha Actual]

                    Doctor(a)
                    **[Nombre del Juez/Magistrado - si no se sabe, usar 'JUEZ DE TUTELA']**
                    **[Nombre del Juzgado - si no se sabe, usar 'JUZGADO DE REPARTO']**
                    [Email del Juzgado - si se sabe]
                    E. S. D.

                    **Referencia:** Acción de Tutela
                    **Radicado:** [Número de Radicado de la tutela]
                    **Accionante:** [Nombre del Accionante]
                    **Accionado:** **REGISTRADURÍA NACIONAL DEL ESTADO CIVIL**
                    [Opcional: Listar otras entidades accionadas o vinculadas si se mencionan]
                    **RNEC Interno:** [Número Interno RNEC si se tiene, si no, omitir esta línea]

                    Respetado Doctor(a) Juez:

                    [Párrafo de presentación: "Yo, [Nombre del Jefe Jurídico], en mi calidad de Jefe de la Oficina Jurídica de la REGISTRADURÍA NACIONAL DEL ESTADO CIVIL (RNEC), conforme al artículo 33 del Decreto 1010 de 2000, presento contestación a la acción de tutela de la referencia..."]

                    **I. PLANTEAMIENTO DE LA ACCIÓN CONSTITUCIONAL**
                    [Los hechos clave de la tutela, basado en <tutela_a_responder>.]

                    **II. PRETENSIONES**
                    [Cita textualmente (en bloque de comillas) las pretensiones del accionante que están en <tutela_a_responder>.]

                    **III. NIVELES DE COMPETENCIA**
                    [Explica por qué la RNEC SÍ es competente para este trámite (ej. Registro Civil, Identificación), usando el Decreto 1010 del <contexto_juridico_neo4j>.]

                    **IV. CONSIDERACIÓN FRENTE A LA SITUACIÓN**
                    [Rellena esta sección siguiendo la REGLA DE RELLENO PARA EL ARGUMENTO.]

                    **V. ANEXOS**
                    [Enlista las pruebas reales que demuestran la acción (ej. "Correo electrónico remisorio por competencia", "Consulta a sistemas SIRC"). Si no hay, indica "No se aportan anexos.".]

                    **VI. PETICIÓN**
                    [Rellena esta sección usando EXACTAMENTE UNA de las 3 redacciones de la REGLA DE PETICIÓN OBLIGATORIA.]

                    **VII. NOTIFICACIONES**
                    [Indica los datos de notificación de la Oficina Jurídica.]

                    Atentamente,

                    (Firma)
                    **[NOMBRE DEL JEFE DE LA OFICINA JURÍDICA]**
                    Jefe de la Oficina Jurídica
                    Registraduría Nacional del Estado Civil

                    ---
                    
                    ---

                    ### REGLAS DE RELLENO (OBLIGATORIAS)

                    **REGLA DE RELLENO PARA IV. CONSIDERACIÓN (EL ARGUMENTO)**
                    Tu argumento debe ser claro y seguir esta lógica:
                    1.  **Presentar Hechos Verificados:** Describe detalladamente lo que la entidad encontró en los sistemas internos (`<verificacion_interna_accionante>`). Contrasta esto con lo que alega la tutela.
                    2.  **Identificar Procedimiento Correcto:** Explica cuál es el trámite administrativo legalmente establecido (ej. "Resolución 10017 de 2021"), usando los datos de `<contexto_juridico_neo4j>`.
                    3.  **Demostrar Acción (El argumento más fuerte):** Informa al juez que la entidad **YA INICIÓ** las acciones para resolver el caso (ej. "Esta Oficina remitió el caso a la Dirección Nacional de Registro Civil", "La dependencia ya abocó conocimiento...").

                    ---

                    **REGLA DE PETICIÓN OBLIGATORIA (PARA SECCIÓN VI. PETICIÓN)**

                    **ADVERTENCIA CRÍTICA:** Debes usar EXACTAMENTE una de estas tres estrategias según el ÁRBOL DE DECISIÓN.

                    **ÁRBOL DE DECISIÓN PARA ELEGIR LA ESTRATEGIA:**
                    1.  ¿Existe un problema/error reconocido (ej: duplicidad, falta de documento)?
                        * NO → Usa **NEGAR** (Estrategia 3)
                        * SÍ → Ve al paso 2
                    2.  ¿El problema ya fue resuelto COMPLETAMENTE antes de la tutela?
                        * SÍ → Usa **CARENCIA ACTUAL DE OBJETO** (Estrategia 1)
                        * NO → Ve al paso 3
                    3.  ¿Se han iniciado acciones administrativas pero aún no se completan?
                        * SÍ → Usa **NO TUTELAR** (Estrategia 2) ← **ESTA ES LA MÁS COMÚN**
                        * NO → Hay un error en tu razonamiento, revisa

                    ---
                    **Estrategia 1: CARENCIA ACTUAL DE OBJETO**
                    * **Cuándo usar:** ÚNICAMENTE si el problema YA FUE COMPLETAMENTE RESUELTO (ej. "YA SE EXPIDIÓ", "YA SE CORRIGIÓ").
                    * **Redacción exacta (copia y pega):**
                        ```
                        Considerando lo anteriormente expuesto, solicito respetuosamente a su Señoría:
                        Que se declare la CARENCIA ACTUAL DE OBJETO de la presente acción de tutela, toda vez que la situación que dio origen a la presente acción constitucional fue resuelta por esta entidad con anterioridad a la notificación de la tutela, cesando así la presunta vulneración de los derechos fundamentales invocados por el accionante.
                        ```

                    **Estrategia 2: NO TUTELAR (LA MÁS COMÚN)**
                    * **Cuándo usar:** Si el trámite YA INICIÓ pero está EN CURSO (ej. "se remitió el caso", "la dependencia abocó conocimiento").
                    * **Redacción exacta (copia y pega):**
                        ```
                        Considerando lo anteriormente expuesto, solicito respetuosamente a su Señoría:
                        Que se NIEGUE el amparo solicitado por el accionante, toda vez que la Registraduría Nacional del Estado Civil ha iniciado las acciones administrativas correspondientes dentro de los plazos legales establecidos, actuando con la debida diligencia y dentro del marco de sus competencias, por lo que no existe vulneración alguna de los derechos fundamentales invocados.
                        ```

                    **Estrategia 3: NEGAR**
                    * **Cuándo usar:** Si la entidad actuó correctamente desde el inicio y NO EXISTE NINGÚN ERROR que corregir (ej. "no hay duplicidad", "no cumple requisitos").
                    * **Redacción exacta (copia y pega):**
                        ```
                        Considerando lo anteriormente expuesto, solicito respetuosamente a su Señoría:
                        Que se NIEGUE el amparo solicitado, toda vez que esta entidad ha actuado conforme a derecho, respetando el debido proceso administrativo y sin que se configure vulneración alguna de los derechos fundamentales invocados por el accionante.
                        ```
                    """
                )
            return base

        # chat (default)
        return (
            "Eres un asistente conversacional útil y preciso.\n"
            "Responde únicamente a la pregunta actual.\n"
            "Usa el contexto disponible solo si aporta valor.\n"
            "No repitas información innecesaria.\n"
            "No analices ni resumas documentos si el usuario no lo pide.\n"
            "No uses HTML ni etiquetas.\n"
            "Si no sabes algo, dilo con claridad."
        )

    def _prompt_edit_generic(self) -> str:
        """
        Prompt genérico para edición de texto existente.
        """
        return """
    ROL: Editor de texto preciso.

    Tu tarea es MODIFICAR un documento existente siguiendo instrucciones explícitas.
    No debes crear contenido nuevo innecesario ni cambiar el estilo general.

    REGLAS:
    - Aplica SOLO los cambios solicitados.
    - No alteres otras partes del texto.
    - No agregues explicaciones ni comentarios.
    - Devuelve el documento COMPLETO con los cambios aplicados.
    - No incluyas etiquetas, títulos ni introducciones.

    ENTRADA DEL USUARIO:
    1. [TEXTO_BASE]: Documento original.
    2. [INSTRUCCIONES]: Cambios específicos a realizar.
    3. [CONTEXTO_OPCIONAL]: Información adicional (si aplica).

    SALIDA:
    - ÚNICAMENTE el texto final editado.
    """

    def _user_payload(self, *, user_query: str, context_bundle: Dict[str, Any]) -> str:
        """
        Construye el mensaje USER de forma genérica para RAG híbrido.
        """

        parts: List[str] = []
        normalized_query = (user_query or "").strip().lower()
        simple_greetings = {
            "hola",
            "buenas",
            "buenos dias",
            "buenas tardes",
            "buenas noches",
            "hello",
            "hi",
        }
        is_simple_greeting = normalized_query in simple_greetings or len(normalized_query) <= 3

        # 1. Consulta del usuario
        parts.append("<tutela_a_responder>")
        parts.append(self._trim_if_needed(user_query or "", 12_000))
        parts.append("</tutela_a_responder>\n")

        if not is_simple_greeting:
            # 2. Contexto estructurado (Neo4j)
            parts.append("<contexto_estructurado>")
            parts.append(
                self._format_graph_results_for_llm(
                    context_bundle.get("graph_results"),
                    operation="general",
                )
            )
            parts.append("</contexto_estructurado>\n")

            # 3. Contexto vectorial (documentos similares)
            parts.append("<contexto_documentos>")
            parts.append(self._safe_json_dumps(context_bundle.get("vector_results")))
            parts.append("</contexto_documentos>\n")

            # 4. Historial de conversación
            parts.append("<historial_conversacion>")
            parts.append(self._safe_json_dumps(context_bundle.get("conversation_context")))
            parts.append("</historial_conversacion>\n")

        parts.append(
            "INSTRUCCIÓN FINAL:\n"
            "Usa el contexto SOLO si es relevante.\n"
            "No inventes información.\n"
            "No uses HTML ni etiquetas.\n"
            "Si el usuario saluda, responde breve y amable."
        )

        payload = "\n".join(parts)
        return self._trim_if_needed(payload, self.max_payload_chars)


    # ------------------------- Ollama health -------------------------

    async def _ensure_generator_ready(self) -> bool:
        try:
            health_status = await self.ollama_service.health_check()
            ok = health_status.get("status") == "healthy"
            if not ok:
                logger.warning("Ollama unhealthy (attempting generation anyway): %s", health_status)
            return True
        except Exception as e:
            logger.warning("Ollama health check failed (attempting generation anyway): %s", e)
            return True
