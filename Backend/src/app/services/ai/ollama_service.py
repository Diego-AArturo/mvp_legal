"""
Servicio Ollama generalizado para generación de texto.

Este servicio proporciona una API limpia y generalizada para generación de texto
usando modelos servidos por Ollama. Los agentes son responsables de su propia
ingeniería de prompts y post-procesamiento. El servicio se enfoca únicamente en
la comunicación con el modelo.
"""

import os
from typing import Any, Dict, List, Optional

from app.clients.ollama_client import OllamaClient
from app.config.settings import Settings
from app.extensions import get_logger

logger = get_logger(__name__)


class OllamaService:
    """
    Servicio generalizado para generación de texto con Ollama.

    Proporciona una API limpia y de bajo nivel para generación de texto.
    Los agentes manejan su propia lógica de negocio, construcción de prompts
    y post-procesamiento.
    """

    def __init__(self, settings: Settings):
        """
        Inicializa el servicio Ollama.

        Args:
            settings: Configuración de la aplicación con parámetros de Ollama
        """
        self.settings = settings
        self.client = OllamaClient(settings)
        self.generation_settings = settings.models.generation
        self.ollama_settings = settings.llm.ollama

        # Configuración de ventana de contexto del modelo
        # Gemma 3 4B IT tiene 32K tokens de contexto total
        self.model_max_context_length = 32000
        self.safety_margin = 500  # Margen de seguridad para evitar límites exactos

        logger.info("OllamaService inicializado en modo generalizado")
        logger.info(
            f"Ventana de contexto del modelo: {self.model_max_context_length} tokens"
        )

    def _estimate_token_count(self, text: str) -> int:
        """
        Estima el número de tokens en un texto.

        Usa una aproximación simple: ~4 caracteres por token para modelos multilingües.
        Esta es una estimación conservadora que funciona razonablemente bien.

        Args:
            text: Texto a estimar

        Returns:
            Número estimado de tokens
        """
        # Aproximación: 1 token ≈ 4 caracteres para texto en español/inglés
        return len(text) // 4 + 10  # +10 como buffer adicional

    def _calculate_safe_max_tokens(
        self, messages: List[Dict[str, str]], requested_max_tokens: int
    ) -> int:
        """
        Calcula un max_tokens seguro que respete los límites del modelo.

        Args:
            messages: Mensajes del prompt
            requested_max_tokens: Tokens solicitados por el usuario

        Returns:
            max_tokens ajustado que respeta la ventana de contexto
        """
        # Estimar tokens del prompt
        prompt_text = "\n".join([msg.get("content", "") for msg in messages])
        estimated_prompt_tokens = self._estimate_token_count(prompt_text)

        # Calcular tokens disponibles para la respuesta
        available_tokens = (
            self.model_max_context_length - estimated_prompt_tokens - self.safety_margin
        )

        # Asegurar que sea positivo y razonable
        available_tokens = max(100, available_tokens)  # Mínimo 100 tokens

        # Tomar el menor entre lo solicitado y lo disponible
        safe_max_tokens = min(requested_max_tokens, available_tokens)

        # Log si hubo ajuste
        if safe_max_tokens < requested_max_tokens:
            logger.warning(
                f" max_tokens ajustado de {requested_max_tokens} a {safe_max_tokens} "
                f"(prompt estimado: {estimated_prompt_tokens} tokens, "
                f"disponible: {available_tokens} tokens, "
                f"límite modelo: {self.model_max_context_length} tokens)"
            )
        else:
            logger.debug(
                f" max_tokens={safe_max_tokens} OK (prompt estimado: {estimated_prompt_tokens} tokens)"
            )

        return safe_max_tokens

    async def generate_text(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model_preference: str = "auto",
        **kwargs,
    ) -> str:
        """
        Genera texto usando mensajes pre-formateados.

        Este es el método principal para generación de texto generalizada.
        Los agentes deben formatear sus propios mensajes y manejar el post-procesamiento.

        Args:
            messages: Lista de mensajes pre-formateados con 'role' y 'content'
            max_tokens: Máximo número de tokens a generar
            temperature: Temperatura de sampling (0.0-1.0)
            model_preference: Preferencia de modelo (actualmente solo Gemma disponible)
            **kwargs: Parámetros adicionales de generación

        Returns:
            Texto generado sin post-procesamiento

        Raises:
            RuntimeError: Si la generación falla
        """
        try:
            # Validar y normalizar mensajes según el modelo activo
            normalized_messages = self._prepare_messages_for_model(messages)

            if not normalized_messages:
                raise ValueError("No se proporcionaron mensajes válidos")

            # Configurar parámetros de generación
            requested_max_tokens = (
                max_tokens or self.generation_settings.generation_max_response_length
            )
            temperature = temperature if temperature is not None else 0.3

            #  VALIDACIÓN AUTOMÁTICA: Ajustar max_tokens para respetar ventana de contexto
            safe_max_tokens = self._calculate_safe_max_tokens(
                normalized_messages, requested_max_tokens
            )

            logger.debug(
                f"Generando texto con {len(normalized_messages)} mensajes, max_tokens={safe_max_tokens} (solicitado: {requested_max_tokens}), temperature={temperature}"
            )

            # Generar respuesta usando el modelo configurado en Ollama
            response = await self.client.generate_with_model(
                messages=normalized_messages,
                max_tokens=safe_max_tokens,
                temperature=temperature,
                **kwargs,
            )

            if not response:
                logger.warning("El modelo Ollama retornó una respuesta vacía")
                return ""

            final_response = response.strip()
            logger.info(
                f"Texto generado exitosamente usando Gemma (longitud: {len(final_response)} caracteres)"
            )

            return final_response

        except ValueError as e:
            logger.error(f"Error de validación en generación de texto: {e}")
            raise RuntimeError(f"Validación de entrada falló: {e}")
        except Exception as e:
            logger.error(f"Error en generación de texto Ollama: {e}")
            raise RuntimeError(f"Generación de texto falló: {e}")

    def _prepare_messages_for_model(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Normaliza mensajes según el modelo activo.

        DeepSeek R1 responde mejor con un único mensaje de usuario
        y sin roles system explícitos para evitar eco del prompt.
        """
        model_name = (self.ollama_settings.model_name or "").lower()
        if model_name.startswith("deepseek-r1"):
            return self._normalize_messages_for_deepseek(messages)
        return self._normalize_messages_for_gemma(messages)

    def _normalize_messages_for_deepseek(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Ajusta mensajes para DeepSeek R1 con un solo prompt de usuario.

        Evita que el modelo repita instrucciones y reduce el eco de sistema.
        """
        if not messages:
            return []

        system_chunks: List[str] = []
        user_chunks: List[str] = []
        assistant_chunks: List[str] = []

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = (msg.get("role") or "user").strip().lower()
            content = (msg.get("content") or "").strip()
            if not content:
                continue

            if role == "system":
                system_chunks.append(content)
            elif role == "assistant":
                assistant_chunks.append(content)
            else:
                user_chunks.append(content)

        parts: List[str] = []
        if system_chunks:
            parts.append("INSTRUCCIONES:")
            parts.append("\n".join(system_chunks))
        if assistant_chunks:
            parts.append("HISTORIAL ASISTENTE:")
            parts.append("\n".join(assistant_chunks))
        if user_chunks:
            parts.append("MENSAJE DEL USUARIO:")
            parts.append("\n".join(user_chunks))

        parts.append(
            "REGLA FINAL: Responde solo con el contenido final solicitado. "
            "No repitas instrucciones ni el texto del prompt."
        )

        merged = "\n\n".join(parts).strip()
        return [{"role": "user", "content": merged}]

    async def generate_with_fallback(
        self,
        messages: List[Dict[str, str]],
        fallback_text: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        Genera texto con fallback personalizado en caso de fallo.

        Args:
            messages: Mensajes pre-formateados para generación
            fallback_text: Texto a retornar si la generación falla
            max_tokens: Máximo número de tokens a generar
            temperature: Temperatura de sampling
            **kwargs: Parámetros adicionales

        Returns:
            Texto generado o fallback_text en caso de fallo
        """
        try:
            return await self.generate_text(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
        except Exception as e:
            logger.warning(f"Generación de texto falló, usando fallback: {e}")
            return fallback_text

    def _normalize_messages_for_gemma(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Normaliza mensajes para compatibilidad con el template de chat de Gemma.

        Gemma solo soporta roles 'user' y 'assistant'. Esta función:
        - Convierte otros roles a 'user' con marcadores claros
        - Fusiona mensajes consecutivos del mismo rol
        - Asegura que la conversación empiece con 'user'

        Args:
            messages: Lista de mensajes con 'role' y 'content'

        Returns:
            Lista de mensajes normalizados para Gemma
        """
        if not messages:
            return []

        normalized = []

        for msg in messages:
            if not isinstance(msg, dict):
                logger.warning(f"Mensaje inválido ignorado: {msg}")
                continue

            role = msg.get("role", "user")
            content = (msg.get("content") or "").strip()

            if not content:
                logger.debug(f"Mensaje con contenido vacío ignorado: role={role}")
                continue

            # Gemma solo soporta user/assistant
            if role not in ("user", "assistant"):
                # Convertir system/tool/otros roles a user con marcador claro
                content = f"[{role.upper()}]\n{content}"
                role = "user"
                logger.debug(
                    f"Rol '{msg.get('role')}' convertido a 'user' con marcador"
                )

            # Fusionar mensajes consecutivos del mismo rol
            if normalized and normalized[-1]["role"] == role:
                normalized[-1]["content"] += f"\n\n{content}"
                logger.debug(f"Mensaje fusionado con el anterior (role={role})")
            else:
                normalized.append({"role": role, "content": content})

        # Asegurar que la conversación empiece con 'user'
        if normalized and normalized[0]["role"] != "user":
            normalized.insert(0, {"role": "user", "content": "[CONTEXTO]"})
            logger.debug(
                "Añadido mensaje inicial de usuario para compatibilidad con Gemma"
            )

        # Mantener historial razonable (aumentado de 6 a 10 para mejor contexto)
        if len(normalized) > 10:
            normalized = normalized[-10:]
            logger.debug(f"Historial truncado a {len(normalized)} mensajes")

        logger.debug(f"Mensajes normalizados: {len(normalized)} mensajes finales")
        return normalized

    async def health_check(self) -> Dict[str, Any]:
        """
        Verifica el estado de salud del servicio Ollama.

        Returns:
            Información del estado de salud del servicio y modelo Gemma
        """
        try:
            client_health = await self.client.health_check()

            overall_status = client_health.get("overall_status", "unknown")
            model_name = (
                client_health.get("model")
                or getattr(self.ollama_settings, "model_name", None)
                or os.getenv("OLLAMA_MODEL_NAME", "ollama-default")
            )
            base_url = client_health.get(
                "base_url", getattr(self.ollama_settings, "base_url", None)
            )
            model_status = "healthy" if overall_status != "unhealthy" else "unhealthy"

            health_info = {
                "service": "ollama_service",
                "status": overall_status,
                "mode": "generalized",
                "version": "2.0.0",
                "client_status": client_health,
                "models": {
                    "ollama": {
                        "name": model_name,
                        "status": model_status,
                        "base_url": base_url,
                    }
                },
                "generation_settings": {
                    "max_tokens": self.generation_settings.generation_max_response_length,
                    "timeout": self.generation_settings.generation_max_wait_time,
                },
                "capabilities": [
                    "text_generation",
                    "conversation_handling",
                    "ollama_model_support",
                    "message_normalization",
                ],
            }

            logger.info(f"Health check completado: status={overall_status}")
            return health_info

        except Exception as e:
            logger.error(f"Health check falló: {e}")
            return {
                "service": "ollama_service",
                "status": "unhealthy",
                "error": str(e),
                "mode": "generalized",
                "version": "2.0.0",
            }

    async def get_model_info(self) -> Dict[str, Any]:
        """
        Obtiene información detallada del modelo disponible.

        Returns:
            Información del modelo Gemma disponible
        """
        try:
            client_health = await self.client.health_check()
            model_name = (
                client_health.get("model")
                or getattr(self.ollama_settings, "model_name", None)
                or os.getenv("OLLAMA_MODEL_NAME", "ollama-default")
            )
            base_url = client_health.get(
                "base_url", getattr(self.ollama_settings, "base_url", None)
            )
            overall_status = client_health.get("overall_status", "unknown")

            return {
                "available_models": [model_name],
                "default_model": model_name,
                "model_details": {
                    model_name: {
                        "name": model_name,
                        "status": (
                            "healthy" if overall_status != "unhealthy" else "unhealthy"
                        ),
                        "base_url": base_url,
                        "capabilities": [
                            "text_generation",
                            "conversation",
                            "instruction_following",
                        ],
                        "supported_roles": ["user", "assistant"],
                        "max_tokens": self.generation_settings.generation_max_response_length,
                    }
                },
            }
        except Exception as e:
            logger.error(f"Error obteniendo información del modelo: {e}")
            return {"available_models": [], "error": str(e)}

    async def validate_messages(self, messages: List[Dict[str, str]]) -> bool:
        """
        Valida que los mensajes tengan el formato correcto.

        Args:
            messages: Lista de mensajes a validar

        Returns:
            True si los mensajes son válidos, False en caso contrario
        """
        if not messages or not isinstance(messages, list):
            logger.warning("Mensajes inválidos: debe ser una lista no vacía")
            return False

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                logger.warning(f"Mensaje {i} inválido: debe ser un diccionario")
                return False

            if "role" not in msg or "content" not in msg:
                logger.warning(f"Mensaje {i} inválido: debe tener 'role' y 'content'")
                return False

            if not isinstance(msg["content"], str) or not msg["content"].strip():
                logger.warning(
                    f"Mensaje {i} inválido: 'content' debe ser string no vacío"
                )
                return False

        logger.debug(f"Validación exitosa: {len(messages)} mensajes válidos")
        return True

    async def initialize(self) -> None:
        """
        Inicializa el servicio (placeholder para futuras mejoras).
        """
        logger.info("OllamaService inicializado exitosamente en modo generalizado")

    async def cleanup(self) -> None:
        """
        Limpia recursos del servicio (placeholder para futuras mejoras).
        """
        logger.info("Limpieza de OllamaService completada")
