from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from app.config.settings import Settings
from app.extensions import get_logger

logger = get_logger(__name__)


class OllamaResult:
    def __init__(
        self,
        text: str,
        model: str,
        finish_reason: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None,
    ):
        self.text = text
        self.model = model
        self.finish_reason = finish_reason
        self.usage = usage or {}


class OllamaClient:
    def __init__(self, settings: Settings):
        llm_cfg = getattr(settings, "llm", None)
        if llm_cfg is None:
            raise RuntimeError("Falta configuración 'settings.llm'")
        ollama_cfg = getattr(llm_cfg, "ollama", None)
        if ollama_cfg is None:
            raise RuntimeError("Falta configuración 'settings.llm.ollama'")

        def require(field: str) -> Any:
            value = getattr(ollama_cfg, field, None)
            if value is None or (isinstance(value, str) and value.strip() == ""):
                raise RuntimeError(f"Falta 'settings.llm.ollama.{field}'")
            return value

        self.base_url = self._normalize_base_url(require("base_url"))  # Ej: http://localhost:11434
        self.model_name = require("model_name")

        try:
            self.timeout = float(require("request_timeout"))
            if self.timeout <= 0:
                raise ValueError
        except Exception:
            raise RuntimeError("Timeout inválido 'settings.llm.ollama.request_timeout' (debe ser > 0)")

        logger.info(f"OllamaClient inicializado (modelo: {self.model_name} @ {self.base_url})")

    def _normalize_base_url(self, base_url: str) -> str:
        if not isinstance(base_url, str) or base_url.strip() == "":
            raise RuntimeError("URL base inválida para Ollama")
        return base_url.strip().rstrip("/")

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> OllamaResult:
        model_to_use = model or self.model_name
        url = f"{self.base_url}/api/chat"

        payload: Dict[str, Any] = {
            "model": model_to_use,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["options"] = {"num_predict": max_tokens}
        if kwargs:
            payload.update(kwargs)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()

                # Validación
                message = data.get("message", {})
                if not isinstance(message, dict) or "content" not in message:
                    raise RuntimeError("Respuesta inválida de Ollama: falta 'message.content'")

                logger.debug(f"Ollama respuesta recibida: {message['content'][:60]}...")
                return OllamaResult(
                    text=message["content"],
                    model=model_to_use,
                    finish_reason=data.get("done_reason"),
                    usage={},  # Ollama no retorna uso
                )
        except httpx.RequestError as e:
            logger.error(f"Error de conexión con Ollama: {e}")
            raise RuntimeError(f"Conexión fallida con Ollama: {e}") from e
        except httpx.HTTPStatusError as e:
            body = e.response.text
            logger.error(f"Error HTTP de Ollama {e.response.status_code}: {body}")
            raise RuntimeError(f"Error en servidor Ollama {e.response.status_code}: {body}") from e
        except Exception as e:
            logger.error(f"Error inesperado en cliente Ollama: {e}")
            raise

    async def generate_with_model(self, messages: List[Dict[str, str]], **kwargs) -> str:
        result = await self.chat_completion(messages=messages, **kwargs)
        return result.text

    async def health_check(self) -> Dict[str, Any]:
        status = {
            "overall_status": "unhealthy",
            "model": self.model_name,
            "base_url": self.base_url,
        }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/")
                status["status_code"] = resp.status_code
                if resp.status_code == 200:
                    status["overall_status"] = "healthy"
        except Exception as e:
            status["error"] = str(e)
        return status

