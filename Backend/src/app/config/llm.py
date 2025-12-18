# app/config/llm.py
"""
LLM (Large Language Model) configuration settings.

This module contains all LLM-related configuration including Ollama settings,
provider selection, and generation parameters. Separated from main settings
for better organization and maintainability.
"""

from functools import lru_cache
from typing import ClassVar, Optional

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.extensions import get_logger

logger = get_logger(__name__)


# ============================================================
# Ollama settings (single-model, strict)
# ============================================================

class OllamaSettings(BaseSettings):
    """
    Configuración estricta de Ollama para setup de modelo único.

    El OllamaClient espera atributos en `settings.llm.ollama`:
    - base_url
    - model_name
    - request_timeout
    """

    # ---- Canonical fields used by OllamaClient ----
    base_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("OLLAMA_BASE_URL"),
    )

    model_name: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("OLLAMA_MODEL_NAME"),
    )

    request_timeout: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("OLLAMA_REQUEST_TIMEOUT"),
    )

    # ---- Generation defaults (used elsewhere in the app) ----
    ollama_default_max_tokens: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("OLLAMA_DEFAULT_MAX_TOKENS"),
    )

    ollama_default_temperature: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("OLLAMA_DEFAULT_TEMPERATURE"),
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -------------------- validators --------------------

    @field_validator("base_url", mode="before")
    @classmethod
    def _normalize_base_url(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        v = v.rstrip("/")
        if not (v.startswith("http://") or v.startswith("https://")):
            v = "http://" + v
        return v

    @field_validator("model_name", mode="before")
    @classmethod
    def _validate_model_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not v:
            raise ValueError("model_name cannot be empty")
        return v

    @model_validator(mode="after")
    def _sanity_checks(self):
        if self.request_timeout is not None and self.request_timeout < 1:
            raise ValueError("request_timeout must be >= 1 second")
        if self.ollama_default_max_tokens is not None and self.ollama_default_max_tokens < 1:
            raise ValueError("ollama_default_max_tokens must be >= 1")
        if self.ollama_default_temperature is not None and not (0.0 <= self.ollama_default_temperature <= 2.0):
            raise ValueError("ollama_default_temperature must be in [0.0, 2.0]")
        return self


# ============================================================
# Provider selection
# ============================================================

class LLMProviderSettings(BaseSettings):
    """
    LLM Provider configuration and selection.
    """

    llm_provider: str = Field(
        default="ollama",
        validation_alias=AliasChoices("LLM_PROVIDER"),
    )

    SUPPORTED_PROVIDERS: ClassVar[list[str]] = ["ollama"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("llm_provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        if v not in cls.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Invalid LLM provider '{v}'. Supported providers: {cls.SUPPORTED_PROVIDERS}"
            )
        return v

    @model_validator(mode="after")
    def validate_provider_requirements(self):
        if self.llm_provider == "ollama":
            logger.info("Using Ollama provider (single-model setup)")
        return self


# ============================================================
# Combined LLM settings
# ============================================================

class LLMSettings(BaseSettings):
    """
    Combined LLM settings that includes provider selection and all LLM configurations.
    Lazily instantiates nested settings to keep startup cost low.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._provider = None
        self._ollama = None

    @property
    def provider(self) -> LLMProviderSettings:
        if self._provider is None:
            self._provider = LLMProviderSettings()
        return self._provider

    @property
    def ollama(self) -> OllamaSettings:
        if self._ollama is None:
            self._ollama = OllamaSettings()
        return self._ollama

    def get_active_provider(self) -> str:
        return self.provider.llm_provider

    def is_ollama_enabled(self) -> bool:
        return self.get_active_provider() == "ollama"


# ============================================================
# Cached accessors
# ============================================================

@lru_cache()
def get_ollama_settings() -> OllamaSettings:
    return OllamaSettings()  # type: ignore[call-arg]


@lru_cache()
def get_llm_provider_settings() -> LLMProviderSettings:
    return LLMProviderSettings()  # type: ignore[call-arg]


@lru_cache()
def get_llm_settings() -> LLMSettings:
    return LLMSettings()  # type: ignore[call-arg]

