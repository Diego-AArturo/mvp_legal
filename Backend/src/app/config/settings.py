"""
Main application settings.

Loads configuration from environment variables or .env.
"""

from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config.databases import DatabaseSettings, get_database_settings
from app.config.llm import LLMSettings, get_llm_settings
from app.config.models import ModelsSettings, get_models_settings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env.
    """

    # General settings
    DEBUG_MODE: bool = False

    # API settings
    API_V1_STR: str = "/api/v1"
    DOCS_URL: str = "/docs"
    REDOC_URL: str = "/redoc"
    OPENAPI_URL: str = "/openapi.json"

    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = []
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]

    # Application info
    title: str = "Petition API"
    version: str = "0.1.0"
    description: str = "A secure petition API with LDAP authentication"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def databases(self) -> DatabaseSettings:
        return get_database_settings()

    @property
    def llm(self) -> LLMSettings:
        return get_llm_settings()

    @property
    def models(self) -> ModelsSettings:
        return get_models_settings()

    # ---- validators ----

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            if v.startswith("[") and v.endswith("]"):
                import json

                return json.loads(v)
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("CORS_ALLOW_METHODS", "CORS_ALLOW_HEADERS", mode="before")
    @classmethod
    def parse_cors_lists(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return ["*"]
            if v.startswith("[") and v.endswith("]"):
                import json

                return json.loads(v)
            return [item.strip() for item in v.split(",") if item.strip()]
        return v


@lru_cache()
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
