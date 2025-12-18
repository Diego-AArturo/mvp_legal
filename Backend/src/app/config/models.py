"""
Configuración de modelos (embeddings y generación).

Proporciona clases de configuración para modelos de embeddings (sentence-transformers)
y generación de texto, incluyendo parámetros de optimización y performance.
"""

from functools import lru_cache
from typing import Any, Dict, Optional

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.extensions import get_logger

logger = get_logger(__name__)


class EmbeddingSettings(BaseSettings):
    """
    Configuración de modelo de embeddings sentence-transformers.

    Contiene configuraciones para generación de embeddings semánticos usando
    la librería sentence-transformers con optimización para CPU y soporte multilingüe.
    """

    # Model Settings
    embedding_model_id: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    embedding_max_seq_length: int = 512
    embedding_normalize_embeddings: bool = True

    # Cache and Authentication
    embedding_cache_dir: Optional[str] = None
    embedding_use_auth_token: bool = False
    embedding_auth_token: Optional[str] = None

    # Performance Settings
    embedding_trust_remote_code: bool = False
    embedding_model_kwargs: Dict[str, Any] = {}
    embedding_encode_kwargs: Dict[str, Any] = {}

    # Pydantic configuration to load from .env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator("embedding_model_id")
    @classmethod
    def validate_embedding_model_id(cls, v):
        """
        Valida que el ID del modelo de embeddings no esté vacío.

        Argumentos:
            v: ID del modelo a validar

        Retorna:
            ID validado

        Lanza:
            ValueError: Si el ID está vacío
        """
        if not v or not v.strip():
            raise ValueError("ID del modelo de embeddings no puede estar vacío")
        return v

    @field_validator(
        "embedding_cache_dir",
        "embedding_auth_token",
        mode="before",
    )
    @classmethod
    def empty_str_to_none(cls, v):
        """
        Convierte strings vacíos a None para campos opcionales.

        También elimina comentarios después de #.

        Argumentos:
            v: Valor a procesar

        Retorna:
            Valor procesado o None si está vacío
        """
        if isinstance(v, str):
            v = v.split("#")[0].strip()
            if not v:
                return None
        return v

    @field_validator("embedding_device", mode="before")
    @classmethod
    def validate_embedding_device(cls, v: str) -> str:
        """
        Valida que el dispositivo de embeddings sea válido.

        Argumentos:
            v: Dispositivo a validar

        Retorna:
            Dispositivo validado en minúsculas

        Lanza:
            ValueError: Si el dispositivo no es válido
        """
        valid_devices = ["cpu", "cuda"] + [f"cuda:{i}" for i in range(8)]
        if v not in valid_devices:
            raise ValueError(f"Dispositivo de embeddings inválido. " f"Debe ser uno de: {valid_devices}")
        return v.lower()

    @model_validator(mode="after")
    def validate_embedding_settings(self):
        """
        Valida configuraciones específicas de embeddings.

        Retorna:
            Self para encadenamiento

        Lanza:
            ValueError: Si las configuraciones son inválidas
        """
        if self.embedding_batch_size < 1:
            raise ValueError("Tamaño de lote de embeddings debe ser al menos 1")

        if self.embedding_max_seq_length < 1:
            raise ValueError("Longitud máxima de secuencia de embeddings debe ser al menos 1")

        return self


class GenerationSettings(BaseSettings):
    """
    Configuración de generación de texto.

    Contiene configuraciones específicas para procesos de generación de texto
    incluyendo tiempos de espera, longitudes de respuesta y estrategias de generación.
    """

    # Generation Agent Configuration
    generation_max_wait_time: int = 30
    generation_context_strategy: str = "comprehensive"
    generation_max_response_length: int = 4000
    generation_enable_citations: bool = True

    # Pydantic configuration to load from .env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator("generation_context_strategy")
    @classmethod
    def validate_context_strategy(cls, v):
        """
        Valida que la estrategia de contexto sea válida.

        Argumentos:
            v: Estrategia a validar

        Retorna:
            Estrategia validada

        Lanza:
            ValueError: Si la estrategia no es válida
        """
        valid_strategies = ["comprehensive", "focused", "balanced"]
        if v not in valid_strategies:
            raise ValueError(f"Invalid context strategy. Must be one of: {valid_strategies}")
        return v

    @model_validator(mode="after")
    def validate_generation_settings(self):
        """Validate generation-specific settings"""
        if self.generation_max_wait_time < 1:
            raise ValueError("Generation max wait time must be at least 1 second")

        if self.generation_max_response_length < 10:
            raise ValueError("Generation max response length must be at least 10 characters")

        return self


class ModelsSettings(BaseSettings):
    """Combined settings for embedding and generation configurations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._embedding = None
        self._generation = None

    @property
    def embedding(self) -> EmbeddingSettings:
        """Get embedding configuration"""
        if self._embedding is None:
            self._embedding = EmbeddingSettings()
        return self._embedding

    @property
    def generation(self) -> GenerationSettings:
        """Get generation configuration"""
        if self._generation is None:
            self._generation = GenerationSettings()
        return self._generation


@lru_cache()
def get_embedding_settings() -> EmbeddingSettings:
    """Get cached embedding settings instance"""
    return EmbeddingSettings()  # type: ignore[call-arg]


@lru_cache()
def get_generation_settings() -> GenerationSettings:
    """Get cached generation settings instance"""
    return GenerationSettings()  # type: ignore[call-arg]


@lru_cache()
def get_models_settings() -> ModelsSettings:
    """Get cached combined models settings instance"""
    return ModelsSettings()  # type: ignore[call-arg]
