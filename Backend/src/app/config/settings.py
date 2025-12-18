"""
Configuración principal de la aplicación.

Define todas las configuraciones de la aplicación cargadas desde variables
de entorno o archivo .env. Incluye configuraciones de autenticación, seguridad,
LDAP, API, CORS y referencias a configuraciones de subsistemas.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config.databases import DatabaseSettings, get_database_settings
from app.config.llm import LLMSettings, get_llm_settings
from app.config.models import ModelsSettings, get_models_settings
from app.config.storage import StorageSettings, get_storage_settings
from app.extensions import get_logger

logger = get_logger(__name__)


class Settings(BaseSettings):
    """
    Configuración de la aplicación cargada desde variables de entorno o archivo .env.

    La configuración incluye:
    - Configuración de autenticación y seguridad (JWT, LDAP)
    - Configuración de API y CORS
    - Referencias a configuraciones de subsistemas (bases de datos, LLM, modelos, almacenamiento)
    """

    # General Settings
    DEBUG_MODE: bool = False
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


    # TLS/SSL Configuration
    LDAP_USE_SSL: bool = False  # si el URL es ldaps:// se forzará True
    LDAP_START_TLS: bool = False
    LDAP_VALIDATE_CERTS: bool = True
    LDAP_REQUIRE_CERT: bool = True
    LDAP_CA_CERTS_PATH: Optional[str] = None
    LDAP_TLS_VERSION: str = "TLSv1.2"

    # Connection timeouts
    LDAP_CONNECTION_TIMEOUT: int = 10
    LDAP_RECEIVE_TIMEOUT: int = 10
    LDAP_NETWORK_TIMEOUT: int = 10
    LDAP_OPERATION_TIMEOUT: int = 10

    # DNS servers for domain resolution
    LDAP_DNS_SERVERS: Optional[str] = None  # Comma-separated list


    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Petition API"
    API_DEBUG: bool = False
    DOCS_URL: str = "/docs"
    REDOC_URL: str = "/redoc"
    OPENAPI_URL: str = "/openapi.json"
    ROOT_PATH: str = ""
    PROXY_PREFIX: str = ""

    # CORS Settings
    BACKEND_CORS_ORIGINS: List[str] = []
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_BURST: int = 100
    RATE_LIMIT_ENABLED: bool = True

    # Cache Settings
    CACHE_TTL: int = 3600  # seconds
    CACHE_PREFIX: str = "petition_api"
    CACHE_TYPE: str = "memory"  # memory, redis
    CACHE_REDIS_URL: Optional[str] = None

    # RabbitMQ Settings
    RABBITMQ_URL: str = "amqp://guest:guest@etl-rabbitmq:5672/"

    # Person Lookup API Settings (Registraduría)
    PERSON_LOOKUP_API_BASE_URL: str = "http://172.20.93.101:5000"
    PERSON_LOOKUP_API_USERNAME: str = "useradmin"
    PERSON_LOOKUP_API_PASSWORD: str = "PassB4s3dt"
    PERSON_LOOKUP_TIMEOUT: float = 10.0  # Timeout estándar para consultas (segundos)
    PERSON_LOOKUP_AUTH_TIMEOUT: float = (
        30.0  # Timeout extendido para autenticación (segundos)
    )

    # Tutela Document Settings
    NOMBRE_JEFE_JURIDICO: Optional[str] = None  # Nombre del jefe jurídico para usar en prompts de tutelas

    # Application Info
    title: str = "Petition API"
    version: str = "0.1.0"
    description: str = "A secure petition API with LDAP authentication"
    contact_name: Optional[str] = None
    contact_email: Optional[str] = None

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

    @property
    def storage(self) -> StorageSettings:
        return get_storage_settings()

    # ---- VALIDADORES ----

    # Limpia valores vacíos/comentarios en campos NO sensibles
    @field_validator(
        "LDAP_SERVER",
        "LDAP_BASE_DN",
        "LDAP_BIND_DN",
        # "LDAP_BIND_PASSWORD",  <-- ¡OJO! intencionalmente EXCLUIDO
        "LDAP_USER_DN_TEMPLATE",
        "JWT_KID",
        "CACHE_REDIS_URL",
        "contact_name",
        "contact_email",
        mode="before",
    )
    @classmethod
    def empty_str_to_none(cls, v):
        if isinstance(v, str):
            # remove inline comments SOLO para campos no sensibles
            v = v.split("#")[0].strip()
            if not v:
                return None
        return v

    # Mantener la contraseña tal cual (permitiendo comillas envolventes)
    @field_validator("LDAP_BIND_PASSWORD", mode="before")
    @classmethod
    def keep_password_verbatim(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            if (s.startswith("'") and s.endswith("'")) or (
                s.startswith('"') and s.endswith('"')
            ):
                s = s[1:-1]
            return s  # NO cortar por '#'
        return v

    @field_validator("LDAP_USERNAME_ATTRIBUTE", mode="before")
    @classmethod
    def default_username_attr(cls, v):
        return v or "sAMAccountName"

    @field_validator("LDAP_SEARCH_FILTER_TEMPLATE", mode="before")
    @classmethod
    def default_search_filter(cls, v):
        return v or "(&(objectClass=user)({attr}={username}))"

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

    @model_validator(mode="after")
    def check_ldap_fields(self):
        ldap_values = [
            self.LDAP_SERVER,
            self.LDAP_BASE_DN,
            self.LDAP_BIND_DN,
            self.LDAP_BIND_PASSWORD,
        ]
        filled = [v is not None for v in ldap_values]
        if 0 < sum(filled) < 4:
            missing_fields = [
                field_name
                for field_name, field_value in [
                    ("LDAP_SERVER", self.LDAP_SERVER),
                    ("LDAP_BASE_DN", self.LDAP_BASE_DN),
                    ("LDAP_BIND_DN", self.LDAP_BIND_DN),
                    ("LDAP_BIND_PASSWORD", self.LDAP_BIND_PASSWORD),
                ]
                if field_value is None
            ]
            missing_list = ", ".join(missing_fields)
            logger.warning(
                f"Authentication is partially configured; missing LDAP settings: {missing_list}. Continuing startup with authentication disabled."
            )
        return self

    @model_validator(mode="after")
    def validate_security_settings(self):
        if self.DEBUG_MODE and not self.ALLOW_INSECURE_AUTH:
            logger.warning("DEBUG_MODE is enabled but ALLOW_INSECURE_AUTH is False")
        if self.FORCE_HTTPS and not self.ALLOW_INSECURE_AUTH:
            logger.info("HTTPS enforcement is enabled")
        if self.MIN_PASSWORD_LENGTH < 8:
            raise ValueError("MIN_PASSWORD_LENGTH must be at least 8 characters")
        return self

@lru_cache()
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
