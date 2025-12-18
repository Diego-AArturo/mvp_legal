"""
Configuración de bases de datos.

Proporciona clases de configuración para Neo4j y PostgreSQL, incluyendo
configuraciones de conexión, pooling, reintentos y parámetros de búsqueda.
"""

import logging
from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Import movido para evitar dependencia circular

logger = logging.getLogger(__name__)


class Neo4jSettings(BaseSettings):
    """
    Configuración de base de datos Neo4j.

    Contiene todas las configuraciones específicas de Neo4j incluyendo conexión,
    pooling de conexiones, lógica de reintentos y preferencias de base de datos.
    """

    # Connection Settings
    neo4j_url: str = Field(alias="NEO4J_URL")
    neo4j_user: str = Field(alias="NEO4J_USER")
    neo4j_password: str = Field(alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")
    neo4j_encryption: bool = True

    # Connection Pool Settings
    neo4j_max_connection_pool_size: int = 50
    neo4j_connection_timeout: int = 20
    neo4j_connection_acquisition_timeout: int = 60

    # Retry and Transaction Settings
    neo4j_max_connection_retries: int = 3
    neo4j_max_transaction_retry_time: int = 30
    neo4j_initial_retry_delay: float = 1.0
    neo4j_retry_delay_multiplier: float = 2.0
    neo4j_retry_delay_jitter: float = 0.2

    # Retrieval/Search Settings (Environment-Driven)
    neo4j_vector_similarity_threshold: float = 0.6
    neo4j_vector_limit: int = 10
    neo4j_relationships_limit: int = 20
    neo4j_paths_top_nodes: int = 3
    neo4j_paths_max_hops: int = 4

    # Pydantic configuration to load from .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @field_validator("neo4j_url", "neo4j_user", "neo4j_password")
    @classmethod
    def validate_required_fields(cls, v):
        """
        Valida que los campos requeridos de conexión no estén vacíos.

        Argumentos:
            v: Valor del campo a validar

        Retorna:
            Valor validado

        Lanza:
            ValueError: Si el campo está vacío
        """
        if not v or not v.strip():
            raise ValueError("Campo de conexión Neo4j no puede estar vacío")
        return v

    @model_validator(mode="after")
    def validate_neo4j_url_format(self):
        """
        Valida que la URL de Neo4j use el protocolo correcto.

        Retorna:
            Self para encadenamiento

        Lanza:
            ValueError: Si la URL no usa bolt:// o bolt+s://
        """
        if not self.neo4j_url.startswith(("bolt://", "bolt+s://")):
            raise ValueError("URL de Neo4j debe comenzar con bolt:// o bolt+s://")
        return self

    @model_validator(mode="after")
    def validate_connection_settings(self):
        """
        Valida configuraciones de pool de conexiones y timeouts.

        Retorna:
            Self para encadenamiento

        Lanza:
            ValueError: Si las configuraciones son inválidas
        """
        if self.neo4j_max_connection_pool_size < 1:
            raise ValueError("Tamaño del pool de conexiones Neo4j debe ser al menos 1")

        if self.neo4j_connection_timeout < 1:
            raise ValueError("Timeout de conexión Neo4j debe ser al menos 1 segundo")

        if self.neo4j_max_connection_retries < 1:
            raise ValueError("Reintentos máximos de conexión Neo4j debe ser al menos 1")

        return self

    @model_validator(mode="after")
    def validate_query_settings(self):
        """
        Valida configuraciones de búsqueda vectorial y descubrimiento de rutas.

        Retorna:
            Self para encadenamiento

        Lanza:
            ValueError: Si las configuraciones son inválidas
        """
        if not 0.0 <= self.neo4j_vector_similarity_threshold <= 1.0:
            raise ValueError("Umbral de similitud vectorial Neo4j debe estar entre 0.0 y 1.0")
        if self.neo4j_vector_limit < 1:
            raise ValueError("Límite de resultados vectoriales Neo4j debe ser al menos 1")
        if self.neo4j_relationships_limit < 1:
            raise ValueError("Límite de relaciones Neo4j debe ser al menos 1")
        if self.neo4j_paths_top_nodes < 1:
            raise ValueError("paths_top_nodes Neo4j debe ser al menos 1")
        if not 1 <= self.neo4j_paths_max_hops <= 6:
            raise ValueError("paths_max_hops Neo4j debe estar entre 1 y 6")
        return self


class PostgreSQLSettings(BaseSettings):
    """
    Configuración de base de datos PostgreSQL.

    Contiene cadena de conexión de PostgreSQL, configuraciones de pooling
    de conexiones y configuraciones específicas de pgvector.
    """

    # Connection Settings
    # Explicitly read from env var POSTGRES_DSN
    postgres_dsn: str = Field(alias="POSTGRES_DSN")

    # Connection Pool Settings
    pg_pool_size: int = 10
    pg_max_overflow: int = 20
    pg_pool_timeout: int = 30
    pg_pool_recycle: int = 3600
    pg_echo_sql: bool = False

    # PgVector Agent Configuration
    pgvector_recent_messages_limit: int = 3
    pgvector_semantic_limit: int = 5
    pgvector_max_total_results: int = 10
    pgvector_similarity_threshold: float = 0.7

    # Pydantic configuration to load from .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @field_validator("postgres_dsn")
    @classmethod
    def validate_postgres_dsn(cls, v):
        """
        Valida que el DSN de PostgreSQL esté correctamente formateado.

        Argumentos:
            v: DSN a validar

        Retorna:
            DSN validado

        Lanza:
            ValueError: Si el DSN está vacío o no usa el formato correcto
        """
        if not v or not v.strip():
            raise ValueError("DSN de PostgreSQL no puede estar vacío")
        if not v.startswith("postgresql://"):
            raise ValueError("DSN de PostgreSQL debe comenzar con postgresql://")
        return v

    @model_validator(mode="after")
    def validate_pool_settings(self):
        """
        Valida configuraciones del pool de conexiones.

        Retorna:
            Self para encadenamiento

        Lanza:
            ValueError: Si las configuraciones son inválidas
        """
        if self.pg_pool_size < 1:
            raise ValueError("Tamaño del pool PostgreSQL debe ser al menos 1")

        if self.pg_max_overflow < 0:
            raise ValueError("Max overflow de PostgreSQL debe ser no negativo")

        if self.pg_pool_timeout < 1:
            raise ValueError("Timeout del pool PostgreSQL debe ser al menos 1 segundo")

        if self.pg_pool_recycle < 60:
            raise ValueError("Reciclaje del pool PostgreSQL debe ser al menos 60 segundos")

        return self

    @model_validator(mode="after")
    def validate_pgvector_settings(self):
        """
        Valida configuraciones específicas de pgvector.

        Retorna:
            Self para encadenamiento

        Lanza:
            ValueError: Si las configuraciones son inválidas
        """
        if self.pgvector_recent_messages_limit < 1:
            raise ValueError("Límite de mensajes recientes PgVector debe ser al menos 1")

        if self.pgvector_semantic_limit < 1:
            raise ValueError("Límite semántico PgVector debe ser al menos 1")

        if self.pgvector_max_total_results < 1:
            raise ValueError("Resultados totales máximos PgVector debe ser al menos 1")

        if not 0.0 <= self.pgvector_similarity_threshold <= 1.0:
            raise ValueError("Umbral de similitud PgVector debe estar entre 0.0 y 1.0")

        return self


class DatabaseSettings(BaseSettings):
    """
    Configuración combinada de bases de datos.

    Incluye configuraciones tanto de Neo4j como de PostgreSQL, permitiendo
    acceso lazy a cada configuración según se necesite.
    """

    def __init__(self, **kwargs):
        """
        Inicializa la configuración combinada de bases de datos.

        Argumentos:
            **kwargs: Argumentos adicionales para BaseSettings
        """
        super().__init__(**kwargs)
        self._neo4j = None
        self._postgresql = None

    @property
    def neo4j(self) -> Neo4jSettings:
        """
        Obtiene la configuración de Neo4j (lazy loading).

        Retorna:
            Instancia de Neo4jSettings
        """
        if self._neo4j is None:
            try:
                self._neo4j = Neo4jSettings()  # type: ignore[call-arg]
            except Exception:
                self._neo4j = Neo4jSettings(
                    NEO4J_URL="bolt://placeholder",
                    NEO4J_USER="user",
                    NEO4J_PASSWORD="pass",
                )
        return self._neo4j

    @property
    def postgresql(self) -> PostgreSQLSettings:
        """Get PostgreSQL configuration"""
        if self._postgresql is None:
            try:
                self._postgresql = PostgreSQLSettings()  # type: ignore[call-arg]
            except Exception:
                self._postgresql = PostgreSQLSettings(POSTGRES_DSN="postgresql://user:pass@localhost/db")
        return self._postgresql


@lru_cache()
def get_neo4j_settings() -> Neo4jSettings:
    """Get cached Neo4j settings instance"""
    try:
        return Neo4jSettings()  # type: ignore[call-arg]
    except Exception:
        # mypy: environment-driven settings may appear missing during static analysis
        return Neo4jSettings(
            NEO4J_URL="bolt://placeholder",
            NEO4J_USER="user",
            NEO4J_PASSWORD="pass",
        )


@lru_cache()
def get_postgresql_settings() -> PostgreSQLSettings:
    """Get cached PostgreSQL settings instance"""
    try:
        return PostgreSQLSettings()  # type: ignore[call-arg]
    except Exception:
        # mypy: environment-driven settings may appear missing during static analysis
        return PostgreSQLSettings(POSTGRES_DSN="postgresql://user:pass@localhost/db")


@lru_cache()
def get_database_settings() -> DatabaseSettings:
    """Get cached combined database settings instance"""
    return DatabaseSettings()  # type: ignore[call-arg]


# Session dependency functions
def get_postgres_session():
    """FastAPI dependency that provides a database session"""
    from app.clients.sql_pgvector_client import get_database_session

    session = get_database_session()
    try:
        yield session
    finally:
        session.close()
