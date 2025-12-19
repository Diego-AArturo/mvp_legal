"""
Cliente para PostgreSQL con soporte pgvector.

Proporciona gestión del ciclo de vida de la conexión a PostgreSQL, creación de sesiones
SQLAlchemy y funciones auxiliares para operaciones de base de datos con soporte vectorial.
""" 


import asyncio
import json
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator, Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, status
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from app.config.databases import get_postgresql_settings
from app.extensions import get_logger
from app.models.tutelas_models import Base  # Importar modelos para asegurar creación de tablas

logger = get_logger(__name__)

# Obtener singleton de configuración
settings = get_postgresql_settings()

# Crear engine de SQLAlchemy con pool de conexiones
# Usa POSTGRES_DSN desde variables de entorno
engine: Engine = create_engine(
    settings.postgres_dsn,
    echo=settings.pg_echo_sql,
    pool_pre_ping=True,
    pool_size=settings.pg_pool_size,
    max_overflow=settings.pg_max_overflow,
    pool_timeout=settings.pg_pool_timeout,
    pool_recycle=settings.pg_pool_recycle,
    poolclass=QueuePool,
    # Additional connection arguments for PostgreSQL
    connect_args={
        "application_name": "petition-api",
        "options": "-c timezone=UTC",
    },
)

# Factory de sesiones para uso con ORM
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,
)


def get_database_session() -> Session:
    """
    Crea una nueva sesión de base de datos.

    Retorna:
        Sesión de SQLAlchemy lista para usar

    Nota:
        Recuerda cerrar la sesión después de usarla o usa el context manager
        session_scope para gestión automática.
    """
    return SessionLocal()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Proporciona un ámbito transaccional alrededor de una serie de operaciones.

    Este context manager maneja automáticamente:
    - Creación de sesión
    - Commit de transacción en caso de éxito
    - Rollback de transacción en caso de error
    - Limpieza de sesión

    Yields:
        Sesión de SQLAlchemy dentro del ámbito transaccional

    Ejemplo:
        with session_scope() as session:
            user = User(name="John")
            session.add(user)
            # La transacción se confirma automáticamente al salir con éxito
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Transacción de base de datos falló: {e}")
        raise
    finally:
        session.close()


def init_database() -> None:
    """
    Inicializa la base de datos creando todas las tablas.

    Esta función debe llamarse durante el inicio de la aplicación
    para asegurar que todas las tablas requeridas existan.
    """
    try:
        # Ensure required extensions exist before ORM creates tables.
        with session_scope() as session:
            session.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
            session.execute(text("CREATE EXTENSION IF NOT EXISTS citext"))
            session.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            session.execute(
                text(
                    """
                    DO $$
                    BEGIN
                      IF NOT EXISTS (
                        SELECT 1 FROM pg_type WHERE typname = 'chat_role'
                      ) THEN
                        CREATE TYPE public.chat_role AS ENUM ('user','assistant','system');
                      END IF;
                    END$$;
                    """
                )
            )
        Base.metadata.create_all(bind=engine)
        logger.info("Tablas de base de datos inicializadas exitosamente")
    except Exception as e:
        logger.error(f"Error al inicializar tablas de base de datos: {e}")
        raise


def close_database() -> None:
    """
    Cierra todas las conexiones de base de datos.

    Esta función debe llamarse durante el apagado de la aplicación
    para cerrar apropiadamente todas las conexiones de base de datos.
    """
    try:
        engine.dispose()
        logger.info("Conexiones de base de datos cerradas exitosamente")
    except Exception as e:
        logger.error(f"Error al cerrar conexiones de base de datos: {e}")
        raise


# Dependencia de FastAPI para sesiones de base de datos
def get_db() -> Generator[Session, None, None]:
    """
    Dependencia de FastAPI para sesiones de base de datos.

    Yields:
        Sesión de SQLAlchemy lista para usar

    Ejemplo:
        @app.get("/users/")
        def get_users(db: Session = Depends(get_db)):
            return db.query(User).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_connection() -> bool:
    """
    Prueba la conexión a la base de datos.

    Retorna:
        True si la conexión es exitosa, False en caso contrario
    """
    try:
        with session_scope() as session:
            session.execute(text("SELECT 1"))
        logger.info("Prueba de conexión a base de datos exitosa")
        return True
    except Exception as e:
        logger.error(f"Prueba de conexión a base de datos falló: {e}")
        return False


def enable_pgvector_extension() -> None:
    """
    Habilita la extensión pgvector en la base de datos.

    Esta función debe llamarse una vez durante la configuración inicial
    para habilitar operaciones de similitud vectorial.

    Nota:
        Requiere privilegios de superusuario o que la extensión esté disponible.
    """
    try:
        with session_scope() as session:
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        logger.info("Extensión pgvector habilitada exitosamente")
    except Exception as e:
        logger.error(f"Error al habilitar extensión pgvector: {e}")
        raise


@asynccontextmanager
async def lifespan_pgvector(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Gestiona el ciclo de vida de la base de datos PostgreSQL + pgvector.

    Inicializa las tablas de base de datos y la extensión pgvector al iniciar,
    cierra las conexiones al apagar.

    Argumentos:
        app: Instancia de FastAPI que está iniciando

    Yields:
        None: Controla el ciclo de vida durante la ejecución de la aplicación

    Lanza:
        HTTPException: Si la inicialización de la base de datos falla
    """
    logger.info("Inicializando base de datos PostgreSQL + pgvector...")

    try:
        # Probar conexión a base de datos
        connection_ok = await asyncio.get_event_loop().run_in_executor(None, test_connection)

        if not connection_ok:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Base de datos PostgreSQL no disponible",
            )

        # Inicializar tablas de base de datos (fatal si falla)
        await asyncio.get_event_loop().run_in_executor(None, init_database)

        # Intentar habilitar extensión pgvector (no fatal si no está disponible)
        try:
            await asyncio.get_event_loop().run_in_executor(None, enable_pgvector_extension)
        except Exception as e:
            logger.warning(
                "Extensión pgvector no disponible; " "continuando sin características vectoriales: %s",
                str(e),
            )

        logger.info("Base de datos PostgreSQL inicializada (pgvector opcional)")

    except Exception as e:
        logger.error(f"Error al inicializar base de datos PostgreSQL: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inicialización de base de datos falló",
        )

    yield

    # Limpieza
    try:
        await asyncio.get_event_loop().run_in_executor(None, close_database)
        logger.info("Conexiones de base de datos PostgreSQL cerradas exitosamente")
    except Exception as e:
        logger.error(f"Error al cerrar base de datos PostgreSQL: {str(e)}")


# ============================================================================
# CLASE: POSTGRESQL CLIENT (Compatibilidad con shared/postgres_client.py)
# ============================================================================

class PostgreSQLClient:
    """Cliente para interactuar con PostgreSQL usando SQLAlchemy."""

    def __init__(self):
        """Inicializar cliente PostgreSQL."""
        self.connection = None  # Mantener para compatibilidad, pero usar SQLAlchemy

    def connect(self) -> bool:
        """Establecer conexión a PostgreSQL."""
        return test_connection()

    def disconnect(self) -> None:
        """Cerrar conexión a PostgreSQL."""
        self.connection = None
        logger.info("PostgreSQL connection closed")

    def is_connected(self) -> bool:
        """Verificar si la conexión está activa."""
        return test_connection()

    def log_event(
        self,
        tutela_id: str,
        tipo_evento: str,
        componente: str,
        filename: str = None,
        processing_time: float = None,
        status: str = "IN_PROGRESS",
        detalles: Dict = None
    ) -> bool:
        """
        Registrar evento en PostgreSQL usando SQLAlchemy.

        Args:
            tutela_id: Hash del documento
            tipo_evento: Tipo de evento (INGESTA, PROCESAMIENTO, COMPLETADO)
            componente: Componente que genera el evento
            filename: Nombre del archivo
            processing_time: Tiempo de procesamiento en ms
            status: Estado del procesamiento
            detalles: Detalles adicionales

        Returns:
            bool: True si se registró exitosamente
        """
        if not all([tutela_id, tipo_evento, componente]):
            logger.error(
                f"Missing required arguments: tutela_id={tutela_id}, "
                f"tipo_evento={tipo_evento}, componente={componente}"
            )
            return False

        if detalles is None:
            detalles = {}

        detalles_json = json.dumps(detalles) if detalles else None

        sql = text("""
            INSERT INTO processing_logs 
            (tutela_id, tipo_evento, componente, filename, processing_time, status, detalles, created_at)
            VALUES (:tutela_id, :tipo_evento, :componente, :filename, :processing_time, :status, :detalles, CURRENT_TIMESTAMP)
        """)

        try:
            with session_scope() as session:
                session.execute(
                    sql,
                    {
                        "tutela_id": tutela_id,
                        "tipo_evento": tipo_evento,
                        "componente": componente,
                        "filename": filename,
                        "processing_time": int(processing_time) if processing_time else None,
                        "status": status,
                        "detalles": detalles_json,
                    }
                )
            logger.info(f"Event logged: {tipo_evento} for tutela {tutela_id}")
            return True
        except Exception as e:
            logger.error(f"Error logging event: {e}")
            return False

    def check_duplicate(self, tutela_id: str) -> bool:
        """
        Verificar si un documento ya fue procesado.

        Args:
            tutela_id: Hash del documento

        Returns:
            bool: True si ya existe, False si es nuevo
        """
        query = text("SELECT COUNT(*) FROM processing_logs WHERE tutela_id = :tutela_id")

        try:
            with session_scope() as session:
                result = session.execute(query, {"tutela_id": tutela_id})
                count = result.scalar()
                return count > 0
        except Exception as e:
            logger.error(f"Error checking duplicate: {e}")
            return False

    def get_latest_status(self, tutela_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtener el último estado de un documento.

        Args:
            tutela_id: Hash del documento

        Returns:
            dict: Último estado o None si no existe
        """
        query = text("""
            SELECT tipo_evento, componente, status, created_at
            FROM v_latest_document_status 
            WHERE tutela_id = :tutela_id
        """)

        try:
            with session_scope() as session:
                result = session.execute(query, {"tutela_id": tutela_id})
                row = result.fetchone()

                if row:
                    return {
                        'tipo_evento': row[0],
                        'componente': row[1],
                        'status': row[2],
                        'created_at': row[3]
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting latest status: {e}")
            return None

    def get_document_history(self, tutela_id: str) -> List[Dict[str, Any]]:
        """
        Obtener historial completo de un documento.

        Args:
            tutela_id: Hash del documento

        Returns:
            list: Lista de eventos del documento
        """
        query = text("""
            SELECT id, tutela_id, tipo_evento, componente,
                   filename, processing_time, status, detalles, created_at
            FROM processing_logs
            WHERE tutela_id = :tutela_id
            ORDER BY created_at
        """)

        try:
            with session_scope() as session:
                result = session.execute(query, {"tutela_id": tutela_id})
                rows = result.fetchall()

                logs = []
                for row in rows:
                    log = {
                        'id': row[0],
                        'tutela_id': row[1],
                        'tipo_evento': row[2],
                        'componente': row[3],
                        'filename': row[4],
                        'processing_time': row[5],
                        'status': row[6],
                        'detalles': json.loads(row[7]) if row[7] else None,
                        'created_at': row[8]
                    }
                    logs.append(log)

                return logs
        except Exception as e:
            logger.error(f"Error getting document history: {e}")
            return []

    def get_etl_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del sistema ETL.

        Returns:
            dict: Estadísticas del sistema
        """
        query = text("""
            SELECT 
                componente,
                COUNT(*) as total_events,
                COUNT(DISTINCT tutela_id) as unique_documents,
                AVG(processing_time) as avg_processing_time,
                COUNT(CASE WHEN status IN ('COMPLETADO', 'SUCCESS') THEN 1 END) as successful_events
            FROM processing_logs
            WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
            GROUP BY componente
            ORDER BY total_events DESC
        """)

        try:
            with session_scope() as session:
                result = session.execute(query)
                rows = result.fetchall()

                stats = []
                for row in rows:
                    stats.append({
                        'componente': row[0],
                        'total_events': row[1],
                        'unique_documents': row[2],
                        'avg_processing_time': float(row[3]) if row[3] else 0,
                        'successful_events': row[4]
                    })

                total_events = sum(s['total_events'] for s in stats)
                total_documents = sum(s['unique_documents'] for s in stats)
                successful_events = sum(s['successful_events'] for s in stats)
                success_rate = (successful_events / total_events * 100) if total_events > 0 else 0

                return {
                    'components': stats,
                    'total_events': total_events,
                    'total_documents': total_documents,
                    'success_rate': success_rate
                }
        except Exception as e:
            logger.error(f"Error getting ETL stats: {e}")
            return {}

    def close(self):
        """Cerrar conexión."""
        self.disconnect()


def get_postgres_client() -> PostgreSQLClient:
    """Obtener instancia del cliente PostgreSQL."""
    return PostgreSQLClient()
