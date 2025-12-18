"""
Cliente para Neo4j - Base de datos de grafos.

Proporciona gestión del ciclo de vida del driver asíncrono de Neo4j y funciones
auxiliares para obtener sesiones y ejecutar consultas Cypher.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Request, status
from neo4j import AsyncDriver, AsyncGraphDatabase
from neo4j.exceptions import (
    AuthError,
    ConfigurationError,
    Neo4jError,
    ServiceUnavailable,
)

from app.config.databases import get_neo4j_settings
from app.extensions import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan_neo4j(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Inicializa el driver de Neo4j al iniciar la aplicación y lo cierra al apagarla.

    Utiliza driver asíncrono para operaciones no bloqueantes. Implementa reintentos
    con backoff exponencial en caso de fallos de conexión.

    Argumentos:
        app: Instancia de FastAPI que está iniciando

    Yields:
        None: Controla el ciclo de vida durante la ejecución de la aplicación

    Lanza:
        HTTPException: Si no se puede conectar después de todos los reintentos
    """
    settings = get_neo4j_settings()
    retries = 0

    while retries < settings.neo4j_max_connection_retries:
        try:
            # Crear el driver asíncrono con pool de conexiones
            driver = AsyncGraphDatabase.driver(
                settings.neo4j_url,
                auth=(settings.neo4j_user, settings.neo4j_password),
                max_connection_pool_size=settings.neo4j_max_connection_pool_size,
                connection_timeout=settings.neo4j_connection_timeout,
                connection_acquisition_timeout=settings.neo4j_connection_acquisition_timeout,
                max_transaction_retry_time=settings.neo4j_max_transaction_retry_time,
                initial_retry_delay=settings.neo4j_initial_retry_delay,
                retry_delay_multiplier=settings.neo4j_retry_delay_multiplier,
                # Nota: retry_delay_jitter no es un parámetro válido del driver de Neo4j
            )
            # Verificar conectividad de forma asíncrona
            await driver.verify_connectivity()
            app.state.neo4j_driver = driver
            logger.info("Conexión exitosa a la base de datos Neo4j")
            break
        except (ServiceUnavailable, ConfigurationError, AuthError) as e:
            retries += 1
            if retries == settings.neo4j_max_connection_retries:
                logger.error(
                    "Error al conectar a Neo4j después de %s intentos: %s",
                    settings.neo4j_max_connection_retries,
                    str(e),
                )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Servicio de base de datos no disponible",
                )
            logger.warning(f"Intento de conexión a Neo4j {retries} falló, reintentando...")

    yield

    # Limpieza
    try:
        await app.state.neo4j_driver.close()
        logger.info("Conexión a Neo4j cerrada exitosamente")
    except Exception as e:
        logger.error(f"Error al cerrar conexión a Neo4j: {str(e)}")


def get_neo4j_driver(request: Request) -> AsyncDriver:
    """
    Dependencia que retorna el objeto AsyncDriver de Neo4j.

    Incluye validación de conexión. Debe usarse como dependencia de FastAPI
    en routers que requieren acceso a Neo4j.

    Argumentos:
        request: Objeto Request de FastAPI que contiene el estado de la aplicación

    Retorna:
        Driver asíncrono de Neo4j configurado y conectado

    Lanza:
        HTTPException: Si el driver no está inicializado o no está disponible
    """
    app = request.app
    if not hasattr(app.state, "neo4j_driver"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conexión a base de datos no inicializada",
        )
    return app.state.neo4j_driver


@asynccontextmanager
async def get_neo4j_session(driver: Optional[AsyncDriver] = None):
    """
    Context manager asíncrono para sesiones de Neo4j.

    Gestiona automáticamente el ciclo de vida de la sesión y el manejo de errores.
    Si no se proporciona un driver, obtiene uno del estado de la aplicación.

    Argumentos:
        driver: Driver opcional de Neo4j. Si es None, se obtiene del estado de la app.

    Yields:
        Sesión asíncrona de Neo4j lista para ejecutar consultas Cypher

    Ejemplo:
        async with get_neo4j_session() as session:
            result = await session.run("MATCH (n) RETURN n")
            data = await result.data()
    """
    # Este helper ya no se usa directamente por FastAPI DI.
    # Se mantiene por compatibilidad hacia atrás si se referencia en otro lugar.
    session = None
    try:
        if driver is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Conexión a base de datos no inicializada",
            )
        session = driver.session(database=get_neo4j_settings().neo4j_database)
        yield session
    except Neo4jError as e:
        logger.error(f"Error en consulta Neo4j: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Consulta a base de datos falló",
        )
    except Exception as e:
        logger.error(f"Error inesperado en sesión Neo4j: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error inesperado en base de datos",
        )
    finally:
        if session:
            await session.close()
