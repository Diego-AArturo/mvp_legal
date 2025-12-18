"""
Punto de entrada principal de la aplicación FastAPI para Petition API.

Este módulo configura y inicializa la aplicación FastAPI, gestiona el ciclo de vida
de los clientes (PostgreSQL, Neo4j, MinIO, servicios de embeddings) y establece
los manejadores de excepciones globales.

Sigue el patrón recomendado de FastAPI para aplicaciones con múltiples dependencias
externas y gestión de recursos compartidos.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, Request, status
from fastapi.exceptions import HTTPException as FastAPIHTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.routers import router as api_router
from app.api.v1.routers.core.health_router import health_check
from app.clients.embedding_client import lifespan_embedding
from app.clients.minio_client import lifespan_minio
from app.clients.neo4j_client import lifespan_neo4j
from app.clients.sql_pgvector_client import lifespan_pgvector
from app.config.settings import Settings, get_settings
from app.extensions import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Gestor del ciclo de vida de la aplicación que coordina todos los clientes.

    Inicializa los clientes en orden de dependencia:
    1. PostgreSQL + pgvector (base de datos principal)
    2. Neo4j (base de datos de grafos)
    3. MinIO (almacenamiento de objetos)
    4. Servicio de embeddings (modelos de embeddings)

    Nota: Ollama se ejecuta como servicio externo y no requiere gestión de ciclo de vida.

    Argumentos:
        app: Instancia de FastAPI que está iniciando

    Yields:
        None: Controla el ciclo de vida durante la ejecución de la aplicación

    Lanza:
        Exception: Propaga cualquier error durante la inicialización
    """
    logger.info("Iniciando Petition API...")

    try:
        # Inicializar clientes en orden de dependencia (anidados para garantizar orden)
        async with lifespan_pgvector(app):
            logger.info("Base de datos PostgreSQL + pgvector inicializada")

            async with lifespan_neo4j(app):
                logger.info("Cliente Neo4j inicializado")

                async with lifespan_minio(app):
                    logger.info("Cliente MinIO inicializado")

                    async with lifespan_embedding(app):
                        logger.info("Servicio de embeddings inicializado")

                        # Ollama se ejecuta como servicio externo independiente
                        # No requiere gestión de ciclo de vida en esta aplicación
                        logger.info("Servicio externo Ollama configurado")
                        yield

    except Exception as e:
        logger.error(
            f"Error durante el inicio de la aplicación: {str(e)}",
            exc_info=True,
        )
        raise
    finally:
        logger.info("Cerrando Petition API...")


def create_app(app_settings: Optional[Settings] = None) -> FastAPI:
    """
    Crea y configura la aplicación FastAPI.

    Esta función centraliza toda la configuración de la aplicación incluyendo:
    - Configuración de FastAPI (metadatos, documentación)
    - Middleware (CORS, redirección HTTPS)
    - Registro de routers
    - Manejadores de excepciones globales

    Argumentos:
        app_settings: Configuración opcional de la aplicación. Si no se proporciona,
                      se obtiene de las variables de entorno mediante get_settings().

    Retorna:
        Instancia configurada de FastAPI lista para ejecutarse.
    """
    if app_settings is None:
        app_settings = get_settings()

    async def https_redirect_middleware(request: Request, call_next):
        """
        Middleware que fuerza la redirección de HTTP a HTTPS.

        Solo actúa si FORCE_HTTPS está habilitado y ALLOW_INSECURE_AUTH
        está deshabilitado. Esto asegura que todas las solicitudes se
        realicen sobre conexiones seguras.
        """
        if app_settings.FORCE_HTTPS and not app_settings.ALLOW_INSECURE_AUTH:
            if request.url.scheme == "http":
                https_url = request.url.replace(scheme="https")
                return JSONResponse(
                    status_code=status.HTTP_301_MOVED_PERMANENTLY,
                    content={"detail": "HTTPS required"},
                    headers={"Location": str(https_url)},
                )
        response = await call_next(request)
        return response

    # Crear instancia de FastAPI con configuración básica
    app = FastAPI(
        title=app_settings.title,
        version=app_settings.version,
        description=app_settings.description,
        lifespan=lifespan,  # Usa el gestor de ciclo de vida coordinado
        docs_url=app_settings.DOCS_URL,
        redoc_url=app_settings.REDOC_URL,
        openapi_url=app_settings.OPENAPI_URL,
        # Configuración de la interfaz Swagger UI
        swagger_ui_parameters={
            "defaultModelsExpandDepth": -1,
            "defaultModelExpandDepth": 1,
            "displayRequestDuration": True,
            "docExpansion": "none",
            "filter": True,
            "showExtensions": True,
            "showCommonExtensions": True,
            "syntaxHighlight.theme": "monokai",
            "tryItOutEnabled": True,
        },
    )
    app.debug = app_settings.DEBUG_MODE

    # Verificar y registrar el estado de la autenticación
    try:
        if not app_settings.SECRET_KEY or not app_settings.LDAP_SERVER:
            logger.warning("Autenticación deshabilitada: faltan SECRET_KEY o " "configuración LDAP. La API iniciará con acceso anónimo " "para endpoints no protegidos.")
    except Exception:
        # La configuración puede no estar completamente inicializada; ignorar
        logger.warning("Verificación de configuración de autenticación falló; " "continuando el inicio.")

    # Configurar middleware CORS si hay orígenes permitidos
    if app_settings.BACKEND_CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=app_settings.BACKEND_CORS_ORIGINS,
            allow_credentials=app_settings.CORS_ALLOW_CREDENTIALS,
            allow_methods=app_settings.CORS_ALLOW_METHODS,
            allow_headers=app_settings.CORS_ALLOW_HEADERS,
        )
        logger.info(f"CORS configurado para {len(app_settings.BACKEND_CORS_ORIGINS)} orígenes")

    # Registrar middleware de redirección HTTPS
    app.middleware("http")(https_redirect_middleware)

    # Registrar router principal de la API v1
    app.include_router(api_router, prefix=app_settings.API_V1_STR)
    logger.info(f"Router API registrado bajo el prefijo: {app_settings.API_V1_STR}")

    @app.get("/")
    async def root():
        """
        Endpoint raíz que proporciona información básica de la API.

        Retorna:
            Diccionario con información de la API y enlaces a la documentación.
        """
        return {
            "message": "Petition API",
            "version": app_settings.version,
            "docs": app_settings.DOCS_URL,
            "redoc": app_settings.REDOC_URL,
        }

    # Reutilizar el mismo handler del health_router para /health
    app.add_api_route("/health", endpoint=health_check, methods=["GET"], tags=["health"])

    # Manejadores de excepciones globales
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """
        Manejador global para excepciones no capturadas.

        Registra el error completo y retorna una respuesta genérica 500
        para evitar exponer detalles internos del sistema al cliente.
        """
        logger.error(
            f"Error no manejado: {exc}",
            exc_info=True,
            extra={"path": request.url.path},
        )
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "detail": "Unexpected error"},
        )

    @app.exception_handler(FastAPIHTTPException)
    async def http_exception_handler(request: Request, exc: FastAPIHTTPException):
        """
        Manejador para excepciones HTTP de FastAPI.

        Formatea las excepciones HTTP de manera consistente con información
        del código de estado y detalles del error.
        """
        logger.warning(
            f"Error HTTP {exc.status_code}: {exc.detail}",
            extra={"path": request.url.path},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTPException",
                "detail": exc.detail,
                "status_code": exc.status_code,
            },
        )

    logger.info(f"Aplicación FastAPI creada: {app_settings.title} v{app_settings.version}")
    return app


# Instancia global de la aplicación FastAPI
# Se crea al importar el módulo y es utilizada por el servidor ASGI (uvicorn)
app = create_app()
