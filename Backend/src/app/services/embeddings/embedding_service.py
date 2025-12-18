"""
Servicio de embeddings para generar vectores semánticos con sentence-transformers.

Genera embeddings multilingües usando paraphrase-multilingual-MiniLM-L12-v2 con optimizaciones
para CPU, soporte de batches y lematización automática.

IMPORTANTE: el texto se lematiza antes de generar embeddings para mejorar la calidad de búsquedas
semánticas; los textos originales siguen almacenándose tal cual en Neo4j/Postgres.
"""

from __future__ import annotations

import asyncio
import os
from typing import List, Union, cast

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config.models import EmbeddingSettings, get_embedding_settings
from app.extensions import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Servicio para generar embeddings semánticos con sentence-transformers.

    Proporciona generación multilingüe optimizada para CPU, procesamiento por lotes
    y administración de recursos con carga diferida.

    CARACTERÍSTICAS:
    - Lematización automática del texto antes de generar embeddings
    - Soporte eficiente para batch processing
    - Manejo de recursos con lazy loading y bloqueo thread-safe
    """

    def __init__(self, settings: EmbeddingSettings | None = None, enable_lemmatization: bool = True):
        """
        Inicializa el servicio de embeddings.

        Argumentos:
            settings: Configuración de embeddings
            enable_lemmatization: Si True, aplica lematización antes de generar embeddings (predeterminado True)
        """
        self.settings = settings or get_embedding_settings()
        self.model: SentenceTransformer | None = None
        self.model_loaded = False
        self._model_lock = asyncio.Lock()
        self.enable_lemmatization = enable_lemmatization
        
        # Lazy import para evitar dependencias circulares
        self._lemmatizer = None
        self._lemmatizer_initialized = False

        logger.info(
            f"Servicio de embeddings inicializado con el modelo: {self.settings.embedding_model_id}",
            extra={"lemmatization_enabled": enable_lemmatization},
        )

    def _get_lemmatizer(self):
        """Obtiene la instancia del lemmatizer (lazy loading)."""
        if not self._lemmatizer_initialized:
            try:
                from app.services.text_lemmatizer import get_lemmatizer
                self._lemmatizer = get_lemmatizer()
                self._lemmatizer_initialized = True
                
                if self._lemmatizer.is_available():
                    logger.info("Lemmatizer inicializado correctamente")
                else:
                    logger.warning("Lemmatizer no disponible, embeddings usarán texto original")
            except Exception as e:
                logger.warning(f"Error inicializando lemmatizer: {e}. Embeddings usarán texto original")
                self._lemmatizer = None
                self._lemmatizer_initialized = True
        
        return self._lemmatizer

    async def initialize_model(self) -> None:
        """
        Inicializa el modelo de sentence transformer de forma segura.

        Carga el modelo con optimización para CPU, registra la configuración y maneja errores.
        """
        if self.model_loaded:
            return

        async with self._model_lock:
            if self.model_loaded:
                return

            try:
                logger.info(f"Cargando modelo de embeddings: {self.settings.embedding_model_id}")

                # Registrar configuración relevante para diagnóstico
                logger.info(
                    "Configuración de embeddings: device=%s, cache_dir=%s, "
                    "HF_HOME=%s, TRANSFORMERS_CACHE=%s, SENTENCE_TRANSFORMERS_HOME=%s, "
                    "HF_HUB_OFFLINE=%s, HF_ENDPOINT=%s",
                    self.settings.embedding_device,
                    self.settings.embedding_cache_dir,
                    os.getenv("HF_HOME"),
                    os.getenv("TRANSFORMERS_CACHE"),
                    os.getenv("SENTENCE_TRANSFORMERS_HOME"),
                    os.getenv("HF_HUB_OFFLINE"),
                    os.getenv("HF_ENDPOINT"),
                )

                # Preparar kwargs del modelo
                model_kwargs = {
                    "device": self.settings.embedding_device,
                    **self.settings.embedding_model_kwargs,
                }

                # Agregar directorio de caché si está especificado
                if self.settings.embedding_cache_dir:
                    model_kwargs["cache_folder"] = self.settings.embedding_cache_dir

                # Agregar token de autenticación si está especificado
                if self.settings.embedding_use_auth_token and self.settings.embedding_auth_token:
                    model_kwargs["use_auth_token"] = self.settings.embedding_auth_token

                # Fallback a CPU si CUDA fue solicitado pero no está disponible
                try:
                    import torch  # type: ignore[import-not-found]

                    cuda_available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
                except Exception:
                    cuda_available = False

                requested_device = str(self.settings.embedding_device).lower()
                if requested_device.startswith("cuda") and not cuda_available:
                    logger.warning("CUDA solicitado pero no está disponible; usando CPU como fallback")
                    model_kwargs["device"] = "cpu"

                # Cargar modelo en thread para evitar bloqueo
                self.model = await asyncio.to_thread(
                    SentenceTransformer,
                    self.settings.embedding_model_id,
                    **model_kwargs,
                )

                # Set max sequence length
                if self.model is not None and hasattr(self.model, "max_seq_length"):
                    self.model.max_seq_length = self.settings.embedding_max_seq_length

                self.model_loaded = True
                logger.info(
                    f"Modelo de embeddings cargado correctamente en el dispositivo: {self.settings.embedding_device}"
                )

            except Exception as e:
                logger.error(f"Error cargando el modelo de embeddings: {e}")
                raise RuntimeError(f"No se pudo inicializar el modelo de embeddings: {e}") from e

    async def generate_embeddings(self, texts: Union[str, List[str]], normalize: bool | None = None) -> Union[List[float], List[List[float]]]:
        """
        Genera embeddings para uno o varios textos.

        IMPORTANTE: Si enable_lemmatization=True, los textos se lematizan previo a la generación para
        normalizar palabras y mejorar la calidad de las búsquedas.

        Argumentos:
            texts: Texto individual o lista de textos
            normalize: Si se normalizan los embeddings (usa la configuración por defecto si es None)

        Retorna:
            Embedding único o lista de embeddings
        """
        if not self.model_loaded:
            await self.initialize_model()

        if self.model is None:
            raise RuntimeError("Embedding model not initialized")

        # Handle single text vs list; narrow to List[str] for typing
        if isinstance(texts, str):
            is_single_text = True
            text_list: List[str] = [texts]
        else:
            is_single_text = False
            text_list = cast(List[str], texts)

        if not text_list:
            return [] if not is_single_text else []

        try:
            # PASO 1: Lematizar textos si está habilitado
            processed_texts = text_list
            if self.enable_lemmatization:
                lemmatizer = self._get_lemmatizer()
                if lemmatizer and lemmatizer.is_available():
                    logger.debug(
                        "Lematizando textos antes de generar embeddings",
                        extra={"text_count": len(text_list)}
                    )
                    
                    # Lematizar en batch para mayor eficiencia
                    processed_texts = await asyncio.to_thread(
                        lemmatizer.lemmatize_batch,
                        text_list,
                        lowercase=True,
                        batch_size=50
                    )
                    
                    logger.debug(
                        "Textos lematizados",
                        extra={
                            "original_sample": text_list[0][:50] if text_list else "",
                            "lemmatized_sample": processed_texts[0][:50] if processed_texts else "",
                        },
                    )
                else:
                    logger.debug("Lemmatizer no disponible, usando textos originales")
            else:
                logger.debug("Lematización deshabilitada, usando textos originales")
            
            # PASO 2: Generar embeddings con los textos procesados
            # Preparar kwargs de codificación
            encode_kwargs = {
                "batch_size": self.settings.embedding_batch_size,
                "normalize_embeddings": (normalize if normalize is not None else self.settings.embedding_normalize_embeddings),
                "convert_to_numpy": True,
                **self.settings.embedding_encode_kwargs,
            }

            # Generar embeddings en un hilo para evitar bloqueo
            model: SentenceTransformer = self.model  # referencia local no opcional
            embeddings = await asyncio.to_thread(lambda: model.encode(processed_texts, **encode_kwargs))

            # Convertir los arrays numpy a listas
            if isinstance(embeddings, np.ndarray):
                embeddings_list = embeddings.tolist()
            else:
                embeddings_list = embeddings

            # Retornar embedding único o lista según el input
            if is_single_text:
                return embeddings_list[0] if embeddings_list else []  # type: ignore
            else:
                return embeddings_list  # type: ignore

        except Exception as e:
            logger.error(f"Error al generar embeddings: {e}")
            raise RuntimeError(f"No se pudieron generar los embeddings: {e}") from e

    async def generate_embedding(self, text: str, normalize: bool | None = None) -> List[float]:
        """
        Genera un embedding para un texto individual.

        Argumentos:
            text: Texto de entrada
            normalize: Si se normaliza el vector

        Retorna:
            Embedding como lista de floats
        """
        result = await self.generate_embeddings(text, normalize)
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], (int, float)):
            return result  # type: ignore
        elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
            return result[0]  # type: ignore
        else:
            return []

    async def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Calcula la similitud coseno entre dos textos.

        Argumentos:
            text1: Primer texto
            text2: Segundo texto

        Retorna:
            Puntuación de similitud coseno
        """
        embeddings = await self.generate_embeddings([text1, text2], normalize=True)

        if len(embeddings) != 2:
            raise ValueError("No se pudieron generar embeddings para ambos textos")

        # Compute cosine similarity
        emb1 = np.array(embeddings[0])
        emb2 = np.array(embeddings[1])

        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)

    def get_embedding_dimension(self) -> int:
        """
        Obtiene la dimensión de los embeddings.

        Retorna:
            Dimensión del vector de embeddings
        """
        if not self.model_loaded or self.model is None:
            # Return known dimension for paraphrase-multilingual-MiniLM-L12-v2
            return 384

        # Get dimension from model
        return self.model.get_sentence_embedding_dimension()  # type: ignore

    async def cleanup(self) -> None:
        """Libera los recursos asociados al modelo."""
        if self.model is not None:
            # Move model to CPU to free GPU memory if applicable
            if hasattr(self.model, "to"):
                self.model.to("cpu")

            self.model = None
            self.model_loaded = False
            logger.info("Modelo de embeddings limpiado")


# Global embedding service instance
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Obtiene o crea la instancia global del servicio de embeddings."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


async def cleanup_embedding_service() -> None:
    """Libera la instancia global del servicio de embeddings."""
    global _embedding_service
    if _embedding_service is not None:
        await _embedding_service.cleanup()
        _embedding_service = None
