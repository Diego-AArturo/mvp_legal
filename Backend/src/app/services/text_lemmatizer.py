"""
Text Lemmatizer Service - Servicio centralizado de lematización para español.

Este servicio proporciona lematización de texto en español usando spaCy.
Se utiliza principalmente para normalizar texto antes de generar embeddings,
mejorando la calidad de las búsquedas semánticas.

IMPORTANTE: La lematización solo se aplica al texto usado para embeddings.
Los textos originales se mantienen sin modificar para persistencia en Neo4j.
"""

import logging
from typing import List, Optional

try:
    import spacy
    from spacy.language import Language

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy no está instalado. Lematización deshabilitada.")

from app.extensions import get_logger

logger = get_logger(__name__)


class TextLemmatizerService:
    """
    Servicio de lematización de texto para español.

    Usa spaCy con el modelo es_core_news_sm para realizar lematización eficiente.
    El servicio está optimizado para documentos legales colombianos.
    """

    def __init__(self, model_name: str = "es_core_news_sm"):
        """
        Inicializa el servicio de lematización.

        Argumentos:
            model_name: Nombre del modelo spaCy a cargar (por defecto es_core_news_sm)
        """
        self.model_name = model_name
        self._nlp: Optional[Language] = None
        self._is_initialized = False

        logger.info(
            "TextLemmatizerService creado (inicialización lazy)",
            extra={"model_name": model_name, "spacy_available": SPACY_AVAILABLE},
        )

    def _initialize(self) -> None:
        """Inicializa el modelo spaCy (lazy loading)."""
        if self._is_initialized:
            return

        if not SPACY_AVAILABLE:
            logger.error(
                "spaCy no disponible. Instalar con: pip install spacy && python -m spacy download es_core_news_sm"
            )
            self._is_initialized = True
            return

        try:
            logger.info(f"Cargando modelo spaCy: {self.model_name}")

            # Cargar modelo con pipeline optimizado para lematización
            # Deshabilitamos componentes innecesarios para mejor rendimiento
            self._nlp = spacy.load(
                self.model_name,
                disable=["ner", "parser"],  # Solo necesitamos tagger y lemmatizer
            )

            logger.info(
                "Modelo spaCy cargado exitosamente",
                extra={
                    "model_name": self.model_name,
                    "pipeline": self._nlp.pipe_names if self._nlp else [],
                },
            )

            self._is_initialized = True

        except OSError as e:
            logger.error(
                f"Error cargando modelo spaCy '{self.model_name}': {e}",
                extra={"model_name": self.model_name, "error": str(e)},
            )
            logger.info(
                f"Instalar con: python -m spacy download {self.model_name}"
            )
            self._is_initialized = True  # Marcar como inicializado para no reintentar

    def is_available(self) -> bool:
        """
        Verifica si el servicio de lematización está disponible.

        Retorna:
            True si spaCy está disponible y el modelo está cargado
        """
        if not self._is_initialized:
            self._initialize()

        return self._nlp is not None

    def lemmatize(self, text: str, lowercase: bool = True) -> str:
        """
        Lematiza un texto en español.

        Argumentos:
            text: Texto a lematizar
            lowercase: Si True, convierte a minúsculas (recomendado para embeddings)

        Retorna:
            Texto lematizado. Si la lematización falla, retorna el texto original.
        """
        if not text or not text.strip():
            return text

        # Inicializar si es necesario
        if not self._is_initialized:
            self._initialize()

        # Si spaCy no está disponible, retornar texto original
        if not self._nlp:
            logger.warning(
                "Lematización no disponible, retornando texto original",
                extra={"text_length": len(text)},
            )
            return text.lower() if lowercase else text

        try:
            # Procesar texto con spaCy
            doc = self._nlp(text)

            # Extraer lemas
            # Filtrar tokens que no son puntuación ni espacios
            lemmas = [
                token.lemma_
                for token in doc
                if not token.is_punct and not token.is_space
            ]

            # Unir lemas
            lemmatized_text = " ".join(lemmas)

            # Aplicar lowercase si se solicita
            if lowercase:
                lemmatized_text = lemmatized_text.lower()

            logger.debug(
                "Texto lematizado",
                extra={
                    "original_length": len(text),
                    "lemmatized_length": len(lemmatized_text),
                    "token_count": len(lemmas),
                },
            )

            return lemmatized_text

        except Exception as e:
            logger.error(
                f"Error en lematización: {e}",
                extra={"text_length": len(text), "error": str(e)},
            )
            # En caso de error, retornar texto original
            return text.lower() if lowercase else text

    def lemmatize_batch(
        self, texts: List[str], lowercase: bool = True, batch_size: int = 50
    ) -> List[str]:
        """
        Lematiza múltiples textos de forma eficiente usando pipe de spaCy.

        Argumentos:
            texts: Lista de textos a lematizar
            lowercase: Si True, convierte a minúsculas
            batch_size: Tamaño del batch para procesamiento en pipe

        Retorna:
            Lista de textos lematizados
        """
        if not texts:
            return []

        # Inicializar si es necesario
        if not self._is_initialized:
            self._initialize()

        # Si spaCy no está disponible, retornar textos originales
        if not self._nlp:
            logger.warning(
                "Lematización no disponible, retornando textos originales",
                extra={"text_count": len(texts)},
            )
            return [text.lower() if lowercase else text for text in texts]

        try:
            lemmatized_texts = []

            # Procesar en batch usando pipe para mayor eficiencia
            for doc in self._nlp.pipe(texts, batch_size=batch_size):
                # Extraer lemas (sin puntuación ni espacios)
                lemmas = [
                    token.lemma_
                    for token in doc
                    if not token.is_punct and not token.is_space
                ]

                lemmatized_text = " ".join(lemmas)

                if lowercase:
                    lemmatized_text = lemmatized_text.lower()

                lemmatized_texts.append(lemmatized_text)

            logger.info(
                "Batch de textos lematizado",
                extra={
                    "text_count": len(texts),
                    "batch_size": batch_size,
                    "total_tokens": sum(len(t.split()) for t in lemmatized_texts),
                },
            )

            return lemmatized_texts

        except Exception as e:
            logger.error(
                f"Error en lematización batch: {e}",
                extra={"text_count": len(texts), "error": str(e)},
            )
            # En caso de error, retornar textos originales
            return [text.lower() if lowercase else text for text in texts]


# Singleton instance para uso global
_lemmatizer_instance: Optional[TextLemmatizerService] = None


def get_lemmatizer() -> TextLemmatizerService:
    """
    Obtiene la instancia singleton del servicio de lematización.

    Retorna:
        TextLemmatizerService: Instancia del servicio de lematización
    """
    global _lemmatizer_instance

    if _lemmatizer_instance is None:
        _lemmatizer_instance = TextLemmatizerService()

    return _lemmatizer_instance

