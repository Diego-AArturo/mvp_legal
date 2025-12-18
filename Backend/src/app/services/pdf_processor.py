"""
PDF Processor Service - Procesador avanzado de PDFs para tutelas.
Usa PyMuPDF (fitz) para extracción robusta de texto con métricas de calidad.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

import fitz  # PyMuPDF

from app.extensions import get_logger

logger = get_logger(__name__)


class PDFProcessorService:
    """Servicio para procesar PDFs de tutelas con extracción avanzada."""

    def __init__(self):
        """Inicializa el procesador de PDFs."""
        self.logger = logger

    def extract_text_from_pdf(self, pdf_content: bytes, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Extraer texto de PDF usando PyMuPDF con métricas de calidad.

        Argumentos:
            pdf_content: Contenido del PDF en bytes
            filename: Nombre del archivo (opcional, para logs)

        Retorna:
            Dict con texto extraído y metadatos de calidad
        """
        try:
            # Abrir PDF con PyMuPDF
            pdf_document = fitz.open(stream=pdf_content, filetype="pdf")

            extracted_text = ""
            total_pages = len(pdf_document)
            page_texts = []

            self.logger.info(f"Procesando PDF: {filename or 'unknown'} ({total_pages} páginas)")

            for page_num in range(total_pages):
                page = pdf_document[page_num]

                # Extracción de texto nativo
                page_text = page.get_text()

                # Limpiar texto de la página
                cleaned_page_text = self._clean_text(page_text)
                page_texts.append(cleaned_page_text)

                extracted_text += cleaned_page_text + "\n\n"

            pdf_document.close()

            # Calcular métricas de calidad
            quality_metrics = self._calculate_quality_metrics(page_texts)

            result = {
                "success": True,
                "extracted_text": extracted_text.strip(),
                "total_pages": total_pages,
                "text_length": len(extracted_text.strip()),
                "quality_metrics": quality_metrics,
                "metadata": {
                    "filename": filename or "unknown",
                    "pages_processed": total_pages,
                    "extraction_method": "pymupdf",
                    "text_quality": self._classify_quality(quality_metrics["overall_score"]),
                },
            }

            self.logger.info(f"Extracción completada: {len(extracted_text)} caracteres, calidad: {quality_metrics['overall_score']:.2f}")
            return result

        except Exception as e:
            self.logger.error(f"Error procesando PDF {filename or 'unknown'}: {e}")
            return {
                "success": False,
                "extracted_text": "",
                "error": str(e),
                "metadata": {"filename": filename or "unknown", "extraction_method": "failed"},
            }

    def _clean_text(self, text: str) -> str:
        """Limpiar y normalizar texto extraído."""
        if not text:
            return ""

        # Eliminar caracteres de control
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # Normalizar espacios múltiples
        text = re.sub(r" +", " ", text)

        # Normalizar saltos de línea múltiples
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

        # Eliminar líneas muy cortas (probablemente ruido)
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            # Solo líneas con más de 2 caracteres
            if len(line) > 2:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _calculate_quality_metrics(self, page_texts: list[str]) -> Dict[str, Any]:
        """Calcular métricas de calidad del texto extraído."""
        if not page_texts:
            return {"overall_score": 0.0, "avg_words_per_page": 0, "avg_chars_per_page": 0, "empty_pages": 0}

        total_words = 0
        total_chars = 0
        empty_pages = 0

        for page_text in page_texts:
            words = page_text.split()
            word_count = len(words)
            char_count = len(page_text)

            total_words += word_count
            total_chars += char_count

            if word_count < 10:  # Página con muy poco contenido
                empty_pages += 1

        avg_words_per_page = total_words / len(page_texts) if page_texts else 0
        avg_chars_per_page = total_chars / len(page_texts) if page_texts else 0

        # Calcular score de calidad (0.0 - 1.0)
        quality_score = 0.0

        # Criterio 1: Palabras por página (30%)
        if avg_words_per_page > 200:
            quality_score += 0.3
        elif avg_words_per_page > 100:
            quality_score += 0.2
        elif avg_words_per_page > 50:
            quality_score += 0.1

        # Criterio 2: Caracteres por página (30%)
        if avg_chars_per_page > 1000:
            quality_score += 0.3
        elif avg_chars_per_page > 500:
            quality_score += 0.2
        elif avg_chars_per_page > 200:
            quality_score += 0.1

        # Criterio 3: Páginas vacías (40%)
        empty_ratio = empty_pages / len(page_texts) if page_texts else 1.0
        quality_score += (1.0 - empty_ratio) * 0.4

        return {
            "overall_score": round(quality_score, 2),
            "avg_words_per_page": round(avg_words_per_page, 1),
            "avg_chars_per_page": round(avg_chars_per_page, 1),
            "empty_pages": empty_pages,
            "total_pages": len(page_texts),
        }

    def _classify_quality(self, score: float) -> str:
        """Clasificar la calidad del texto extraído."""
        if score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "medium"
        else:
            return "low"
