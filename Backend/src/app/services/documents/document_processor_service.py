# -*- coding: utf-8 -*-
"""
Servicio de procesamiento de documentos normativos.

Este servicio maneja la extracción de texto, limpieza, normalización a markdown
y segmentación de documentos normativos individuales.
"""

from __future__ import annotations

import logging
import re
import zipfile
from collections import Counter
from typing import List, Optional

try:
    import docx2txt
except ImportError as e:
    docx2txt = None
    logging.error(f"Error importando libreria docx2txt: {e}")

try:
    import fitz  # PyMuPDF
except ImportError as e:
    fitz = None
    logging.error(f"Error importando libreria PyMuPDF (fitz): {e}")

from app.extensions import get_logger

logger = get_logger(__name__)


class DocumentProcessorService:
    """
    Servicio para procesar documentos normativos individuales.

    Maneja la extracción de texto, limpieza, normalización a markdown
    y segmentación de documentos según rutas jerárquicas.
    """

    def __init__(self):
        """Inicializa el procesador de documentos."""
        self.logger = logger

    def extract_text_from_file(self, archivo_entrada) -> Optional[str]:
        """
        Extrae el contenido de texto plano de un archivo PDF, Word (.docx) o Word antiguo (.doc).
        Esta versión es más robusta: usa PyMuPDF (fitz) como respaldo para PDFs corruptos.

        Argumentos:
            archivo_entrada: Puede ser una ruta de archivo (string) o un objeto en memoria (BytesIO).

        Retorna:
            str: Texto extraído del documento o None si hay error
        """
        texto_extraido = ""
        nombre_archivo = ""

        # Determinar si la entrada es una ruta o un objeto en memoria
        input_stream = archivo_entrada
        if isinstance(archivo_entrada, str):
            nombre_archivo = archivo_entrada.lower()
        else:
            if hasattr(archivo_entrada, "filename") and archivo_entrada.filename:
                nombre_archivo = archivo_entrada.filename.lower()
            elif hasattr(archivo_entrada, "name") and archivo_entrada.name:
                nombre_archivo = archivo_entrada.name.lower()
            if hasattr(archivo_entrada, "file"):
                input_stream = archivo_entrada.file

        try:
            if nombre_archivo.endswith(".pdf"):
                if fitz is None:
                    self.logger.error("PyMuPDF (fitz) no esta disponible para procesar PDFs.")
                    return None
                # Metodo de respaldo: PyMuPDF (fitz), robusto para archivos danados
                texto_fallback = ""
                doc_fitz = None

                if isinstance(archivo_entrada, str):
                    doc_fitz = fitz.open(archivo_entrada)
                else:
                    # Si es un stream en memoria, hay que asegurarse de que este al inicio
                    if hasattr(input_stream, "seek"):
                        input_stream.seek(0)
                    # PyMuPDF no acepta SpooledTemporaryFile directamente; usar bytes.
                    if hasattr(input_stream, "read"):
                        raw_bytes = input_stream.read()
                        doc_fitz = fitz.open(stream=raw_bytes, filetype="pdf")
                    else:
                        doc_fitz = fitz.open(stream=input_stream, filetype="pdf")

                with doc_fitz:
                    for pagina in doc_fitz:
                        texto_pagina = pagina.get_text()
                        if texto_pagina:
                            texto_pagina_limpio = self._clean_encoding(texto_pagina)
                            texto_fallback += texto_pagina_limpio + "\n"

                if texto_fallback:
                    self.logger.info(
                        f"[EXITO] Texto extraido exitosamente de '{nombre_archivo}' usando PyMuPDF (fitz)."
                    )
                    return texto_fallback
                raise IOError(
                    f"No se pudo extraer texto de '{nombre_archivo}' con ninguno de los metodos."
                )

            if nombre_archivo.endswith(".docx") or nombre_archivo.endswith(".doc"):
                if docx2txt is None:
                    self.logger.error("docx2txt no esta disponible para procesar documentos Word.")
                    return None
                try:
                    texto_extraido = docx2txt.process(input_stream)
                except zipfile.BadZipFile as e:
                    self.logger.warning(
                        f"[ADVERTENCIA] El archivo '{nombre_archivo}' parece ser un .doc antiguo o corrupto. {e}"
                    )
                    return None
            else:
                self.logger.error(f"Formato de archivo no soportado: {nombre_archivo}")
                return None

            self.logger.info(
                f"[EXITO] Texto extraido exitosamente de '{nombre_archivo}'"
            )
            return texto_extraido

        except Exception as e:
            self.logger.error(f"[ERROR] Error final al procesar '{nombre_archivo}': {e}")
            return None

    def _clean_encoding(self, text: str) -> str:
        """
        Limpia el texto de caracteres no válidos en UTF-8 y normaliza la codificación.

        Argumentos:
            text: Texto que puede contener caracteres problemáticos

        Retorna:
            str: Texto limpio y válido en UTF-8
        """
        if not text:
            return ""

        try:
            # Intentar codificar/decodificar para detectar problemas
            text.encode("utf-8")
            return text
        except UnicodeEncodeError:
            # Si hay problemas de codificación, limpiar caracteres problemáticos
            try:
                # Convertir a bytes y luego decodificar con manejo de errores
                if isinstance(text, str):
                    # Reemplazar caracteres problemáticos
                    clean_text = text.encode("utf-8", errors="replace").decode("utf-8")
                    return clean_text
                else:
                    # Si es bytes, decodificar con manejo de errores
                    return text.decode("utf-8", errors="replace")
            except Exception as e:
                self.logger.warning(f"Error limpiando codificación: {e}")
                # Como último recurso, filtrar solo caracteres ASCII imprimibles
                return "".join(char for char in str(text) if ord(char) < 128 and char.isprintable() or char.isspace())

    def clean_text_to_file(self, texto_sucio: str) -> str:
        """
        Realiza una limpieza avanzada del texto extraído. Esta versión combina dos estrategias:
        1. Reglas fijas (regex) para eliminar patrones comunes de encabezados/pies de página.
        2. Detección por frecuencia para eliminar líneas repetitivas específicas del documento.

        Argumentos:
            texto_sucio: El texto plano extraído directamente del archivo.

        Retorna:
            str: El texto limpio, sin ruido ni encabezados/pies de página.
        """
        # Limpieza básica inicial
        texto_procesado = re.sub(r"\n\s*\n", "\n", texto_sucio)
        texto_procesado = re.sub(r"^\s+", "", texto_procesado, flags=re.MULTILINE)

        lineas = texto_procesado.splitlines()

        # Limpieza por reglas fijas (regex)
        lineas_filtradas_1 = []
        patrones_fijos = [
            # Eliminar patrones de paginación
            re.compile(r"^(Página|Page)\s+\d+(\s+de\s+\d+)?\s*$", re.IGNORECASE),
            # Eliminar líneas que solo contienen un número
            re.compile(r"^\s*\d+\s*$"),
            # Eliminar líneas con formato de fecha y nombre de documento repetitivo
            re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}.*DECRETO|LEY", re.IGNORECASE),
        ]

        for linea in lineas:
            es_ruido_fijo = False
            for patron in patrones_fijos:
                if patron.match(linea.strip()):
                    es_ruido_fijo = True
                    break
            if not es_ruido_fijo:
                lineas_filtradas_1.append(linea)

        # Detección de encabezados/pies de página por frecuencia
        lineas_candidatas = [linea.strip() for linea in lineas_filtradas_1 if 0 < len(linea.strip()) < 100]

        frecuencias = Counter(lineas_candidatas)

        # Si una línea aparece más de 2 veces, es probablemente un encabezado/pie de página
        encabezados_a_eliminar = {linea for linea, count in frecuencias.items() if count > 2 and not re.match(r"^(ART[ÍI]CULO|PAR[ÁA]GRAFO|T[ÍI]TULO|CAP[ÍI]TULO|PARTE)", linea, re.IGNORECASE)}

        if encabezados_a_eliminar:
            self.logger.info(f"[BUSCAR] Se detectaron y eliminarán {len(encabezados_a_eliminar)} líneas de encabezado/pie de página repetitivas.")

        # Filtrado final del texto
        lineas_finales = [linea for linea in lineas_filtradas_1 if linea.strip() not in encabezados_a_eliminar]

        texto_final = "\n".join(lineas_finales)
        self.logger.info("[EXITO] Limpieza avanzada (por reglas y frecuencia) completada.")
        return texto_final

    def normalize_to_markdown(self, texto_limpio: str) -> str:
        """
        Convierte un texto legal plano en un documento Markdown estructurado.
        Esta versión mejorada incluye un filtro para eliminar el "ruido" y una
        lógica para separar los títulos del contenido que aparece en la misma línea.

        Argumentos:
            texto_limpio: El contenido del documento como una sola cadena de texto.

        Retorna:
            str: El contenido formateado en Markdown, limpio y estructurado.
        """
        # Patrones para identificar y eliminar líneas de "ruido" irrelevantes
        NOISE_PATTERNS = [
            r"^Ir al portal SUIN-Juriscol",
            r"^Ayúdanos a mejorar",
            r"^Guardar en PDF o imprimir la norma",
            r"^Responder Encuesta",
            r"^DIARIO OFICIAL\. AÑO.*",
            r"^\s*R E S U M E N\s+D E\s+.*",
            r"^\s*E S TA D O\s+D E\s+V I G E N C I A.*",
            r"^Curso SUIN-Juriscol.*",
            r"^Inscripciones abiertas.*",
            r"^\d{1,2}/\d{1,2}/\d{2,4},\s+\d{1,2}:\d{1,2}",
            r"^https?://www.suin-juriscol.gov.co.*",
            r"^Los datos publicados en SUIN-Juriscol.*",
            r".*\[(Mostrar|Ocultar)\]",
            r"^\s*J U R I S P R U D E N C I A\s*$",
            r"^\s*T E X T O\s+C O R R E S P O N D I E N T E\s+A\s*$",
            r"^El Presidente de la República de Colombia,",
            r"^en ejercicio de las facultades.*",
            r"^DECRETA:\s*$",
            r"^Publíquese y cúmplase.",
            r"^Dado en Bogotá, D. E.,.*",
            r"^(MISAEL PASTRANA BORRERO|El Ministro de Justicia|El Jefe del Departamento).*",
        ]

        # Jerarquía de reglas
        HIERARCHY_RULES = {
            r"^(DECRETO\s+\d+\s+DE\s+\d{4})": "#",
            r"^(LIBRO\s+[\dIVXLC]+)": "#",
            r"^(PARTE\s+[\dIVXLC]+)": "##",
            r"^(T[ÍI]TULO\s+[\dIVXLC]+)": "###",
            r"^(CAP[ÍI]TULO\s+.*)": "####",
            r"^(ART[ÍI]CULO\s+[\d\.°\-\s]+)": "#####",
            r"^(PAR[ÁA]GRAFO(?:[\s\d°]+|\s+TRANSITORIO)?\.?)": "######",
        }

        lineas_originales = texto_limpio.splitlines()
        lineas_markdown = []

        for linea in lineas_originales:
            linea_limpia = linea.strip()
            if not linea_limpia:
                continue

            # Aplicar el filtro de ruido
            es_ruido = False
            for patron_ruido in NOISE_PATTERNS:
                if re.match(patron_ruido, linea_limpia, re.IGNORECASE):
                    es_ruido = True
                    break

            if es_ruido:
                continue

            # Aplicar las reglas de jerarquía
            es_titulo = False
            for patron, prefijo_md in HIERARCHY_RULES.items():
                match = re.match(patron, linea_limpia, re.IGNORECASE)
                if match:
                    titulo_texto = match.group(1).strip()
                    contenido_texto = linea_limpia[len(match.group(0)) :].strip()

                    lineas_markdown.append(f"{prefijo_md} {titulo_texto}")
                    lineas_markdown.append("")

                    if contenido_texto:
                        lineas_markdown.append(contenido_texto)

                    es_titulo = True
                    break

            if not es_titulo:
                lineas_markdown.append(linea)

        return "\n".join(lineas_markdown)

    def get_markdown_level(self, line: str) -> int:
        """
        Calcula el nivel jerárquico de una línea de encabezado Markdown.
        Devuelve 1 para '#', 2 para '##', etc.
        Devuelve un número muy alto (999) si la línea no es un encabezado.
        """
        clean_line = line.strip()
        if clean_line.startswith("#"):
            level = 0
            for char in clean_line:
                if char == "#":
                    level += 1
                else:
                    break
            return level
        return 999

    def clean_markdown_title(self, title: str) -> str:
        """
        Elimina los caracteres de formato Markdown (#) y los espacios extra de una línea
        para obtener el texto puro del título.
        """
        return re.sub(r"^#+\s*", "", title).strip()

    def segment_by_title(self, markdown_text: str, searched_title: str) -> str:
        """
        Segmenta un documento Markdown, extrayendo el contenido desde un título
        específico hasta el siguiente título del mismo nivel jerárquico o superior.

        Argumentos:
            markdown_text: El contenido completo del documento en formato Markdown.
            searched_title: El texto del título que el usuario desea encontrar.

        Retorna:
            str: El fragmento del documento Markdown correspondiente a la sección solicitada.
        """
        lines = markdown_text.splitlines()
        clean_searched_title = searched_title.strip()

        start_index = -1
        target_level = -1

        # Encontrar el título de inicio y su nivel jerárquico
        for i, line in enumerate(lines):
            current_clean_title = self.clean_markdown_title(line)
            if clean_searched_title.lower() in current_clean_title.lower():
                start_index = i
                target_level = self.get_markdown_level(line)
                break

        if start_index == -1:
            return ""

        # Encontrar el final de la sección
        end_index = len(lines)

        for i in range(start_index + 1, len(lines)):
            current_level = self.get_markdown_level(lines[i])
            if current_level <= target_level:
                end_index = i
                break

        # Extraer y devolver el fragmento
        selected_fragment = lines[start_index:end_index]
        return "\n".join(selected_fragment)

    def segment_by_path(self, markdown_text: str, path: List[str]) -> str:
        """
        Realiza una segmentación jerárquica encadenada a través de una ruta de títulos.

        Argumentos:
            markdown_text: El contenido completo del documento en formato Markdown.
            path: Una lista de títulos que representa la ruta jerárquica.

        Retorna:
            str: El fragmento final del documento Markdown después de aplicar toda la ruta.
        """
        self.logger.info(f"Buscando por la ruta jerárquica: {' -> '.join(path)}")
        current_text = markdown_text

        for i, title in enumerate(path):
            self.logger.info(f"  -> Nivel {i + 1}: Buscando '{title}'...")
            fragment = self.segment_by_title(current_text, title)

            if not fragment:
                self.logger.error(f"  [ERROR] No se pudo encontrar el título '{title}' en el nivel {i + 1} de la ruta.")
                return ""

            current_text = fragment

        self.logger.info("[EXITO] Ruta jerárquica encontrada exitosamente.")
        return current_text

    def process_document(self, file_path: str, section_path: List[str]) -> Optional[str]:
        """
        Procesa un archivo de normativa, lo limpia, lo normaliza a markdown y lo segmenta según una ruta de títulos.

        Argumentos:
            file_path: La ruta del archivo de normativa.
            section_path: Una lista de títulos que representa la ruta jerárquica.

        Retorna:
            str: El fragmento final del documento Markdown después de aplicar toda la ruta.
        """
        try:
            text = self.extract_text_from_file(file_path)
            if text is None:
                return None

            clean_text = self.clean_text_to_file(text)
            md = self.normalize_to_markdown(clean_text)
            if not section_path:
                return md

            section = self.segment_by_path(md, section_path)
            return section

        except Exception as e:
            self.logger.error(f"[ERROR] No se pudo procesar el archivo: {file_path} error {e}")
            return None

