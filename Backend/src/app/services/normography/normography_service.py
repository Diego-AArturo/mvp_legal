# -*- coding: utf-8 -*-
"""
Servicio de normografía para Neo4j.

Este servicio maneja la actualización y carga masiva de documentos normativos
en la base de datos Neo4j, integrándose con los clientes existentes del sistema.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from langchain.text_splitter import MarkdownHeaderTextSplitter
except ImportError as e:
    raise ImportError(f"Error importando LangChain: {e}")

from neo4j import AsyncDriver
from neo4j.exceptions import AuthError, ServiceUnavailable

from app.config.settings import Settings
from app.extensions import get_logger
from app.schemas.normography_schemas import (
    BulkLoadResponse,
    NormographyUpdateResponse,
)
from app.services.documents.document_processor_service import DocumentProcessorService
from app.services.embeddings.embedding_service import EmbeddingService

logger = get_logger(__name__)


class NormographyService:
    """
    Servicio para actualización y carga de normografía en Neo4j.

    Maneja la creación de estructuras de grafo, actualización de secciones específicas
    y carga masiva de documentos normativos.
    """

    def __init__(
        self,
        driver: AsyncDriver,
        settings: Settings,
        embedding_service: EmbeddingService,
    ):
        """
        Inicializa el servicio de normografía.

        Argumentos:
            driver: Driver de Neo4j
            settings: Configuración del sistema
            embedding_service: Servicio de embeddings
        """
        self.driver = driver
        self.settings = settings
        self.embedding_service = embedding_service
        self.database = settings.databases.neo4j.neo4j_database
        self.document_processor = DocumentProcessorService()

        # Configuración de headers para splitting
        self.headers_to_split_on = [
            ("#", "Categoria"),
            ("##", "SubCategoria"),
            ("###", "ElementoNorma"),
            ("####", "Articulo"),
        ]

        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on)

        logger.info("Servicio de normografía inicializado correctamente")

    def _update_structure_graph(self, markdown_text: str, parent_node_id: Optional[str] = None) -> Tuple[List[Dict], List[Tuple]]:
        """
        Convierte un fragmento de texto Markdown en una lista de nodos y relaciones.

        MEJORADO: Maneja formato de texto plano con títulos en líneas separadas
        y contenido multi-línea.

        Argumentos:
            markdown_text: Texto Markdown a procesar
            parent_node_id: ID del nodo padre (opcional)

        Retorna:
            Tupla con (lista de nodos, lista de relaciones)
        """
        new_nodes = []
        new_relations = []

        local_context = {}
        nodes_by_id = {}

        lines = markdown_text.splitlines()
        first_node_created = False

        # Variables para manejar contenido multi-línea
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            #  NUEVOS PATRONES: Manejar formato markdown y texto plano

            # Detectar encabezados markdown (### TÍTULO I)
            markdown_header_match = re.match(r"^(#{1,6})\s+(.+)", line)
            if markdown_header_match:
                header_level = len(markdown_header_match.group(1))
                header_text = markdown_header_match.group(2).strip()
                node_type, node_name = self._classify_header_by_content_and_level(header_text, header_level)

                if node_type:
                    # Buscar contenido en las siguientes líneas
                    content_lines = []
                    j = i + 1
                    while j < len(lines) and not self._is_header_line(lines[j]):
                        if lines[j].strip():
                            content_lines.append(lines[j].strip())
                        j += 1

                    node_text = " ".join(content_lines)
                    node = self._create_node(node_type, node_name, node_text, local_context, parent_node_id)
                    if node:
                        new_nodes.append(node)
                        nodes_by_id[node["id"]] = node
                        self._update_context(node_type, node["id"], local_context)

                        # Crear relación
                        parent_id = self._get_parent_for_node_type(node_type, local_context, parent_node_id)
                        if not first_node_created and parent_node_id:
                            new_relations.append((node["id"], "PERTENECE_A", parent_node_id))
                            first_node_created = True
                        elif parent_id:
                            new_relations.append((node["id"], "PERTENECE_A", parent_id))

                    i = j - 1  # Saltar las líneas procesadas

            # Detectar patrones de texto plano
            else:
                # Patrones para texto plano (sin #)
                match_norma = re.match(
                    r"^(DECRETO LEY|LEY|DECRETO|RESOLUCIÓN|CIRCULAR|SENTENCIA|AUTO)\s+(.+)",
                    line,
                    re.IGNORECASE,
                )
                match_titulo = re.match(r"^T[ÍI]TULO\s+([\ \dIVXLC]+)", line, re.IGNORECASE)
                match_capitulo = re.match(r"^CAP[ÍI]TULO\s+([\ \dIVXLC]+)", line, re.IGNORECASE)
                match_libro = re.match(r"^LIBRO\s+([\ \dIVXLC]+)", line, re.IGNORECASE)
                match_parte = re.match(r"^PARTE\s+([\ \dIVXLC]+)", line, re.IGNORECASE)
                match_articulo = re.match(r"^Art[íi]culo\s+(\d+[°º]?)\.(.*)", line, re.IGNORECASE)
                match_paragrafo = re.match(r"^Par[áa]grafo\.?\s*(.*)", line, re.IGNORECASE)

                node = None
                node_type = None

                if match_norma:
                    node_type = match_norma.group(1).replace(" ", "")
                    node_name = line
                    # Buscar contenido en líneas siguientes hasta el próximo título
                    content_lines = []
                    j = i + 1
                    while j < len(lines) and not self._is_header_line(lines[j]):
                        if lines[j].strip():
                            content_lines.append(lines[j].strip())
                        j += 1
                    node_text = " ".join(content_lines)

                elif match_titulo:
                    node_type = "Titulo"
                    node_name = line
                    # Buscar descripción en las siguientes líneas no vacías
                    description_lines = []
                    j = i + 1
                    while j < len(lines) and not self._is_header_line(lines[j]):
                        if lines[j].strip():
                            description_lines.append(lines[j].strip())
                        j += 1
                    node_text = " ".join(description_lines)

                elif match_capitulo:
                    node_type = "Capitulo"
                    node_name = line
                    description_lines = []
                    j = i + 1
                    while j < len(lines) and not self._is_header_line(lines[j]):
                        if lines[j].strip():
                            description_lines.append(lines[j].strip())
                        j += 1
                    node_text = " ".join(description_lines)

                elif match_libro:
                    node_type = "Libro"
                    node_name = line
                    description_lines = []
                    j = i + 1
                    while j < len(lines) and not self._is_header_line(lines[j]):
                        if lines[j].strip():
                            description_lines.append(lines[j].strip())
                        j += 1
                    node_text = " ".join(description_lines)

                elif match_parte:
                    node_type = "Parte"
                    node_name = line
                    description_lines = []
                    j = i + 1
                    while j < len(lines) and not self._is_header_line(lines[j]):
                        if lines[j].strip():
                            description_lines.append(lines[j].strip())
                        j += 1
                    node_text = " ".join(description_lines)

                elif match_articulo:
                    node_type = "Articulo"
                    node_name = f"Artículo {match_articulo.group(1)}"
                    # El contenido del artículo puede estar en la misma línea después del punto
                    node_text = match_articulo.group(2).strip() if match_articulo.group(2) else ""
                    # Buscar contenido adicional en líneas siguientes
                    additional_lines = []
                    j = i + 1
                    while j < len(lines) and not self._is_header_line(lines[j]):
                        if lines[j].strip():
                            additional_lines.append(lines[j].strip())
                        j += 1
                    if additional_lines:
                        if node_text:
                            node_text += " " + " ".join(additional_lines)
                        else:
                            node_text = " ".join(additional_lines)

                elif match_paragrafo:
                    node_type = "Paragrafo"
                    node_name = "Parágrafo"
                    node_text = match_paragrafo.group(1).strip() if match_paragrafo.group(1) else ""
                    # Buscar contenido adicional
                    additional_lines = []
                    j = i + 1
                    while j < len(lines) and not self._is_header_line(lines[j]):
                        if lines[j].strip():
                            additional_lines.append(lines[j].strip())
                        j += 1
                    if additional_lines:
                        if node_text:
                            node_text += " " + " ".join(additional_lines)
                        else:
                            node_text = " ".join(additional_lines)

                if node_type:
                    node = self._create_node(node_type, node_name, node_text, local_context, parent_node_id)
                    if node:
                        new_nodes.append(node)
                        nodes_by_id[node["id"]] = node
                        self._update_context(node_type, node["id"], local_context)

                        # Crear relación
                        parent_id = self._get_parent_for_node_type(node_type, local_context, parent_node_id)
                        if not first_node_created and parent_node_id:
                            new_relations.append((node["id"], "PERTENECE_A", parent_node_id))
                            first_node_created = True
                        elif parent_id:
                            new_relations.append((node["id"], "PERTENECE_A", parent_id))

                    # Saltar las líneas que ya procesamos
                    if "j" in locals():
                        i = j - 1

            i += 1

        return new_nodes, new_relations

    def _is_header_line(self, line: str) -> bool:
        """
        Determina si una línea es un encabezado (título, capítulo, artículo, etc.)
        """
        line = line.strip()
        if not line:
            return False

        # Patrones de encabezados
        header_patterns = [
            r"^#{1,6}\s+",  # Markdown headers
            r"^(DECRETO LEY|LEY|DECRETO|RESOLUCIÓN|CIRCULAR|SENTENCIA|AUTO)\s+",
            r"^T[ÍI]TULO\s+[\dIVXLC]+",
            r"^CAP[ÍI]TULO\s+[\dIVXLC]+",
            r"^LIBRO\s+[\dIVXLC]+",
            r"^PARTE\s+[\dIVXLC]+",
            r"^Art[íi]culo\s+\d+[°º]?\.",
            r"^Par[áa]grafo\.?",
        ]

        for pattern in header_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True

        return False

    def _classify_header_by_content_and_level(self, header_text: str, header_level: int = None) -> Tuple[Optional[str], str]:
        """
        Clasifica un encabezado basado en su contenido y opcionalmente su nivel.
        """
        header_upper = header_text.upper()

        if re.match(
            r"^(DECRETO LEY|LEY|DECRETO|RESOLUCIÓN|CIRCULAR|SENTENCIA|AUTO)\s+",
            header_upper,
        ):
            match = re.match(
                r"^(DECRETO LEY|LEY|DECRETO|RESOLUCIÓN|CIRCULAR|SENTENCIA|AUTO)\s+(.+)",
                header_upper,
            )
            if match:
                node_type = match.group(1).replace(" ", "")
                return node_type, header_text

        elif re.match(r"^T[ÍI]TULO\s+", header_upper):
            return "Titulo", header_text

        elif re.match(r"^CAP[ÍI]TULO\s+", header_upper):
            return "Capitulo", header_text

        elif re.match(r"^LIBRO\s+", header_upper):
            return "Libro", header_text

        elif re.match(r"^PARTE\s+", header_upper):
            return "Parte", header_text

        elif re.match(r"^ART[ÍI]CULO\s+", header_upper):
            return "Articulo", header_text

        elif re.match(r"^PAR[ÁA]GRAFO", header_upper):
            return "Paragrafo", header_text

        # Fallback basado en nivel si está disponible
        elif header_level:
            if header_level <= 2:
                return "Titulo", header_text
            elif header_level == 3:
                return "Capitulo", header_text
            elif header_level >= 4:
                return "Articulo", header_text

        return None, header_text

    def _create_node(
        self,
        node_type: str,
        node_name: str,
        node_text: str,
        local_context: dict,
        parent_node_id: Optional[str],
    ) -> Optional[Dict]:
        """
        Crea un nodo con el formato correcto.
        """
        if node_type in ["Titulo", "Capitulo", "Libro", "Parte", "Seccion"]:
            parent_context_id = local_context.get("norma_id", parent_node_id or "doc_root")
            node_id = f"{node_type.lower()}_{parent_context_id}_{node_name.replace(' ', '_').replace('(', '').replace(')', '').lower()[:30]}"
        elif node_type == "Articulo":
            parent_context_id = local_context.get(
                "seccion_id",
                local_context.get("norma_id", parent_node_id or "doc_root"),
            )
            node_id = f"articulo_{parent_context_id}_{node_name.replace(' ', '_').replace('.', '').replace('°', '').lower()[:20]}"
        elif node_type == "Paragrafo":
            parent_context_id = local_context.get("articulo_id", "articulo_desconocido")
            node_id = f"paragrafo_{parent_context_id}_{node_name.replace(' ', '_').replace('.', '').lower()[:20]}"
        elif node_type in ["Ley", "Decreto", "DecretoLey"]:
            node_id = f"{node_type.lower()}_{node_name.replace(' ', '_').replace('(', '').replace(')', '').lower()[:50]}"
        else:
            return None

        return {
            "id": node_id,
            "tipo": node_type,
            "nombre": node_name,
            "texto": node_text,
        }

    def _update_context(self, node_type: str, node_id: str, local_context: dict):
        """
        Actualiza el contexto local con el nuevo nodo.
        """
        if node_type in ["Ley", "Decreto", "DecretoLey"]:
            local_context["norma_id"] = node_id
            local_context.pop("seccion_id", None)
            local_context.pop("articulo_id", None)
        elif node_type in ["Titulo", "Capitulo", "Libro", "Parte", "Seccion"]:
            local_context["seccion_id"] = node_id
            local_context.pop("articulo_id", None)
        elif node_type == "Articulo":
            local_context["articulo_id"] = node_id

    def _get_parent_for_node_type(self, node_type: str, local_context: dict, parent_node_id: Optional[str]) -> Optional[str]:
        """
        Determina el nodo padre apropiado para un tipo de nodo.
        """
        if node_type in ["Titulo", "Capitulo", "Libro", "Parte", "Seccion"]:
            return local_context.get("norma_id", parent_node_id)
        elif node_type == "Articulo":
            return local_context.get("seccion_id") or local_context.get("norma_id", parent_node_id)
        elif node_type == "Paragrafo":
            return local_context.get("articulo_id")
        else:
            return parent_node_id

    async def _delete_subtree(self, tx, target_node_id: str):
        """Elimina un nodo y todos sus descendientes (sub-árbol)."""
        logger.info(f"Eliminando sub-árbol a partir del nodo: {target_node_id}...")
        query = """
        MATCH (target {id: $target_node_id})
        OPTIONAL MATCH (target)<-[:PERTENECE_A*]-(descendant)
        DETACH DELETE target, descendant
        """
        await tx.run(query, target_node_id=target_node_id)
        logger.info("[EXITO] Sub-árbol eliminado.")

    async def _update_graph_section(
        self,
        tx,
        category: str,
        law_name: str,
        path_to_section: List[str],
        new_section_markdown: str,
    ):
        """
        Orquesta el proceso completo de actualización en una única transacción.
        Si la ruta de nodos padre no existe, la crea dinámicamente.
        """
        logger.info("\n--- Iniciando Transacción de Actualización ---")

        # Asegurar que la categoría y la ley existan
        logger.info(f" Creando/verificando categoría: {category}")
        category_result = await tx.run(
            """
            MERGE (c {nombre: $category})
            ON CREATE SET c.id = $id
            WITH c
            CALL apoc.create.addLabels(c, ['Categoria', 'Normografia']) YIELD node
            RETURN count(node) as created, c.id as category_id
        """,
            category=category,
            id=f"cat_{category.replace(' ', '_').lower()[:50]}",
        )

        category_record = await category_result.single()
        logger.info(f" Categoría procesada: {category_record}")

        logger.info(f" Creando/verificando ley: {law_name}")
        law_result = await tx.run(
            """
            MATCH (c:Categoria {nombre: $category})
            MERGE (n {nombre: $law_name})
            ON CREATE SET n.id = $id
            WITH n, c
            CALL apoc.create.addLabels(n, ['Ley', 'Normografia']) YIELD node
            MERGE (n)-[:PERTENECE_A]->(c)
            RETURN n.id as id, labels(n) as labels
        """,
            category=category,
            law_name=law_name,
            id=f"{law_name.replace(' ', '_').lower()[:50]}",
        )

        law_record = await law_result.single()
        logger.info(f" Ley procesada: {law_record}")
        current_parent_id = law_record["id"]

        # Recorrer la ruta intermedia
        path_parent_nodes = path_to_section[:-1]
        for title in path_parent_nodes:
            find_query = """
            MATCH (parent {id: $parent_id})<-[:PERTENECE_A]-(child)
            WHERE child.nombre CONTAINS $title
            RETURN child.id as id
            """
            child_result = await tx.run(find_query, parent_id=current_parent_id, title=title)
            child_record = await child_result.single()

            if child_record:
                current_parent_id = child_record["id"]
            else:
                logger.warning(f"[ADVERTENCIA] Nodo intermedio '{title}' no encontrado. Creándolo...")
                node_type = "Seccion"
                new_node_id = f"seccion_{current_parent_id}_{title.replace(' ', '_').lower()[:30]}"
                create_query = f"""
                MATCH (parent {{id: $parent_id}})
                CREATE (child:{node_type} {{id: $id, nombre: $title, texto: ''}})
                CREATE (child)-[:PERTENECE_A]->(parent)
                RETURN child.id as id
                """
                new_child_result = await tx.run(
                    create_query,
                    parent_id=current_parent_id,
                    id=new_node_id,
                    title=title,
                )
                new_child_record = await new_child_result.single()
                current_parent_id = new_child_record["id"]

        parent_id = current_parent_id

        # Encontrar y eliminar el nodo objetivo si existe
        target_id_to_delete = None
        if path_to_section:
            target_title = path_to_section[-1]
            find_target_query = """
            MATCH (parent {id: $parent_id})<-[:PERTENECE_A]-(target)
            WHERE target.nombre CONTAINS $title
            RETURN target.id as id
            """
            target_result = await tx.run(find_target_query, parent_id=parent_id, title=target_title)
            target_record = await target_result.single()
            if target_record:
                target_id_to_delete = target_record["id"]

        if target_id_to_delete:
            await self._delete_subtree(tx, target_id_to_delete)

        # Procesar y cargar la nueva estructura
        logger.info("Procesando nuevo fragmento Markdown...")
        new_nodes, new_relations = self._update_structure_graph(new_section_markdown, parent_id)

        nodes_created = 0
        relations_created = 0

        if new_nodes:
            logger.info(f"Cargando {len(new_nodes)} nuevos nodos...")
            node_query = """
            UNWIND $nodes as node_data
            MERGE (n {id: node_data.id})
            SET n += node_data
            WITH n, node_data
            // Añade la etiqueta específica Y la etiqueta genérica al mismo tiempo
            CALL apoc.create.addLabels(n, [node_data.tipo, 'Normografia']) YIELD node
            RETURN count(node)
            """
            node_result = await tx.run(node_query, nodes=new_nodes)
            node_record = await node_result.single()
            nodes_created = node_record[0] if node_record else 0

        if new_relations:
            logger.info(f"Cargando {len(new_relations)} nuevas relaciones...")
            rel_query = """
            UNWIND $rels as rel_data
            MATCH (origen {id: rel_data[0]})
            MATCH (destino {id: rel_data[2]})
            CALL apoc.merge.relationship(origen, rel_data[1], {}, {}, destino) YIELD rel
            RETURN count(rel)
            """
            rel_result = await tx.run(rel_query, rels=new_relations)
            rel_record = await rel_result.single()
            relations_created = rel_record[0] if rel_record else 0

        logger.info("--- Transacción de Actualización Completada ---")
        return nodes_created, relations_created

    async def update_normography_section(self, category: str, law_name: str, path: List[str], new_markdown_content: str) -> NormographyUpdateResponse:
        """
        Actualiza una sección específica de normografía en Neo4j.
        MEJORADO: Detecta y procesa documentos completos cuando path está vacío.

        Argumentos:
            category: Categoría de la normografía
            law_name: Nombre de la ley
            path: Ruta de la sección a actualizar ([] = documento completo)
            new_markdown_content: Contenido nuevo en formato Markdown

        Retorna:
            NormographyUpdateResponse con el resultado de la operación
        """
        try:
            #  NUEVA LÓGICA: Detectar si es documento completo o sección específica
            is_full_document = not path or len(path) == 0

            if is_full_document:
                logger.info(" Procesando documento completo (sin sección específica)")
                return await self._process_full_document(
                    category=category,
                    law_name=law_name,
                    markdown_content=new_markdown_content,
                )
            else:
                logger.info(f" Procesando sección específica: {' -> '.join(path)}")
                async with self.driver.session(database=self.database) as session:
                    nodes_created, relations_created = await session.execute_write(
                        self._update_graph_section,
                        category=category,
                        law_name=law_name,
                        path_to_section=path,
                        new_section_markdown=new_markdown_content,
                    )

                    return NormographyUpdateResponse(
                        success=True,
                        nodes_created=nodes_created,
                        relations_created=relations_created,
                    )

        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"[ERROR] Error de conexión a Neo4j: {e}")
            return NormographyUpdateResponse(success=False, error_message=f"Error de conexión a Neo4j: {str(e)}")
        except Exception as e:
            logger.error(f"[ERROR] Error inesperado: {e}")
            return NormographyUpdateResponse(success=False, error_message=f"Error inesperado: {str(e)}")

    async def _process_full_document(self, category: str, law_name: str, markdown_content: str) -> NormographyUpdateResponse:
        """
        Procesa un documento completo usando una estrategia híbrida:
        1. Pre-procesa con LangChain MarkdownHeaderTextSplitter
        2. Crea estructura jerárquica con nuestro sistema personalizado
        3. Genera embeddings automáticamente
        """
        try:
            logger.info(" Iniciando procesamiento de documento completo")

            # Paso 1: Limpiar y normalizar el markdown
            normalized_content = self.normalize_markdown(markdown_content)

            # Paso 2: Pre-procesar con LangChain para identificar secciones principales
            chunks = self._split_document_into_sections(normalized_content)
            logger.info(f" Documento dividido en {len(chunks)} secciones principales")

            # Paso 3: Crear estructura de grafo completa
            all_nodes, all_relations = self._create_full_document_structure(chunks, law_name, category)

            # Paso 4: Generar embeddings para todos los nodos
            await self._generate_embeddings_for_nodes(all_nodes)

            # Paso 5: Cargar todo en Neo4j en una transacción
            async with self.driver.session(database=self.database) as session:
                nodes_created, relations_created = await session.execute_write(
                    self._bulk_create_document_structure,
                    category=category,
                    law_name=law_name,
                    nodes=all_nodes,
                    relations=all_relations,
                )

            logger.info(f" Documento completo procesado: {nodes_created} nodos, {relations_created} relaciones")

            return NormographyUpdateResponse(
                success=True,
                nodes_created=nodes_created,
                relations_created=relations_created,
            )

        except Exception as e:
            logger.error(f" Error procesando documento completo: {e}", exc_info=True)
            return NormographyUpdateResponse(
                success=False,
                error_message=f"Error procesando documento completo: {str(e)}",
            )

    def normalize_markdown(self, texto: str) -> str:
        """
        Normaliza el documento aplicando reglas de reemplazo en un orden estricto
        para garantizar la correcta jerarquía de encabezados Markdown.
        """
        reglas_de_normalizacion = [
            # Convertir categorías principales a H1 (#)
            (r"^####\s+\*\*(.*?)\*\*", r"# \1"),
            # Convertir secciones de análisis a H2 (##)
            (
                r"^##\s(Análisis Exhaustivo.*?|Flujo Práctico.*?|Situaciones Especiales.*?)",
                r"## \1",
            ),
            # Convertir sub-temas numerados a H3 (###)
            (r"^####\s+\d+\.\s(.*?)$", r"### \1"),
            # Convertir el resto de los subtítulos H3 originales a H4 (####)
            (r"^###\s(.*?)$", r"#### \1"),
        ]

        texto_procesado = texto
        for patron, reemplazo in reglas_de_normalizacion:
            texto_procesado = re.sub(patron, reemplazo, texto_procesado, flags=re.MULTILINE)

        return texto_procesado

    def clean_documents(self, documentos_sucios: List[Any]) -> List[Dict[str, Any]]:
        """Limpia los documentos obtenidos del splitter."""
        documentos_limpios = []
        for doc in documentos_sucios:
            # Limpieza del contenido
            texto_limpio = re.sub(r"^\s*(\* ####|####|\*)\s*", "", doc.page_content, flags=re.MULTILINE).strip()

            # Limpieza de los metadatos
            metadata_limpia = {}
            for clave, valor in doc.metadata.items():
                valor_limpio = re.sub(r"^\d+\\\.?\s*", "", valor).strip()
                metadata_limpia[clave] = valor_limpio

            if texto_limpio or metadata_limpia:
                documentos_limpios.append({"metadata": metadata_limpia, "texto": texto_limpio})

        return documentos_limpios

    def _normalize_tipo_norma(self, tipo_raw: str) -> str:
        """
        Normaliza el tipo de norma a formato estándar.
        
        Argumentos:
            tipo_raw: Tipo de norma en formato crudo (puede estar en mayúsculas, minúsculas, etc.)
        
        Retorna:
            Tipo normalizado (Ley, DecretoLey, Decreto, etc.)
        """
        tipo_clean = tipo_raw.replace(" ", "").strip()
        tipo_upper = tipo_clean.upper()
        
        # Casos especiales
        if tipo_upper == "DECRETOLEY" or tipo_upper == "DECRETO_LEY":
            return "DecretoLey"
        elif tipo_upper == "DECRETO":
            return "Decreto"
        elif tipo_upper in ["LEY", "LEYES"]:
            return "Ley"
        elif tipo_upper in ["RESOLUCIÓN", "RESOLUCION"]:
            return "Resolución"
        elif tipo_upper in ["CIRCULAR", "CIRCULARES"]:
            return "Circular"
        elif tipo_upper in ["SENTENCIA", "SENTENCIAS"]:
            return "Sentencia"
        elif tipo_upper in ["AUTO", "AUTOS"]:
            return "Auto"
        elif tipo_upper in ["CONSTITUCIÓN", "CONSTITUCION"]:
            return "Constitución"
        elif tipo_upper in ["TRATADO", "TRATADOS", "CONVENIO", "CONVENIOS", "CONVENCIÓN", "CONVENCION"]:
            return "Tratado"
        else:
            # Formato estándar: primera letra mayúscula, resto minúsculas
            return tipo_clean.capitalize()

    def _generate_norma_id(self, tipo_norma: str, nombre_norma: str) -> str:
        """
        Genera un ID único y consistente para una norma basado en su tipo y nombre.
        Extrae el número/identificador principal de la norma para evitar duplicados.
        
        Argumentos:
            tipo_norma: Tipo normalizado de la norma
            nombre_norma: Nombre completo de la norma
        
        Retorna:
            ID único para la norma
        """
        # Extraer número/identificador principal de la norma
        # Patrones: "Ley 130 de 1994", "Decreto Ley 1260 de 1970", "Sentencia SU-214 de 2016"
        match_numero = re.search(
            r"(?:Ley|Decreto\s+Ley|Decreto|Resolución|Circular|Sentencia|Auto|Constitución|Tratado|Convenio|Convención)\s+([A-Z0-9\-]+)",
            nombre_norma,
            re.IGNORECASE
        )
        
        if match_numero:
            numero_principal = match_numero.group(1).upper().replace("-", "_")
            tipo_lower = tipo_norma.lower()
            return f"{tipo_lower}_{numero_principal}"
        else:
            # Fallback: usar nombre normalizado
            nombre_normalizado = nombre_norma.replace(" ", "_").replace("(", "").replace(")", "").lower()[:50]
            return f"{tipo_norma.lower()}_{nombre_normalizado}"

    def _consolidate_node(self, nodo_existente: Dict, nuevo_texto: str, nuevo_nombre: str = None) -> Dict:
        """
        Consolida información de un nodo existente con nueva información.
        
        Argumentos:
            nodo_existente: Nodo existente en el diccionario
            nuevo_texto: Nuevo texto a agregar
            nuevo_nombre: Nuevo nombre (opcional, solo si es más descriptivo)
        
        Retorna:
            Nodo consolidado
        """
        # Consolidar texto: agregar nuevo texto si no está ya incluido
        texto_existente = nodo_existente.get("texto", "")
        if nuevo_texto and nuevo_texto.strip():
            if texto_existente:
                # Evitar duplicar texto idéntico
                if nuevo_texto.strip() not in texto_existente:
                    # Agregar separador y nuevo texto
                    nodo_existente["texto"] = f"{texto_existente}\n\n{nuevo_texto.strip()}"
            else:
                nodo_existente["texto"] = nuevo_texto.strip()
        
        # Actualizar nombre si el nuevo es más descriptivo (más largo o contiene más información)
        if nuevo_nombre and len(nuevo_nombre) > len(nodo_existente.get("nombre", "")):
            nodo_existente["nombre"] = nuevo_nombre
        
        return nodo_existente

    def create_graph_structure(self, lista_documentos: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Tuple]]:
        """
        Versión mejorada que UTILIZA la jerarquía markdown extraída por el splitter.
        Respeta la estructura jerárquica: Categoria -> SubCategoria -> ElementoNorma -> Articulo
        Consolida información de nodos duplicados para evitar pérdida de datos.
        """
        lista_de_nodos = []
        lista_de_relaciones = []

        contexto_actual = {}
        nodos_por_id = {}
        # Mapa adicional para detectar duplicados por nombre normalizado
        nodos_por_nombre_normalizado = {}

        for doc in lista_documentos:
            metadata = doc["metadata"]
            texto = doc["texto"]

            # Nivel 1: Categoría (#) - Ej: "Normografía Legal Colombiana"
            categoria_id = None
            if "Categoria" in metadata:
                nombre_cat = metadata["Categoria"]
                id_cat = f"cat_{nombre_cat.replace(' ', '_').lower()[:50]}"

                if id_cat not in nodos_por_id:
                    nodo_cat = {
                        "id": id_cat,
                        "tipo": "Categoria",
                        "nombre": nombre_cat,
                        "texto": nombre_cat,
                    }
                    lista_de_nodos.append(nodo_cat)
                    nodos_por_id[id_cat] = nodo_cat
                categoria_id = id_cat
                contexto_actual["padre_id"] = id_cat

            # Nivel 2: SubCategoría (##) - Ej: "Registro Civil de Nacimiento, Matrimonio y Defunción"
            subcategoria_id = None
            if "SubCategoria" in metadata:
                nombre_subcat = metadata["SubCategoria"]
                id_subcat = f"subcat_{nombre_subcat.replace(' ', '_').replace('(', '').replace(')', '').lower()[:50]}"
                
                if id_subcat not in nodos_por_id:
                    nodo_subcat = {
                        "id": id_subcat,
                        "tipo": "SubCategoria",
                        "nombre": nombre_subcat,
                        "texto": nombre_subcat,
                    }
                    lista_de_nodos.append(nodo_subcat)
                    nodos_por_id[id_subcat] = nodo_subcat
                    # Conectar a categoría
                    if categoria_id:
                        lista_de_relaciones.append((id_subcat, "PERTENECE_A", categoria_id))
                
                subcategoria_id = id_subcat
                contexto_actual["padre_id"] = id_subcat

            # Nivel 3: ElementoNorma (###) - AQUÍ ESTÁN LAS LEYES/DECRETOS/SENTENCIAS/RESOLUCIONES
            norma_id = None
            if "ElementoNorma" in metadata:
                nombre_norma = metadata["ElementoNorma"]
                
                # Detectar tipo de norma desde el nombre
                match_norma = re.match(
                    r"^(Decreto Ley|Ley|Decreto|Resolución|Circular|Sentencia|Auto|Constitución|Tratado|Convenio|Convención|DECRETO LEY|LEY|DECRETO|RESOLUCIÓN|CIRCULAR|SENTENCIA|AUTO|CONSTITUCIÓN|TRATADO|CONVENIO|CONVENCIÓN)\s+.*",
                    nombre_norma,
                    re.IGNORECASE
                )
                
                if match_norma:
                    tipo_raw = match_norma.group(1)
                    tipo_nodo = self._normalize_tipo_norma(tipo_raw)
                    
                    # Generar ID único y consistente para la norma
                    id_norma = self._generate_norma_id(tipo_nodo, nombre_norma)
                    
                    # Verificar si ya existe un nodo con el mismo ID o nombre normalizado
                    nombre_normalizado_key = f"{tipo_nodo}_{nombre_norma.lower().strip()}"
                    
                    if id_norma in nodos_por_id:
                        # Nodo duplicado: consolidar información
                        nodo_existente = nodos_por_id[id_norma]
                        nodo_existente = self._consolidate_node(nodo_existente, texto.strip(), nombre_norma)
                        logger.info(f"[CONSOLIDACION] Información consolidada para norma existente: {tipo_nodo} - {nombre_norma}")
                    elif nombre_normalizado_key in nodos_por_nombre_normalizado:
                        # Nodo con nombre similar: usar el ID existente y consolidar
                        id_existente = nodos_por_nombre_normalizado[nombre_normalizado_key]
                        nodo_existente = nodos_por_id[id_existente]
                        nodo_existente = self._consolidate_node(nodo_existente, texto.strip(), nombre_norma)
                        id_norma = id_existente
                        logger.info(f"[CONSOLIDACION] Información consolidada para norma similar: {tipo_nodo} - {nombre_norma}")
                    else:
                        # Nuevo nodo
                        nodo_norma = {
                            "id": id_norma,
                            "tipo": tipo_nodo,
                            "nombre": nombre_norma,
                            "texto": texto.strip() if texto.strip() else nombre_norma,
                        }
                        lista_de_nodos.append(nodo_norma)
                        nodos_por_id[id_norma] = nodo_norma
                        nodos_por_nombre_normalizado[nombre_normalizado_key] = id_norma
                        
                        # Conectar a padre (SubCategoría o Categoría)
                        padre_id = subcategoria_id or categoria_id
                        if padre_id:
                            # Verificar que la relación no exista ya
                            relacion_existente = (id_norma, "PERTENECE_A", padre_id)
                            if relacion_existente not in lista_de_relaciones:
                                lista_de_relaciones.append(relacion_existente)
                        logger.info(f"[JERARQUIA] Creada norma desde metadata: {tipo_nodo} - {nombre_norma}")
                
                norma_id = id_norma
                contexto_actual["norma_id"] = id_norma
                contexto_actual["padre_id"] = id_norma

            # Nivel 4: Artículo (####) - PERO primero verificar si es realmente una norma principal
            if "Articulo" in metadata:
                nombre_articulo = metadata["Articulo"]
                
                # VALIDACIÓN CRÍTICA: Verificar si el "Articulo" es realmente una norma principal
                match_norma_en_articulo = re.match(
                    r"^(Decreto Ley|Ley|Decreto|Resolución|Circular|Sentencia|Auto|Constitución|Tratado|Convenio|Convención|DECRETO LEY|LEY|DECRETO|RESOLUCIÓN|CIRCULAR|SENTENCIA|AUTO|CONSTITUCIÓN|TRATADO|CONVENIO|CONVENCIÓN)\s+.*",
                    nombre_articulo,
                    re.IGNORECASE
                )
                
                if match_norma_en_articulo:
                    # Es realmente una norma, no un artículo - procesar como norma
                    tipo_raw = match_norma_en_articulo.group(1)
                    tipo_nodo = self._normalize_tipo_norma(tipo_raw)
                    
                    # Generar ID único y consistente
                    id_norma = self._generate_norma_id(tipo_nodo, nombre_articulo)
                    nombre_normalizado_key = f"{tipo_nodo}_{nombre_articulo.lower().strip()}"
                    
                    if id_norma in nodos_por_id:
                        # Nodo duplicado: consolidar información
                        nodo_existente = nodos_por_id[id_norma]
                        nodo_existente = self._consolidate_node(nodo_existente, texto.strip(), nombre_articulo)
                        logger.info(f"[CONSOLIDACION] Norma detectada en metadata Articulo (duplicado): {tipo_nodo} - {nombre_articulo}")
                    elif nombre_normalizado_key in nodos_por_nombre_normalizado:
                        # Nodo con nombre similar: usar el ID existente y consolidar
                        id_existente = nodos_por_nombre_normalizado[nombre_normalizado_key]
                        nodo_existente = nodos_por_id[id_existente]
                        nodo_existente = self._consolidate_node(nodo_existente, texto.strip(), nombre_articulo)
                        id_norma = id_existente
                        logger.info(f"[CONSOLIDACION] Norma detectada en metadata Articulo (similar): {tipo_nodo} - {nombre_articulo}")
                    else:
                        # Nueva norma detectada en metadata Articulo
                        nodo_norma = {
                            "id": id_norma,
                            "tipo": tipo_nodo,
                            "nombre": nombre_articulo,
                            "texto": texto.strip() if texto.strip() else nombre_articulo,
                        }
                        lista_de_nodos.append(nodo_norma)
                        nodos_por_id[id_norma] = nodo_norma
                        nodos_por_nombre_normalizado[nombre_normalizado_key] = id_norma
                        
                        # Conectar a padre (SubCategoría o Categoría)
                        padre_id = subcategoria_id or categoria_id
                        if padre_id:
                            relacion_existente = (id_norma, "PERTENECE_A", padre_id)
                            if relacion_existente not in lista_de_relaciones:
                                lista_de_relaciones.append(relacion_existente)
                        logger.info(f"[JERARQUIA] Norma detectada en metadata Articulo: {tipo_nodo} - {nombre_articulo}")
                    
                    norma_id = id_norma
                    contexto_actual["norma_id"] = id_norma
                    contexto_actual["padre_id"] = id_norma
                else:
                    # Es realmente un artículo
                    match_art = re.match(r"^Art[íi]culo\s+(\d+)", nombre_articulo, re.IGNORECASE)
                    numero_art = match_art.group(1) if match_art else "0"
                    
                    padre_actual = norma_id or subcategoria_id or categoria_id or "doc_root"
                    # Generar ID único basado en la norma padre y número de artículo
                    if norma_id:
                        id_articulo = f"articulo_{norma_id}_{numero_art}"
                    else:
                        contador_articulos = contexto_actual.get(f"articulos_count_{padre_actual}", 0) + 1
                        contexto_actual[f"articulos_count_{padre_actual}"] = contador_articulos
                        id_articulo = f"articulo_{padre_actual}_{contador_articulos}_{numero_art}"
                    
                    if id_articulo in nodos_por_id:
                        # Artículo duplicado: consolidar información
                        nodo_existente = nodos_por_id[id_articulo]
                        nodo_existente = self._consolidate_node(nodo_existente, texto.strip(), nombre_articulo)
                        logger.info(f"[CONSOLIDACION] Información consolidada para artículo existente: {nombre_articulo}")
                    else:
                        # Nuevo artículo
                        nodo_articulo = {
                            "id": id_articulo,
                            "tipo": "Articulo",
                            "nombre": nombre_articulo,
                            "numero_original": numero_art,
                            "texto": texto.strip() if texto.strip() else nombre_articulo,
                        }
                        lista_de_nodos.append(nodo_articulo)
                        nodos_por_id[id_articulo] = nodo_articulo
                        
                        # Conectar a norma padre
                        if norma_id:
                            relacion_existente = (id_articulo, "PERTENECE_A", norma_id)
                            if relacion_existente not in lista_de_relaciones:
                                lista_de_relaciones.append(relacion_existente)
                        elif padre_actual != "doc_root":
                            relacion_existente = (id_articulo, "PERTENECE_A", padre_actual)
                            if relacion_existente not in lista_de_relaciones:
                                lista_de_relaciones.append(relacion_existente)
                        logger.info(f"[JERARQUIA] Creado artículo desde metadata: {nombre_articulo}")

            # Procesar contenido adicional del texto (artículos dentro de normas, secciones, etc.)
            if texto.strip() and norma_id:
                # Procesar el texto para encontrar artículos, secciones, etc. dentro de la norma
                lineas = texto.split("\n")
                nodo_actual = None
                texto_acumulado = []

                for i, linea in enumerate(lineas):
                    linea = linea.strip()
                    if not linea:
                        continue

                    # Detectar headers markdown en el contenido
                    markdown_header_match = re.match(r"^(#{1,6})\s+(.+)", linea)
                    if markdown_header_match:
                        header_level = len(markdown_header_match.group(1))
                        header_text = markdown_header_match.group(2).strip()
                        
                        # Si encontramos un header, guardar el texto acumulado del nodo anterior
                        if nodo_actual and texto_acumulado:
                            texto_previo = " ".join(texto_acumulado).strip()
                            if texto_previo:
                                if nodo_actual.get("texto"):
                                    if texto_previo not in nodo_actual["texto"]:
                                        nodo_actual["texto"] = f"{nodo_actual['texto']}\n\n{texto_previo}"
                                else:
                                    nodo_actual["texto"] = texto_previo
                        
                        # Procesar según el nivel del header
                        if header_level == 4:  # #### = Artículo (pero verificar si es norma)
                            # Primero verificar si es realmente una norma
                            match_norma_en_header = re.match(
                                r"^(Decreto Ley|Ley|Decreto|Resolución|Circular|Sentencia|Auto|Constitución|Tratado|Convenio|Convención|DECRETO LEY|LEY|DECRETO|RESOLUCIÓN|CIRCULAR|SENTENCIA|AUTO|CONSTITUCIÓN|TRATADO|CONVENIO|CONVENCIÓN)\s+.*",
                                header_text,
                                re.IGNORECASE
                            )
                            
                            if match_norma_en_header:
                                # Es una norma, procesar como tal
                                tipo_raw = match_norma_en_header.group(1)
                                tipo_nodo = self._normalize_tipo_norma(tipo_raw)
                                id_norma_header = self._generate_norma_id(tipo_nodo, header_text)
                                nombre_normalizado_key = f"{tipo_nodo}_{header_text.lower().strip()}"
                                
                                texto_previo = " ".join(texto_acumulado).strip() if texto_acumulado else ""
                                
                                if id_norma_header in nodos_por_id:
                                    nodo_existente = nodos_por_id[id_norma_header]
                                    nodo_existente = self._consolidate_node(nodo_existente, texto_previo, header_text)
                                    nodo_actual = nodo_existente
                                    logger.info(f"[CONSOLIDACION] Norma detectada en header nivel 4 (duplicado): {tipo_nodo} - {header_text}")
                                elif nombre_normalizado_key in nodos_por_nombre_normalizado:
                                    id_existente = nodos_por_nombre_normalizado[nombre_normalizado_key]
                                    nodo_existente = nodos_por_id[id_existente]
                                    nodo_existente = self._consolidate_node(nodo_existente, texto_previo, header_text)
                                    nodo_actual = nodo_existente
                                    id_norma_header = id_existente
                                    logger.info(f"[CONSOLIDACION] Norma detectada en header nivel 4 (similar): {tipo_nodo} - {header_text}")
                                else:
                                    nodo_norma = {
                                        "id": id_norma_header,
                                        "tipo": tipo_nodo,
                                        "nombre": header_text,
                                        "texto": texto_previo if texto_previo else header_text,
                                    }
                                    lista_de_nodos.append(nodo_norma)
                                    nodos_por_id[id_norma_header] = nodo_norma
                                    nodos_por_nombre_normalizado[nombre_normalizado_key] = id_norma_header
                                    nodo_actual = nodo_norma
                                    
                                    padre_id = subcategoria_id or categoria_id
                                    if padre_id:
                                        relacion_existente = (id_norma_header, "PERTENECE_A", padre_id)
                                        if relacion_existente not in lista_de_relaciones:
                                            lista_de_relaciones.append(relacion_existente)
                                    logger.info(f"[JERARQUIA] Norma detectada en header nivel 4: {tipo_nodo} - {header_text}")
                                
                                texto_acumulado = []
                                continue
                            
                            # Es realmente un artículo
                            match_art = re.match(r"^Art[íi]culo\s+(\d+)", header_text, re.IGNORECASE)
                            if match_art:
                                numero_art = match_art.group(1)
                                id_articulo = f"articulo_{norma_id}_{numero_art}"
                                
                                texto_previo = " ".join(texto_acumulado).strip() if texto_acumulado else ""
                                
                                if id_articulo in nodos_por_id:
                                    # Artículo duplicado: consolidar
                                    nodo_existente = nodos_por_id[id_articulo]
                                    nodo_existente = self._consolidate_node(nodo_existente, texto_previo, header_text)
                                    nodo_actual = nodo_existente
                                else:
                                    nodo_actual = {
                                        "id": id_articulo,
                                        "tipo": "Articulo",
                                        "nombre": header_text,
                                        "numero_original": numero_art,
                                        "texto": texto_previo if texto_previo else "",
                                    }
                                    lista_de_nodos.append(nodo_actual)
                                    nodos_por_id[id_articulo] = nodo_actual
                                    relacion_existente = (id_articulo, "PERTENECE_A", norma_id)
                                    if relacion_existente not in lista_de_relaciones:
                                        lista_de_relaciones.append(relacion_existente)
                                
                                texto_acumulado = []
                                continue
                        
                        elif header_level == 3:  # ### = Sección/Título/Capítulo
                            match_seccion = re.match(r"^(Título|Libro|Capítulo|Parte)\s+", header_text, re.IGNORECASE)
                            if match_seccion:
                                seccion_text = match_seccion.group(1)
                                if seccion_text == "Título":
                                    tipo_seccion = "Titulo"
                                elif seccion_text == "Capítulo":
                                    tipo_seccion = "Capitulo"
                                elif seccion_text == "Libro":
                                    tipo_seccion = "Libro"
                                elif seccion_text == "Parte":
                                    tipo_seccion = "Parte"
                                else:
                                    tipo_seccion = "Seccion"
                                
                                padre_actual = norma_id or "doc_root"
                                contador_secciones = contexto_actual.get(f"secciones_count_{padre_actual}", 0) + 1
                                contexto_actual[f"secciones_count_{padre_actual}"] = contador_secciones
                                
                                id_seccion = f"{tipo_seccion.lower()}_{padre_actual}_{contador_secciones}"
                                
                                if id_seccion not in nodos_por_id:
                                    nodo_actual = {
                                        "id": id_seccion,
                                        "tipo": tipo_seccion,
                                        "nombre": header_text,
                                        "texto": "",
                                    }
                                    lista_de_nodos.append(nodo_actual)
                                    nodos_por_id[id_seccion] = nodo_actual
                                    lista_de_relaciones.append((id_seccion, "PERTENECE_A", padre_actual))
                                
                                texto_acumulado = []
                                continue

                    # Patrones de texto plano (fallback si no hay headers markdown)
                    match_articulo = re.match(
                        r"^(ARTÍCULO|Art\.|Artículo|Arts\.)\s+([0-9.]+)",
                        linea,
                        re.IGNORECASE,
                    )
                    match_seccion = re.match(r"^(Título|Libro|Capítulo|Parte)\s+.*", linea)
                    match_paragrafo = re.match(
                        r"^(PARÁGRAFO|PÁRRAFO)(\s+PRIMERO|\s+SEGUNDO|\s+TERCERO|\s+CUARTO|\s+QUINTO|\s+[IVX]+|\s+\d+)?\s*\.?",
                        linea,
                        re.IGNORECASE,
                    )

                    if any([match_articulo, match_seccion, match_paragrafo]):
                        if nodo_actual and texto_acumulado:
                            nodo_actual["texto"] = " ".join(texto_acumulado).strip()
                        texto_acumulado = []

                    if match_articulo:
                        numero_articulo = match_articulo.group(2)
                        nombre_original = linea.strip()
                        padre_actual = contexto_actual.get("seccion_id", norma_id or "doc_root")
                        
                        # Generar ID único basado en la norma padre y número de artículo
                        if norma_id:
                            id_nodo = f"articulo_{norma_id}_{numero_articulo.replace('.', '_')}"
                        else:
                            contador_articulos = contexto_actual.get(f"articulos_count_{padre_actual}", 0) + 1
                            contexto_actual[f"articulos_count_{padre_actual}"] = contador_articulos
                            id_nodo = f"articulo_{padre_actual}_{contador_articulos}_{numero_articulo.replace('.', '_')}"

                        padre_id = contexto_actual.get("seccion_id") or norma_id or contexto_actual.get("padre_id")
                        contexto_actual["articulo_id"] = id_nodo

                        if id_nodo in nodos_por_id:
                            # Artículo duplicado: consolidar información
                            nodo_existente = nodos_por_id[id_nodo]
                            nodo_existente = self._consolidate_node(nodo_existente, "", nombre_original)
                            nodo_actual = nodo_existente
                        else:
                            nodo_actual = {
                                "id": id_nodo,
                                "tipo": "Articulo",
                                "nombre": nombre_original,
                                "numero_original": numero_articulo,
                                "texto": "",
                            }
                            lista_de_nodos.append(nodo_actual)
                            nodos_por_id[id_nodo] = nodo_actual
                            if padre_id:
                                relacion_existente = (id_nodo, "PERTENECE_A", padre_id)
                                if relacion_existente not in lista_de_relaciones:
                                    lista_de_relaciones.append(relacion_existente)

                    elif match_seccion:
                        seccion_text = match_seccion.group(1)
                        if seccion_text == "Título":
                            tipo_nodo = "Titulo"
                        elif seccion_text == "Capítulo":
                            tipo_nodo = "Capitulo"
                        elif seccion_text == "Libro":
                            tipo_nodo = "Libro"
                        elif seccion_text == "Parte":
                            tipo_nodo = "Parte"
                        else:
                            tipo_nodo = "Seccion"

                        nombre_original = linea.strip()
                        padre_actual = norma_id or "doc_root"
                        contador_secciones = contexto_actual.get(f"secciones_count_{padre_actual}", 0) + 1
                        contexto_actual[f"secciones_count_{padre_actual}"] = contador_secciones
                        id_nodo = f"{tipo_nodo.lower()}_{padre_actual}_{contador_secciones}"

                        nodo_actual = {
                            "id": id_nodo,
                            "tipo": tipo_nodo,
                            "nombre": nombre_original,
                            "texto": "",
                        }
                        padre_id = norma_id or contexto_actual.get("padre_id")
                        contexto_actual["seccion_id"] = id_nodo

                        if id_nodo not in nodos_por_id:
                            lista_de_nodos.append(nodo_actual)
                            nodos_por_id[id_nodo] = nodo_actual
                            if padre_id:
                                lista_de_relaciones.append((id_nodo, "PERTENECE_A", padre_id))

                    elif match_paragrafo:
                        tipo_nodo = "Paragrafo"
                        nombre_original = linea.strip()
                        padre_id = contexto_actual.get("articulo_id")
                        if padre_id:
                            contador_paragrafos = contexto_actual.get(f"paragrafos_count_{padre_id}", 0) + 1
                            contexto_actual[f"paragrafos_count_{padre_id}"] = contador_paragrafos
                            id_nodo = f"paragrafo_{padre_id}_{contador_paragrafos}"

                            nodo_actual = {
                                "id": id_nodo,
                                "tipo": tipo_nodo,
                                "nombre": nombre_original,
                                "texto": "",
                            }

                            if id_nodo not in nodos_por_id:
                                lista_de_nodos.append(nodo_actual)
                                nodos_por_id[id_nodo] = nodo_actual
                                lista_de_relaciones.append((id_nodo, "PERTENECE_A", padre_id))

                    else:
                        if nodo_actual:
                            texto_acumulado.append(linea)

                # Guardar el último nodo
                if nodo_actual and texto_acumulado:
                    nodo_actual["texto"] = " ".join(texto_acumulado).strip()

        logger.info(f"[JERARQUIA] Estructura creada: {len(lista_de_nodos)} nodos, {len(lista_de_relaciones)} relaciones")
        return lista_de_nodos, lista_de_relaciones

    def _clean_neo4j_values(self, valor: Any) -> Any:
        """
        Limpia recursivamente valores para que sean compatibles con Neo4j
        Elimina: None, arrays vacíos, arrays con None, arrays anidados
        """
        if valor is None:
            return None

        if isinstance(valor, list):
            valores_limpios = [self._clean_neo4j_values(v) for v in valor if v is not None]
            valores_limpios = [v for v in valores_limpios if v is not None]

            if not valores_limpios:
                return None

            if len(valores_limpios) > 0 and isinstance(valores_limpios[0], list):
                return valores_limpios[0] if valores_limpios[0] else None

            return valores_limpios

        return valor

    async def _create_nodes_final(self, tx, nodos: List[Dict]) -> int:
        """Versión FINAL que elimina TODOS los valores problemáticos para Neo4j."""
        logger.info(f"Cargando {len(nodos)} nodos (limpiando valores None/nulos)...")

        # Agrupar nodos por tipo
        nodos_por_tipo = {}
        for nodo in nodos:
            tipo = nodo.get("tipo", "Entity")
            if tipo not in nodos_por_tipo:
                nodos_por_tipo[tipo] = []
            nodos_por_tipo[tipo].append(nodo)

        total_creados = 0

        # Procesar cada tipo por separado
        for tipo, nodos_tipo in nodos_por_tipo.items():
            logger.info(f"  [DOCUMENTO] Procesando {len(nodos_tipo)} nodos de tipo {tipo}...")

            # Limpiar nodos para este tipo
            nodos_limpios = []
            for nodo in nodos_tipo:
                nodo_limpio = {}

                for key, value in nodo.items():
                    if key == "tipo":
                        continue

                    if key == "embeddings":
                        embedding_limpio = self._clean_neo4j_values(value)
                        if embedding_limpio is not None:
                            nodo_limpio["embedding"] = embedding_limpio
                        continue

                    valor_limpio = self._clean_neo4j_values(value)
                    if valor_limpio is not None:
                        nodo_limpio[key] = valor_limpio

                if "id" in nodo_limpio:
                    nodos_limpios.append(nodo_limpio)

            if not nodos_limpios:
                logger.warning(f"    [ADVERTENCIA] No hay nodos válidos para tipo {tipo}")
                continue

            # Consulta Cypher con doble etiqueta
            query_nodos = """
            UNWIND $nodos as nodo
            MERGE (n {id: nodo.id})
            SET n += nodo
            WITH n
            // Añade la etiqueta específica Y la etiqueta genérica al mismo tiempo
            CALL apoc.create.addLabels(n, [$tipo, 'Normografia']) YIELD node
            RETURN count(node) as created
            """

            result = await tx.run(query_nodos, nodos=nodos_limpios, tipo=tipo)
            result_record = await result.single()
            creados = result_record["created"] if result_record else 0
            total_creados += creados
            logger.info(f"    [EXITO] {creados} nodos {tipo} procesados")

        logger.info(f"Se procesaron {total_creados} nodos en total.")
        return total_creados

    async def _create_relations_corrected(self, tx, relaciones: List[Tuple]) -> int:
        """Carga relaciones usando APOC para crear tipos de relación dinámicos."""
        logger.info(f"Creando {len(relaciones)} relaciones con tipos dinámicos...")

        query_relaciones = """
        UNWIND $relaciones as rel
        MATCH (origen {id: rel.origen_id})
        MATCH (destino {id: rel.destino_id})
        CALL apoc.merge.relationship(origen, rel.tipo, {}, {}, destino) YIELD rel as r
        RETURN count(r) as created
        """

        relaciones_dict = [{"origen_id": r[0], "tipo": r[1], "destino_id": r[2]} for r in relaciones]

        result = await tx.run(query_relaciones, relaciones=relaciones_dict)
        result_record = await result.single()
        creados = result_record["created"] if result_record else 0
        logger.info(f"Se crearon {creados} relaciones.")
        return creados

    async def upload_bulk_data(self, nodos: List[Dict], relaciones: List[Tuple]) -> Tuple[int, int]:
        """
        Función principal FINAL que elimina todos los valores problemáticos y carga datos masivamente.
        """
        try:
            async with self.driver.session(database=self.database) as session:
                logger.info("[INICIAR] Iniciando carga FINAL de datos a Neo4j...")

                # Crear constraints básicos
                logger.info("[LISTA] Creando constraints...")
                constraints = [
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Categoria) REQUIRE n.id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Ley) REQUIRE n.id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:DecretoLey) REQUIRE n.id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Decreto) REQUIRE n.id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Seccion) REQUIRE n.id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Titulo) REQUIRE n.id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Capitulo) REQUIRE n.id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Libro) REQUIRE n.id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Parte) REQUIRE n.id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Articulo) REQUIRE n.id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Paragrafo) REQUIRE n.id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Sentencia) REQUIRE n.id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Circular) REQUIRE n.id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Resolución) REQUIRE n.id IS UNIQUE",
                ]

                for constraint in constraints:
                    await session.run(constraint)

                # Ejecutar carga
                total_nodos = await session.execute_write(self._create_nodes_final, nodos)
                total_relaciones = await session.execute_write(self._create_relations_corrected, relaciones)

                logger.info("\\n[EXITO] ¡Carga completada exitosamente!")
                logger.info(f"   [ESTADISTICAS] Nodos procesados: {total_nodos}")
                logger.info(f"   [ESTADISTICAS] Relaciones creadas: {total_relaciones}")

                return total_nodos, total_relaciones

        except Exception as e:
            logger.error(f"[ERROR] Ocurrió un error durante la carga: {str(e)}", exc_info=True)
            raise

    async def process_markdown_file(self, file_path: str, skip_normalization: bool = False) -> BulkLoadResponse:
        """
        Procesa un archivo Markdown y lo carga en Neo4j.

        Argumentos:
            file_path: Ruta al archivo Markdown
            skip_normalization: Si es True, omite la normalización del archivo

        Retorna:
            BulkLoadResponse con el resultado de la operación
        """
        start_time = time.time()

        try:
            # Leer el archivo
            with open(file_path, "r", encoding="utf-8") as f:
                documentos = f.read()

            # Normalizar solo si no se omite
            if skip_normalization:
                texto_normalizado = documentos
                normalized_path = None
                logger.info("[ARCHIVO] Normalización omitida, usando archivo original")
            else:
                texto_normalizado = self.normalize_markdown(documentos)
                # Opcional: guardar el archivo normalizado
                normalized_path = file_path.replace(".md", "_normalizado.md")
                with open(normalized_path, "w", encoding="utf-8") as f:
                    f.write(texto_normalizado)
                logger.info(f"[ARCHIVO] Archivo normalizado guardado en: {normalized_path}")

            # Dividir el texto en chunks
            chunks = self.markdown_splitter.split_text(texto_normalizado)
            documentos_limpios = self.clean_documents(chunks)

            # Crear estructura de grafo
            nodos_finales, relaciones_finales = self.create_graph_structure(documentos_limpios)

            # Generar embeddings para TODOS los nodos
            await self.embedding_service.initialize_model()
            for nodo in nodos_finales:
                # Preparar texto para embedding: usar texto si existe, sino usar nombre
                texto_para_embedding = ""
                if "texto" in nodo and nodo["texto"].strip():
                    texto_para_embedding = nodo["texto"].strip()
                elif "nombre" in nodo and nodo["nombre"].strip():
                    texto_para_embedding = nodo["nombre"].strip()
                else:
                    # Fallback: usar tipo + id
                    texto_para_embedding = f"{nodo.get('tipo', 'Nodo')} {nodo.get('id', 'sin_id')}"

                # Generar embedding para este nodo
                if texto_para_embedding:
                    try:
                        embeddings = await self.embedding_service.generate_embeddings([texto_para_embedding])
                        nodo["embeddings"] = embeddings[0] if embeddings else None
                        logger.debug(f"Embedding generado para nodo {nodo['id']}: {len(embeddings[0]) if embeddings else 0} dimensiones")
                    except Exception as e:
                        logger.warning(f"Error generando embedding para nodo {nodo['id']}: {e}")
                        nodo["embeddings"] = None
                else:
                    logger.warning(f"No se pudo generar texto para embedding del nodo {nodo['id']}")
                    nodo["embeddings"] = None

            # Validar que todos los nodos tengan texto y embeddings
            nodos_validados = self._validate_nodes_completeness(nodos_finales)

            # Cargar en Neo4j
            total_nodos, total_relaciones = await self.upload_bulk_data(nodos_validados, relaciones_finales)

            processing_time = time.time() - start_time

            logger.info(f"[EXITO] Procesamiento completado: {len(nodos_finales)} nodos, {len(relaciones_finales)} relaciones")

            return BulkLoadResponse(
                success=True,
                nodes_processed=total_nodos,
                relations_created=total_relaciones,
                processing_time=processing_time,
                normalized_file_path=normalized_path,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"[ERROR] Error procesando archivo {file_path}: {e}")
            return BulkLoadResponse(success=False, processing_time=processing_time, error_message=str(e))

    def _validate_nodes_completeness(self, nodos: List[Dict]) -> List[Dict]:
        """
        Valida y completa los nodos para asegurar que todos tengan texto y embeddings.
        """
        nodos_validados = []

        for nodo in nodos:
            # Asegurar que el nodo tenga texto
            if not nodo.get("texto") or not nodo["texto"].strip():
                # Si no tiene texto, usar el nombre como texto mínimo
                if nodo.get("nombre"):
                    nodo["texto"] = nodo["nombre"]
                else:
                    nodo["texto"] = f"{nodo.get('tipo', 'Nodo')} {nodo.get('id', 'sin_identificador')}"
                logger.info(f"Nodo {nodo['id']} completado con texto mínimo: {nodo['texto'][:50]}...")

            # Verificar embeddings
            if not nodo.get("embeddings"):
                logger.warning(f"Nodo {nodo['id']} no tiene embeddings, será procesado sin vector")
            else:
                logger.debug(f"Nodo {nodo['id']} validado con texto ({len(nodo['texto'])} chars) y embeddings ({len(nodo['embeddings'])} dims)")

            nodos_validados.append(nodo)

        logger.info(f"Validación completada: {len(nodos_validados)} nodos preparados para carga")
        return nodos_validados

    def find_node_by_original_name(self, nodos: List[Dict], nombre_original: str, tipo: str = None) -> List[Dict]:
        """
        Busca nodos por su nombre original, manteniendo la trazabilidad legal.

        Argumentos:
            nodos: Lista de nodos donde buscar
            nombre_original: Nombre exacto como aparece en el documento original
            tipo: Tipo de nodo opcional para filtrar

        Retorna:
            Lista de nodos que coinciden con el nombre original
        """
        resultados = []

        for nodo in nodos:
            if nodo.get("nombre") == nombre_original:
                if tipo is None or nodo.get("tipo") == tipo:
                    resultados.append(nodo)

        logger.info(f"Búsqueda por nombre original '{nombre_original}': {len(resultados)} coincidencias")
        return resultados

    def _split_document_into_sections(self, markdown_content: str) -> List[Dict[str, Any]]:
        """
        Usa LangChain MarkdownHeaderTextSplitter para dividir el documento en secciones.
        """
        # Configurar headers para dividir por títulos principales
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
            ("######", "Header 6"),
        ]

        # Crear splitter
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,  # Mantener headers para identificar tipos
        )

        # Dividir documento
        docs = markdown_splitter.split_text(markdown_content)

        # Convertir a formato interno
        chunks = []
        for doc in docs:
            chunk = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "headers": self._extract_headers_from_metadata(doc.metadata),
            }
            chunks.append(chunk)

        return chunks

    def _extract_headers_from_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        """
        Extrae la jerarquía de headers del metadata de LangChain.
        """
        headers = []
        for i in range(1, 7):  # Header 1 to Header 6
            header_key = f"Header {i}"
            if header_key in metadata:
                headers.append(metadata[header_key])
        return headers

    def _create_full_document_structure(self, chunks: List[Dict[str, Any]], law_name: str, category: str) -> Tuple[List[Dict], List[Tuple]]:
        """
        Crea la estructura completa de nodos y relaciones para todo el documento.
        """
        all_nodes = []
        all_relations = []

        # Crear nodo de la ley principal con ID consistente (usar prefijo "ley_")
        law_node = {
            "id": f"ley_{law_name.replace(' ', '_').replace('(', '').replace(')', '').lower()[:50]}",
            "tipo": "Ley",
            "nombre": law_name,
            "texto": f"Ley completa: {law_name}",
        }
        all_nodes.append(law_node)

        # Procesar cada chunk y crear nodos jerárquicos
        for chunk in chunks:
            nodes, relations = self._process_chunk_to_nodes(chunk, law_node["id"], law_name)
            all_nodes.extend(nodes)
            all_relations.extend(relations)
        
        # Asegurar que todos los nodos tengan texto antes de generar embeddings
        for node in all_nodes:
            if not node.get("texto") or not node["texto"].strip():
                if node.get("nombre") and node["nombre"].strip():
                    node["texto"] = node["nombre"].strip()
                else:
                    node["texto"] = f"{node.get('tipo', 'Nodo')} {node.get('id', 'sin_id')}"
                logger.warning(f"Nodo {node.get('id')} sin texto, usando fallback: {node['texto'][:50]}...")

        return all_nodes, all_relations

    def _process_chunk_to_nodes(self, chunk: Dict[str, Any], law_id: str, law_name: str) -> Tuple[List[Dict], List[Tuple]]:
        """
        Convierte un chunk en nodos y relaciones usando nuestro sistema mejorado.
        """
        content = chunk["content"]
        headers = chunk.get("headers", [])

        # Usar nuestro sistema existente pero mejorado
        nodes, relations = self._update_structure_graph(content, law_id)

        # Ajustar relaciones basándose en la jerarquía de headers
        if headers and nodes:
            # El primer nodo debería conectarse al nivel jerárquico correcto
            first_node = nodes[0]
            # Lógica para determinar el padre correcto basándose en headers
            parent_id = self._determine_parent_from_headers(headers, law_id)
            if parent_id != law_id and relations:
                # Actualizar la relación del primer nodo
                relations = [(first_node["id"], "PERTENECE_A", parent_id)] + relations[1:]

        return nodes, relations

    def _determine_parent_from_headers(self, headers: List[str], law_id: str) -> str:
        """
        Determina el nodo padre correcto basándose en la jerarquía de headers.
        """
        # Por ahora, conectar directamente a la ley
        # En el futuro, se puede implementar lógica más sofisticada
        return law_id

    async def _generate_embeddings_for_nodes(self, nodes: List[Dict]) -> None:
        """
        Genera embeddings para todos los nodos usando el servicio centralizado.
        """
        logger.info(f" Generando embeddings para {len(nodes)} nodos")

        await self.embedding_service.initialize_model()

        # Procesar en lotes
        batch_size = 10
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]

            # Preparar textos para embedding
            texts_for_embedding = []
            for node in batch:
                # Usar texto si existe, sino usar nombre
                text_for_embedding = ""
                if node.get("texto") and node["texto"].strip():
                    text_for_embedding = node["texto"].strip()
                elif node.get("nombre") and node["nombre"].strip():
                    text_for_embedding = node["nombre"].strip()
                else:
                    text_for_embedding = f"{node.get('tipo', 'Nodo')} {node.get('id', 'sin_id')}"

                texts_for_embedding.append(text_for_embedding)

            try:
                # Generar embeddings
                embeddings = await self.embedding_service.generate_embeddings(texts_for_embedding, normalize=True)

                # Asignar embeddings a nodos
                for j, embedding in enumerate(embeddings):
                    if j < len(batch):
                        batch[j]["embedding"] = embedding

            except Exception as e:
                logger.error(f"Error generando embeddings para lote: {e}")
                # Continuar sin embeddings para este lote

    async def _bulk_create_document_structure(
        self,
        tx,
        category: str,
        law_name: str,
        nodes: List[Dict],
        relations: List[Tuple],
    ) -> Tuple[int, int]:
        """
        Crea toda la estructura del documento en una sola transacción.
        CORREGIDO: Preserva tanto el texto como los embeddings.
        """
        logger.info(" Creando estructura completa en Neo4j")

        # Crear categoría si no existe
        logger.info(f" [BULK] Creando/verificando categoría: {category}")
        category_result = await tx.run(
            """
            MERGE (c {nombre: $category})
            ON CREATE SET c.id = $id
            WITH c
            CALL apoc.create.addLabels(c, ['Categoria', 'Normografia']) YIELD node
            RETURN count(node) as created, c.id as category_id, labels(c) as labels
        """,
            category=category,
            id=f"cat_{category.replace(' ', '_').lower()[:50]}",
        )

        category_record = await category_result.single()
        logger.info(f" [BULK] Categoría procesada: {category_record}")

        nodes_created = 0
        relations_created = 0

        # Crear todos los nodos
        if nodes:
            logger.info(f" Creando {len(nodes)} nodos")
            for node in nodes:
                # Copia completa incluyendo embedding
                node_properties = dict(node)
                
                # Asegurar que el embedding esté en el formato correcto (lista de floats)
                if "embedding" in node_properties:
                    embedding = node_properties["embedding"]
                    if embedding is not None:
                        # Convertir a lista si no lo es
                        if not isinstance(embedding, list):
                            logger.warning(f"Nodo {node.get('id')} tiene embedding en formato incorrecto: {type(embedding)}")
                            node_properties.pop("embedding", None)
                        else:
                            # Verificar que sea una lista de números
                            try:
                                node_properties["embedding"] = [float(x) for x in embedding]
                                logger.debug(f"Nodo {node.get('id')} tiene embedding con {len(embedding)} dimensiones")
                            except (ValueError, TypeError) as e:
                                logger.error(f"Error convirtiendo embedding del nodo {node.get('id')}: {e}")
                                node_properties.pop("embedding", None)
                    else:
                        node_properties.pop("embedding", None)
                else:
                    logger.warning(f"Nodo {node.get('id')} NO tiene embedding")

                # Crear nodo con todas las propiedades
                node_query = """
                MERGE (n {id: $id})
                SET n += $properties
                WITH n
                CALL apoc.create.addLabels(n, [$tipo, 'Normografia']) YIELD node
                RETURN count(node)
                """

                result = await tx.run(
                    node_query,
                    {
                        "id": node["id"],
                        "properties": node_properties,  # Incluye texto, nombre, tipo, embedding
                        "tipo": node.get("tipo", "Nodo"),
                    },
                )

                record = await result.single()
                if record:
                    nodes_created += record[0]
                
                # Verificar que el embedding se guardó (para debugging)
                if "embedding" in node_properties and node_properties.get("embedding") is not None:
                    verify_query = """
                    MATCH (n {id: $id})
                    RETURN CASE WHEN n.embedding IS NOT NULL THEN size(n.embedding) ELSE 0 END as embedding_size
                    """
                    verify_result = await tx.run(verify_query, {"id": node["id"]})
                    verify_record = await verify_result.single()
                    if verify_record and verify_record["embedding_size"] == 0:
                        logger.error(f"ADVERTENCIA: Embedding NO se guardó para nodo {node.get('id')}")
                    elif verify_record:
                        logger.debug(f"Embedding guardado correctamente para nodo {node.get('id')}: {verify_record['embedding_size']} dimensiones")

        # Crear todas las relaciones
        if relations:
            logger.info(f" Creando {len(relations)} relaciones")
            for source_id, rel_type, target_id in relations:
                rel_query = """
                MATCH (source {id: $source_id})
                MATCH (target {id: $target_id})
                MERGE (source)-[r:PERTENECE_A]->(target)
                RETURN count(r)
                """

                result = await tx.run(rel_query, {"source_id": source_id, "target_id": target_id})

                record = await result.single()
                if record:
                    relations_created += record[0]

        # Conectar la ley a la categoría
        connect_query = """
        MATCH (c:Categoria {nombre: $category})
        MATCH (l:Ley {nombre: $law_name})
        MERGE (l)-[:PERTENECE_A]->(c)
        RETURN count(*)
        """

        await tx.run(connect_query, {"category": category, "law_name": law_name})

        logger.info(f" Estructura creada: {nodes_created} nodos, {relations_created} relaciones")
        return nodes_created, relations_created

    async def migrate_existing_section_labels(self) -> Dict[str, int]:
        """
        Migra nodos existentes con etiqueta genérica 'Seccion' a etiquetas específicas
        basándose en el contenido de su nombre.

        Retorna:
            Dict con estadísticas de la migración
        """
        try:
            async with self.driver.session(database=self.database) as session:
                logger.info(" Iniciando migración de etiquetas de secciones existentes...")

                # Consulta para obtener todos los nodos Seccion existentes
                find_sections_query = """
                MATCH (n:Seccion)
                WHERE n.nombre IS NOT NULL
                RETURN elementId(n) as element_id, n.nombre as nombre, labels(n) as current_labels
                ORDER BY n.nombre
                """

                result = await session.run(find_sections_query)
                sections_to_migrate = []

                async for record in result:
                    element_id = record["element_id"]
                    nombre = record["nombre"]
                    current_labels = record["current_labels"]

                    # Determinar la etiqueta específica basada en el nombre
                    new_label = None
                    if "título" in nombre.lower() or "titulo" in nombre.lower():
                        new_label = "Titulo"
                    elif "capítulo" in nombre.lower() or "capitulo" in nombre.lower():
                        new_label = "Capitulo"
                    elif "libro" in nombre.lower():
                        new_label = "Libro"
                    elif "parte" in nombre.lower():
                        new_label = "Parte"

                    if new_label and new_label not in current_labels:
                        sections_to_migrate.append(
                            {
                                "element_id": element_id,
                                "nombre": nombre,
                                "new_label": new_label,
                                "current_labels": current_labels,
                            }
                        )

                logger.info(f" Encontrados {len(sections_to_migrate)} nodos para migrar")

                # Migrar cada nodo
                migration_stats = {
                    "titulo_migrated": 0,
                    "capitulo_migrated": 0,
                    "libro_migrated": 0,
                    "parte_migrated": 0,
                    "total_migrated": 0,
                    "errors": 0,
                }

                for section in sections_to_migrate:
                    try:
                        # Añadir la nueva etiqueta específica manteniendo las existentes
                        migration_query = """
                        MATCH (n)
                        WHERE elementId(n) = $element_id
                        CALL apoc.create.addLabels(n, [$new_label]) YIELD node
                        RETURN labels(node) as updated_labels
                        """

                        migration_result = await session.run(
                            migration_query,
                            element_id=section["element_id"],
                            new_label=section["new_label"],
                        )

                        migration_record = await migration_result.single()
                        if migration_record:
                            updated_labels = migration_record["updated_labels"]
                            logger.info(f" Migrado: {section['nombre']} -> {section['new_label']} (etiquetas: {updated_labels})")

                            # Actualizar estadísticas
                            label_key = f"{section['new_label'].lower()}_migrated"
                            migration_stats[label_key] += 1
                            migration_stats["total_migrated"] += 1

                    except Exception as e:
                        logger.error(f" Error migrando {section['nombre']}: {e}")
                        migration_stats["errors"] += 1

                logger.info(" Migración completada!")
                logger.info(f" Estadísticas finales: {migration_stats}")

                return migration_stats

        except Exception as e:
            logger.error(f" Error durante la migración: {e}")
            return {"error": str(e), "total_migrated": 0}
