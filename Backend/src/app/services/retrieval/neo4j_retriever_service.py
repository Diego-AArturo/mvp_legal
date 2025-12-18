"""
Servicio de recuperación Neo4j para contexto legal y análisis de relaciones.

Proporciona capacidades de búsqueda semántica sobre documentos legales, regulaciones
y relaciones institucionales almacenadas en la base de datos de grafos Neo4j.
"""

import time
from typing import Any, Dict, List

from neo4j import AsyncDriver, AsyncSession

from app.config.settings import Settings
from app.extensions import get_logger
from app.schemas.graph_search_schemas import GraphSearchResponse, GraphSearchResult
from app.services.embeddings.embedding_service import EmbeddingService

logger = get_logger(__name__)


class Neo4jRetrieverService:
    """
    Servicio de recuperación Neo4j para contexto legal y análisis de relaciones.

    Proporciona capacidades de búsqueda semántica sobre documentos legales, regulaciones
    y relaciones institucionales almacenadas en la base de datos de grafos Neo4j.
    """

    def __init__(
        self,
        driver: AsyncDriver,
        settings: Settings,
        embedding_service: EmbeddingService,
    ):
        """
        Inicializa el servicio de recuperación Neo4j.

        Argumentos:
            driver: Driver asíncrono de Neo4j
            settings: Configuración de la aplicación
            embedding_service: Servicio de embeddings para generar vectores
        """
        self.driver = driver
        self.settings = settings
        self.embedding_service = embedding_service
        self.database = settings.databases.neo4j.neo4j_database

    async def semantic_search(
        self, query: str, limit: int = 10, similarity_threshold: float = 0.7
    ) -> GraphSearchResponse:
        """
        Realiza búsqueda semántica para contexto legal basada en la consulta del usuario.

        Argumentos:
            query: Consulta del usuario para buscar
            limit: Número máximo de resultados a retornar
            similarity_threshold: Puntuación mínima de similitud para resultados

        Retorna:
            GraphSearchResponse con resultados de búsqueda legal
        """
        start_time = time.time()
        try:
            async with self.driver.session(database=self.database) as session:
                query_embedding = await self._embed_query(query)
                # Leer umbrales y límites desde configuración
                s = self.settings.databases.neo4j
                limit = min(limit, s.neo4j_vector_limit)
                similarity_threshold = max(0.0, min(1.0, similarity_threshold))
                # Si se usó el valor por defecto, reemplazar con configuración de env
                if similarity_threshold == 0.7:
                    similarity_threshold = s.neo4j_vector_similarity_threshold

                # Combinar múltiples estrategias de búsqueda
                entities = await self._search_legal_entities(
                    session,
                    query,
                    query_embedding,
                    limit,
                    similarity_threshold,
                )
                relationships = await self._search_legal_relationships(
                    session,
                    query,
                    query_embedding,
                    min(limit, s.neo4j_relationships_limit),
                    similarity_threshold,
                )
                paths = await self._discover_legal_paths(
                    session,
                    query,
                    query_embedding,
                    limit,
                    similarity_threshold,
                )
                # Convertir entidades a GraphSearchResult
                results = []
                for entity in entities:
                    if isinstance(entity, dict):
                        result = GraphSearchResult(
                            id=entity.get("id", ""),
                            tipo=entity.get("tipo", ""),
                            nombre=entity.get("nombre", ""),
                            texto_relevante=entity.get("contenido", "")[
                                :500
                            ],  # Limitar texto
                            conceptos_clave=entity.get("conceptos_clave", []),
                            normas_citadas=entity.get("normas_citadas", []),
                            score=entity.get("score", 0.0),
                            metadata=entity.get("metadata", {}),
                        )
                        results.append(result)

                return GraphSearchResponse(
                    results=results,
                    total_found=len(results),
                    execution_time=(time.time() - start_time),
                    search_strategy="normografia_vector",
                    metadata={
                        "query_used": "normografia_vector",
                        "relationships_found": len(relationships),
                        "paths_found": len(paths),
                    },
                )

        except Exception as e:
            logger.error(f"Error in Neo4j semantic search: {e}")
            return GraphSearchResponse(
                results=[],
                total_found=0,
                execution_time=(time.time() - start_time),
                search_strategy="normografia_vector_error",
                metadata={"error": str(e)},
            )

    async def _search_legal_entities(
        self,
        session: AsyncSession,
        query: str,
        query_embedding: List[float],
        limit: int,
        similarity_threshold: float,
    ) -> List[Dict[str, Any]]:
        """
        Search for legal entities (laws, regulations, institutions) related to query.

        Este método replica la lógica del neo4j_agent para búsqueda vectorial.

        Argumentos:
            session: Neo4j async session
            query: Search query
            query_embedding: Vector embedding de la consulta
            limit: Maximum results
            similarity_threshold: Umbral de similitud mínimo

        Retorna:
            List of legal entities with metadata
        """
        entities = []

        # Búsqueda exclusivamente en norma_embeddings para nodos de normografía
        try:
            norma_query = """
            WITH $embedding AS embedding, toInteger($limit) AS limit_int
            CALL db.index.vector.queryNodes('norma_embeddings', limit_int, embedding)
            YIELD node AS item, score
            WHERE score >= $threshold AND ('Normografia' IN labels(item) OR 'Norma' IN labels(item))
            RETURN item.id AS id,
                   item.nombre AS nombre,
                   item.texto AS texto,
                   item.tipo AS tipo_norma,
                   score,
                   labels(item) AS etiquetas
            ORDER BY score DESC
            """

            norma_result = await session.run(
                norma_query,
                embedding=[float(x) for x in query_embedding],
                limit=limit,
                threshold=similarity_threshold,
            )

            async for record in norma_result:
                if record["id"] and record["score"] is not None:
                    entities.append({
                        "id": record["id"],
                        "score": float(record["score"]),
                        "tipo": record.get("tipo_norma") or "Normografia",
                        "nombre": record.get("nombre", "Norma sin nombre"),
                        "contenido": record.get("texto", ""),
                        "metadata": {
                            "source_type": "norma",
                            "tipo_norma": record.get("tipo_norma"),
                            "labels": record.get("etiquetas") or [],
                        },
                        "conceptos_clave": [],
                        "normas_citadas": [],
                    })
        except Exception as norma_error:
            logger.warning(f"Error querying norma_embeddings: {norma_error}")

        # Ordenar por score y limitar resultados
        entities.sort(key=lambda x: x["score"], reverse=True)
        return entities[:limit]

    async def _search_legal_relationships(
        self,
        session: AsyncSession,
        query: str,
        query_embedding: List[float],
        limit: int,
        similarity_threshold: float,
    ) -> List[Dict[str, Any]]:
        """
        Search for legal relationships and dependencies related to query.

        Argumentos:
            session: Neo4j async session
            query: Search query
            limit: Maximum results

        Retorna:
            List of legal relationships with context
        """
        # Obtener nodos similares únicamente desde norma_embeddings (normografía)
        node_ids = []

        # Buscar normas
        try:
            norma_query = """
            CALL db.index.vector.queryNodes('norma_embeddings', $limit, $embedding)
            YIELD node, score
            WHERE score >= $threshold AND ('Normografia' IN labels(node) OR 'Norma' IN labels(node))
            RETURN node.id AS id
            LIMIT $limit
            """
            norma_result = await session.run(
                norma_query,
                embedding=[float(x) for x in query_embedding],
                limit=limit,
                threshold=similarity_threshold,
            )
            async for record in norma_result:
                if record["id"]:
                    node_ids.append(record["id"])
        except Exception:
            pass

        if not node_ids:
            return []

        # Expandir relaciones desde los nodos encontrados
        cypher_query = """
        MATCH (node) WHERE node.id IN $node_ids
          AND ('Normografia' IN labels(node) OR 'Norma' IN labels(node))
        MATCH (node)-[r]-(m)
        WHERE 'Normografia' IN labels(m) OR 'Norma' IN labels(m)
        RETURN {
          source: {id: node.id, labels: labels(node),
                   properties: properties(node)},
          relationship: {type: type(r), properties: properties(r)},
          target: {id: m.id, labels: labels(m),
                   properties: properties(m)}
        } AS relationships
        LIMIT $limit
        """

        result = await session.run(
            cypher_query,
            node_ids=node_ids,
            limit=limit,
        )
        records = await result.data()
        return [record["relationships"] for record in records]

    async def _discover_legal_paths(
        self,
        session: AsyncSession,
        query: str,
        query_embedding: List[float],
        limit: int,
        similarity_threshold: float,
    ) -> List[Dict[str, Any]]:
        """
        Discover legal paths and hierarchies related to query.

        Argumentos:
            session: Neo4j async session
            query: Search query
            limit: Maximum results

        Retorna:
            List of legal paths with navigation information
        """
        s = self.settings.databases.neo4j

        # Primero obtener nodos similares
        node_ids = []

        try:
            norma_query = """
            CALL db.index.vector.queryNodes('norma_embeddings', $limit, $embedding)
            YIELD node, score
            WHERE score >= $threshold AND ('Normografia' IN labels(node) OR 'Norma' IN labels(node))
            RETURN node.id AS id
            ORDER BY score DESC
            LIMIT $limit
            """
            norma_result = await session.run(
                norma_query,
                embedding=[float(x) for x in query_embedding],
                limit=limit,
                threshold=similarity_threshold,
            )
            async for record in norma_result:
                if record["id"]:
                    node_ids.append(record["id"])
        except Exception:
            pass

        if not node_ids:
            return []

        # Buscar caminos cortos entre los nodos encontrados
        cypher_query = """
        MATCH (a) WHERE a.id IN $node_ids
          AND ('Normografia' IN labels(a) OR 'Norma' IN labels(a))
        MATCH (b) WHERE b.id IN $node_ids AND a.id < b.id
          AND ('Normografia' IN labels(b) OR 'Norma' IN labels(b))
        MATCH p = shortestPath((a)-[*..3]-(b))
        WHERE p IS NOT NULL
        RETURN {
          path: [n IN nodes(p) | {id: n.id, labels: labels(n),
                                   properties: properties(n)}],
          relationships: [r IN relationships(p) | type(r)],
          length: length(p)
        } AS paths
        LIMIT $limit
        """

        result = await session.run(
            cypher_query,
            node_ids=node_ids[:s.neo4j_paths_top_nodes],
            limit=limit,
        )
        records = await result.data()
        return [record["paths"] for record in records]

    async def _generate_legal_insights(
        self,
        session: AsyncSession,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Generate legal insights based on discovered entities and relationships.

        Argumentos:
            session: Neo4j async session
            entities: Discovered legal entities
            relationships: Discovered legal relationships

        Retorna:
            List of legal insights and recommendations
        """
        insights: List[str] = []
        total_entities = len(entities)
        total_relationships = len(relationships)

        if not (entities or relationships):
            return ["No significant graph context detected for this query."]

        if total_entities:
            insights.append(f"Identified {total_entities} relevant graph entities.")
        if total_relationships:
            insights.append(f"Recorded {total_relationships} total relationships.")

        # Entity label distribution as a proxy for context focus
        entity_labels: Dict[str, int] = {}
        for entity in entities:
            label = (
                entity.get("label")
                or entity.get("properties", {}).get("type")
                or entity.get("tipo")
            )
            if label:
                entity_labels[label] = entity_labels.get(label, 0) + 1
        if entity_labels:
            top_entities = sorted(
                entity_labels.items(), key=lambda item: item[1], reverse=True
            )[:3]
            label_summary = ", ".join([f"{label}: {count}" for label, count in top_entities])
            insights.append(f"Dominant entity labels: {label_summary}.")

        # Relationship type counts and centrality
        relationship_types: Dict[str, int] = {}
        node_degree: Dict[str, int] = {}
        node_context: Dict[str, Dict[str, Any]] = {}
        compliance_indicators: Dict[str, int] = {}

        for relationship in relationships:
            rel_type = relationship.get("relationship", {}).get("type", "unknown")
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
            for role in ("source", "target"):
                node_data = relationship.get(role, {})
                node_id = node_data.get("id")
                if not node_id:
                    continue
                node_degree[node_id] = node_degree.get(node_id, 0) + 1
                if node_id not in node_context:
                    node_context[node_id] = {
                        "labels": node_data.get("labels", []),
                        "score": relationship.get("score", 0.0),
                    }
            # simple compliance heuristics
            if any(
                keyword in rel_type.lower()
                for keyword in ("compliance", "regulation", "regula", "norma")
            ):
                compliance_indicators[rel_type] = (
                    compliance_indicators.get(rel_type, 0) + 1
                )

        if relationship_types:
            top_rels = sorted(
                relationship_types.items(), key=lambda item: item[1], reverse=True
            )[:3]
            rel_summary = ", ".join([f"{rel}: {count}" for rel, count in top_rels])
            insights.append(f"Relationship focus by type: {rel_summary}.")

        if compliance_indicators:
            comp_summary = ", ".join(
                [f"{rel}: {count}" for rel, count in compliance_indicators.items()]
            )
            insights.append(
                f"Regulatory/compliance relationships detected ({comp_summary}); review mandatories."
            )

        if node_degree:
            sorted_nodes = sorted(node_degree.items(), key=lambda item: item[1], reverse=True)
            top_nodes = sorted_nodes[:2]
            for node_id, degree in top_nodes:
                context = node_context.get(node_id, {})
                label = context.get("labels", [])
                label_text = "/".join(label) if label else "node"
                insights.append(
                    f"Node {node_id} ({label_text}) participates in {degree} relationships; central hub in this subgraph."
                )

        # Precedent/authority inference
        precedent_types = [
            rel_type
            for rel_type in relationship_types
            if "precedent" in rel_type.lower() or "juris" in rel_type.lower()
        ]
        if precedent_types:
            insights.append(
                f"Inferred precedent-focused threads for relationships ({', '.join(precedent_types)}); consider applicability."
            )

        return insights

    async def _embed_query(self, text: str) -> List[float]:
        """
        Generate a query embedding compatible with Neo4j vector indexes.

        Uses the configured embedding service to generate semantic embeddings
        for the given text query.

        Argumentos:
            text: Input text to embed

        Retorna:
            Embedding vector as list of floats

        Note:
            Neo4j vector indexes expect specific dimensions. The
            paraphrase-multilingual-MiniLM-L12-v2 model generates 384-dimensional
            embeddings. If your Neo4j indexes expect different dimensions (e.g., 1536),
            you may need to update your index configuration or use a different model.
        """
        try:
            # Generate embedding using the embedding service
            embedding = await self.embedding_service.generate_embedding(
                text, normalize=True
            )

            if not embedding:
                logger.error(f"Empty embedding generated for text: {text[:50]}...")
                # Return zero vector as fallback
                return [0.0] * self.embedding_service.get_embedding_dimension()

            logger.debug(f"Generated embedding of dimension {len(embedding)} for query")
            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding for Neo4j query: {e}")
            # Return zero vector as fallback
            dimension = self.embedding_service.get_embedding_dimension()
            logger.warning(f"Using zero embedding fallback with dimension {dimension}")
            return [0.0] * dimension

    async def search_by_jurisdiction(
        self, query: str, jurisdiction: str, limit: int = 10
    ) -> GraphSearchResponse:
        """
        Search for legal context within a specific jurisdiction.

        Argumentos:
            query: Search query
            jurisdiction: Legal jurisdiction to filter by
            limit: Maximum results

        Retorna:
            GraphSearchResponse filtered by jurisdiction
        """
        start_time = time.time()
        try:
            async with self.driver.session(database=self.database) as session:
                embedding = await self._embed_query(query)
                # Ensure embedding is a list of floats
                embedding = [float(x) for x in embedding]

                # Buscar normas con filtro de jurisdicción
                norma_cypher = """
                CALL db.index.vector.queryNodes('norma_embeddings', $k, $embedding)
                YIELD node, score
                WITH node, score WHERE score >= $threshold
                  AND ('Normografia' IN labels(node) OR 'Norma' IN labels(node)) AND (
                    (exists(node.jurisdiccion) AND toLower(node.jurisdiccion) CONTAINS toLower($jur)) OR
                    (exists(node.jurisdiction) AND toLower(node.jurisdiction) CONTAINS toLower($jur)) OR
                    (exists(node.region) AND toLower(node.region) CONTAINS toLower($jur))
                  )
                RETURN {
                  id: node.id, labels: labels(node), score: score,
                  properties: properties(node)
                } AS entity
                ORDER BY score DESC
                LIMIT $limit
                """

                params = {
                    "embedding": embedding,
                    "k": limit,
                    "threshold": 0.6,
                    "jur": jurisdiction,
                    "limit": limit,
                }

                entities = []

                # Ejecutar búsqueda en normas
                try:
                    norma_recs = await (await session.run(norma_cypher, params)).data()
                    entities.extend([r["entity"] for r in norma_recs])
                except Exception as e:
                    logger.warning(f"Error searching normas by jurisdiction: {e}")

                # Ordenar por score y limitar
                entities.sort(key=lambda x: x.get("score", 0), reverse=True)
                entities = entities[:limit]
                # Expand a few relationships for context
                relationships: List[Dict[str, Any]] = []
                if entities:
                    ids = [e["id"] for e in entities[: min(5, len(entities))]]
                    rels_query = """
                    MATCH (n)-[r]-(m) WHERE n.id IN $ids
                      AND ('Normografia' IN labels(n) OR 'Norma' IN labels(n))
                      AND ('Normografia' IN labels(m) OR 'Norma' IN labels(m))
                    RETURN {
                      source: {id: n.id, labels: labels(n)},
                      relationship: {type: type(r),
                                     properties: properties(r)},
                      target: {id: m.id, labels: labels(m)}
                    } AS rel
                    LIMIT $limit
                    """
                    try:
                        rel_recs = await (
                            await session.run(rels_query, ids=ids, limit=limit)
                        ).data()
                        relationships = [r["rel"] for r in rel_recs]
                    except Exception as e:
                        logger.warning(f"Error expanding relationships: {e}")

                # Convertir entidades a GraphSearchResult
                results = []
                for entity in entities:
                    if isinstance(entity, dict):
                        props = entity.get("properties", {}) if isinstance(entity.get("properties"), dict) else {}
                        result = GraphSearchResult(
                            id=entity.get("id", ""),
                            tipo=props.get("tipo", ""),
                            nombre=props.get("nombre", ""),
                            texto_relevante=str(props.get("texto", ""))[:500],
                            score=entity.get("score", 0.0),
                            metadata=entity.get("metadata", {}),
                        )
                        results.append(result)

                return GraphSearchResponse(
                    results=results,
                    total_found=len(results),
                    execution_time=(time.time() - start_time),
                    search_strategy="jurisdiction_filter",
                    metadata={
                        "jurisdiction": jurisdiction,
                        "entities_found": len(entities),
                        "relationships_found": len(relationships),
                    },
                )
        except Exception as e:
            logger.error(f"Jurisdiction search failed: {e}")
            return GraphSearchResponse(
                results=[],
                total_found=0,
                execution_time=(time.time() - start_time),
                search_strategy="jurisdiction_error",
                metadata={"error": str(e), "jurisdiction": jurisdiction},
            )

    async def search_by_legal_domain(
        self, query: str, domain: str, limit: int = 10
    ) -> GraphSearchResponse:
        """
        Search for legal context within a specific legal domain.

        Argumentos:
            query: Search query
            domain: Legal domain (civil, criminal, administrative, etc.)
            limit: Maximum results

        Retorna:
            GraphSearchResponse filtered by legal domain
        """
        start_time = time.time()
        try:
            async with self.driver.session(database=self.database) as session:
                embedding = await self._embed_query(query)
                # Ensure embedding is a list of floats
                embedding = [float(x) for x in embedding]

                # Buscar normas con filtro de dominio
                norma_cypher = """
                CALL db.index.vector.queryNodes('norma_embeddings', $k, $embedding)
                YIELD node, score
                WITH node, score WHERE score >= $threshold
                  AND ('Normografia' IN labels(node) OR 'Norma' IN labels(node)) AND (
                    (exists(node.dominio) AND toLower(node.dominio) CONTAINS toLower($dom)) OR
                    (exists(node.domain) AND toLower(node.domain) CONTAINS toLower($dom)) OR
                    (exists(node.tipo) AND toLower(node.tipo) CONTAINS toLower($dom)) OR
                    (exists(node.materia) AND toLower(node.materia) CONTAINS toLower($dom))
                  )
                RETURN {
                  id: node.id, labels: labels(node), score: score,
                  properties: properties(node)
                } AS entity
                ORDER BY score DESC
                LIMIT $limit
                """

                params = {
                    "embedding": embedding,
                    "k": limit,
                    "threshold": 0.6,
                    "dom": domain,
                    "limit": limit,
                }

                entities = []

                # Ejecutar búsqueda en normas
                try:
                    norma_recs = await (await session.run(norma_cypher, params)).data()
                    entities.extend([r["entity"] for r in norma_recs])
                except Exception as e:
                    logger.warning(f"Error searching normas by domain: {e}")

                # Ordenar por score y limitar
                entities.sort(key=lambda x: x.get("score", 0), reverse=True)
                entities = entities[:limit]

                # Convertir entidades a GraphSearchResult
                results = []
                for entity in entities:
                    if isinstance(entity, dict):
                        props = entity.get("properties", {}) if isinstance(entity.get("properties"), dict) else {}
                        result = GraphSearchResult(
                            id=entity.get("id", ""),
                            tipo=props.get("tipo", ""),
                            nombre=props.get("nombre", ""),
                            texto_relevante=str(props.get("texto", ""))[:500],
                            score=entity.get("score", 0.0),
                            metadata=entity.get("metadata", {}),
                        )
                        results.append(result)

                return GraphSearchResponse(
                    results=results,
                    total_found=len(results),
                    execution_time=(time.time() - start_time),
                    search_strategy="legal_domain_filter",
                    metadata={"domain": domain, "entities_found": len(entities)},
                )
        except Exception as e:
            logger.error(f"Domain search failed: {e}")
            return GraphSearchResponse(
                results=[],
                total_found=0,
                execution_time=(time.time() - start_time),
                search_strategy="legal_domain_error",
                metadata={"error": str(e), "domain": domain},
            )

    async def get_legal_precedents(
        self, query: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find legal precedents and jurisprudence related to query.

        Argumentos:
            query: Search query
            limit: Maximum results

        Retorna:
            List of legal precedents with citation information
        """
        try:
            async with self.driver.session(database=self.database) as session:
                embedding = await self._embed_query(query)
                # Ensure embedding is a list of floats
                embedding = [float(x) for x in embedding]

                # Buscar normas que sean precedentes
                norma_cypher = """
                CALL db.index.vector.queryNodes('norma_embeddings', $limit, $embedding)
                YIELD node, score
                WITH node, score WHERE score >= $threshold
                  AND ('Normografia' IN labels(node) OR 'Norma' IN labels(node)) AND (
                    (exists(node.tipo) AND toLower(node.tipo) CONTAINS 'sentencia') OR
                    (exists(node.tipo) AND toLower(node.tipo) CONTAINS 'jurisprud') OR
                    (exists(node.category) AND toLower(node.category) CONTAINS 'precedent')
                  )
                RETURN {
                  id: node.id, labels: labels(node), score: score,
                  properties: properties(node)
                } AS precedent
                ORDER BY score DESC
                LIMIT $limit
                """

                params = {"embedding": embedding, "limit": limit, "threshold": 0.6}

                precedents = []

                # Ejecutar búsqueda en normas
                try:
                    norma_recs = await (await session.run(norma_cypher, params)).data()
                    precedents.extend([r["precedent"] for r in norma_recs])
                except Exception as e:
                    logger.warning(f"Error searching norma precedents: {e}")

                # Ordenar por score y limitar
                precedents.sort(key=lambda x: x.get("score", 0), reverse=True)
                return precedents[:limit]
        except Exception as e:
            logger.error(f"Precedent search failed: {e}")
            return []

    async def search_similar_chunks(
        self, query_text: str, limit: int = 10, similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar normas based on text similarity using embeddings.

        Argumentos:
            query_text: Text to search for similar normas
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score

        Retorna:
            List of similar normas with metadata
        """
        try:
            query_embedding = await self._embed_query(query_text)
            # Ensure embedding is a list of floats
            query_embedding = [float(x) for x in query_embedding]

            async with self.driver.session(database=self.database) as session:
                # Search for similar normas using vector similarity
                cypher_query = """
                MATCH (n)
                WHERE ('Normografia' IN labels(n) OR 'Norma' IN labels(n)) AND n.embedding IS NOT NULL
                WITH n, gds.similarity.cosine(n.embedding, $query_embedding) AS similarity
                WHERE similarity >= $threshold
                RETURN n.id as norma_id,
                       n.nombre as nombre,
                       n.texto as texto,
                       n.tipo as tipo,
                       similarity
                ORDER BY similarity DESC
                LIMIT $limit
                """

                result = await session.run(
                    cypher_query,
                    {
                        "query_embedding": query_embedding,
                        "threshold": similarity_threshold,
                        "limit": limit,
                    },
                )

                chunks = []
                async for record in result:
                    chunk = {
                        "norma_id": record["norma_id"],
                        "nombre": record["nombre"],
                        "texto": record["texto"],
                        "tipo": record["tipo"],
                        "similarity_score": record["similarity"],
                    }
                    chunks.append(chunk)

                logger.info(f"Found {len(chunks)} similar normas for query")
                return chunks

        except Exception as e:
            logger.error(f"Error in similar normas search: {e}")
            return []

    async def search_by_legal_entities(
        self, entity_names: List[str], limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for normas related to specific legal entities.

        Argumentos:
            entity_names: List of legal entity names to search for
            limit: Maximum number of results

        Retorna:
            List of normas related to the legal entities
        """
        if not entity_names:
            return []

        try:
            async with self.driver.session(database=self.database) as session:
                # Search for normas that mention legal entities, norms, or concepts
                cypher_query = """
                MATCH (n)
                WHERE ('Normografia' IN labels(n) OR 'Norma' IN labels(n))
                  AND ANY(entity IN $entity_names WHERE
                    (exists(n.nombre) AND n.nombre CONTAINS entity) OR
                    (exists(n.texto) AND n.texto CONTAINS entity)
                  )
                WITH n,
                     SIZE([entity IN $entity_names WHERE
                           (exists(n.nombre) AND n.nombre CONTAINS entity) OR
                           (exists(n.texto) AND n.texto CONTAINS entity)]) as relevance_score
                RETURN n.id as norma_id,
                       n.nombre as nombre,
                       n.texto as texto,
                       n.tipo as tipo,
                       relevance_score
                ORDER BY relevance_score DESC
                LIMIT $limit
                """

                result = await session.run(
                    cypher_query, {"entity_names": entity_names, "limit": limit}
                )

                chunks = []
                async for record in result:
                    chunk = {
                        "norma_id": record["norma_id"],
                        "nombre": record["nombre"],
                        "texto": record["texto"],
                        "tipo": record["tipo"],
                        "relevance_score": record["relevance_score"],
                    }
                    chunks.append(chunk)

                logger.info(
                    f"Found {len(chunks)} normas for legal entities: {entity_names}"
                )
                return chunks

        except Exception as e:
            logger.error(f"Error in legal entities search: {e}")
            return []
