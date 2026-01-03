# MVP Legal Backend

Servicios FastAPI para chat con streaming SSE y flujos de generacion/orquestacion multi-agente.

## Arquitectura (resumen)

- **LangGraph**: grafo principal (`generation_graph`) orquesta nodos `orchestrator_plan`, `pgvector`, `neo4j`, `generation` y `finalize`.
- **Agentes**:
  - **OrchestratorAgent**: valida estado, aplica directivas y enruta; en `chat_edit` exige que el cliente indique la operacion (`chat` o `edit`).
  - **GenerationAgent**: llama al LLM (via `OllamaService`) con prompts distintos para `chat`, `edit`, `compose`.
  - **PgVectorAgent**: persiste/recupera contexto conversacional y busqueda semantica.
  - **Neo4jAgent**: analisis de grafo/legal (cuando hay driver configurado).
- **Servicios soporte**: `EmbeddingService`, `OllamaService`, y los settings en `app/config`.

## Endpoints principales (segun `src/app/api/v1/routers`)

### Conversaciones (`/generation`)

1) **POST `/generation/tutela`** Flujo completo (no SSE) para procesar un texto de tutela.

   - **Body**:
     ```json
     {
       "tutela_text": "texto completo...",
       "first_interaction": true,
       "conversation_id": "opcional-uuid",
     }
     ```
   - **Respuesta (JSON)**:
     ```json
     {
       "status": "success",
       "conversation_id": "uuid",
       "final_response": {
         "content": "respuesta generada"
       },
       "pgvector_inserted": true,
       "pertinence_valid": true,
       "neo4j_context": {
         "results": [],
         "total_found": 0,
         "search_strategy": "unknown",
         "metadata": {},
         "execution_time": 0.0
       },
       "execution_time": 12.34
     }
     ```
2) **POST `/generation/chat/stream`** Chat SSE. El cliente **elige** el modo (chat vs edit); no hay auto-clasificacion.

   - **Body**:
     ```json
     {
       "conversation_id": "uuid existente",
       "message": "texto del usuario",
       "mode": "chat" | "edit"  // tambien acepta "edith" y lo normaliza a "edit"
     }
     ```
   - **Flags/parametros**:
     - `mode`: `chat` -> operacion `chat` (`goal: direct_answer`); `edit` -> operacion `edit` (`goal: edit_draft`).
     - El grafo puede usar PgVector (memoria) y Neo4j (contexto legal) segun disponibilidad.
     - Eventos SSE: `start`, `orchestrator`, `node`, `complete`, `warn`, `error`.
   - **Respuesta (SSE)**:
     - Stream `text/event-stream` con eventos `start`, `orchestrator`, `node`, `complete`, `warn`, `error`.
     - Cada `data:` es JSON; el evento final `complete` incluye:
       ```json
       {
         "status": "success",
         "conversation_id": "uuid",
         "final_response": {
           "content": "respuesta generada"
         },
         "execution_time": 8.7,
         "render_mode": "chat_message",
         "message_kind": "chat_response",
         "workflow_summary": {
           "goal": "direct_answer",
           "flow_mode": "chat_edit",
           "operation": "chat",
           "context_sources_used": ["pgvector"],
           "uses_neo4j": false,
           "uses_pgvector": true,
           "response_type": "conversational"
         }
       }
       ```
   - **Consumo SSE (curl)**:
     ```bash
     curl -N -H "Content-Type: application/json" \
       -X POST http://localhost:8000/generation/chat/stream \
       -d '{"conversation_id":"<uuid>","message":"Hola","mode":"chat"}'
     ```

     Use un cliente SSE para parsear eventos (`event:` + `data:`). Cada `data` es JSON.

### Tutelas (`/tutelas`)

1) **GET `/tutelas/{tutela_id}/conversation-id`** Genera o retorna el `conversation_id` estable (UUID v5) para una tutela externa.

   - **Path params**:
     - `tutela_id`: string (cualquier identificador externo de tutela).
   - **Respuesta (JSON)**:
     ```json
     {
       "tutela_id": "id-externo",
       "conversation_id": "uuid",
       "created_at": "2025-12-19T01:02:03.456789"
     }
     ```
   - **Notas**:
     - Si no existia, se crea un registro en `app_conversations` con `metadata.source = "tutelas_router"`.

2) **POST `/tutelas/upload-pdf`** Sube un PDF y extrae texto con PyMuPDF.

   - **Body (multipart/form-data)**:
     - `file`: archivo PDF (maximo 10MB).
   - **Respuesta (JSON)**:
     ```json
     {
       "success": true,
       "extracted_text": "texto extraido",
       "filename": "archivo.pdf",
       "file_size": 12345,
       "pages_count": 3,
       "message": "Texto extraido correctamente"
     }
     ```
   - **Errores**:
     - `400`: archivo no PDF, sin texto o demasiado grande.
     - `500`: error interno al procesar el PDF.

### Core (`/health`)

- **GET `/health/`**, **`/health/ready`**, **`/health/live`**: checks basicos para liveness/readiness/health.

### Datos (`/rag`)

- Endpoints de RAG/normografia para carga y procesamiento de documentos legales.

1) **POST `/rag/normography/process-document`** Procesa un documento normativo por ruta en disco.

   - **Body**:
     ```json
     {
       "file_path": "C:\\ruta\\archivo.pdf",
       "section_path": ["TITULO I", "CAPITULO II"]
     }
     ```
   - **Respuesta (JSON)**:
     ```json
     {
       "success": true,
       "section_content": "markdown extraido",
       "processing_stats": {
         "file_path": "C:\\ruta\\archivo.pdf",
         "section_path": ["TITULO I", "CAPITULO II"],
         "content_length": 1234
       }
     }
     ```

2) **POST `/rag/normography/update-section`** Actualiza una seccion especifica en Neo4j.

   - **Body**:
     ```json
     {
       "category": "Derecho laboral",
       "law_name": "Ley 100 de 1993",
       "path": ["TITULO I", "CAPITULO II"],
       "markdown_content": "### Articulo 1\nTexto..."
     }
     ```
   - **Respuesta (JSON)**:
     ```json
     {
       "success": true,
       "nodes_created": 10,
       "relations_created": 9
     }
     ```

3) **POST `/rag/normography/bulk-load`** Carga masiva desde Markdown.

   - **Body**:
     ```json
     {
       "file_path": "C:\\ruta\\normas.md",
       "skip_normalization": false
     }
     ```
   - **Respuesta (JSON)**:
     ```json
     {
       "success": true,
       "nodes_processed": 1200,
       "relations_created": 1180,
       "processing_time": 12.34,
       "normalized_file_path": "C:\\ruta\\normas_normalizado.md"
     }
     ```

4) **POST `/rag/normography/process-pdf-and-upload`** Procesa un PDF/DOCX y lo carga en Neo4j.

   - **Body (multipart/form-data)**:
     - `file`: archivo PDF/DOCX/DOC
     - `category`: string
     - `law_name`: string
     - `section_path`: JSON string (ej. `["TITULO I","CAPITULO II"]`)
   - **Respuesta (JSON)**:
     ```json
     {
       "success": true,
       "nodes_created": 120,
       "relations_created": 115
     }
     ```

5) **POST `/rag/normography/sync-to-postgres`** Sincroniza Neo4j -> PostgreSQL.

   - **Body**: ninguno
   - **Respuesta (JSON)**:
     ```json
     {
       "success": true,
       "message": "Sincronizacion completada exitosamente",
       "nodes_synced": 1200,
       "edges_synced": 1180
     }
     ```

6) **POST `/rag/normography/migrate-section-labels`** Migra etiquetas genericas de seccion.

   - **Body**: ninguno
   - **Respuesta (JSON)**:
     ```json
     {
       "success": true,
       "message": "Migracion completada exitosamente",
       "stats": {
         "titulo_migrated": 10,
         "capitulo_migrated": 8,
         "libro_migrated": 2,
         "parte_migrated": 1,
         "total_migrated": 21,
         "errors": 0
       }
     }
     ```

7) **POST `/rag/normography/regenerate-embeddings`** Regenera embeddings en Neo4j.

   - **Body**: ninguno
   - **Respuesta (JSON)**:
     ```json
     {
       "success": true,
       "message": "Embeddings regenerados exitosamente usando EmbeddingService centralizado",
       "stats": {
         "processed": 1000,
         "updated": 990,
         "errors": 10
      }
    }
    ```

8) **POST `/rag/neo4j/search`** Busqueda semantica en Neo4j.

   - **Body**:
     ```json
     {
       "query": "consulta legal",
       "limit": 10,
       "similarity_threshold": 0.7
     }
     ```
   - **Respuesta (JSON)**:
     ```json
     {
       "results": {
         "results": [],
         "total_found": 0,
         "execution_time": 0.12,
         "search_strategy": "normografia_vector",
         "metadata": {}
       }
     }
     ```

9) **POST `/rag/neo4j/generate`** Generacion RAG con contexto Neo4j.

   - **Body**:
     ```json
     {
       "query": "consulta legal",
       "limit": 10,
       "similarity_threshold": 0.7,
       "max_tokens": 1200,
       "temperature": 0.7,
       "model_preference": "auto"
     }
     ```
   - **Respuesta (JSON)**:
     ```json
     {
       "text": "respuesta generada",
       "context_summary": {
         "entities": 3,
         "has_relationships": true
      }
    }
    ```

### Normografia (`/normography`)

1) **GET `/normography/laws/graph`** Obtiene nodos y relaciones del grafo legal.

   - **Query params**:
     - `limit`: int (default 500)
     - `offset`: int (default 0)
     - `edge_limit`: int (default 5000)
     - `edge_offset`: int (default 0)
     - `include_disabled`: bool (default true)
   - **Respuesta (JSON)**:
     ```json
     {
       "nodes": [
         {
           "node_uid": "uuid",
           "element_id": "elementId(n)",
           "label": "Ley",
           "graph_id": "ley_100_de_1993",
           "nombre": "Ley 100 de 1993",
           "texto": "texto...",
           "tipo": "Ley",
           "enabled": true
         }
       ],
       "edges": [
         {
           "edge_id": 1,
           "rel_type": "PERTENECE_A",
           "start_node": "uuid",
           "end_node": "uuid",
           "enabled": true
         }
       ],
       "meta": {
         "count_nodes": 1,
         "count_edges": 1,
         "include_disabled": true,
         "limits": {
           "nodes": 500,
           "edges": 5000
         }
       }
     }
     ```

2) **PATCH `/normography/laws/{node_uid}`** Actualiza una ley en Neo4j y PostgreSQL.

   - **Path params**:
     - `node_uid`: UUID del nodo
   - **Body**:
     ```json
     {
       "nombre": "Ley 100 de 1993 (actualizada)",
       "texto": "texto actualizado...",
       "tipo": "Ley",
       "source": "fuente",
       "tematica": "salud",
       "resumen_tematica": "resumen",
       "enabled": true,
       "properties": {
         "origen": "manual"
       }
     }
     ```
   - **Respuesta (JSON)**:
     ```json
     {
       "success": true,
       "node_uid": "uuid",
       "element_id": "elementId(n)",
       "message": "Ley actualizada exitosamente en Neo4j y PostgreSQL"
     }
     ```

## Notas de uso

- `mode` es obligatorio en `chat/stream` para flujos `chat_edit`; si falta, el orquestador devuelve error de directiva.
- El `conversation_id` se normaliza a UUID si viene vacio o en otro formato.
- El backend asume servicios de embeddings y, opcionalmente, Neo4j configurados en `Settings`.

## Ejemplo rapido (Python SSE)

```python
import httpx, sseclient

payload = {"conversation_id": "123e4567-e89b-12d3-a456-426614174000",
           "message": "Necesito editar el borrador",
           "mode": "edit"}

with httpx.stream("POST", "http://localhost:8000/generation/chat/stream", json=payload) as r:
    client = sseclient.SSEClient(r.iter_text())
    for event in client.events():
        print(event.event, event.data)
```
