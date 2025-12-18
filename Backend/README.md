# MVP Legal Backend

Servicios FastAPI para chat con streaming SSE y flujos de generación/orquestación multi‑agente.

## Arquitectura (resumen)

- **LangGraph**: grafo principal (`generation_graph`) orquesta nodos `orchestrator_plan`, `pgvector`, `neo4j`, `generation` y `finalize`.
- **Agentes**:
  - **OrchestratorAgent**: valida estado, aplica directivas y enruta; en `chat_edit` exige que el cliente indique la operación (`chat` o `edit`).
  - **GenerationAgent**: llama al LLM (vía `OllamaService`) con prompts distintos para `chat`, `edit`, `compose`.
  - **PgVectorAgent**: persiste/recupera contexto conversacional y búsqueda semántica.
  - **Neo4jAgent**: análisis de grafo/legal (cuando hay driver configurado).
- **Servicios soporte**: `EmbeddingService`, `OllamaService`, y los settings en `app/config`.

## Endpoints principales (según `src/app/api/v1/routers`)

### Conversaciones (`/generation`)

1) **POST `/generation/tutela`**Flujo completo (no SSE) para procesar un texto de tutela.

   - **Body**:
     ```json
     {
       "tutela_text": "texto completo...",
       "first_interaction": true,
       "conversation_id": "opcional-uuid",
       "registraduria_role": "opcional"
     }
     ```
   - **Respuesta**: `TutelaResponse` con `conversation_id`, `final_response`, `pgvector_inserted`, `pertinence_valid`, `neo4j_context`, `execution_time`.
2) **POST `/generation/chat/stream`**Chat SSE. El cliente **elige** el modo (chat vs edit); no hay auto‑clasificación.

   - **Body**:
     ```json
     {
       "conversation_id": "uuid existente",
       "message": "texto del usuario",
       "mode": "chat" | "edit"  // también acepta "edith" y lo normaliza a "edit"
     }
     ```
   - **Flags/parámetros**:
     - `mode`: `chat` → operación `chat` (`goal: direct_answer`); `edit` → operación `edit` (`goal: edit_draft`).
     - El grafo puede usar PgVector (memoria) y Neo4j (contexto legal) según disponibilidad.
     - Eventos SSE: `start`, `orchestrator`, `node`, `complete`, `warn`, `error`.
   - **Consumo SSE (curl)**:
     ```bash
     curl -N -H "Content-Type: application/json" \
       -X POST http://localhost:8000/generation/chat/stream \
       -d '{"conversation_id":"<uuid>","message":"Hola","mode":"chat"}'
     ```

     Use un cliente SSE para parsear eventos (`event:` + `data:`). Cada `data` es JSON.

### Core (`/health`)

- **GET `/health/`**, **`/health/ready`**, **`/health/live`**: checks básicos para liveness/readiness/health.

### Datos (`/rag`)

- Endpoints de RAG/normografía para carga y procesamiento de documentos legales (bulk load). Ver `src/app/api/v1/routers/data/rag_router.py` para los payloads específicos (incluye clases `BulkLoadRequest`, `DocumentProcessRequest`, etc.).

## Notas de uso

- `mode` es obligatorio en `chat/stream` para flujos `chat_edit`; si falta, el orquestador devuelve error de directiva.
- El `conversation_id` se normaliza a UUID si viene vacío o en otro formato.
- El backend asume servicios de embeddings y, opcionalmente, Neo4j configurados en `Settings`.

## Ejemplo rápido (Python SSE)

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
