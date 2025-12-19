"""
Router minimal de tutelas para generar/consultar conversation_id.

Este router evita dependencias del dominio de tutelas eliminado y solo crea
registros en app_conversations usando un UUID estable (v5) derivado del tutela_id.
"""

import json
import uuid as _uuid
from datetime import datetime
from typing import Any, Dict

import fitz
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config.databases import get_postgres_session
from app.extensions import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/tutelas", tags=["tutelas"])


@router.get("/health")
def tutelas_health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "tutelas",
        "timestamp": datetime.utcnow().isoformat(),
    }


def _ensure_conversation(
    db: Session,
    conversation_id: str,
    *,
    title: str,
    metadata: Dict[str, Any],
) -> None:
    insert_stmt = text(
        """
        INSERT INTO app_conversations (
            id, user_id, title, status, started_at, updated_at, metadata
        )
        VALUES (:id, NULL, :title, 'open', now(), now(), CAST(:metadata AS jsonb))
        ON CONFLICT (id) DO NOTHING
        """
    )
    db.execute(
        insert_stmt,
        {
            "id": conversation_id,
            "title": title,
            "metadata": json.dumps(metadata or {}),
        },
    )
    db.commit()


@router.get("/{tutela_id}/conversation-id")
def get_tutela_conversation_id(
    tutela_id: str,
    request: Request,
    db: Session = Depends(get_postgres_session),
):
    """
    Genera o retorna conversation_id para una tutela externa.

    - conversation_id = UUID v5 estable basado en tutela_id
    - se crea un registro en app_conversations si no existe
    """
    try:
        conversation_id = str(_uuid.uuid5(_uuid.NAMESPACE_URL, str(tutela_id)))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="tutela_id invalido",
        )

    exists = db.execute(
        text("SELECT id, started_at FROM app_conversations WHERE id = :cid"),
        {"cid": conversation_id},
    ).fetchone()

    if not exists:
        _ensure_conversation(
            db,
            conversation_id,
            title=f"Tutela {tutela_id}",
            metadata={
                "source": "tutelas_router",
                "external_tutela_id": str(tutela_id),
                "client": request.client.host if request.client else None,
            },
        )
        created_at = None
    else:
        created_at = exists[1]

    return {
        "tutela_id": str(tutela_id),
        "conversation_id": conversation_id,
        "created_at": created_at.isoformat() if created_at else None,
    }


@router.post("/upload-pdf")
async def upload_pdf_and_extract_text(
    file: UploadFile = File(...),
):
    """
    Sube un archivo PDF y extrae su contenido textual.

    Procesa un archivo PDF subido por el usuario y extrae todo el texto contenido
    en el documento. Valida el formato del archivo y su tamano maximo (10MB).
    El texto extraido puede ser usado posteriormente para crear o actualizar tutelas.

    **Argumentos**:
    - file: Archivo PDF a procesar (maximo 10MB)

    **Retorna**:
    - Diccionario con texto extraido, informacion del archivo y estadisticas

    **Codigos de Estado**:
    - 200: PDF procesado exitosamente
    - 400: Archivo invalido, sin texto o excede tamano maximo
    - 500: Error al procesar el PDF

    **Notas**:
    - Solo se aceptan archivos PDF
    - El tamano maximo es 10MB
    - El PDF debe contener texto legible (no solo imagenes)
    """
    try:
        # Validar que el archivo es PDF
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")

        # Validar tamano maximo (10 MB)
        file_content = await file.read()
        if len(file_content) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="El tamano del archivo debe ser inferior a 10 MB",
            )

        # Extraer contenido textual del PDF
        try:
            pdf_doc = fitz.open(stream=file_content, filetype="pdf")
            extracted_text = ""

            for page in pdf_doc:
                extracted_text += (page.get_text() or "") + "\n"

            # Limpiar el texto obtenido
            extracted_text = extracted_text.strip()
            pages_count = pdf_doc.page_count
            pdf_doc.close()

            if not extracted_text:
                raise HTTPException(
                    status_code=400, detail="No fue posible extraer texto del PDF"
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"No se pudo extraer texto del PDF: {e}")
            raise HTTPException(
                status_code=400,
                detail="Error al extraer texto del PDF. Verifica que el documento contenga texto legible.",
            )

        return {
            "success": True,
            "extracted_text": extracted_text,
            "filename": file.filename,
            "file_size": len(file_content),
            "pages_count": pages_count,
            "message": "Texto extraido correctamente",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ocurrio un error al procesar la carga del PDF: {e}")
        raise HTTPException(status_code=500, detail="Error interno al procesar el PDF")
