"""
RAGSmith – Documents router
Handles file upload and triggers the async processing pipeline.
"""

import os
import logging
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, status

from database import get_connection
from models.schemas import DocumentResponse
from services.processor import extract_text, chunk_text, build_or_update_index

# The embedding model is always all-MiniLM-L6-v2 regardless of which LLM the project uses
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

router = APIRouter()
logger = logging.getLogger("ragsmith.documents")

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".doc", ".rst", ".csv"}
MAX_FILE_SIZE_MB = 50


def _get_project_or_404(conn, project_id: int):
    row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")
    return row


def _process_document(doc_id: int, project_id: int, file_bytes: bytes, filename: str, model: str):
    """Background task: extract → chunk → embed → index.
    NOTE: `model` is the project's LLM model (e.g. 'mistral'). Embeddings always
    use EMBEDDING_MODEL ('all-MiniLM-L6-v2'), which is a dedicated sentence-transformer.
    """
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE documents SET status = 'processing' WHERE id = ?", (doc_id,)
        )
        conn.commit()

        # Extract text
        text = extract_text(file_bytes, filename)
        if not text:
            raise ValueError("No text could be extracted from the document.")

        # Chunk
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("Document produced no chunks after processing.")

        # Embed & index — always use the sentence-transformer, NOT the LLM model name
        num_indexed = build_or_update_index(
            project_id=project_id,
            new_chunks=chunks,
            doc_id=doc_id,
            filename=filename,
            embedding_model=EMBEDDING_MODEL,
        )

        conn.execute(
            "UPDATE documents SET status = 'ready', num_chunks = ? WHERE id = ?",
            (num_indexed, doc_id),
        )
        conn.commit()
        logger.info("Document %d processed: %d chunks indexed", doc_id, num_indexed)

    except Exception as exc:
        logger.exception("Failed to process document %d: %s", doc_id, exc)
        conn.execute(
            "UPDATE documents SET status = 'error', error_msg = ? WHERE id = ?",
            (str(exc)[:500], doc_id),
        )
        conn.commit()
    finally:
        conn.close()


@router.get("/{project_id}", response_model=List[DocumentResponse])
def list_documents(project_id: int):
    conn = get_connection()
    try:
        _get_project_or_404(conn, project_id)
        rows = conn.execute(
            "SELECT * FROM documents WHERE project_id = ? ORDER BY created_at DESC",
            (project_id,),
        ).fetchall()
        return [
            DocumentResponse(
                id=r["id"],
                project_id=r["project_id"],
                filename=r["filename"],
                num_chunks=r["num_chunks"],
                status=r["status"],
                error_msg=r["error_msg"],
                created_at=r["created_at"],
            )
            for r in rows
        ]
    finally:
        conn.close()


@router.post("/{project_id}/upload", response_model=DocumentResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    project_id: int,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    import pathlib

    ext = pathlib.Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {MAX_FILE_SIZE_MB} MB limit.",
        )

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    conn = get_connection()
    try:
        project = _get_project_or_404(conn, project_id)

        # Persist upload
        upload_dir = f"data/uploads/{project_id}"
        os.makedirs(upload_dir, exist_ok=True)
        safe_filename = file.filename.replace("/", "_").replace("\\", "_")
        file_path = os.path.join(upload_dir, safe_filename)
        with open(file_path, "wb") as f_out:
            f_out.write(file_bytes)

        cur = conn.execute(
            "INSERT INTO documents (project_id, filename, file_path, status) VALUES (?, ?, ?, 'pending')",
            (project_id, safe_filename, file_path),
        )
        conn.commit()
        doc_id = cur.lastrowid

        # Queue background processing
        background_tasks.add_task(
            _process_document,
            doc_id=doc_id,
            project_id=project_id,
            file_bytes=file_bytes,
            filename=safe_filename,
            model=project["model"],
        )

        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return DocumentResponse(
            id=row["id"],
            project_id=row["project_id"],
            filename=row["filename"],
            num_chunks=row["num_chunks"],
            status=row["status"],
            error_msg=row["error_msg"],
            created_at=row["created_at"],
        )
    finally:
        conn.close()


@router.get("/{project_id}/doc/{doc_id}", response_model=DocumentResponse)
def get_document_status(project_id: int, doc_id: int):
    conn = get_connection()
    try:
        _get_project_or_404(conn, project_id)
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ? AND project_id = ?",
            (doc_id, project_id),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")
        return DocumentResponse(
            id=row["id"],
            project_id=row["project_id"],
            filename=row["filename"],
            num_chunks=row["num_chunks"],
            status=row["status"],
            error_msg=row["error_msg"],
            created_at=row["created_at"],
        )
    finally:
        conn.close()


@router.delete("/{project_id}/doc/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(project_id: int, doc_id: int):
    """
    Removes document record. Note: FAISS index rebuild is required
    for full removal from the vector store (scheduled for v2).
    """
    conn = get_connection()
    try:
        _get_project_or_404(conn, project_id)
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ? AND project_id = ?",
            (doc_id, project_id),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")

        # Remove uploaded file
        if row["file_path"] and os.path.exists(row["file_path"]):
            os.remove(row["file_path"])

        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        logger.info("Deleted document id=%d from project id=%d", doc_id, project_id)
    finally:
        conn.close()


@router.post("/{project_id}/doc/{doc_id}/retry", response_model=DocumentResponse)
def retry_document(project_id: int, doc_id: int, background_tasks: BackgroundTasks):
    """Re-process a document that previously failed (status='error')."""
    conn = get_connection()
    try:
        project = _get_project_or_404(conn, project_id)
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ? AND project_id = ?",
            (doc_id, project_id),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")
        if row["status"] not in ("error", "pending"):
            raise HTTPException(
                status_code=409,
                detail=f"Document status is '{row['status']}', only 'error' or 'pending' documents can be retried.",
            )

        file_path = row["file_path"]
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail="Original file not found on disk. Please re-upload the document.",
            )

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        # Reset status
        conn.execute(
            "UPDATE documents SET status='pending', error_msg=NULL, num_chunks=0 WHERE id=?",
            (doc_id,),
        )
        conn.commit()

        background_tasks.add_task(
            _process_document,
            doc_id=doc_id,
            project_id=project_id,
            file_bytes=file_bytes,
            filename=row["filename"],
            model=project["model"],
        )

        updated = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return DocumentResponse(
            id=updated["id"],
            project_id=updated["project_id"],
            filename=updated["filename"],
            num_chunks=updated["num_chunks"],
            status=updated["status"],
            error_msg=updated["error_msg"],
            created_at=updated["created_at"],
        )
    finally:
        conn.close()
