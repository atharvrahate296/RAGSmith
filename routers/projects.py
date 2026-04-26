"""
RAGSmith – Projects router
CRUD for RAG projects. Uses db helpers for SQLite/Postgres compatibility.
"""

import logging
from typing import List

from fastapi import APIRouter, HTTPException, status

from database import get_connection, db_execute, db_fetchone, db_fetchall, db_insert, ph
from models.schemas import ProjectCreate, ProjectUpdate, ProjectResponse
from services.processor import delete_project_index

router = APIRouter()
logger = logging.getLogger("ragsmith.projects")


def _to_response(row: dict) -> ProjectResponse:
    return ProjectResponse(
        id=row["id"],
        name=row["name"],
        description=row.get("description") or "",
        provider=row.get("provider") or "ollama",
        model=row["model"],
        top_k=row["top_k"],
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        document_count=row.get("doc_count", 0),
    )


@router.get("/", response_model=List[ProjectResponse])
def list_projects():
    conn = get_connection()
    try:
        rows = db_fetchall(conn, """
            SELECT p.*, COUNT(d.id) as doc_count
            FROM projects p
            LEFT JOIN documents d ON d.project_id = p.id
            GROUP BY p.id
            ORDER BY p.created_at DESC
        """)
        return [_to_response(r) for r in rows]
    finally:
        conn.close()


@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
def create_project(body: ProjectCreate):
    conn = get_connection()
    try:
        from config import get_settings
        cfg = get_settings()
        
        # Determine default model based on provider
        if body.provider == "groq":
            default_model = "gemma:2b" # fallback
        else:
            default_model = "mistral:7b" # fallback

        existing = db_fetchone(conn, f"SELECT id FROM projects WHERE name={ph()}", (body.name,))
        if existing:
            raise HTTPException(status_code=409, detail=f"Project '{body.name}' already exists.")

        effective_model = body.model or default_model
        doc_id = db_insert(conn,
            f"INSERT INTO projects (name, description, provider, model, top_k) VALUES ({ph()},{ph()},{ph()},{ph()},{ph()})",
            (body.name, body.description or "", body.provider or "ollama", effective_model, body.top_k or 5))

        row = db_fetchone(conn, f"SELECT *, 0 as doc_count FROM projects WHERE id={ph()}", (doc_id,))
        logger.info("Created project '%s' (id=%d)", body.name, doc_id)
        return _to_response(row)
    finally:
        conn.close()


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: int):
    conn = get_connection()
    try:
        row = db_fetchone(conn, f"""
            SELECT p.*, COUNT(d.id) as doc_count
            FROM projects p
            LEFT JOIN documents d ON d.project_id = p.id
            WHERE p.id = {ph()}
            GROUP BY p.id
        """, (project_id,))
        if not row:
            raise HTTPException(status_code=404, detail="Project not found")
        return _to_response(row)
    finally:
        conn.close()


@router.patch("/{project_id}", response_model=ProjectResponse)
def update_project(project_id: int, body: ProjectUpdate):
    conn = get_connection()
    try:
        row = db_fetchone(conn, f"SELECT * FROM projects WHERE id={ph()}", (project_id,))
        if not row:
            raise HTTPException(status_code=404, detail="Project not found")

        fields, values = [], []
        if body.name is not None:
            fields.append(f"name={ph()}")
            values.append(body.name)
        if body.description is not None:
            fields.append(f"description={ph()}")
            values.append(body.description)
        if body.provider is not None:
            fields.append(f"provider={ph()}")
            values.append(body.provider)
        if body.model is not None:
            fields.append(f"model={ph()}")
            values.append(body.model)
        if body.top_k is not None:
            fields.append(f"top_k={ph()}")
            values.append(body.top_k)

        if fields:
            from config import get_settings
            if get_settings().db_driver == "postgres":
                fields.append("updated_at=NOW()")
            else:
                fields.append("updated_at=datetime('now')")
            values.append(project_id)
            db_execute(conn,
                f"UPDATE projects SET {', '.join(fields)} WHERE id={ph()}",
                values, commit=True)

        updated = db_fetchone(conn, f"""
            SELECT p.*, COUNT(d.id) as doc_count
            FROM projects p
            LEFT JOIN documents d ON d.project_id = p.id
            WHERE p.id = {ph()}
            GROUP BY p.id
        """, (project_id,))
        return _to_response(updated)
    finally:
        conn.close()


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: int):
    conn = get_connection()
    try:
        row = db_fetchone(conn, f"SELECT id FROM projects WHERE id={ph()}", (project_id,))
        if not row:
            raise HTTPException(status_code=404, detail="Project not found")
        db_execute(conn, f"DELETE FROM projects WHERE id={ph()}", (project_id,), commit=True)
        delete_project_index(project_id)
        logger.info("Deleted project id=%d", project_id)
    finally:
        conn.close()
