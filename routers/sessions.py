"""
RAGSmith – Chat Sessions router
Manages multi-session chat history for projects with full conversation context.
"""

import logging
from typing import List
from fastapi import APIRouter, HTTPException
from database import get_connection, db_fetchone, db_fetchall, db_execute, db_insert, ph
from models.schemas import (
    ChatSessionCreate, ChatSessionUpdate, ChatSessionResponse, QueryLogResponse
)

router = APIRouter()
logger = logging.getLogger("ragsmith.sessions")

def _get_session_or_404(conn, session_id: int):
    row = db_fetchone(conn, f"SELECT * FROM chat_sessions WHERE id={ph()}", (session_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return row

@router.post("/", response_model=ChatSessionResponse)
def create_session(body: ChatSessionCreate):
    conn = get_connection()
    try:
        # Validate project exists
        project = db_fetchone(conn, f"SELECT id, provider, model FROM projects WHERE id={ph()}", (body.project_id,))
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        provider = body.provider or project.get("provider") or "ollama"
        model = body.model or project.get("model")

        session_id = db_insert(conn,
            f"INSERT INTO chat_sessions (project_id, name, provider, model) VALUES ({ph()},{ph()},{ph()},{ph()})",
            (body.project_id, body.name, provider, model))
        
        row = db_fetchone(conn, f"SELECT * FROM chat_sessions WHERE id={ph()}", (session_id,))
        logger.info("Created chat session id=%d name='%s' provider='%s'", session_id, body.name, provider)
        return ChatSessionResponse(**row)
    finally:
        conn.close()

@router.get("/project/{project_id}", response_model=List[ChatSessionResponse])
def list_sessions(project_id: int):
    """List all chat sessions for a project, ordered by most recent first."""
    conn = get_connection()
    try:
        # Validate project exists
        project = db_fetchone(conn, f"SELECT id FROM projects WHERE id={ph()}", (project_id,))
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        rows = db_fetchall(conn,
            f"SELECT * FROM chat_sessions WHERE project_id={ph()} ORDER BY updated_at DESC",
            (project_id,))
        return [ChatSessionResponse(**r) for r in rows]
    finally:
        conn.close()

@router.get("/{session_id}", response_model=ChatSessionResponse)
def get_session(session_id: int):
    """Get details of a specific chat session."""
    conn = get_connection()
    try:
        row = _get_session_or_404(conn, session_id)
        return ChatSessionResponse(**row)
    finally:
        conn.close()

@router.patch("/{session_id}", response_model=ChatSessionResponse)
def update_session(session_id: int, body: ChatSessionUpdate):
    """Update session name and/or model."""
    conn = get_connection()
    try:
        _get_session_or_404(conn, session_id)
        updates = []
        params = []
        
        if body.name is not None:
            if not body.name.strip():
                raise HTTPException(status_code=400, detail="Session name cannot be empty")
            updates.append(f"name={ph()}")
            params.append(body.name)
        
        if body.provider is not None:
            updates.append(f"provider={ph()}")
            params.append(body.provider)

        if body.model is not None:
            updates.append(f"model={ph()}")
            params.append(body.model)
        
        if not updates:
            raise HTTPException(status_code=400, detail="No updates provided")
        
        params.append(session_id)
        db_execute(conn,
            f"UPDATE chat_sessions SET {', '.join(updates)}, updated_at=datetime('now') WHERE id={ph()}",
            tuple(params), commit=True)
        
        row = db_fetchone(conn, f"SELECT * FROM chat_sessions WHERE id={ph()}", (session_id,))
        logger.info("Updated session id=%d", session_id)
        return ChatSessionResponse(**row)
    finally:
        conn.close()

@router.delete("/{session_id}")
def delete_session(session_id: int):
    """Delete a chat session and all its associated query logs."""
    conn = get_connection()
    try:
        _get_session_or_404(conn, session_id)
        db_execute(conn, f"DELETE FROM chat_sessions WHERE id={ph()}", (session_id,), commit=True)
        logger.info("Deleted session id=%d", session_id)
        return {"detail": "Session deleted successfully"}
    finally:
        conn.close()

@router.get("/{session_id}/history", response_model=List[QueryLogResponse])
def get_session_history(session_id: int):
    """Get full conversation history for a session in chronological order."""
    conn = get_connection()
    try:
        # Verify session exists
        _get_session_or_404(conn, session_id)
        
        rows = db_fetchall(conn,
            f"""SELECT id, project_id, session_id, query_text, response, model, num_chunks, 
                       grounding_score, query_relevance, created_at 
                FROM query_logs WHERE session_id={ph()} ORDER BY created_at ASC""",
            (session_id,))
        
        return [QueryLogResponse(
            id=r["id"],
            project_id=r["project_id"],
            session_id=r["session_id"],
            query_text=r["query_text"],
            response=r["response"],
            model=r["model"],
            num_chunks=r["num_chunks"],
            created_at=str(r["created_at"]),
            grounding_score=float(r.get("grounding_score") or 0.0),
            query_relevance=float(r.get("query_relevance") or 0.0)
        ) for r in rows]
    finally:
        conn.close()
