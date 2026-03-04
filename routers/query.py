"""
RAGSmith – Query router
Executes the full RAG pipeline: embed → search → generate.
"""

import logging
from typing import List

from fastapi import APIRouter, HTTPException

from database import get_connection
from models.schemas import QueryRequest, QueryResponse, ChunkResult, QueryLogResponse
from services.processor import search_index
from services.llm import generate_answer, check_ollama_running

router = APIRouter()
logger = logging.getLogger("ragsmith.query")

# Embedding model is always the sentence-transformer, never the LLM model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _get_project_or_404(conn, project_id: int):
    row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")
    return row


@router.post("/{project_id}", response_model=QueryResponse)
def query_project(project_id: int, body: QueryRequest):
    """
    Full RAG pipeline:
      1. Embed query (SentenceTransformers)
      2. Top-k FAISS similarity search
      3. Local LLM generation via Ollama
      4. Log and return answer
    """
    conn = get_connection()
    try:
        project = _get_project_or_404(conn, project_id)

        # Check that there's at least one ready document
        ready_count = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE project_id = ? AND status = 'ready'",
            (project_id,),
        ).fetchone()[0]

        if ready_count == 0:
            raise HTTPException(
                status_code=422,
                detail="No indexed documents in this project. Upload and process a document first.",
            )

        model = project["model"]
        top_k = project["top_k"]

        # Verify Ollama is running
        if not check_ollama_running():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Ollama is not running. Please start Ollama: https://ollama.com "
                    f"and pull the model: ollama pull {model}"
                ),
            )

        # Retrieve relevant chunks
        results = search_index(
            project_id=project_id,
            query_text=body.query,
            top_k=top_k,
            embedding_model=EMBEDDING_MODEL,
        )

        if not results:
            answer = (
                "No relevant information found in the knowledge base for your query. "
                "Try rephrasing or uploading more documents."
            )
            sources: List[ChunkResult] = []
        else:
            # Generate answer
            answer = generate_answer(
                query=body.query,
                context_chunks=results,
                model=model,
            )
            sources = [
                ChunkResult(text=text[:500], score=round(score, 4), doc_filename=filename)
                for text, score, filename in results
            ]

        # Log to DB
        conn.execute(
            "INSERT INTO query_logs (project_id, query_text, response, num_chunks) VALUES (?, ?, ?, ?)",
            (project_id, body.query, answer, len(results)),
        )
        conn.commit()

        return QueryResponse(
            query=body.query,
            answer=answer,
            sources=sources,
            model=model,
        )

    finally:
        conn.close()


@router.get("/{project_id}/history", response_model=List[QueryLogResponse])
def query_history(project_id: int, limit: int = 20):
    """Retrieve recent query logs for a project."""
    conn = get_connection()
    try:
        _get_project_or_404(conn, project_id)
        rows = conn.execute(
            """
            SELECT * FROM query_logs
            WHERE project_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (project_id, min(limit, 100)),
        ).fetchall()
        return [
            QueryLogResponse(
                id=r["id"],
                project_id=r["project_id"],
                query_text=r["query_text"],
                response=r["response"],
                num_chunks=r["num_chunks"],
                created_at=r["created_at"],
            )
            for r in rows
        ]
    finally:
        conn.close()
