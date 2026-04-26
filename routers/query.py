"""
RAGSmith – Query router
Executes the full v2.0 RAG pipeline:
  embed → hybrid search (FAISS + BM25 + RRF) → cross-encoder rerank → generate → evaluate
"""

import logging
from typing import List

from fastapi import APIRouter, HTTPException

from database import get_connection, db_fetchone, db_fetchall, db_execute, db_insert, ph
from models.schemas import (
    QueryRequest, QueryResponse, ChunkResult, QueryLogResponse,
)
from services.processor import search_index
from services.reranker import rerank
from services.evaluator import evaluate_response
from services.llm import generate_answer

router = APIRouter()
logger = logging.getLogger("ragsmith.query")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _get_project_or_404(conn, project_id: int) -> dict:
    row = db_fetchone(conn, f"SELECT * FROM projects WHERE id={ph()}", (project_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")
    return row


@router.post("/{project_id}", response_model=QueryResponse)
def query_project(project_id: int, body: QueryRequest):
    conn = get_connection()
    try:
        project = _get_project_or_404(conn, project_id)

        ready_count = db_fetchone(conn,
            f"SELECT COUNT(*) as cnt FROM documents WHERE project_id={ph()} AND status='ready'",
            (project_id,))["cnt"]

        if not ready_count:
            raise HTTPException(status_code=422,
                detail="No indexed documents in this project. Upload and process a document first.")

        model = project["model"]
        top_k = project["top_k"]

        # ── Stage 1: Hybrid Search (FAISS + BM25 + RRF) — retrieve top-20 ────
        candidates = search_index(
            project_id=project_id,
            query_text=body.query,
            top_k=top_k,
            embedding_model=EMBEDDING_MODEL,
            hybrid_recall_n=20,
        )

        if not candidates:
            answer = ("No relevant information found in the knowledge base. "
                      "Try rephrasing or uploading more documents.")
            sources: List[ChunkResult] = []
            eval_result = None
        else:
            # ── Stage 2: Cross-Encoder Re-ranking — cut to top_k ────────────
            reranked = rerank(query=body.query, candidates=candidates, top_k=top_k)

            # ── Stage 3: LLM Generation ────────────────────────────────────
            # Convert to (text, score, filename) tuples that generate_answer expects
            llm_chunks = [(c["text"], c["rrf_score"], c["filename"]) for c in reranked]
            try:
                answer = generate_answer(query=body.query, context_chunks=llm_chunks, model=model)
            except (ConnectionError, RuntimeError) as exc:
                raise HTTPException(status_code=503, detail=str(exc))

            # ── Stage 4: Confidence Evaluation ────────────────────────────
            eval_result = evaluate_response(
                query=body.query,
                answer=answer,
                chunk_texts=[c["text"] for c in reranked],
                embedding_model=EMBEDDING_MODEL,
            )

            # Build source list — mark the top attributed chunk
            top_idx = eval_result.top_chunk_index
            sources = []
            for i, c in enumerate(reranked):
                sources.append(ChunkResult(
                    text=c["text"][:500],
                    score=round(c["rrf_score"], 4),
                    doc_filename=c["filename"],
                    dense_score=c.get("dense_score", 0.0),
                    bm25_score=c.get("bm25_score", 0.0),
                    rrf_score=c.get("rrf_score", 0.0),
                    rerank_score=round(c.get("rerank_score", 0.0), 4),
                    original_rank=c.get("original_rank", i),
                    is_top_source=(i == top_idx),
                ))

        # ── Persist to query_logs ─────────────────────────────────────────────
        grounding = eval_result.grounding_score if eval_result else 0.0
        relevance = eval_result.query_relevance  if eval_result else 0.0
        confidence = eval_result.confidence_label if eval_result else "low"

        db_insert(conn,
            f"""INSERT INTO query_logs
                (project_id, query_text, response, num_chunks, grounding_score, query_relevance)
                VALUES ({ph()},{ph()},{ph()},{ph()},{ph()},{ph()})""",
            (project_id, body.query, answer, len(sources), grounding, relevance))

        return QueryResponse(
            query=body.query,
            answer=answer,
            sources=sources,
            model=model,
            grounding_score=grounding,
            query_relevance=relevance,
            confidence_label=confidence,
        )
    finally:
        conn.close()


@router.get("/{project_id}/history", response_model=List[QueryLogResponse])
def query_history(project_id: int, limit: int = 20):
    conn = get_connection()
    try:
        _get_project_or_404(conn, project_id)
        rows = db_fetchall(conn,
            f"SELECT * FROM query_logs WHERE project_id={ph()} ORDER BY created_at DESC LIMIT {ph()}",
            (project_id, min(limit, 100)))
        return [QueryLogResponse(
            id=r["id"], project_id=r["project_id"],
            query_text=r["query_text"], response=r["response"],
            num_chunks=r["num_chunks"], created_at=str(r["created_at"]),
            grounding_score=float(r.get("grounding_score") or 0.0),
            query_relevance=float(r.get("query_relevance") or 0.0),
        ) for r in rows]
    finally:
        conn.close()
