"""
RAGSmith – Pydantic request/response schemas
"""

from typing import Optional, List
from pydantic import BaseModel, Field


# ── Projects ──────────────────────────────────────────────────────────────────

class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Unique project name")
    description: Optional[str] = Field("", max_length=500)
    provider: Optional[str] = Field("ollama", description="LLM provider: groq or ollama")
    model: Optional[str] = Field("mistral:7b", description="Model name to use for generation")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of chunks to retrieve")


class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    provider: Optional[str] = None
    model: Optional[str] = None
    top_k: Optional[int] = Field(None, ge=1, le=20)


class ProjectResponse(BaseModel):
    id: int
    name: str
    description: str
    provider: str
    model: str
    top_k: int
    created_at: str
    updated_at: str
    document_count: int = 0


# ── Documents ─────────────────────────────────────────────────────────────────

class DocumentResponse(BaseModel):
    id: int
    project_id: int
    filename: str
    num_chunks: int
    status: str
    error_msg: Optional[str]
    created_at: str


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[int] = None
    model: Optional[str] = None


class ChunkResult(BaseModel):
    text: str
    score: float               # primary score (rrf_score for display)
    doc_filename: str
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    original_rank: int = 0     # rank before re-ranking (0-indexed)
    is_top_source: bool = False  # True for the most attributed chunk


class RetrievalDetails(BaseModel):
    """Detailed retrieval metrics for transparency."""
    total_chunks_retrieved: int
    chunks: List[ChunkResult]
    retrieval_time_ms: float = 0.0
    rerank_time_ms: float = 0.0


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[ChunkResult]
    retrieval_details: Optional[RetrievalDetails] = None  # Hidden by default, shown on toggle
    model: str
    session_id: Optional[int] = None
    grounding_score: float = 0.0
    query_relevance: float = 0.0
    confidence_label: str = "low"   # "high" | "medium" | "low"


# ── Chat Sessions ─────────────────────────────────────────────────────────────

class ChatSessionCreate(BaseModel):
    project_id: int
    name: str = Field(..., min_length=1, max_length=100)
    provider: Optional[str] = None
    model: Optional[str] = None


class ChatSessionUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    provider: Optional[str] = None
    model: Optional[str] = None


class ChatSessionResponse(BaseModel):
    id: int
    project_id: int
    name: str
    provider: Optional[str]
    model: Optional[str]
    created_at: str
    updated_at: str


# ── Query Logs ────────────────────────────────────────────────────────────────

class QueryLogResponse(BaseModel):
    id: int
    project_id: int
    session_id: Optional[int]
    query_text: str
    response: str
    model: Optional[str]
    num_chunks: int
    created_at: str
    grounding_score: float = 0.0
    query_relevance: float = 0.0


# ── Export ────────────────────────────────────────────────────────────────────

class ExportResponse(BaseModel):
    project_name: str
    export_path: str
    message: str
