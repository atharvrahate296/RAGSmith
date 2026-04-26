"""
RAGSmith – Cross-Encoder Re-ranker

Two-stage retrieval architecture:
  Stage 1 (Recall)   – Hybrid BM25 + FAISS search → top-20 candidates
  Stage 2 (Precision) – Cross-Encoder scores each (query, chunk) pair →
                         re-sorted top-k returned to the LLM

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  ~70 MB, downloads once, cached by sentence-transformers.
  Trained on MS MARCO passage ranking (500k+ annotated pairs).
  No fine-tuning required for general document QA.
"""

import logging
from typing import List, Dict

logger = logging.getLogger("ragsmith.reranker")

# Singleton — loaded once per process, not per request
_cross_encoder = None
_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_cross_encoder():
    """Lazy-load and cache the cross-encoder model."""
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading cross-encoder: %s", _RERANKER_MODEL)
            _cross_encoder = CrossEncoder(_RERANKER_MODEL)
            logger.info("Cross-encoder loaded ✓")
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc
    return _cross_encoder


def rerank(query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
    """
    Re-rank candidate chunks using cross-encoder relevance scores.

    Parameters
    ----------
    query      : The user's query string
    candidates : List of dicts, each must have at least {"text": str, ...}
                 (as returned by the hybrid search in processor.py)
    top_k      : Number of top results to return after re-ranking

    Returns
    -------
    List of dicts (length ≤ top_k), each enriched with:
        rerank_score  : float  — cross-encoder relevance logit
        original_rank : int    — position in pre-rerank list (0-indexed)
    Sorted by rerank_score descending.
    """
    if not candidates:
        return []

    model = _get_cross_encoder()

    # Build (query, chunk_text) pairs for batch scoring
    pairs = [(query, c["text"]) for c in candidates]

    logger.debug("Re-ranking %d candidates …", len(pairs))
    scores = model.predict(pairs)  # ndarray of floats

    # Attach scores and original rank
    for i, (candidate, score) in enumerate(zip(candidates, scores)):
        candidate["rerank_score"] = float(score)
        candidate["original_rank"] = i

    # Sort by cross-encoder score descending
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    return candidates[:top_k]
