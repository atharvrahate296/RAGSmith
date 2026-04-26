"""
RAGSmith – Confidence Scoring & Grounding Evaluation

Embedding-based post-generation faithfulness measurement.

After the LLM produces an answer, this module:
  1. Embeds the answer and all retrieved chunks
  2. Computes grounding_score  = cosine(answer, mean(chunks))
  3. Computes query_relevance  = cosine(query,  answer)
  4. Computes per-chunk attribution scores
  5. Returns a labelled EvaluationResult

No extra models, no extra LLM calls.  Uses the same all-MiniLM-L6-v2
already loaded by services/embeddings.py.  Overhead: ~15ms.

Thresholds (empirical heuristics — not hard guarantees):
  ≥ 0.75 → "high"   (well-grounded)
  ≥ 0.50 → "medium" (partially grounded)
  < 0.50 → "low"    (possible hallucination)
"""

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np

from services.embeddings import embed_texts

logger = logging.getLogger("ragsmith.evaluator")

HIGH_THRESHOLD   = 0.75
MEDIUM_THRESHOLD = 0.50


@dataclass
class EvaluationResult:
    grounding_score:   float
    query_relevance:   float
    confidence_label:  str           # "high" | "medium" | "low"
    top_chunk_index:   int           # index of most attributed chunk
    per_chunk_scores:  List[float] = field(default_factory=list)


def confidence_label(score: float) -> str:
    """Map a grounding score to a human-readable confidence label."""
    if score >= HIGH_THRESHOLD:
        return "high"
    elif score >= MEDIUM_THRESHOLD:
        return "medium"
    else:
        return "low"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D L2-normalised vectors."""
    # embed_texts already L2-normalises, so dot product == cosine similarity
    return float(np.dot(a, b))


def evaluate_response(
    query: str,
    answer: str,
    chunk_texts: List[str],
    embedding_model: str = "all-MiniLM-L6-v2",
) -> EvaluationResult:
    """
    Compute grounding and relevance metrics for a generated answer.

    Parameters
    ----------
    query          : The original user query
    answer         : The LLM-generated answer
    chunk_texts    : List of retrieved chunk texts used as context
    embedding_model: SentenceTransformer model name

    Returns
    -------
    EvaluationResult dataclass
    """
    if not answer or not chunk_texts:
        return EvaluationResult(
            grounding_score=0.0,
            query_relevance=0.0,
            confidence_label="low",
            top_chunk_index=0,
            per_chunk_scores=[],
        )

    try:
        # Embed query, answer, and all chunks in a single batch for efficiency
        all_texts = [query, answer] + chunk_texts
        all_vecs  = embed_texts(all_texts, model_name=embedding_model)

        query_vec  = all_vecs[0]            # shape (D,)
        answer_vec = all_vecs[1]            # shape (D,)
        chunk_vecs = all_vecs[2:]           # shape (N, D)

        # 1. Per-chunk attribution scores
        per_chunk = [_cosine(answer_vec, cv) for cv in chunk_vecs]

        # 2. Grounding score = cosine(answer, mean_chunk)
        mean_chunk_vec = chunk_vecs.mean(axis=0)
        # Re-normalise mean vector (mean of unit vectors is not unit)
        norm = np.linalg.norm(mean_chunk_vec)
        if norm > 0:
            mean_chunk_vec = mean_chunk_vec / norm
        grounding = _cosine(answer_vec, mean_chunk_vec)

        # 3. Query-answer relevance
        relevance = _cosine(query_vec, answer_vec)

        # 4. Top attributed chunk
        top_idx = int(np.argmax(per_chunk)) if per_chunk else 0

        label = confidence_label(grounding)

        logger.debug(
            "Evaluation — grounding: %.3f  relevance: %.3f  label: %s",
            grounding, relevance, label,
        )

        return EvaluationResult(
            grounding_score=round(grounding, 4),
            query_relevance=round(relevance, 4),
            confidence_label=label,
            top_chunk_index=top_idx,
            per_chunk_scores=[round(s, 4) for s in per_chunk],
        )

    except Exception as exc:
        logger.warning("Evaluation failed (non-fatal): %s", exc)
        return EvaluationResult(
            grounding_score=0.0,
            query_relevance=0.0,
            confidence_label="low",
            top_chunk_index=0,
            per_chunk_scores=[],
        )
