"""
RAGSmith – Hybrid Retriever
BM25 sparse retrieval + Reciprocal Rank Fusion with FAISS dense results.

BM25 index is built over tokenised chunk texts (whitespace split — fast, no NLTK).
RRF merges two ranked lists with the standard k=60 smoothing constant.
"""

import logging
import pickle
from typing import List, Tuple, Dict

logger = logging.getLogger("ragsmith.retriever")

RRF_K = 60  # smoothing constant — standard default


# ── BM25 ──────────────────────────────────────────────────────────────────────

def build_bm25_index(chunks: List[str]):
    """
    Build a BM25Okapi index from a list of chunk texts.
    Tokenisation: simple whitespace split (matches query tokenisation).

    Returns
    -------
    BM25Okapi instance
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise RuntimeError(
            "rank_bm25 not installed. Run: pip install rank_bm25"
        ) from exc

    tokenised = [text.lower().split() for text in chunks]
    return BM25Okapi(tokenised)


def bm25_search(bm25_index, query: str, top_n: int) -> List[Tuple[int, float]]:
    """
    Score all chunks against the query and return the top-n results.

    Returns
    -------
    List of (chunk_index, bm25_score) sorted descending by score.
    """
    tokens = query.lower().split()
    scores = bm25_index.get_scores(tokens)  # ndarray, one score per chunk

    # Pair with indices and sort
    scored = [(i, float(s)) for i, s in enumerate(scores)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_ranked: List[Tuple[int, float]],
    bm25_ranked: List[Tuple[int, float]],
    k: int = RRF_K,
) -> List[Dict]:
    """
    Merge dense and BM25 ranked lists using RRF.

    Parameters
    ----------
    dense_ranked : [(chunk_idx, dense_score), ...] sorted best-first
    bm25_ranked  : [(chunk_idx, bm25_score),  ...] sorted best-first
    k            : RRF smoothing constant (default 60)

    Returns
    -------
    List of dicts sorted by rrf_score descending:
        {
          "idx": int,
          "dense_score": float,   # 0.0 if not in dense results
          "bm25_score":  float,   # 0.0 if not in BM25 results
          "rrf_score":   float,
        }
    """
    # Build lookup: chunk_idx → (rank, score) for each list
    dense_lookup: Dict[int, Tuple[int, float]] = {
        idx: (rank, score) for rank, (idx, score) in enumerate(dense_ranked)
    }
    bm25_lookup: Dict[int, Tuple[int, float]] = {
        idx: (rank, score) for rank, (idx, score) in enumerate(bm25_ranked)
    }

    # Union of all candidate indices
    all_idx = set(dense_lookup) | set(bm25_lookup)

    results = []
    for idx in all_idx:
        dense_rank, dense_score = dense_lookup.get(idx, (len(dense_ranked), 0.0))
        bm25_rank,  bm25_score  = bm25_lookup.get(idx,  (len(bm25_ranked),  0.0))

        rrf_score = 1.0 / (k + dense_rank + 1) + 1.0 / (k + bm25_rank + 1)

        results.append({
            "idx":         idx,
            "dense_score": dense_score,
            "bm25_score":  bm25_score,
            "rrf_score":   rrf_score,
        })

    results.sort(key=lambda x: x["rrf_score"], reverse=True)
    return results


# ── Persistence helpers ───────────────────────────────────────────────────────

def bm25_path(project_id: int) -> str:
    return f"data/chunks/project_{project_id}_bm25.pkl"


def save_bm25_index(project_id: int, bm25_index) -> None:
    import os
    path = bm25_path(project_id)
    os.makedirs("data/chunks", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bm25_index, f)
    logger.debug("BM25 index saved → %s", path)


def load_bm25_index(project_id: int):
    """Load BM25 index from disk. Returns None if not found."""
    import os
    path = bm25_path(project_id)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def delete_bm25_index(project_id: int) -> None:
    import os
    path = bm25_path(project_id)
    if os.path.exists(path):
        os.remove(path)
        logger.info("Deleted BM25 index: %s", path)
