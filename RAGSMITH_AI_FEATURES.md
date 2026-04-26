# ⬡ RAGSmith — AI Techniques Documentation

> **Version 2.0 · Advanced Retrieval Pipeline**  
> Hybrid Search · Cross-Encoder Re-ranking · Confidence Evaluation

---

## Overview

RAGSmith v2.0 extends the base RAG pipeline with three production-grade AI techniques that directly improve retrieval quality, answer precision, and output trustworthiness. These are not cosmetic additions — each technique addresses a concrete failure mode of naive RAG systems.

```
User Query
    │
    ├──► FAISS Dense Search ─────────────────┐
    │                                         ▼
    └──► BM25 Sparse Search ──────► RRF Fusion (Top-20)    [Hybrid Search]
                                             │
                                             ▼
                                   Cross-Encoder Re-ranker  [Re-ranking]
                                   (Top-20 → Top-K)
                                             │
                                             ▼
                                   Ollama LLM Generation
                                             │
                                             ▼
                                   Confidence Evaluator     [Evaluation]
                                             │
                                             ▼
                                   Response + Scores → UI
```

---

## Feature 1 — Hybrid Search (BM25 + Dense Retrieval)

### What Problem It Solves

Pure dense/vector search is semantically powerful but blind to exact keyword matches. If a user queries `"What is IndexFlatIP?"`, a dense retriever finds chunks *about* vector indexes — but may completely miss the chunk that *literally contains* `IndexFlatIP`. BM25 catches what dense search misses, and vice versa.

### How It Works

Two retrieval systems run in parallel on every query:

**Dense Retrieval (existing)**
The query is embedded using `all-MiniLM-L6-v2` and compared against chunk embeddings stored in FAISS using cosine similarity (inner product on L2-normalised vectors).

**BM25 Sparse Retrieval (new)**
BM25 (Best Match 25) is a classical probabilistic ranking algorithm. It scores each chunk based on:
- Term Frequency (TF) — how often query terms appear in the chunk
- Inverse Document Frequency (IDF) — how rare those terms are across all chunks
- Document length normalisation — prevents longer chunks from dominating

No embeddings. No neural network. Pure statistics. Zero additional compute cost.

### Reciprocal Rank Fusion (RRF)

The two ranked lists are merged using RRF — a parameter-free score fusion algorithm:

```
RRF_score(chunk) = 1 / (k + rank_dense) + 1 / (k + rank_bm25)
```

Where `k = 60` is a smoothing constant that reduces the impact of very high ranks. The final candidate list is sorted by RRF score descending.

**Example:**

| Chunk | Dense Rank | BM25 Rank | RRF Score |
|-------|-----------|-----------|-----------|
| A     | 1         | 8         | 1/61 + 1/68 = 0.0311 |
| B     | 3         | 1         | 1/63 + 1/61 = 0.0322 |
| C     | 2         | 2         | 1/62 + 1/62 = 0.0323 |

Chunk C wins because it performs consistently well on both signals — even without being top-1 in either. This is the key insight of RRF.

### Implementation Notes

**New dependency:** `rank_bm25` (pure Python, no compute)

**Files changed:**
- `services/processor.py` — `build_or_update_index()` builds BM25 index alongside FAISS, pickled as `project_{id}_bm25.pkl`
- `services/processor.py` — `search_index()` runs both retrievers and applies RRF fusion
- `services/retriever.py` (new) — BM25 wrapper and RRF logic

**Storage:** BM25 index is serialised with `pickle` alongside existing `chunks.pkl`. Negligible disk footprint.

**UI exposure:** Each retrieved source chunk displays its dense score, BM25 score, and final RRF score — making the dual-signal retrieval directly visible.

---

## Feature 2 — Cross-Encoder Re-ranking

### What Problem It Solves

Your bi-encoder (FAISS) encodes the query and each chunk **independently**. This is fast but approximate — the model never sees the query and chunk together, so it cannot model fine-grained relevance interactions. A chunk might be semantically close to a query in vector space but not actually answer it well.

Re-ranking adds a precise second pass over the top candidates using a model that **reads both query and chunk together**.

### The Two-Stage Architecture

```
Stage 1 — Recall  (Hybrid Search, fast):
  Retrieve Top-20 candidates from FAISS + BM25

Stage 2 — Precision  (Re-ranking, accurate):
  Score each of the 20 candidates with Cross-Encoder
  Re-sort by cross-encoder score
  Return Top-K to the LLM
```

Stage 1 casts a wide net for recall. Stage 2 surgically reorders for precision. This is the standard production architecture used by systems like Cohere Rerank, Jina Reranker, and most enterprise search systems.

### The Model

**`cross-encoder/ms-marco-MiniLM-L-6-v2`** (HuggingFace)

- Size: ~70 MB, downloaded once and cached
- Training data: MS MARCO — Microsoft's large-scale passage ranking dataset with 500k+ annotated query-passage pairs
- Architecture: BERT-style encoder, input is `[CLS] query [SEP] passage [SEP]`
- Output: single relevance logit (higher = more relevant)
- Speed: milliseconds per chunk on CPU; near-instant on RTX 3050

No fine-tuning required. The model works out-of-the-box for general passage ranking.

### Why This Matters

Consider a top-5 result from FAISS:

| FAISS Rank | Chunk Summary | Cross-Encoder Score | Final Rank |
|-----------|--------------|-------------------|-----------|
| 1 | Tangentially related topic | 0.31 | 4 |
| 2 | Exact answer, different phrasing | 0.94 | 1 |
| 3 | Background context | 0.67 | 2 |
| 4 | Unrelated but similar embedding | 0.12 | 5 |
| 5 | Supporting detail | 0.58 | 3 |

The LLM now receives the best chunk first, which directly improves answer quality — especially for models with limited context windows.

### Implementation Notes

**No new dependencies** — `CrossEncoder` is part of `sentence-transformers`, already in `requirements.txt`

**New file:** `services/reranker.py`
- Singleton model loader (loaded once per process, not per query)
- `rerank(query, chunks)` → returns chunks sorted by cross-encoder score

**Files changed:**
- `services/processor.py` / `routers/query.py` — hybrid search retrieves top-20, reranker cuts to `top_k`
- `models/schemas.py` — `ChunkResult` gains `rerank_score: float` and `original_rank: int`

**UI exposure:** Each source chunk shows its original FAISS rank alongside its re-ranked position. A chunk that jumped from rank #9 to #1 visibly demonstrates the technique working.

---

## Feature 3 — Confidence Scoring & Grounding Evaluation

### What Problem It Solves

A RAG system can generate fluent, confident-sounding answers that are completely unsupported by the retrieved context. There is currently no signal in the pipeline to distinguish a well-grounded answer from a hallucinated one. This feature computes that signal automatically after every generation.

### Approach: Embedding-Based Grounding

After the LLM generates an answer, the answer text is embedded and compared against the retrieved chunk embeddings using cosine similarity. The intuition: if the LLM faithfully used the context, the answer's embedding will be geometrically close to the source chunk embeddings. If it hallucinated, the answer drifts away.

This approach uses your existing `all-MiniLM-L6-v2` model. No second LLM call. No additional models. Near-zero latency overhead.

### Three Computed Metrics

**1. Answer Grounding Score**
```
grounding_score = cosine_similarity(
    embed(answer),
    mean(embed(chunk) for chunk in retrieved_chunks)
)
```
Measures how semantically close the generated answer is to the retrieved context as a whole. This is the primary hallucination signal.

**2. Per-Chunk Attribution Score**
```
attribution[i] = cosine_similarity(embed(answer), embed(chunk_i))
```
Identifies which specific chunk contributed most to the answer. The highest-scoring chunk is the "most responsible source" — useful for surfacing the most relevant citation.

**3. Query-Answer Relevance**
```
query_relevance = cosine_similarity(embed(query), embed(answer))
```
Measures whether the answer actually addresses the question. An answer can be perfectly grounded in the context but still fail to answer what was asked.

### Score Interpretation

| Grounding Score | Signal | UI Display |
|----------------|--------|-----------|
| ≥ 0.75 | High confidence — answer well-supported by sources | 🟢 Green badge |
| 0.50 – 0.74 | Medium confidence — partial grounding | 🟡 Yellow badge |
| < 0.50 | Low confidence — possible hallucination | 🔴 Red badge |

Thresholds are empirically derived heuristics, not hard guarantees. This is a known limitation of embedding-based evaluation vs. LLM-as-judge approaches (like RAGAS), which require an additional generation step.

### Implementation Notes

**No new dependencies** — uses existing `embed_texts()` from `services/embeddings.py`

**New file:** `services/evaluator.py`
- `evaluate_response(query, answer, chunks)` → returns `EvaluationResult` dataclass
- Fields: `grounding_score`, `query_relevance`, `top_chunk_index`, `per_chunk_scores`

**Files changed:**
- `routers/query.py` — calls evaluator after `generate_answer()`
- `models/schemas.py` — `QueryResponse` gains `grounding_score`, `query_relevance`, `confidence_label`
- `database.py` — `query_logs` table gains `grounding_score`, `query_relevance` columns for trend tracking

**UI exposure:**
- Confidence badge (green/yellow/red) displayed alongside every answer
- Grounding score shown as a labelled progress bar
- Most attributed chunk highlighted in the sources list
- History view shows score trends across past queries

---

## End-to-End Query Latency Budget

| Stage | Component | Estimated Latency |
|-------|-----------|------------------|
| Embedding | Query embed (MiniLM) | ~10ms |
| Retrieval | FAISS search | ~5ms |
| Retrieval | BM25 search | ~5ms |
| Fusion | RRF computation | <1ms |
| Re-ranking | Cross-encoder (20 chunks, CPU) | ~200–400ms |
| Generation | Ollama LLM | ~2–10s (model dependent) |
| Evaluation | Grounding score (embed + cosine) | ~15ms |
| **Total overhead vs baseline** | | **~400–450ms** |

The LLM generation dominates latency by an order of magnitude. The AI technique overhead is negligible in practice.

---

## Dependencies Added

| Package | Purpose | License | Compute |
|---------|---------|---------|---------|
| `rank_bm25` | BM25 sparse retrieval | Apache 2.0 | CPU only, trivial |
| *(none)* | Cross-Encoder re-ranking | — | Uses existing `sentence-transformers` |
| *(none)* | Confidence scoring | — | Uses existing `sentence-transformers` |

Only one new package required across all three features.

---

## Academic References

- Robertson, S. & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval.
- Cormack, G., Clarke, C., & Buettcher, S. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.* SIGIR.
- Nogueira, R. & Cho, K. (2019). *Passage Re-ranking with BERT.* arXiv:1901.04085.
- Es, S. et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* arXiv:2309.15217.

---

## Architecture Summary

RAGSmith v2.0 implements a **three-stage retrieval pipeline** that mirrors the architecture of production-grade search systems:

1. **Hybrid recall** — BM25 + Dense retrieval with RRF fusion ensures no relevant chunk is missed, covering both lexical and semantic similarity signals
2. **Cross-encoder precision** — Re-ranking reorders candidates using deep query-chunk interaction modeling, ensuring the LLM receives the most relevant context first
3. **Grounded generation** — Embedding-based confidence scoring provides an automatic faithfulness signal after every generation, making hallucination visible rather than silent

Each stage is independently valuable, composable, and designed to degrade gracefully — if re-ranking is disabled, hybrid search still improves over dense-only; if evaluation is skipped, retrieval quality is unaffected.

---

*RAGSmith — Build your knowledge, own your AI.*
