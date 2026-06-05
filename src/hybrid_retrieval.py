"""
Hybrid retrieval: FAISS semantic search + BM25 keyword search + score fusion.

The two retrievers are complementary:
  - FAISS catches semantic similarity (paraphrases, synonyms)
  - BM25 catches exact term matches (numbers, ticker symbols, product names)

Scores are combined with Reciprocal Rank Fusion (RRF), which avoids the need
to normalise incompatible score scales.  An optional cross-encoder reranker
can be added later without changing the interface.
"""

from __future__ import annotations
import math
from typing import List, Dict, Any

import numpy as np
from rank_bm25 import BM25Okapi


# ── Tokenisation ──────────────────────────────────────────────────────────────

def _tokenise(text: str) -> List[str]:
    return text.lower().split()


# ── BM25 retrieval ────────────────────────────────────────────────────────────

def build_bm25_index(chunks: List[Dict[str, Any]]) -> BM25Okapi:
    corpus = [_tokenise(c["text"]) for c in chunks]
    return BM25Okapi(corpus)


def retrieve_bm25(
    question: str,
    chunks: List[Dict[str, Any]],
    bm25: BM25Okapi,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    tokens = _tokenise(question)
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {**chunks[i], "bm25_score": float(scores[i]), "bm25_rank": rank + 1}
        for rank, i in enumerate(top_indices)
    ]


# ── FAISS retrieval ───────────────────────────────────────────────────────────

def retrieve_faiss(
    question: str,
    chunks: List[Dict[str, Any]],
    index,
    model,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    q_emb = model.encode([question], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_emb, top_k)
    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx == -1:
            continue
        results.append(
            {**chunks[idx], "faiss_distance": float(dist), "faiss_rank": rank + 1}
        )
    return results


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def fuse_results(
    faiss_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    top_k: int = 4,
    faiss_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Merge FAISS and BM25 results with Reciprocal Rank Fusion.

    faiss_weight / bm25_weight control the blend.  Defaults favour semantic
    search slightly, which works well for free-text business questions.
    """
    scores: Dict[int, float] = {}
    meta: Dict[int, Dict] = {}

    for r in faiss_results:
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + faiss_weight * _rrf_score(r["faiss_rank"])
        meta[cid] = r

    for r in bm25_results:
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + bm25_weight * _rrf_score(r["bm25_rank"])
        if cid not in meta:
            meta[cid] = r

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return [
        {
            **meta[cid],
            "hybrid_score": score,
            "distance": meta[cid].get("faiss_distance", 0.0),
        }
        for cid, score in ranked
    ]


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve_hybrid(
    question: str,
    chunks: List[Dict[str, Any]],
    faiss_index,
    bm25_index: BM25Okapi,
    model,
    top_k: int = 4,
    faiss_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Run FAISS + BM25 retrieval and fuse results.

    Returns top_k chunks sorted by hybrid score (highest first),
    with keys: chunk_id, page_number, text, distance, hybrid_score.
    """
    faiss_r = retrieve_faiss(question, chunks, faiss_index, model, top_k=top_k * 3)
    bm25_r  = retrieve_bm25(question, chunks, bm25_index, top_k=top_k * 3)
    return fuse_results(faiss_r, bm25_r, top_k=top_k,
                        faiss_weight=faiss_weight, bm25_weight=bm25_weight)
