"""
reranker.py

Repository Result Fusion.

Responsibility
--------------
Merge dense (embedding) and sparse (BM25) search results into a single
ranked list using Reciprocal Rank Fusion (RRF).

Why RRF instead of a cross-encoder reranker
--------------------------------------------
A cross-encoder reranker would score every candidate with a second
transformer forward pass over each (query, chunk) pair -- real accuracy
upside, but a second model to load and run on every single query. RRF
gets most of the benefit at near-zero cost: it's pure arithmetic over
rank *positions*, so it needs no score normalization between two
retrievers whose scores aren't on the same scale (cosine distance vs a
BM25 term-frequency score).

    score(chunk) = sum( 1 / (k + rank_r(chunk)) )  over every retriever r
                                                     that returned it

A chunk both retrievers agree on scores higher than one only one
retriever found, even at rank 1 in that single list -- agreement between
two independent retrievers is itself evidence of relevance, and that's
the entire reason RRF beats a plain top-k union or dedup-by-first-seen.

This module DOES NOT:
    ✗ Call the embedding model
    ✗ Query ChromaDB or the BM25 index
    ✗ Call the LLM
"""

from typing import List

from .. import config


def reciprocal_rank_fusion(
    result_lists: List[List[dict]],
    weights: List[float] = None,
    key: str = "chunk_id",
    k: int = config.RRF_K,
) -> List[dict]:
    """
    Fuse multiple already-ranked (best-first) lists of chunk dicts into
    one list, sorted by descending fused score. Each returned chunk keeps
    its original fields plus `fused_score`.

    `weights` (one per list, same order, default all 1.0) matters in
    practice for this corpus: plain unweighted RRF was measured (see the
    19-query benchmark) to actively *hurt* conceptual/descriptive queries
    relative to dense-only. BM25 on a domain-specific codebase gives its
    HIGHEST scores to incidental rare-word overlap (a CLI example that
    happens to define a `--verbose` flag outscores the actual `Option`
    class on a query about options) rather than true relevance, because
    the query's meaningful terms ("option", "command") are also the
    corpus's own ubiquitous vocabulary and so carry almost no IDF weight.
    Unweighted RRF rewards any agreement between retrievers equally, so
    that confident-but-wrong sparse signal was outranking a correct,
    confident dense-only result. Down-weighting sparse keeps the real win
    -- exact-identifier queries sparse gets right that dense doesn't --
    without letting sparse noise veto a good dense match. See
    config.RRF_DENSE_WEIGHT / config.RRF_SPARSE_WEIGHT.
    """

    if weights is None:
        weights = [1.0] * len(result_lists)

    scores: dict = {}
    chunks_by_key: dict = {}

    for weight, results in zip(weights, result_lists):
        for rank, chunk in enumerate(results, start=1):
            chunk_key = chunk[key]
            scores[chunk_key] = scores.get(chunk_key, 0.0) + weight / (k + rank)
            chunks_by_key.setdefault(chunk_key, chunk)

    fused = [
        {**chunks_by_key[chunk_key], "fused_score": score}
        for chunk_key, score in scores.items()
    ]

    fused.sort(key=lambda c: c["fused_score"], reverse=True)
    return fused
