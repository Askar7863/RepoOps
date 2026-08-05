"""
retriever.py

Repository Retrieval Pipeline.

Responsibility
--------------
Retrieve the most relevant code chunks for a query by combining dense
(embedding) and sparse (BM25 keyword) search.

Pipeline
--------
User Query
    -> Embed query (with the asymmetric-model query prefix)
    -> Dense search (ChromaDB, cosine)      -----+
    -> Sparse search (BM25)                 -----+--> RRF fuse
    -> Confidence check -> low? widen the sparse query and re-fuse once
    -> Return top-k fused chunks

Confidence is checked but not surfaced as a hard gate: a low-confidence
result still gets returned (a partial, uncertain answer beats a blocked
one) -- the widen retry is a best-effort improvement, not a requirement.

This module DOES NOT:
    ✗ Call the LLM
    ✗ Build prompts
    ✗ Clone repositories
"""

from typing import List

from .. import config
from .embeddings import embed_query
from .chroma_store import get_collection
from .sparse_store import search_sparse
from .reranker import reciprocal_rank_fusion


def _dense_search(repository_name: str, query: str, top_k: int) -> List[dict]:
    collection = get_collection(repository_name)
    query_embedding = embed_query(query)

    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        {"chunk_id": chunk_id, "content": document, "dense_distance": distance, **metadata}
        for chunk_id, document, metadata, distance in zip(ids, docs, metas, distances)
    ]


def _top_score(fused: List[dict]) -> float:
    return fused[0]["fused_score"] if fused else 0.0


def _promote_exact_matches(fused: List[dict], sparse_results: List[dict]) -> List[dict]:
    """
    An exact identifier match (see sparse_store.py's chunk-name boost) is
    a decisive signal, but RRF's rank-based scoring can't reliably surface
    it at a *safe* sparse weight: that weight has to stay low so BM25's
    general relevance noise doesn't outrank a correct descriptive-query
    dense result (measured directly -- see config.py's RRF_SPARSE_WEIGHT
    comment), and that same low weight caps how much credit even an
    unambiguous exact-name match can earn through the rank formula alone.
    So it's promoted explicitly here instead of asking one scalar weight
    to do both jobs.
    """

    exact_ids = {c["chunk_id"] for c in sparse_results if c.get("exact_name_match")}
    if not exact_ids:
        return fused

    promoted = [c for c in fused if c["chunk_id"] in exact_ids]
    rest = [c for c in fused if c["chunk_id"] not in exact_ids]
    return promoted + rest


def retrieve_chunks(
    repository_name: str,
    query: str,
    top_k: int = 5,
) -> List[dict]:
    """
    Retrieve the most relevant repository chunks.

    Parameters
    ----------
    repository_name : str
        Chroma collection / BM25 index name.

    query : str
        User question.

    top_k : int
        Number of chunks to return.

    Returns
    -------
    list
        Chunks sorted by fused relevance (`score`, descending -- higher is
        more relevant, unlike the raw cosine distance this used to expose).
    """

    weights = [config.RRF_DENSE_WEIGHT, config.RRF_SPARSE_WEIGHT]

    dense_results = _dense_search(repository_name, query, config.DENSE_TOP_K)
    sparse_results = search_sparse(repository_name, query, config.SPARSE_TOP_K, widen=False)

    fused = reciprocal_rank_fusion([dense_results, sparse_results], weights=weights)

    if _top_score(fused) < config.RRF_CONFIDENCE_THRESHOLD:
        widened_sparse = search_sparse(repository_name, query, config.SPARSE_TOP_K, widen=True)
        widened_fused = reciprocal_rank_fusion([dense_results, widened_sparse], weights=weights)

        if _top_score(widened_fused) > _top_score(fused):
            fused = widened_fused
            sparse_results = widened_sparse

    fused = _promote_exact_matches(fused, sparse_results)

    return [
        {
            "chunk_id": chunk["chunk_id"],
            "content": chunk["content"],
            "score": chunk["fused_score"],
            "repo": chunk["repo"],
            "path": chunk["path"],
            "language": chunk["language"],
            "chunk_type": chunk["chunk_type"],
            "chunk_name": chunk["chunk_name"],
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
        }
        for chunk in fused[:top_k]
    ]
