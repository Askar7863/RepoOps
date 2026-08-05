"""
indexer.py

Repository Indexing Pipeline.

Responsibility
--------------
Orchestrate repository indexing across both retrieval paths.

Pipeline
--------
Chunks
    ↓
Reset any previous index for this repo (idempotent re-indexing)
    ↓
Generate Embeddings  ---->  Store dense vectors (ChromaDB)
                     ---->  Store sparse index (BM25)
    ↓
Return Indexing Statistics

Resetting first matters: chunk IDs are deterministic (path + name + line
range), so re-indexing the same repo without clearing the old collection
first would just add the new chunks alongside stale ones from the
previous run rather than replacing them.

This module DOES NOT:
    ✗ Clone repositories
    ✗ Scan repositories
    ✗ Chunk source code
"""

from pathlib import Path

from .embeddings import embed_documents
from .chroma_store import reset_collection, store_embeddings, delete_chunks_by_path
from .sparse_store import build_sparse_index, delete_sparse_index, get_all_chunks


def index_repository(
    repository_name: str,
    chunks: list,
):
    """
    Index an entire repository.

    Parameters
    ----------
    repository_name : str
        Repository name (collection / index name)

    chunks : list
        Output from github.chunker

    Returns
    -------
    dict
        Indexing statistics.
    """

    ####################################################
    # Reset previous index for this repo, if any
    ####################################################

    reset_collection(repository_name)
    delete_sparse_index(repository_name)

    ####################################################
    # Generate Embeddings
    ####################################################

    embedded_chunks = embed_documents(chunks)

    ####################################################
    # Store dense vectors (ChromaDB) + sparse index (BM25)
    ####################################################

    vectors_stored = store_embeddings(
        collection_name=repository_name,
        embedded_chunks=embedded_chunks,
    )

    sparse_indexed = build_sparse_index(
        collection_name=repository_name,
        chunks=chunks,
    )

    ####################################################
    # Return Statistics
    ####################################################

    return {

        "repository": repository_name,

        "total_chunks": len(chunks),

        "embedded_chunks": len(embedded_chunks),

        "vectors_stored": vectors_stored,

        "sparse_indexed": sparse_indexed,

        "collection": repository_name,

        "status": "Indexed Successfully"

    }


def reindex_file(
    repository_name: str,
    repo_root: Path,
    relative_path: str,
    chunks: list,
):
    """
    Re-chunk, re-embed, and re-store just ONE file's chunks -- used after
    code_writer_tool changes a file on disk, so a later request can still
    find what's actually there now.

    Without this, the index is a one-time snapshot taken when the repo
    was first loaded: any file code_writer_tool creates or edits
    afterward is invisible to retrieval forever, and a request to modify
    it falls back to whatever coincidentally scores best among the
    *original* indexed content -- a wrong, unrelated file, silently. A
    full index_repository() re-run would fix that too, but can take many
    minutes on a large repo; this touches only the one changed file.

    Parameters
    ----------
    repository_name : str
        Repository name (collection / index name)

    repo_root : Path
        The repo's local clone root (state["local_path"])

    relative_path : str
        The changed file's path, relative to repo_root

    chunks : list
        Freshly chunked content for just this file (empty list if the
        file has no chunkable category, or no chunks were produced)
    """

    ####################################################
    # Dense: drop this file's stale vectors, embed + add the new ones
    ####################################################

    delete_chunks_by_path(repository_name, relative_path)

    embedded_chunks = embed_documents(chunks) if chunks else []

    if embedded_chunks:
        store_embeddings(
            collection_name=repository_name,
            embedded_chunks=embedded_chunks,
        )

    ####################################################
    # Sparse: BM25 has no incremental delete, so rebuild from the
    # updated full corpus -- pure CPU tokenization/construction, no LLM
    # calls, fast even for a repo with thousands of chunks.
    ####################################################

    existing_chunks = [c for c in get_all_chunks(repository_name) if c["path"] != relative_path]

    sparse_indexed = build_sparse_index(
        collection_name=repository_name,
        chunks=existing_chunks + chunks,
    )

    return {
        "repository": repository_name,
        "path": relative_path,
        "chunks_indexed": len(chunks),
        "sparse_indexed": sparse_indexed,
    }