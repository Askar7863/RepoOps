"""
chroma_store.py

Repository Vector Store.

Responsibility
--------------
Store repository embeddings inside ChromaDB.

This module:
    ✓ Initialize ChromaDB
    ✓ Create/Open collection
    ✓ Insert embeddings
    ✓ Delete collection (optional)

This module DOES NOT:
    ✗ Generate embeddings
    ✗ Search vectors
    ✗ Rerank results
"""

from pathlib import Path

import chromadb


###########################################################
# Persistent Database Location
###########################################################

CHROMA_PATH = Path("chroma_db")

client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)


###########################################################
# Create / Load Collection
###########################################################

def get_collection(collection_name: str):
    """
    Create or load a Chroma collection.
    """

    return client.get_or_create_collection(
        name=collection_name
    )


###########################################################
# Store Repository Chunks
###########################################################

def store_embeddings(
    collection_name: str,
    embedded_chunks: list,
):
    """
    Store repository embeddings.

    Parameters
    ----------
    collection_name : str

    embedded_chunks : list
        Output from embeddings.py
    """

    collection = get_collection(collection_name)

    ids = [chunk["chunk_id"] for chunk in embedded_chunks]
    documents = [chunk["content"] for chunk in embedded_chunks]
    embeddings = [chunk["embedding"] for chunk in embedded_chunks]
    metadatas = [
        {
            "repo": chunk["repo"],
            "path": chunk["path"],
            "language": chunk["language"],
            "chunk_type": chunk["chunk_type"],
            "chunk_name": chunk["chunk_name"],
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
        }
        for chunk in embedded_chunks
    ]

    # collection.add() rejects a single call larger than the backend's max
    # batch size (confirmed 5461 on this project's client) rather than
    # chunking it itself -- a repo with enough chunks to cross that would
    # otherwise fail indexing entirely, partway through this call.
    batch_size = client.get_max_batch_size()

    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"Adding {len(ids)} vectors...")

    return collection.count()


###########################################################
# Delete Collection
###########################################################

def delete_collection(collection_name: str):

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # nothing to delete -- e.g. first time this repo is indexed


###########################################################
# Reset Collection
###########################################################

def reset_collection(collection_name: str):
    """
    Drop and recreate a collection. Re-indexing the same repo without this
    would re-add chunks under (mostly) the same deterministic IDs and
    accumulate duplicates/stale entries from the previous run instead of
    replacing them.
    """

    delete_collection(collection_name)
    return get_collection(collection_name)


###########################################################
# Delete Chunks For One File
###########################################################

def delete_chunks_by_path(collection_name: str, path: str) -> None:
    """
    Remove every stored chunk belonging to one file -- used to clear out
    its stale vectors before re-adding freshly re-chunked ones after the
    file changes on disk (see app/rag/indexer.py::reindex_file).
    """

    collection = get_collection(collection_name)

    try:
        collection.delete(where={"path": path})
    except Exception:
        pass  # nothing stored for this path yet -- e.g. a brand new file