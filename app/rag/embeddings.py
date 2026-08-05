"""
embeddings.py

Repository Embedding Generator.

Responsibility
--------------
Generate vector embeddings for repository code chunks and queries, using
one local model for both.

This module:
    ✓ Loads the embedding model once, offline-first
    ✓ Converts code chunks into embeddings (batched, with per-chunk fallback)
    ✓ Converts a query into an embedding (with the model's query prefix)

This module DOES NOT:
    ✗ Store vectors
    ✗ Query ChromaDB
    ✗ Retrieve documents

Note on the query prefix
-------------------------
BAAI/bge-* models are trained asymmetrically: the *query* side needs an
instruction prefix ("Represent this sentence for searching relevant
passages: "), the *document* side never does. Getting this backwards
doesn't raise an error -- it just quietly makes every retrieval worse.
embed_documents() and embed_query() are kept as separate functions so
that mistake isn't possible to make by accident.
"""

import threading

from sentence_transformers import SentenceTransformer

from .. import config
from ..utils.logger import log_error, log_warning


# --------------------------------------------------------------------
# Load Embedding Model (Loaded only once, offline-first)
# --------------------------------------------------------------------
# local_files_only avoids a hub round-trip on every load when the model is
# already cached -- otherwise every process start pays a network check for
# a file that never changed.

try:
    embedding_model = SentenceTransformer(config.LOCAL_EMBED_MODEL, local_files_only=True)
except Exception:
    embedding_model = SentenceTransformer(config.LOCAL_EMBED_MODEL)

assert embedding_model.get_embedding_dimension() == config.EMBED_DIMENSIONS, (
    f"{config.LOCAL_EMBED_MODEL} outputs "
    f"{embedding_model.get_embedding_dimension()}-d vectors, "
    f"but config.EMBED_DIMENSIONS is {config.EMBED_DIMENSIONS}"
)

# A CPU model call from concurrent request threads is cheap insurance
# against any thread-safety edge case in the underlying inference stack --
# effectively free serialization for what's already a fast, small model.
_ENCODE_LOCK = threading.Lock()


def _encode(texts: list[str]):
    with _ENCODE_LOCK:
        return embedding_model.encode(texts, normalize_embeddings=True)


def count_embedding_tokens(text: str) -> int:
    """
    Token count using the embedding model's OWN tokenizer -- the number
    that actually determines whether embed_documents() will silently
    truncate this text at encode time. Used by chunker.py to decide
    whether a chunk needs re-splitting; an LLM tokenizer (e.g. tiktoken)
    would give the wrong number here since it's a different vocabulary.
    """

    return len(embedding_model.tokenizer(text, truncation=False)["input_ids"])


# --------------------------------------------------------------------
# Document embeddings (batched, with per-chunk fallback)
# --------------------------------------------------------------------

def embed_documents(repository_chunks: list[dict]) -> list[dict]:
    """
    Generate embeddings for repository chunks (no query prefix).

    Batched for throughput; if a whole batch fails, retries that batch's
    chunks one at a time so a single bad chunk can't take the rest down.

    Parameters
    ----------
    repository_chunks : list
        Output produced by chunker.py

    Returns
    -------
    list
        Each element contains metadata + embedding vector. Chunks whose
        embedding fails even individually are dropped (logged, not raised).
    """

    embedded_chunks = []
    batch_size = config.LOCAL_EMBED_BATCH_SIZE

    for start in range(0, len(repository_chunks), batch_size):
        batch = repository_chunks[start:start + batch_size]
        texts = [chunk["content"] for chunk in batch]

        try:
            vectors = _encode(texts)
            embedded_chunks.extend(
                {**chunk, "embedding": vector.tolist()}
                for chunk, vector in zip(batch, vectors)
            )
            continue
        except Exception as exc:
            log_warning(f"Batch embed failed ({exc}); retrying chunks individually")

        for chunk in batch:
            try:
                vector = _encode([chunk["content"]])[0]
                embedded_chunks.append({**chunk, "embedding": vector.tolist()})
            except Exception as exc:
                log_error(f"Dropping chunk {chunk.get('chunk_id')}: embedding failed ({exc})")

    return embedded_chunks


# --------------------------------------------------------------------
# Query embedding (with the asymmetric-model query prefix)
# --------------------------------------------------------------------

def embed_query(query: str) -> list[float]:
    """
    Generate an embedding for a user query, with the model's query-side
    instruction prefix applied. Never use this for indexed content.
    """

    prefixed = f"{config.LOCAL_EMBED_QUERY_PREFIX}{query}"
    return _encode([prefixed])[0].tolist()
