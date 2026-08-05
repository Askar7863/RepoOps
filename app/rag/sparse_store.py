"""
sparse_store.py

Repository Sparse (Keyword) Index.

Responsibility
--------------
Maintain a BM25 keyword index alongside the dense vector store (see
chroma_store.py) -- one lightweight, pure-CPU index per repository,
persisted to disk so it survives a process restart.

Why this exists
----------------
Dense embeddings are weak on exact-identifier queries: a query naming a
specific function ("resolve_redirects") is a token-overlap problem, not a
semantic-similarity one, and a general-purpose embedding model doesn't
reliably win at that. BM25 does. Fusing both (see reranker.py) covers
both query styles without needing a bigger/slower model -- BM25 itself is
just term-frequency arithmetic, no inference cost at all.

Tokenization
------------
The corpus is tokenized as whole identifiers AND their decomposed
sub-words ("resolve_redirects" indexes as "resolve_redirects", "resolve",
"redirects") so both a literal-symbol query and a natural-language query
can match. A plain query is tokenized literally, on the theory that if the
user typed the exact identifier, an exact match should win outright;
retriever.py falls back to the widened tokenization (decomposed +
stopwords stripped) only if confidence is low.

This module DOES NOT:
    ✗ Generate embeddings
    ✗ Perform dense vector search
    ✗ Fuse dense + sparse results (see reranker.py)
"""

from pathlib import Path
import pickle
import re
from typing import List, Optional

from rank_bm25 import BM25Okapi


BM25_PATH = Path("bm25_db")
BM25_PATH.mkdir(parents=True, exist_ok=True)

_METADATA_FIELDS = (
    "chunk_id", "content", "repo", "path", "language",
    "chunk_type", "chunk_name", "start_line", "end_line",
)

# Per-process cache so repeated queries against the same repo don't re-read
# and re-unpickle the index from disk every time.
_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "for", "with", "at", "by", "from", "as",
    "and", "or", "but", "if", "so", "than", "then", "that", "this",
    "it", "its", "do", "does", "did", "how", "what", "which", "who",
    "can", "could", "should", "would", "will", "does",
}


def _split_identifier(token: str) -> List[str]:
    words = []
    for part in token.split("_"):
        if not part:
            continue
        words.extend(w.lower() for w in _CAMEL_BOUNDARY_RE.sub(" ", part).split() if w)
    return words


def _tokenize_corpus(text: str) -> List[str]:
    """
    Whole identifiers, lowercased -- deliberately NOT decomposed into
    sub-words. A rare compound identifier like `resolve_redirects` is a
    far more discriminative token kept whole than it would be exploded
    into generic words ("resolve", "redirects") that recur throughout an
    entire codebase and would drown out the one chunk that matters.
    Decomposition is reserved for the widened query retry only (see
    tokenize_query) precisely so it doesn't water down every index entry.
    """

    return [raw.lower() for raw in _WORD_RE.findall(text)]


def tokenize_query(query: str, widen: bool = False) -> List[str]:
    """
    Strict mode (default): literal tokens as typed -- rewards an exact
    identifier match without diluting the corpus's own specificity.
    Widened mode (used on a low-confidence retry, see retriever.py): adds
    decomposed sub-words and strips stopwords, so a query like "resolve
    redirects" or "resolveRedirects" still overlaps a corpus that happens
    to mention "resolve" and "redirect" separately elsewhere in that chunk.
    """

    tokens = []

    for raw in _WORD_RE.findall(query):
        lowered = raw.lower()

        if widen and lowered in _STOPWORDS:
            continue

        tokens.append(lowered)

        if widen:
            parts = _split_identifier(raw)
            if len(parts) > 1:
                tokens.extend(p for p in parts if p not in _STOPWORDS)

    return tokens


# ---------------------------------------------------------------------------
# Build / persist / load
# ---------------------------------------------------------------------------

def _index_path(collection_name: str) -> Path:
    return BM25_PATH / f"{collection_name}.pkl"


def build_sparse_index(collection_name: str, chunks: List[dict]) -> int:
    """
    Tokenize and index every chunk's content, persisting the result so it
    survives a process restart. Returns the number of chunks indexed.
    """

    corpus_tokens = [_tokenize_corpus(chunk["content"]) for chunk in chunks]
    bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None

    bundle = {
        "bm25": bm25,
        "chunks": [{field: chunk[field] for field in _METADATA_FIELDS} for chunk in chunks],
    }

    with open(_index_path(collection_name), "wb") as f:
        pickle.dump(bundle, f)

    _CACHE[collection_name] = bundle
    return len(chunks)


def delete_sparse_index(collection_name: str) -> None:
    _CACHE.pop(collection_name, None)
    path = _index_path(collection_name)
    if path.exists():
        path.unlink()


def get_all_chunks(collection_name: str) -> List[dict]:
    """
    Every chunk currently in this repo's sparse index (metadata + content,
    no BM25 scores). rank_bm25 has no incremental delete/update, so
    updating the index for one changed file means rebuilding it from the
    full corpus -- this is how a caller gets that full corpus back out
    (see app/rag/indexer.py::reindex_file).
    """

    bundle = _load_bundle(collection_name)
    return list(bundle["chunks"]) if bundle else []


def _load_bundle(collection_name: str) -> Optional[dict]:
    if collection_name in _CACHE:
        return _CACHE[collection_name]

    path = _index_path(collection_name)
    if not path.exists():
        return None

    with open(path, "rb") as f:
        bundle = pickle.load(f)

    _CACHE[collection_name] = bundle
    return bundle


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_sparse(collection_name: str, query: str, top_k: int, widen: bool = False) -> List[dict]:
    """
    Rank every indexed chunk against the query by BM25 score, with an
    explicit boost for a chunk whose own name exactly matches a query
    token. That boost matters in practice: a function's *definition*
    typically mentions its own name only once (the signature line), while
    a caller referencing it by name can easily match just as often or
    more -- so raw term-frequency scoring doesn't reliably favor the
    definition over its call sites. An exact identifier query
    ("resolve_redirects") is exactly the case sparse retrieval exists to
    win outright, so that specific signal is boosted rather than left to
    term-frequency chance.

    Chunks that share no tokens with the query (score <= 0) are dropped
    rather than padding the ranked list with noise.
    """

    bundle = _load_bundle(collection_name)
    if bundle is None or bundle["bm25"] is None or not bundle["chunks"]:
        return []

    tokens = tokenize_query(query, widen=widen)
    if not tokens:
        return []

    scores = bundle["bm25"].get_scores(tokens)
    chunks = bundle["chunks"]

    token_set = set(tokens)
    exact_name_bonus = max(float(scores.max()) if len(scores) else 0.0, 1.0) * 2
    exact_match_indices = set()

    for i, chunk in enumerate(chunks):
        if chunk["chunk_name"].lower() in token_set:
            scores[i] += exact_name_bonus
            exact_match_indices.add(i)

    ranked_indices = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)

    return [
        {**chunks[i], "sparse_score": float(scores[i]), "exact_name_match": i in exact_match_indices}
        for i in ranked_indices[:top_k]
        if scores[i] > 0
    ]
