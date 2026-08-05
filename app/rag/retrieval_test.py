"""
retrieval_test.py

Retrieval accuracy regression test.

Responsibility
--------------
Run a small labeled (query, expected_path, expected_chunk_name) set
against the already-indexed repositories in repos/ and report Hit@1/3/5
and MRR -- the same methodology the reference design calls for: a hit
counts on file + chunk name matching (chunk IDs are non-deterministic
across a re-index, so they can't be the match key), and dense-only vs
sparse-only vs fused are reported separately so a change to the fusion
weights can be judged against real numbers instead of by feel.

Usage
-----
Requires click/flask/requests already cloned into repos/ and indexed
(POST /api/repositories/load, or app.rag.indexer.index_repository
directly) under those collection names. Run with:

    python -m app.rag.retrieval_test

This module DOES NOT:
    ✗ Clone or index repositories itself
    ✗ Modify any stored index
"""

from .retriever import _dense_search, retrieve_chunks
from .sparse_store import search_sparse
from .. import config


# (repo, query, expected_path_substring, expected_chunk_name)
LABELED_QUERIES = [
    ("click", "class that represents a single command line option like --verbose", "core.py", "Option"),
    ("click", "class that represents a command line command with a callback function", "core.py", "Command"),
    ("click", "class used to render and update a progress bar in the terminal", "_termui_impl.py", "ProgressBar"),
    ("click", "function that prompts the user to confirm an action with yes or no", "termui.py", "confirm"),
    ("click", "function used to print output to the terminal, click's version of print", "utils.py", "echo"),
    ("click", "class that represents an invocation context holding parsed parameters and parent context", "core.py", "Context"),
    ("click", "class for a command that has multiple named subcommands", "core.py", "Group"),

    ("flask", "the main Flask application class that ties together routing config and request handling", "app.py", "Flask"),
    ("flask", "function that renders a Jinja2 template to a string using the app's template environment", "templating.py", "render_template"),
    ("flask", "function that converts Python data into a JSON HTTP response", "json", "jsonify"),
    ("flask", "class that signs and verifies session cookies using itsdangerous", "sessions.py", "SecureCookieSessionInterface"),
    ("flask", "class used to register a set of routes and handlers as a reusable component of an app", "blueprints.py", "Blueprint"),
    ("flask", "class representing an HTTP response object returned from a view function", "wrappers.py", "Response"),

    ("requests", "class representing an HTTP session that persists cookies and connections across requests", "sessions.py", "Session"),
    ("requests", "function that sends an HTTP GET request", "api.py", "get"),
    ("requests", "class that represents an HTTP response including status code and body", "models.py", "Response"),
    ("requests", "class that adapts HTTP connections using urllib3, handling retries and connection pooling", "adapters.py", "HTTPAdapter"),
    ("requests", "class representing a fully prepared HTTP request ready to be sent over the wire", "models.py", "PreparedRequest"),
    ("requests", "mixin class that resolves and follows HTTP redirect responses", "sessions.py", "SessionRedirectMixin"),

    # Exact-identifier queries -- this is the case sparse exists to win.
    ("requests", "resolve_redirects", "sessions.py", "resolve_redirects"),
    ("click", "_pipepager", "_termui_impl.py", "_pipepager"),
]


def _rank_of(results, path_sub, chunk_name):
    for i, r in enumerate(results, start=1):
        if path_sub in r["path"] and r["chunk_name"] == chunk_name:
            return i
    return None


def _metrics(ranks):
    n = len(ranks)
    hit1 = sum(1 for r in ranks if r == 1)
    hit3 = sum(1 for r in ranks if r and r <= 3)
    hit5 = sum(1 for r in ranks if r and r <= 5)
    mrr = sum((1.0 / r) if r else 0.0 for r in ranks) / n
    return {"n": n, "hit@1": hit1, "hit@3": hit3, "hit@5": hit5, "mrr": round(mrr, 3)}


def run(top_k: int = 10) -> dict:
    """Report dense-only, sparse-only, and fused metrics separately."""

    dense_ranks, sparse_ranks, fused_ranks = [], [], []

    for repo, query, path_sub, chunk_name in LABELED_QUERIES:
        dense = _dense_search(repo, query, top_k)
        sparse = search_sparse(repo, query, top_k, widen=False)
        fused = retrieve_chunks(repo, query, top_k=top_k)

        dense_ranks.append(_rank_of(dense, path_sub, chunk_name))
        sparse_ranks.append(_rank_of(sparse, path_sub, chunk_name))
        fused_ranks.append(_rank_of(fused, path_sub, chunk_name))

    return {
        "dense_only": _metrics(dense_ranks),
        "sparse_only": _metrics(sparse_ranks),
        "fused": _metrics(fused_ranks),
    }


if __name__ == "__main__":
    results = run()
    for name, metrics in results.items():
        print(f"{name:<12} {metrics}")
