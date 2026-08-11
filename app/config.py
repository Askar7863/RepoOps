"""Configuration — environment variables and paths for RepoOps."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-small")

# ---------------------------------------------------------------------------
# LLM provider toggle
# ---------------------------------------------------------------------------
# The ONE place to switch which chat-completion provider every tool/router/
# verifier call in the app uses -- set LLM_PROVIDER=anthropic in .env (or
# leave unset/"azure" for the original behavior). See app/agent/llm_client.py
# for how this is actually applied: every existing call site keeps calling
# `get_client().chat.completions.create(...)` completely unchanged, and
# `model=CHAT_MODEL` below resolves to the right model string either way, so
# switching providers never means touching any of the ~10 call sites across
# router.py/verifier.py/the tools -- only this config value (or the env var
# behind it) needs to change.
#
# Embeddings (app/rag/embeddings.py) are a separate, always-local model and
# are NOT affected by this toggle -- only chat/completion calls switch.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "azure").strip().lower()  # "azure" | "anthropic"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
ANTHROPIC_CHAT_MODEL = os.getenv("ANTHROPIC_CHAT_MODEL", "claude-sonnet-4-6")

CHAT_MODEL = ANTHROPIC_CHAT_MODEL if LLM_PROVIDER == "anthropic" else AZURE_CHAT_DEPLOYMENT

# The always-fast tier for internal routing/classification calls (see
# llm_client.get_fast_client) -- deliberately NOT affected by LLM_PROVIDER.
# These are narrow, forced-schema decisions ("which tool", "in scope or
# not") where a bigger/slower model adds latency, not quality -- measured
# directly: a single query chains several of these before ever reaching
# the one call whose output the user actually reads.
FAST_CHAT_MODEL = AZURE_CHAT_DEPLOYMENT

# ---------------------------------------------------------------------------
# RAG pipeline tunables (app/github/chunker.py, app/rag/*)
# ---------------------------------------------------------------------------

# Embedding model — must stay in sync with EMBED_DIMENSIONS. Swapping models
# means re-indexing everything, not just changing this value.
LOCAL_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIMENSIONS = 384

# BAAI/bge-* models are trained asymmetrically: queries need this instruction
# prefix, indexed content never does. Omitting it doesn't error — it just
# quietly makes every retrieval worse.
LOCAL_EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

LOCAL_EMBED_BATCH_SIZE = 8

# Oversized single chunks (a big class, a long doc section) get re-split
# before embedding instead of silently truncating at the model's 512-token
# window. Measured with the embedding model's OWN tokenizer (see
# app/rag/embeddings.py::count_embedding_tokens) -- not an LLM tokenizer,
# which would undercount and let chunks through that still get truncated.
# 512 minus room for [CLS]/[SEP] and a safety margin.
MAX_CHUNK_TOKENS = 480

# Fixed-size fallback chunker — used only when a file has no chunkable
# structure (unsupported language, parse failure, no functions/classes found).
# Same embedding-tokenizer basis as MAX_CHUNK_TOKENS.
CHUNK_TOKEN_SIZE = 450
CHUNK_TOKEN_OVERLAP = 50

# Hybrid retrieval: dense (embeddings) + sparse (BM25) fused with
# Reciprocal Rank Fusion. See app/rag/reranker.py.
DENSE_TOP_K = 30
SPARSE_TOP_K = 30
RRF_K = 60

# Sparse is weighted well below dense by default -- calibrated against a
# 21-query labeled set spanning both descriptive ("class that represents
# an HTTP session") and exact-identifier ("resolve_redirects") queries,
# swept from 0.0 to 0.4 (see conversation/benchmark notes). Finding:
# unweighted (1.0) or even lightly-weighted (0.4/0.2/0.1) sparse
# measurably HURTS descriptive queries. BM25 on a narrow, domain-specific
# codebase confidently rewards incidental keyword overlap ("--verbose" in
# an unrelated CLI example) over true relevance, because the query's
# meaningful words are also the corpus's own ubiquitous vocabulary and so
# carry almost no IDF weight -- and because RRF's rank gap between a
# correct dense rank-1 and an incorrect rank-2 is tiny (~0.00026 at
# RRF_K=60), it doesn't take much sparse credit to flip that ordering.
#
# Note: exact-identifier queries do NOT depend on this weight -- a chunk
# whose own name matches a query token is promoted explicitly in
# retriever.py's _promote_exact_matches, decisively, independent of the
# weight below (rank-based RRF credit alone couldn't reliably surface an
# unambiguous match at a weight low enough to be safe for descriptive
# queries). What this weight controls now is a narrower, genuine signal:
# rewarding a chunk dense AND sparse both independently rank well without
# an exact name match. 0.05 keeps almost all of dense-only's
# descriptive-query accuracy (matched exactly on Hit@3/Hit@5 in the
# sweep) while still counting that agreement for something.
RRF_DENSE_WEIGHT = 1.0
RRF_SPARSE_WEIGHT = 0.05

# If the top fused score doesn't clear this, retry once with a widened
# sparse query (stopwords stripped, identifiers split) before giving up.
# Scaled to RRF_DENSE_WEIGHT/RRF_SPARSE_WEIGHT: a dense-only rank-1 match
# (1.0/(RRF_K+1) ~= 0.0164) alone should already count as high confidence,
# so this sits comfortably below that rather than at the old unweighted
# ceiling -- with sparse weighted down, that ceiling no longer reflects
# what a genuinely confident match scores.
RRF_CONFIDENCE_THRESHOLD = 0.012
