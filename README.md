# RepoOps

An agent that clones a GitHub repository, indexes it with a hybrid RAG
pipeline, and lets you chat with it — including writing and modifying
code directly in the cloned repo.

## Setup

```bash
python -m venv venv
venv\Scripts\activate      # Windows; use `source venv/bin/activate` on macOS/Linux
pip install -r requirements.txt

cp .env.example .env       # then fill in your Azure OpenAI credentials

cd frontend && npm install && npm run build && cd ..   # builds the React frontend once

python -m app.main
```

The frontend lives in `frontend/` (React + TypeScript + Tailwind + shadcn/ui)
and is a separate project from the Python backend. `npm run build` produces
`frontend/dist/`, which `app/main.py` serves directly -- no separate frontend
server needed for normal use. For frontend development with hot reload, run
`npm run dev` inside `frontend/` (proxies `/api/*` to the backend on 8020) and
use <http://localhost:5173> instead while iterating on UI code.

That runs uvicorn with `--reload` restricted to `app/` only (see the
`if __name__ == "__main__"` block at the bottom of `main.py`). If you'd
rather invoke uvicorn directly, pass the same restriction explicitly --
otherwise its file watcher covers the whole project root, including
`repos/`, `chroma_db/`, `bm25_db/`, and `.active_repository`, all of
which the app writes to at runtime, and every one of those writes would
trigger a mid-request restart:

```bash
uvicorn app.main:app --reload --reload-dir app --port 8020
```

Open <http://127.0.0.1:8020>, paste a GitHub repo URL, click **Load
Repository**, then ask about it in the chat.

**Only ever run one server process.** Starting it a second time in a new
terminal where the venv wasn't re-activated resolves `python`/`uvicorn`
to a system-wide install with none of this project's dependencies --
that second process looks alive but fails every real request in a
confusing way. `main.py` guards against this: importing it under the
wrong interpreter now fails immediately with a clear message instead of
a cryptic `ModuleNotFoundError` deep inside `chunker.py`.

## Architecture

```
app/main.py                 -- FastAPI entrypoint (the only file you run)
  POST /api/repositories/load    clone -> scan -> filter -> chunk -> index
  POST /api/chat                 run one message through the agent graph
  GET  /*                        serves the built frontend (frontend/dist/)

frontend/                    -- React + TypeScript + Tailwind + shadcn/ui,
                                 a separate project built with `npm run build`
                                 (see Setup above); not part of the Python app

app/active_repo.py           -- which repo is currently loaded, persisted
                                 to .active_repository at the project root
                                 (not an in-memory variable -- if more than
                                 one server process ever runs, each would
                                 otherwise have its own disconnected idea
                                 of the active repo)

app/github/                 -- ingestion
  clone.py       git clone a repo URL into repos/<name>/
  scanner.py     walk every file in the clone
  filter.py      keep only source/doc/config files, drop build artifacts
  language.py    detect each file's language + chunking category
  chunker.py     tree-sitter AST chunking (code), heading-split (docs),
                 top-level-key split (config); token-bounded fixed-size
                 fallback everywhere else; find_symbol_bounds() locates
                 an existing function/class's exact source lines for
                 the code writer tool

app/rag/                    -- retrieval
  embeddings.py     local sentence-transformers model (query vs document
                    encoding are separate -- BGE models need a query prefix)
  chroma_store.py   dense vector store (ChromaDB)
  sparse_store.py   BM25 keyword index, persisted alongside it
  reranker.py       Reciprocal Rank Fusion across dense + sparse
  retriever.py      hybrid retrieve: dense + sparse -> fuse -> confidence
                    check -> exact-identifier promotion
  indexer.py        orchestrates embeddings + both stores, idempotently;
                    reindex_file() incrementally re-indexes just ONE file
                    after code_writer_tool changes it, so it's findable
                    by a later request without a full repo reload
  retrieval_test.py labeled-query regression test (`python -m app.rag.retrieval_test`)

app/agent/                  -- orchestration (LangGraph)
  graph.py       the compiled StateGraph: resolve_local_path ->
                 classify_intent -> route_to_tool -> verify_answer,
                 retrying route_to_tool once on failed verification
                 (never for a tool with side effects -- see tools/__init__.py)
  pipeline.py    run_agent(state) -- the entrypoint main.py calls
  router.py      LLM function-calling classifier; declares the 3 tools below
  verifier.py    judges whether a tool's reply is grounded and adequate
  state.py       the AgentState shape passed through the graph
  llm_client.py  Azure OpenAI client factory
  tools/               all three tools below are fully implemented --
                       this router declares nothing it doesn't build
    code_writer.py    writes/modifies real files on disk
                      (create_file / append_to_file / replace_symbol for
                      code; any operation replaces a whole-document file
                      like HTML/CSS/JSON in place, never appends one),
                      then re-indexes just that file (see reindex_file)
    repo_summarizer.py summarizes the repo, or answers a
                      general/usage question about it, grounded directly
                      in its markdown docs (read whole, not RAG-chunked)
                      plus a structural overview from already-indexed
                      chunk metadata. Deliberately handles both "give me
                      a summary" and "how do I use X" -- one tool, not
                      two overlapping ones for the classifier to confuse
                      (the router used to declare these separately as
                      repo_summarizer + ask_repo)
    general_chat.py   baseline conversational fallback (not one of the
                      3 declared tools -- this is what the router falls
                      back to when no tool matches)
    repo_health.py    READ-ONLY health check, every
                      number computed in Python and handed to the LLM
                      as fact (never a free-form impression):
                        - CODE HEALTH: language/file distribution,
                          oversized functions, TODO/FIXME/HACK density,
                          duplicate symbol names across files, a
                          Python-only docstring-coverage heuristic
                        - REPO ACTIVITY & MAINTENANCE: from `git log`
                          on the local clone -- commit frequency,
                          last-commit recency, contributor count and
                          bus-factor, hotspot files, a 12-week commit
                          activity series
                        - REPOSITORY SIZE: on-disk bytes + line count
                        - DEPENDENCIES: parsed from whatever manifests
                          are present (requirements.txt, package.json,
                          pyproject.toml via tomli, Pipfile, Cargo.toml
                          -- go.mod is detected but not parsed, see
                          _dependencies for why)
                        - SECURITY: a STATIC pattern scan only (secret-
                          shaped strings, committed .env, 
                          ignore
                          presence) -- explicitly NOT a vulnerability/
                          CVE audit, labeled as such everywhere it's
                          surfaced
                        - CODE COMPLEXITY: an approximate per-function
                          complexity via control-flow keyword counting
                          -- NOT real AST-based cyclomatic complexity
                      Six composite 0-100 scores (documentation,
                      testing, security, complexity, activity, bus-
                      factor) combine into one overall Health Score,
                      each via an explicit, disclosed formula -- never
                      asked of the LLM to invent. Fully offline, no
                      network calls. Distinct axis from
                      repo_summarizer_tool: WHAT the repo is/does vs.
                      HOW GOOD/well-maintained it is.
    code_writer_test.py          10 structural regression cases (path
                                  escape, missing symbol, existing-file
                                  collision, full-behavior preservation,
                                  create-then-modify, ...) -- run with
                                  `python -m app.agent.tools.code_writer_test`
    code_writer_website_test.py  5 more, verifying actual generated-code
                                  quality (real HTML parse, JS syntax via
                                  `node --check`, unicode round-trip) --
                                  run with `python -m
                                  app.agent.tools.code_writer_website_test`
    repo_health_test.py           65 cases: deterministic unit checks on
                                  every metric function against click's
                                  real git history and index (docstring
                                  heuristic on multi-line signatures,
                                  test/example path filtering, dependency
                                  parsing cross-checked against a hand-
                                  count of click's own pyproject.toml,
                                  secret-scan placeholder-vs-real-looking
                                  distinction, health-score reweighting
                                  when sub-scores are missing, and edge
                                  cases for every metric), router
                                  disambiguation vs. repo_summarizer_tool,
                                  and full-pipeline report-shape + read-
                                  only-guarantee checks -- run with
                                  `python -m app.agent.tools.repo_health_test`
    __init__.py       wires the 3 tool names to their handlers, plus
                      general_chat as the router's fallback

app/utils/logger.py         -- console logging used throughout ingestion
```

## Tools

`router.py` declares exactly three tools, and all three are fully
implemented -- there is no "declared but not built yet" tier in this
codebase. Adding a new tool means writing its handler in
`app/agent/tools/`, adding its schema to `TOOL_SCHEMAS`, and pointing
`TOOL_DISPATCH` at it in `router.py`.
