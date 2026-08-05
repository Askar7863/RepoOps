"""
main.py

RepoOps — FastAPI Entrypoint.

Responsibility
--------------
The central file: it wires the whole system together and is what
actually runs. Two things happen here, and nothing else --

  1. POST /api/repositories/load  -- clone a GitHub repo and index it:
     scan -> filter -> chunk -> embed + store (dense in ChromaDB, sparse
     in BM25). Each step is implemented in app/github/ and app/rag/;
     this endpoint only orchestrates and logs them.

  2. POST /api/chat  -- run one message through the LangGraph agent
     (app/agent/graph.py): classify intent -> dispatch to a tool
     (app/agent/tools/) -> verify the answer, retrying once if it fails
     verification. code_writer_tool and repo_summarizer_tool are
     implemented; every other declared tool reports itself as
     not-yet-built rather than silently misrouting.

This module DOES NOT:
    ✗ Implement cloning, chunking, indexing, retrieval, or tool logic
      itself (see app/github/, app/rag/, app/agent/)
"""

import sys
from pathlib import Path

# Guard against running under the wrong Python interpreter -- this has
# repeatedly happened in practice: a second server started in a new
# terminal where the venv wasn't actually activated resolves `python`/
# `uvicorn` to a system-wide install with none of this project's
# dependencies, and every request against that process fails with a
# confusing, unrelated-looking ModuleNotFoundError deep inside chunker.py
# or similar. Checked with only stdlib (sys, pathlib) so this guard itself
# can never be the thing that fails to import.
_EXPECTED_VENV = (Path(__file__).resolve().parent.parent / "venv").resolve()

if Path(sys.prefix).resolve() != _EXPECTED_VENV:
    print(
        "\nERROR: this process is not running from this project's own venv.\n"
        f"  Expected interpreter under: {_EXPECTED_VENV}\n"
        f"  Actually running from:      {Path(sys.prefix).resolve()}\n\n"
        "This causes a second server process to run with no dependencies "
        "installed, which looks alive but fails every real request.\n\n"
        "Fix -- activate the venv first, or run the venv's python directly:\n"
        f"  {_EXPECTED_VENV / 'Scripts' / 'python.exe'} -m app.main\n"
    )
    sys.exit(1)

from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .active_repo import get_active_repository, set_active_repository
from .github.clone import clone_repository, REPO_STORAGE
from .github.scanner import scan_repository
from .github.filter import filter_repository_files
from .github.chunker import chunk_repository
from .rag.indexer import index_repository
from .rag.chroma_store import delete_collection
from .rag.sparse_store import delete_sparse_index
from .agent.pipeline import run_agent
from .agent.tools.repo_health import compute_health, narrate_health

from .utils.logger import (
    log_error,
    log_step,
    log_info,
    log_success,
    log_count,
    log_list,
)


app = FastAPI(title="RepoOps")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class RepositoryRequest(BaseModel):
    repo_url: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/api/chat")
def chat(req: ChatRequest):
    state = {
        "message": req.message,
        "active_repository": get_active_repository() or "",
        "sources": [],
        "tool_used": "general_chat",
        "error": None,
    }

    state = run_agent(state)

    return {
        "reply": state["reply"],
        "tool_used": state["tool_used"],
        "sources": state["sources"],
        "verified": state.get("verified"),
        "verification_reason": state.get("verification_reason"),
    }


@app.post("/api/repositories/health")
def repository_health():
    """
    Direct health-check endpoint for the UI's Health Check button --
    deliberately bypasses run_agent/classify_intent entirely. A button
    click is already an unambiguous intent; spending an LLM call to
    have the router guess it back would just add latency and cost for
    no benefit. Calls the same compute_health/narrate_health functions
    repo_health_tool's chat path uses, so there is exactly one place
    that computes these numbers -- this just returns the structured
    numbers ALONGSIDE the narrative (the chat path only returns the
    narrative), since a real dashboard UI needs the raw numbers to
    render stat tiles, not just prose to display in a chat bubble.
    """

    active = get_active_repository()
    if not active:
        raise HTTPException(
            status_code=400,
            detail="No repository is loaded yet. Load a GitHub repository first.",
        )

    local_path = REPO_STORAGE / active

    health = compute_health(active, local_path)

    try:
        narrative = narrate_health(health["report_text"])
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not generate the health report narrative: {exc}",
        ) from exc

    return {
        "repository": active,
        "health_score": health["health_score"],
        "sub_scores": health["sub_scores"],
        "code_health": health["code_health"],
        "git_info": health["git_info"],
        "size_info": health["size_info"],
        "dependency_info": health["dependency_info"],
        "security_info": health["security_info"],
        "complexity_info": health["complexity_info"],
        "documentation_score": health["documentation_score"],
        "testing_score": health["testing_score"],
        "narrative": narrative,
    }


@app.post("/api/repositories/load")
def load_repository(req: RepositoryRequest):
    """Clone a GitHub repository and index it end to end."""

    repo_url = req.repo_url.strip()

    parsed = urlparse(repo_url)

    if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() not in {
        "github.com",
        "www.github.com",
    }:
        raise HTTPException(
            status_code=400,
            detail="Enter a valid GitHub repository URL.",
        )

    previous_repository = get_active_repository()

    try:

        ###################################################
        # STEP 1 - CLONE REPOSITORY
        ###################################################

        log_step("STEP 1 - CLONE REPOSITORY")

        local_path = clone_repository(repo_url)

        log_success("Repository cloned successfully")
        log_info(f"Location : {local_path}")

        ###################################################
        # STEP 2 - SCAN REPOSITORY
        ###################################################

        log_step("STEP 2 - SCAN REPOSITORY")

        files = scan_repository(local_path)

        log_count("Total Files", len(files))

        ###################################################
        # STEP 3 - FILTER FILES
        ###################################################

        log_step("STEP 3 - FILTER FILES")

        filtered_files = filter_repository_files(files)

        log_count("Useful Files", len(filtered_files))

        log_list(
            "Sample Files",
            [
                str(file.relative_to(local_path))
                for file in filtered_files
            ],
        )

        ###################################################
        # STEP 4 - CHUNK REPOSITORY
        ###################################################

        log_step("STEP 4 - CHUNK REPOSITORY")

        chunks = chunk_repository(
            filtered_files=filtered_files,
            repo_root=local_path,
        )

        log_count("Total Chunks", len(chunks))

        log_success("Chunking completed successfully")

        ###################################################
        # STEP 5 - GENERATE EMBEDDINGS & STORE (DENSE + SPARSE)
        ###################################################

        log_step("STEP 5 - GENERATE EMBEDDINGS & STORE (DENSE + SPARSE)")

        index_result = index_repository(
            repository_name=local_path.name,
            chunks=chunks,
        )

        log_success("Repository indexed successfully")
        log_info(f"Collection : {index_result['collection']}")
        log_count("Vectors Stored", index_result["vectors_stored"])
        log_count("Sparse Indexed", index_result["sparse_indexed"])

        set_active_repository(local_path.name)

        ###################################################
        # STEP 6 - DROP THE PREVIOUS REPOSITORY'S INDEX
        ###################################################
        # Only one repository's data should ever be queryable at a time --
        # the agent must not be able to answer from a previously loaded
        # repo. Retrieval was already scoped to the active collection by
        # name, so this couldn't leak through a chat request; but leaving
        # a stale index sitting on disk after switching repos is exactly
        # the kind of stale state that bit us once already (see: two
        # server processes disagreeing on the active repo). Deleting it
        # here removes that risk at the source instead of only guarding
        # against it at query time.

        if previous_repository and previous_repository != local_path.name:

            log_step("STEP 6 - DROP PREVIOUS REPOSITORY INDEX")

            delete_collection(previous_repository)
            delete_sparse_index(previous_repository)

            log_success(f"Removed index for previous repository: {previous_repository}")

    except Exception as exc:

        log_step("PIPELINE FAILED")
        log_error(str(exc))

        raise HTTPException(
            status_code=400,
            detail=f"Could not process repository: {exc}",
        ) from exc

    return {
        "success": True,
        "repo_url": repo_url,
        "local_path": str(local_path),

        "total_files": len(files),
        "filtered_files": len(filtered_files),
        "total_chunks": len(chunks),

        "message": "Repository indexed successfully.",

        "indexing": index_result,

        "sample_files": [
            str(file.relative_to(local_path))
            for file in filtered_files[:10]
        ],

        "sample_chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "chunk_name": chunk["chunk_name"],
                "chunk_type": chunk["chunk_type"],
                "path": chunk["path"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
            }
            for chunk in chunks[:5]
        ],
    }


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


if __name__ == "__main__":
    import uvicorn

    # reload_dirs restricts the dev-reloader to this package only. Without
    # it, uvicorn --reload watches the whole project root -- including
    # repos/ (where code_writer_tool writes), chroma_db/, bm25_db/, and
    # .active_repository, all of which the app itself writes to at
    # runtime. Every one of those writes looks like a source change and
    # triggers a restart mid-request. Watching only app/ (the actual
    # source code) fixes that while keeping --reload for real edits.
    uvicorn.run("app.main:app", host="127.0.0.1", port=8020, reload=True, reload_dirs=["app"])
