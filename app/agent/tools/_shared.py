"""
_shared.py

Shared Tool Helpers.

Responsibility
--------------
Common building blocks every tool handler needs: a guarded LLM call, the
standard state shortcuts for the cases that show up identically across
tools (no repository loaded, LLM unreachable), path resolution, and
fetching a symbol's true complete source (built for code_writer_tool --
a RAG chunk can be a fragment of an oversized function, so editing from
a partial view invites reconstructing the missing parts instead of
working from what's really there).

This module DOES NOT:
    ✗ Implement any specific tool's logic
    ✗ Classify or route (see router.py)
"""

from pathlib import Path
from typing import Optional

from ... import config
from ...utils.logger import log_error
from ...github.chunker import chunk_code_file, chunk_config_file, chunk_doc_file, find_symbol_bounds
from ...github.language import detect_category
from ..llm_client import get_client


DEFAULT_TEMPERATURE = 0.2

NO_REPO_REPLY = "No repository is loaded yet. Load a GitHub repository first, then ask again."

OUTLINE_HANDLERS = {
    "code": chunk_code_file,
    "doc": chunk_doc_file,
    "config": chunk_config_file,
}


def call_llm(system: str, user_content: str, temperature: float = DEFAULT_TEMPERATURE) -> str:
    """Plain (non-tool-calling) chat completion. Raises on failure -- callers
    are expected to catch and route through llm_error()."""

    try:
        resp = get_client().chat.completions.create(
            model=config.AZURE_CHAT_DEPLOYMENT,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        )
        return resp.choices[0].message.content
    except Exception as exc:
        log_error(str(exc))
        raise


def no_repo(state: dict, tool_name: str) -> dict:
    state["reply"] = NO_REPO_REPLY
    state["sources"] = []
    state["tool_used"] = tool_name
    return state


def llm_error(state: dict, tool_name: str, exc: Exception) -> dict:
    state["reply"] = "Sorry, I couldn't reach the language model. Please try again."
    state["sources"] = []
    state["tool_used"] = tool_name
    state["error"] = str(exc)
    return state


def resolve_target(local_path: Path, file_path: str) -> Optional[Path]:
    """Resolve file_path against the repo root; None if it would escape the repo."""

    root = local_path.resolve()
    candidate = (root / file_path).resolve()

    if candidate != root and root not in candidate.parents:
        return None

    return candidate


def full_symbol_source(local_path: Path, chunk: dict) -> Optional[dict]:
    """
    Fetch a chunk's symbol complete and unfragmented, straight from disk.

    Chunking splits an oversized function/class into multiple pieces
    ("echo", "echo#2", "echo#3", ...) so no single piece truncates at the
    embedder's token limit -- correct for retrieval, but it means a
    RAG-retrieved chunk can be only a fraction of the real function.
    Handing an LLM a fragment and asking it to reason about "the complete"
    version of it (whether to edit it or diagnose a bug in it) invites it
    to reconstruct the missing parts from general knowledge instead of
    working from what's really there. find_symbol_bounds reads the
    actual tree-sitter boundaries, independent of that splitting, so this
    always returns the true, complete current source for a real code
    symbol -- or None if the chunk isn't a locatable code symbol (a doc/
    config chunk, or one find_symbol_bounds can't find, e.g. it was
    renamed since indexing).
    """

    target = resolve_target(local_path, chunk["path"])
    if target is None or not target.exists():
        return None

    symbol_name = chunk["chunk_name"].split("#")[0]
    bounds = find_symbol_bounds(target, local_path, symbol_name)
    if bounds is None:
        return None

    start_line, end_line = bounds
    lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()

    return {
        "path": chunk["path"],
        "symbol_name": symbol_name,
        "start_line": start_line,
        "end_line": end_line,
        "content": "\n".join(lines[start_line - 1: end_line]),
    }


def file_outline(local_path: Path, relative_path: str) -> str:
    """
    A compact map of every symbol/section in a file -- name, kind, line
    range, no content -- so the LLM can see the file's whole structure
    instead of just whatever handful of chunks a global cross-repo search
    happened to surface (which may all come from other files entirely,
    leaving it blind to the rest of the actual file in question).

    Re-chunks the file fresh (cheap -- one file, not the whole repo) so
    this always reflects the file's current real content, not a stale
    index. Returns "" if the file has no known category or can't be
    parsed -- callers should treat that as "no outline available", not
    an error.
    """

    target = resolve_target(local_path, relative_path)
    if target is None or not target.exists():
        return ""

    category = detect_category(target)
    handler = OUTLINE_HANDLERS.get(category)
    if handler is None:
        return ""

    try:
        chunks = handler(target, local_path)
    except Exception:
        return ""

    if not chunks:
        return ""

    # Merge a run of oversized-split fragments ("echo", "echo#2", "echo#3")
    # back into one logical entry -- this is a structural map of the
    # file, not a re-listing of the embedding index, so splitting that
    # exists purely to avoid embedder truncation shouldn't appear as
    # three separate functions named echo. Only merge a chunk into the
    # PREVIOUS row, and only when its own name carries the "#N" suffix
    # that marks it as a continuation -- never merge by name+kind alone,
    # since two genuinely different symbols routinely share a name (every
    # class's own __init__, __repr__, ...) and merging those would report
    # a wildly wrong line range spanning unrelated classes.
    rows: list = []

    for c in chunks:
        name, kind = c["chunk_name"], c["chunk_type"]
        is_fragment = "#" in name
        base_name = name.split("#")[0]

        if is_fragment and rows and rows[-1]["name"] == base_name and rows[-1]["kind"] == kind:
            rows[-1]["end"] = max(rows[-1]["end"], c["end_line"])
        else:
            rows.append({
                "name": base_name, "kind": kind, "start": c["start_line"], "end": c["end_line"],
                "replaceable": c.get("replaceable", True),
            })

    return "\n".join(
        f"  {r['kind']:<14} {r['name']:<30} lines {r['start']}-{r['end']}"
        + ("" if r["replaceable"] else "  (embedded callback fragment -- do not replace_symbol this; append instead)")
        for r in rows
    )
