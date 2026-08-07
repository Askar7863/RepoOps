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

# ---------------------------------------------------------------------------
# Personas -- an audience LENS on top of a tool's existing narration, never
# a change to the underlying facts. Every tool here already separates
# "compute the real numbers/content" from "have the LLM narrate them" --
# this plugs into that same seam: the SAME computed data goes in regardless
# of persona, only how it gets explained changes. A persona must never be
# allowed to change which numbers, sources, or code get produced -- if it
# ever does, that's a bug, not a feature.
# ---------------------------------------------------------------------------

PERSONA_LENSES = {
    "student": (
        "AUDIENCE LENS -- STUDENT: this reader is learning the codebase, not "
        "already fluent in it. Favor CONCEPTUAL explanation over technical "
        "mechanism -- explain what something is for and why it matters, before "
        "(or instead of) exactly how it's implemented under the hood. Define any "
        "technical term, acronym, or pattern name the first time you use it, in "
        "plain language -- never assume it's already understood. Prefer everyday "
        "analogies over jargon where they genuinely help. Where useful, point out "
        "where a newcomer could look next to learn more. Keep the tone approachable "
        "and patient, not clinical -- this is someone building a mental model, not "
        "shipping a fix right now. Less technical, more conceptual, throughout."
    ),
    "developer": (
        "AUDIENCE LENS -- DEVELOPER: this reader is actively working in this "
        "codebase and already comfortable with programming concepts and this "
        "project's terminology. Be technical and precise -- don't spend words "
        "defining things a working developer already knows. Lead with the "
        "concrete, actionable answer, citing exact files/functions/lines where "
        "relevant. Keep it efficient: skip conceptual scene-setting unless it's "
        "directly needed to act on the answer."
    ),
    "architect": (
        "AUDIENCE LENS -- ARCHITECT: this reader is assessing the system as a "
        "whole, not one task. Frame the answer in terms of structure, trade-offs, "
        "and consequences -- what this means for maintainability, risk, and where "
        "to invest effort -- rather than line-level implementation detail. Where "
        "the data actually supports it, make an explicit recommendation or call "
        "out a risk rather than only describing. Prioritize the few things that "
        "matter strategically over an exhaustive walkthrough."
    ),
}


def persona_lens(persona: Optional[str]) -> str:
    """
    The narration-framing paragraph for one persona, or "" for no persona
    (None) or an unrecognized value -- deliberately a true no-op in that
    case, not a fallback to some default paragraph, so any caller that
    never passes a persona keeps today's EXACT prompt, unaffected by this
    feature existing at all.
    """

    if not persona:
        return ""
    return PERSONA_LENSES.get(persona, "")


def call_llm(
    system: str,
    user_content: str,
    temperature: float = DEFAULT_TEMPERATURE,
    persona: Optional[str] = None,
    conversation_context: Optional[str] = None,
) -> str:
    """Plain (non-tool-calling) chat completion. Raises on failure -- callers
    are expected to catch and route through llm_error().

    conversation_context (see memory.py) is prepended to user_content, not
    system -- it's per-turn grounding about what was already said, not a
    change to the tool's own rules, so the strict "use only these
    computed numbers" system prompts stay untouched by this feature."""

    lens = persona_lens(persona)
    full_system = f"{system}\n\n{lens}" if lens else system
    full_user_content = f"{conversation_context}\n\n{user_content}" if conversation_context else user_content

    try:
        resp = get_client().chat.completions.create(
            model=config.AZURE_CHAT_DEPLOYMENT,
            temperature=temperature,
            messages=[
                {"role": "system", "content": full_system},
                {"role": "user", "content": full_user_content},
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
