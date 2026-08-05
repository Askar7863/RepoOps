"""
code_writer.py

Code Writer Tool.

Responsibility
--------------
The one fully implemented repository-action tool: given a natural-language
request, decide what file needs touching, whether that means creating it
or modifying an existing symbol inside it, generate the code, and
actually write it to disk in the cloned repository.

Pipeline
--------
User request
    -> Retrieve reference code (existing hybrid RAG pipeline, for style
       AND to locate the real file_path/symbol_name of an existing target
       -- see full_symbol_source, file_outline)
    -> LLM commits to one concrete plan: file_path, operation,
       symbol_name (if replacing), code
    -> Validate the target path resolves inside the repo (no path escape)
    -> Apply the write:
         new file          -- written as-is, regardless of which
                              operation was nominally requested
         existing CODE file (.py/.cpp/.java/...) --
           replace_symbol    -- swap an existing function/class's exact
                                source lines for the new code, located
                                via the same tree-sitter boundaries
                                chunker.py indexes with (see
                                find_symbol_bounds) -- not a second,
                                independently-guessed match. Refuses
                                (writes nothing, reports why) rather than
                                falling back to append when the symbol
                                can't be safely located -- confirmed by
                                testing that append was NOT actually
                                safe here: code generated for a
                                surgical, in-place replace_symbol is
                                shaped like one (it routinely re-emits
                                verbatim context -- imports, sibling
                                declarations -- around the one real
                                change, per the system prompt's own
                                instruction to "preserve every existing
                                branch... verbatim"), so appending it
                                after the existing content duplicates
                                that context instead of adding something
                                new (verified: a real file broke with
                                "Cannot redeclare block-scoped variable"
                                after exactly this fallback fired).
           append_to_file /
           create_file        -- add the new code to the end of the
                                file, since a new function/method
                                genuinely can coexist with what's there
         existing WHOLE-DOCUMENT file (HTML/CSS/JSON/anything without a
         tree-sitter grammar) -- ANY operation replaces the file's
           entire content. Confirmed by testing: for these formats the
           LLM always regenerates the complete document, never a true
           fragment, so appending it after the existing content just
           duplicates the whole file (verified: three concatenated
           copies of one HTML document from a single append).
    -> Re-index just this one file (app.rag.indexer.reindex_file) so a
       later request can find what's actually here now -- otherwise a
       file this tool just touched is invisible to retrieval until the
       whole repo is reloaded, and a later edit request would silently
       fall back to whatever unrelated content scores best in the stale
       index instead (confirmed in practice, not hypothetical)
    -> Report exactly what changed and where

This module DOES NOT:
    ✗ Classify intent (see router.py)
    ✗ Retrieve chunks with its own logic (reuses app.rag.retriever)
"""

import json
from pathlib import Path
from typing import Optional

from ... import config
from ...utils.logger import log_error, log_info, log_warning
from ...github.chunker import find_symbol_bounds
from ...github.language import detect_category
from ...rag.retriever import retrieve_chunks
from ...rag.indexer import reindex_file
from ..llm_client import get_client
from ._shared import no_repo, llm_error, resolve_target, full_symbol_source, file_outline, OUTLINE_HANDLERS


TOOL_NAME = "code_writer_tool"

_VALID_OPERATIONS = {"create_file", "append_to_file", "replace_symbol"}

WRITE_CODE_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "write_code",
            "description": "Commit to exactly one concrete file-level change and produce the code for it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": (
                            "Path to the file, relative to the repository root "
                            "(e.g. 'src/app/utils.py'). Reuse an existing file's exact "
                            "path when modifying it; for a new file, pick a path "
                            "consistent with the repo's existing layout and naming."
                        ),
                    },
                    "operation": {
                        "type": "string",
                        "enum": sorted(_VALID_OPERATIONS),
                        "description": (
                            "create_file: file_path does not exist yet. "
                            "append_to_file: add new code to the end of an existing "
                            "file without touching what's already there. "
                            "replace_symbol: replace one existing function/class's "
                            "full definition with a new version (requires symbol_name)."
                        ),
                    },
                    "symbol_name": {
                        "type": ["string", "null"],
                        "description": (
                            "Exact function or class name to replace. Required when "
                            "operation is replace_symbol; null for every other operation."
                        ),
                    },
                    "code": {
                        "type": "string",
                        "description": (
                            "The code to write: the complete file content for "
                            "create_file, the snippet to append for append_to_file, "
                            "or the complete replacement function/class definition "
                            "for replace_symbol -- never a partial snippet or "
                            "'...rest unchanged...' when replacing a symbol, since it "
                            "replaces the existing definition verbatim."
                        ),
                    },
                    "summary": {
                        "type": "string",
                        "description": "One or two sentences describing the change, for the user.",
                    },
                },
                "required": ["file_path", "operation", "symbol_name", "code", "summary"],
            },
        },
    }
]

WRITE_CODE_SYSTEM_PROMPT = (
    "You are an expert software engineer making a real, direct change to a repository's "
    "source files. When the request targets existing code, locate it using the retrieved "
    "reference chunks' file_path/symbol_name -- never guess a path or symbol name from "
    "memory or convention when a retrieved chunk already names the real one. When a "
    "COMPLETE current implementation is provided, edit it surgically: preserve every "
    "existing branch, error case, and behavior verbatim except what the request explicitly "
    "asks to change -- never rewrite a working function from scratch or simplify it, even "
    "if you could imagine a cleaner version. Match the repo's existing style, naming "
    "conventions, and structure.\n\n"
    "CRITICAL -- a retrieved chunk from a DOCUMENTATION or planning file (a README, a lab/"
    "coursework brief, a design doc, any prose describing what to build) is background "
    "context ONLY, never the file to write to, even when it's the top-ranked match and even "
    "when the request's wording closely echoes that document -- a request to BUILD or "
    "IMPLEMENT something a doc describes means writing the real, executable code somewhere "
    "in the actual source tree (a new file if nothing implements it yet), never editing the "
    "description itself. Only target a documentation file directly when the user explicitly "
    "asks to edit that documentation (e.g. 'update the README', 'add a section to the "
    "coursework doc'). If you're unsure whether a retrieved match is documentation or real "
    "code, its content will read as prose/instructions rather than executable syntax --  "
    "treat any such match as background only.\n\n"
    "Call write_code exactly once with a single, concrete, complete change."
)


def _append(target: Path, code: str) -> dict:
    """
    Append `code` after a blank-line separator. Line numbers are computed
    from actual line LISTS, not newline counting -- counting characters
    gets the arithmetic subtly wrong depending on whether the existing
    file ends with a trailing newline; splitting into lines first doesn't.
    """

    existing_lines = target.read_text(encoding="utf-8", errors="ignore").splitlines() if target.exists() else []
    code_lines = code.rstrip("\n").splitlines()

    if existing_lines:
        new_lines = existing_lines + [""] + code_lines
        start_line = len(existing_lines) + 2  # existing lines, +1 blank separator, +1 to land on the next line
    else:
        new_lines = code_lines
        start_line = 1

    target.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    return {
        "note": "appended to file",
        "start_line": start_line,
        "end_line": start_line + len(code_lines) - 1,
    }


def _replace_symbol(local_path: Path, target: Path, symbol_name: str, code: str) -> Optional[dict]:
    """
    Only ever called for an actual code file (see _apply_write) -- a
    whole-document format is intercepted before reaching here.

    Returns None -- deliberately NOT an append -- when the symbol can't
    be safely located: either it genuinely isn't there (renamed since
    indexing, or the LLM guessed a name), or find_symbol_bounds found it
    but refused it as unsafe (an embedded callback fragment sharing a
    line with a sibling -- see chunker.py's _corrected_bounds). Code
    generated for a replace_symbol is shaped for an in-place swap, not a
    standalone addition -- the system prompt has the LLM re-emit
    surrounding context verbatim around the one real change, so appending
    it instead (the previous behavior) duplicates that context rather
    than adding something new. Confirmed by testing: this exact fallback,
    on a real file, produced a duplicate top-level `const`/import that
    broke the build ("Cannot redeclare block-scoped variable"). The
    caller reports this as a clean failure instead.
    """

    bounds = find_symbol_bounds(target, local_path, symbol_name) if target.exists() else None

    if bounds is None:
        return None

    start_line, end_line = bounds
    code_lines = code.rstrip("\n").splitlines()

    lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
    new_lines = lines[: start_line - 1] + code_lines + lines[end_line:]
    target.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    return {
        "note": f"replaced '{symbol_name}'",
        "start_line": start_line,
        "end_line": start_line + len(code_lines) - 1,
    }


# Plain line-list formats -- no grammar a genuine append could break, unlike
# markdown/HTML/JSON (see the force-replace comment below). Confirmed by
# reproduction: forcing requirements.txt through the same force-replace path
# silently discarded its entire existing dependency list, because the LLM
# (correctly, for an append_to_file request) produced just the one new line
# to add, not a full regenerated file -- there was nothing "whole-document"
# about its behavior here the way there reliably is for markdown/HTML.
_APPEND_SAFE_TEXT_EXTENSIONS = {".txt"}


def _apply_write(
    local_path: Path, target: Path, operation: str, symbol_name: Optional[str], code: str,
) -> Optional[dict]:
    """
    Perform the write; returns what actually happened (may differ from
    the requested operation -- e.g. any operation on an existing
    whole-document file (HTML/CSS/JSON/...) replaces its content rather
    than appending). Returns None, having written NOTHING, when
    operation is replace_symbol and the target symbol can't be safely
    located (see _replace_symbol) -- the caller reports this as a clean
    failure rather than guessing.
    """

    target.parent.mkdir(parents=True, exist_ok=True)
    code = code.rstrip("\n") + "\n"

    if not target.exists():
        # Brand new -- nothing to conflict with, regardless of which
        # operation was nominally requested.
        target.write_text(code, encoding="utf-8")
        return {"note": "created new file", "start_line": 1, "end_line": code.count("\n")}

    # A "whole document" format (HTML/CSS/JSON/etc, or anything without a
    # tree-sitter grammar) has no independently-appendable symbols the
    # way a code file's functions do -- when the LLM touches one of
    # these, `code` is virtually always the complete file regenerated,
    # not a fragment meant to sit alongside what's already there.
    # Confirmed by testing: create_file-on-existing, replace_symbol with
    # no match, AND a directly-requested append_to_file all produced a
    # full regenerated document from the LLM -- appending any of them
    # after the existing content just duplicates the whole file rather
    # than updating it (one test run ended up with three full copies of
    # the same HTML document concatenated). So for these formats, ANY
    # operation on an existing file means "replace it with this,"
    # regardless of which operation was nominally requested. Code files,
    # and plain line-list text files (see _APPEND_SAFE_TEXT_EXTENSIONS),
    # keep the original nuanced behavior below, since a new function --
    # or a new line in a requirements.txt -- genuinely can coexist with
    # the rest of the file.
    if detect_category(target) != "code" and target.suffix.lower() not in _APPEND_SAFE_TEXT_EXTENSIONS:
        target.write_text(code, encoding="utf-8")
        return {
            "note": f"replaced the whole file (non-code format; requested operation was '{operation}')",
            "start_line": 1, "end_line": code.count("\n"),
        }

    if operation == "replace_symbol" and symbol_name:
        return _replace_symbol(local_path, target, symbol_name, code)

    # append_to_file, and create_file's fallback for an existing CODE file
    # (or an append-safe text file)
    return _append(target, code)


def act_code_writer(state: dict) -> dict:
    if state.get("local_path") is None:
        return no_repo(state, TOOL_NAME)

    local_path: Path = state["local_path"]
    query = state["tool_args"].get("query", state["message"])

    reference_chunks = retrieve_chunks(
        repository_name=state["active_repository"],
        query=query,
        top_k=5,
    )

    reference = "\n\n".join(
        f"[{i}] file_path: {chunk['path']}\n"
        f"    symbol_name: {chunk['chunk_name']} ({chunk['chunk_type']})\n"
        f"    relevance_score: {chunk['score']:.4f}\n"
        f"    lines: {chunk['start_line']}-{chunk['end_line']}\n"
        f"    content:\n{chunk['content']}"
        for i, chunk in enumerate(reference_chunks, start=1)
    ) or "(no reference code found)"

    # Retrieval always returns its top_k best-available matches, even when
    # NOTHING in the repo is actually relevant -- there's no absolute score
    # threshold that reliably separates "confident match" from "best of a
    # bad bunch" (a fixed cutoff borrowed from retriever.py's own widen-
    # search heuristic doesn't transfer: in one real, confirmed case, the
    # "winning" chunk scored 0.0164, comfortably above that threshold, and
    # was still just noise -- an entirely unrelated file that happened to
    # rank first for a generic algorithm request with nothing real to
    # match). What actually reveals a weak match is RELATIVE clustering:
    # in that same case, the top 5 scores were 0.0164/0.0161/0.0159/0.0156/
    # 0.0154 -- a top-to-bottom spread of about 6%, i.e. all five chunks
    # were roughly equally (ir)relevant, not a real front-runner. A
    # genuine match (especially an exact-identifier one, which the
    # retriever explicitly promotes) produces a much bigger gap between
    # 1st and 5th place. So the signal handed to the LLM is this spread,
    # not a hardcoded absolute number.
    if len(reference_chunks) >= 2 and reference_chunks[0]["score"] > 0:
        top_score = reference_chunks[0]["score"]
        bottom_score = reference_chunks[-1]["score"]
        spread_pct = round(100 * (top_score - bottom_score) / top_score, 1)
    else:
        spread_pct = None

    # The single most likely target gets its COMPLETE current source
    # fetched fresh from disk (see full_symbol_source) -- the reference
    # chunks above may only be a fragment of it.
    full_target = full_symbol_source(local_path, reference_chunks[0]) if reference_chunks else None

    if full_target:
        current_impl_block = (
            "\n\nCOMPLETE current implementation of the most likely target -- this is "
            f"the real, full, unfragmented content of `{full_target['symbol_name']}` in "
            f"{full_target['path']} (lines {full_target['start_line']}-{full_target['end_line']}). "
            "If the request modifies this symbol, your replacement must preserve every "
            "existing branch, error case, and behavior shown here and change ONLY what "
            "the request asks for -- do not simplify, shorten, or drop anything you don't "
            "see explicitly asked to remove:\n"
            f"{full_target['content']}"
        )

        outline = file_outline(local_path, full_target["path"])
        outline_block = (
            f"\n\nFull structure of {full_target['path']} -- every symbol defined in this "
            "file, for navigation (so you know what else already exists, what pattern/"
            "naming to follow, and what not to duplicate):\n"
            f"{outline}"
        ) if outline else ""
    else:
        current_impl_block = ""
        outline_block = ""

    if spread_pct is not None:
        confidence_note = (
            f"\n\nTop-to-bottom relevance spread across these {len(reference_chunks)} chunks: "
            f"{spread_pct}%. A LOW spread (roughly under 20%) means every chunk above is "
            "roughly equally (ir)relevant -- there is no genuine front-runner, which almost "
            "always means the request is for NEW, standalone functionality that nothing in "
            "this repo actually implements yet, not a change to something existing. In that "
            "case, do NOT default to chunk [1]'s file just because it's ranked first -- pick "
            "(or create) a file whose own PURPOSE actually matches what you're building, "
            "following the repo's real naming/folder conventions, the same way a careful "
            "human contributor would. A HIGH spread means chunk [1] is a clear, confident "
            "match and its file_path is trustworthy."
        )
    else:
        confidence_note = ""

    user_content = (
        "Reference chunks retrieved from the repository, ranked most relevant first:\n"
        f"{reference}"
        f"{confidence_note}"
        f"{outline_block}"
        f"{current_impl_block}\n\n"
        "If the request refers to EXECUTABLE CODE that already exists (\"fix X\", \"modify Y\", "
        "\"add a parameter to Z\") and the top matching chunk IS that code, its file_path and "
        "symbol_name above ARE the real location in this repository -- use those exact values "
        "verbatim, don't guess a path or re-derive a name from general knowledge or convention. "
        "But if the top matching chunk is prose from a documentation/planning file (its "
        "chunk_type will typically read as a doc-style section, not Function/Class/Method, and "
        "its content is sentences describing intent rather than syntax) -- that file_path is "
        "NOT a write target, no matter how closely the request's wording matches it; it's "
        "background material telling you what to build, so pick an appropriate real source "
        "file for the actual implementation instead, following the repo's existing layout. "
        "The chunk content itself is reference for matching style/conventions/requirements -- "
        "don't repeat it back verbatim in your output.\n\n"
        f"Request:\n{query}"
    )

    state["context_used"] = user_content

    try:
        resp = get_client().chat.completions.create(
            model=config.AZURE_CHAT_DEPLOYMENT,
            temperature=0.2,
            tools=WRITE_CODE_SCHEMA,
            tool_choice={"type": "function", "function": {"name": "write_code"}},
            messages=[
                {"role": "system", "content": WRITE_CODE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        plan = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
    except Exception as exc:
        return llm_error(state, TOOL_NAME, exc)

    file_path = (plan.get("file_path") or "").strip().lstrip("/\\")
    operation = plan.get("operation")
    symbol_name = plan.get("symbol_name") or None
    code = plan.get("code") or ""
    summary = plan.get("summary") or ""

    if not file_path or operation not in _VALID_OPERATIONS or not code.strip():
        state["reply"] = (
            "I couldn't determine a concrete file change from that request. "
            "Try naming the file and describing the change more specifically."
        )
        state["sources"] = []
        state["tool_used"] = TOOL_NAME
        state["error"] = "invalid_plan"
        return state

    target = resolve_target(local_path, file_path)

    if target is None:
        state["reply"] = f"Refusing to write outside the repository (`{file_path}` resolves outside it)."
        state["sources"] = []
        state["tool_used"] = TOOL_NAME
        state["error"] = "path_escape"
        return state

    try:
        result = _apply_write(local_path, target, operation, symbol_name, code)
    except Exception as exc:
        log_error(f"code_writer_tool failed to write {target}: {exc}")
        state["reply"] = f"I generated the code but couldn't write it to disk: {exc}"
        state["sources"] = []
        state["tool_used"] = TOOL_NAME
        state["error"] = str(exc)
        return state

    if result is None:
        # replace_symbol targeting a symbol that can't be safely located
        # (see _replace_symbol) -- nothing was written. Reporting this
        # plainly and stopping is deliberate: appending the already-
        # generated replacement code instead (the previous behavior)
        # duplicates whatever surrounding context it re-emitted, which
        # is exactly what broke a real file during testing.
        state["reply"] = (
            f"I couldn't safely locate `{symbol_name}` as an independently-replaceable "
            f"symbol in `{file_path}` -- either it doesn't exist under that exact name, "
            "or it's an embedded fragment (e.g. a callback passed into another call) "
            "that can't be replaced in isolation without risking the code around it. "
            "Try naming a different function/class to change, or describe the change "
            "as something to ADD instead of a replacement."
        )
        state["sources"] = []
        state["tool_used"] = TOOL_NAME
        state["error"] = "symbol_not_replaceable"
        return state

    log_info(f"code_writer_tool: {result['note']} in {file_path}")

    # Keep the RAG index live: without this, a file this tool just wrote
    # is invisible to retrieve_chunks until the whole repo is reloaded,
    # since indexing otherwise only ever runs once at /api/repositories/
    # load time. A later request to modify THIS file would then silently
    # fall back to whatever unrelated content scores best in the stale
    # index instead. Re-chunking just this one file keeps it cheap
    # regardless of overall repo size.
    try:
        category = detect_category(target)
        handler = OUTLINE_HANDLERS.get(category)
        fresh_chunks = handler(target, local_path) if handler else []
        reindex_file(state["active_repository"], local_path, file_path, fresh_chunks)
    except Exception as exc:
        log_warning(f"code_writer_tool: wrote {file_path} but failed to re-index it: {exc}")

    state["reply"] = (
        f"{summary}\n\n**File:** `{file_path}` -- {result['note']} "
        f"(lines {result['start_line']}-{result['end_line']})\n\n"
        f"```\n{code.rstrip(chr(10))}\n```"
    )
    state["sources"] = [{
        "file": file_path,
        "chunk_name": symbol_name or file_path,
        "chunk_type": operation,
        "lines": f"{result['start_line']}-{result['end_line']}",
    }]
    state["tool_used"] = TOOL_NAME

    return state
