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
    -> PLAN which file(s) need to change (_plan_file_targets) -- most
       requests need exactly one, but some genuinely can't be satisfied
       by touching only one file (e.g. "add a new LLM provider" needs
       the new provider file AND the dispatcher/enum it's registered
       in elsewhere in the repo). Planning sees the repo's real,
       authoritative file list, not just the top-5 semantically-nearest
       chunks -- a dispatcher/enum file is glue code, rarely the
       semantic nearest-neighbor of a feature request, so it would
       often never surface through retrieval alone. Falls back to an
       empty plan (today's original single-target behavior, unchanged)
       if this call fails outright.
    -> For every EXISTING file the plan names, read its REAL, current,
       full content straight from disk (_gather_target_contents) --
       "understanding the other files" means the generation step sees
       the actual dispatcher/switch/enum it needs to extend, not a
       guess. Token-budgeted across all of them combined.
    -> LLM commits to a concrete plan for EVERY target as one batch
       (write_code_batch): file_path, operation, symbol_name (if
       replacing), code -- one entry per file, generated together so a
       new provider's registration in file B can reference exactly the
       real file_path/symbol_name generated for the new file A.
    -> Validate every target path resolves inside the repo (no path escape)
    -> Apply the writes, one file at a time, in order. If a change fails
       or would be unsafe, undo every write already made in this batch
       (restore each touched file to its exact original content before
       this run started) and report the failure with NOTHING changed --
       a multi-file request either fully succeeds or leaves the repo
       exactly as it was, never half-wired (e.g. never a new provider
       registered in an enum but not actually reachable through the
       dispatcher because that second write failed). Each individual
       file's write still goes through the same, unchanged per-format
       rules as a single-file request always has:
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
    -> Re-index every file actually written (app.rag.indexer.reindex_file,
       once per file) so a later request can find what's actually here
       now -- otherwise a file this tool just touched is invisible to
       retrieval until the whole repo is reloaded, and a later edit
       request would silently fall back to whatever unrelated content
       scores best in the stale index instead (confirmed in practice,
       not hypothetical)
    -> Report exactly what changed and where, across every file

This module DOES NOT:
    ✗ Classify intent (see router.py)
    ✗ Retrieve chunks with its own logic (reuses app.rag.retriever)
"""

import json
import re
from pathlib import Path
from typing import List, Optional

import tiktoken

from ... import config
from ...utils.logger import log_error, log_info, log_warning
from ...github.chunker import find_symbol_bounds, check_syntax_errors
from ...github.language import detect_category, tree_sitter_language_name, NOT_CODE_EXTENSIONS
from ...rag.retriever import retrieve_chunks
from ...rag.indexer import reindex_file
from ...rag.sparse_store import get_all_chunks
from ..llm_client import get_client
from ._shared import no_repo, llm_error, resolve_target, full_symbol_source, file_outline, OUTLINE_HANDLERS, persona_lens


TOOL_NAME = "code_writer_tool"

_VALID_OPERATIONS = {"create_file", "append_to_file", "replace_symbol"}

# Extract the local name(s) an import statement binds, across the common
# JS/TS/Python forms -- used only to check whether an appended import is
# actually referenced anywhere else in the same file (see the
# "unused_bindings" check in act_code_writer's apply loop). This is a
# real, deterministic check on the ACTUAL written content, not a repeat of
# the same "don't append a dead import" instruction the model already has
# in WRITE_CODE_SYSTEM_PROMPT and was confirmed (in testing) to sometimes
# not follow -- so the tool's own reply can stay honest about it either way.
_IMPORT_BINDING_PATTERNS = [
    re.compile(r'^\s*import\s+(\w+)\s*(?:,\s*\{[^}]*\})?\s+from\s+["\']', re.MULTILINE),  # default
    re.compile(r'^\s*import\s*\{\s*([^}]+?)\s*\}\s*from\s+["\']', re.MULTILINE),           # named
    re.compile(r'^\s*import\s*\*\s*as\s+(\w+)\s+from\s+["\']', re.MULTILINE),              # namespace
    re.compile(r'^\s*from\s+\S+\s+import\s+([^\n]+)', re.MULTILINE),                        # python
    re.compile(r'^\s*import\s+(\w+)\s*$', re.MULTILINE),                                    # python plain
]


def _extract_import_bindings(code: str) -> List[str]:
    names = []
    for pattern in _IMPORT_BINDING_PATTERNS:
        for m in pattern.finditer(code):
            for part in m.group(1).split(","):
                part = part.strip()
                if " as " in part:
                    part = part.split(" as ")[-1].strip()
                if part and part.isidentifier():
                    names.append(part)
    return names


def _unused_appended_bindings(appended_code: str, full_file_content: str) -> List[str]:
    """
    Which names does `appended_code` import that appear nowhere else in
    the file besides their own import line? A name used exactly once in
    the whole file is, by definition, only ever declared -- never called,
    read, or otherwise referenced.
    """
    unused = []
    for name in _extract_import_bindings(appended_code):
        if len(re.findall(r'\b' + re.escape(name) + r'\b', full_file_content)) <= 1:
            unused.append(name)
    return unused

# Same tokenizer repo_summarizer.py uses for its doc budget -- already a
# real dependency, reused here for the same reason: a real token count,
# not a crude character-count proxy, for budgeting how much OTHER
# existing files' full content (see _gather_target_contents) fits in
# the generation prompt alongside everything already there.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text, disallowed_special=()))

_FILE_CHANGE_ITEM_SCHEMA = {
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
            "description": "One or two sentences describing this file's change, for the user.",
        },
    },
    "required": ["file_path", "operation", "symbol_name", "code", "summary"],
}

# One call per planned file, NOT one call producing an array for all of
# them -- tried the array-of-changes shape first and it failed on a real
# multi-file test: cramming several complete code blocks into one JSON
# tool-call payload made the model produce malformed JSON (a real,
# observed "Expecting ':' delimiter" parse failure once the payload got
# large), something the single-file version never hit in extensive
# testing. Generating one file at a time keeps each call's output back
# down to the same size a single-file request always had, and each call
# still gets every OTHER planned file's real content (including a NEW
# file's content already decided earlier in THIS SAME request -- see
# act_code_writer) for coordination, so nothing about cross-file
# awareness is lost by not batching the LLM CALL itself.
WRITE_CODE_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "write_code",
            "description": "Commit to exactly one concrete file-level change and produce the code for it.",
            "parameters": _FILE_CHANGE_ITEM_SCHEMA,
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
    "You'll usually be asked for exactly one file, same as always. Sometimes this is one "
    "call in a SEQUENCE covering a change that genuinely needs more than one file (e.g. a "
    "new provider/plugin plus the dispatcher/enum/registry it must be wired into) -- when "
    "that's the case, you'll be shown every OTHER planned file's real content below, "
    "including one already decided earlier in this SAME multi-file request (marked as "
    "such) even if it's a brand new file that doesn't exist on disk yet. Use that to stay "
    "coordinated with what's already been (or will be) written elsewhere:\n\n"
    "- If THIS file is a NEW file meant to plug into an existing dispatcher/registry "
    "pattern shown in another planned file's content below (e.g. that file destructures "
    "`{ chatWithLLMStream, resetChatHistory, summaryTextWithLLM }` from each sibling "
    "module, or imports a default export with specific properties), your file's own "
    "exports MUST match that exact shape -- same property names, same default-vs-named "
    "export style as the siblings actually use. Do not invent a differently-shaped "
    "function; mirror the real pattern precisely, since the dispatcher will destructure "
    "your new file exactly like every existing one.\n"
    "- `append_to_file` ONLY adds code at the very END of a file, as a new, independently "
    "valid top-level piece (a new function, const, import, class). NEVER use it for a "
    "fragment that only makes sense INSIDE something that already exists -- a bare "
    "`case ...:` clause with no enclosing switch, a bare `else`, a closing brace with no "
    "matching open. That produces invalid syntax exactly as broken as it looks appended "
    "after everything else in the file.\n"
    "- If extending an existing structure (a switch/match statement, a registry object, an "
    "array) requires inserting code in the MIDDLE of a named function/class, use "
    "`replace_symbol` on that enclosing named symbol and reproduce its COMPLETE body with "
    "your addition inserted at the correct place, preserving every existing branch "
    "verbatim -- the same rule as any other replace_symbol.\n"
    "- \"imports\" is NOT a real, independently-replaceable symbol_name, even if you see it "
    "labeled that way in retrieved reference metadata -- it's an aggregated summary of a "
    "file's import lines, not an editable declaration. To add a new import, use "
    "append_to_file (a new `import ...;` line is valid anywhere at a module's top level, "
    "so appending it works); never replace_symbol with symbol_name \"imports\".\n"
    "- A symbol_name like \"block_1\", \"block_2\", \"block_17\" (any \"block_<number>\") is "
    "likewise NOT a real, independently-replaceable declaration -- it's an arbitrary "
    "fixed-size slice of a file that has no tree-sitter structure to chunk by, produced "
    "purely for retrieval. It is explicitly labeled as such wherever it appears above. "
    "Never replace_symbol with a symbol_name shaped like this; treat that file's content "
    "as reference only, the same as you would if no symbol_name were given at all.\n"
    "- If THIS file's needed change has NO enclosing named function/class at all around "
    "the edit point (bare top-level code, so there is nothing a replace_symbol could "
    "safely target), you cannot make that specific edit safely with the operations "
    "available -- and that applies to the file's change AS A WHOLE, not just the part "
    "that happens to be safe. Before appending ANYTHING (an import especially), ask "
    "yourself: once this is added, is it actually USED by something else in this same "
    "change? An import is only useful if the code that references it also gets added "
    "somewhere -- if the only thing you can safely add to this file is the import, while "
    "whatever would actually USE it (a switch/dispatch case) has nowhere safe to go, "
    "adding JUST the import produces an unused, dead import and makes zero real "
    "progress on the request. In that situation, do NOT append the import either -- "
    "instead, set `code` to an empty string and use `summary` to plainly say this file "
    "can't be safely auto-edited and exactly what a human needs to add there by hand (including the "
    "import, if one is needed); you'll still be skipped safely.\n"
    "- If the request asks to DELETE or REMOVE a file entirely, or delete a whole "
    "function/class/symbol outright (make it gone, not replace it with something), that is "
    "not something any of the three operations can do -- there is no delete operation, full "
    "stop. Do NOT approximate deletion by emptying a symbol's body, replacing it with "
    "unrelated code, or regenerating the file as blank -- confirmed as a real failure: doing "
    "this once silently erased a real, load-bearing function while looking like an ordinary "
    "edit. Set `code` to an empty string and phrase `summary` as a plain, direct decline "
    "naming the real limitation, e.g. 'Deleting files/symbols isn't something I can do -- I "
    "can only create a file, append to one, or replace part of one. If you want <name> "
    "removed, delete it yourself; if you meant something I can do instead (like clearing its "
    "content), tell me that and I will.' Never restate the request back as if describing an "
    "action taken (e.g. 'This change deletes X') -- worded that way it reads as confirmation "
    "that something happened, when nothing did.\n\n"
    "Reference the real file_path/symbol_name for a NEW file consistently with how it's "
    "shown to you when it was decided in an earlier call this same request. Produce a "
    "complete, valid change for just this one file.\n\n"
    "Call write_code exactly once with a single, concrete, complete change."
)


# ---------------------------------------------------------------------------
# Multi-file planning -- decide WHICH files a request needs touched, before
# committing to what to write. Most requests plan exactly one file; this
# only matters for the rarer case where a change genuinely can't work by
# touching just one (see the module docstring's Pipeline section).
# ---------------------------------------------------------------------------

_MAX_PLAN_TARGETS = 5
_MAX_REPO_FILES_LISTED = 200

# Total budget across ALL other-existing-target files fetched for one
# request (see _gather_target_contents) -- generous for the real case
# (a dispatcher/enum file is glue code, typically small) while still a
# real ceiling against a plan naming several large files.
EXISTING_TARGET_TOKEN_BUDGET = 12_000

PLAN_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "plan_file_changes",
            "description": (
                "List every file that needs to be created or modified to fully "
                "satisfy this request. Most requests need exactly one file; only "
                "list more when the change genuinely can't work without them too."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "targets": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": _MAX_PLAN_TARGETS,
                        "items": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": (
                                        "Path relative to the repository root. Reuse a "
                                        "real existing path verbatim when touching an "
                                        "existing file; for a new file, follow the "
                                        "repo's real layout/naming conventions."
                                    ),
                                },
                                "reason": {
                                    "type": "string",
                                    "description": "One sentence: why this specific file needs to change.",
                                },
                            },
                            "required": ["file_path", "reason"],
                        },
                    },
                },
                "required": ["targets"],
            },
        },
    },
]

PLAN_SYSTEM_PROMPT = (
    "You are planning a real code change to a repository -- deciding WHICH files need "
    "to be touched, not writing code yet. Most requests need exactly one file: the one "
    "the request is actually about. Only list additional files when the change "
    "genuinely cannot work without them -- the clearest real case is adding something "
    "(a provider, plugin, handler, route) that this repo registers elsewhere: if the "
    "reference chunks or file list below show other, similar things all wired into one "
    "central dispatcher/switch/enum/registry file, a NEW one needs to be wired in there "
    "too, so that file belongs in your plan alongside the new one. Use the real file "
    "list below to name an EXISTING file by its real, exact path -- never invent a "
    "plausible-sounding path you have not actually seen evidence of; if you cannot find "
    "a real dispatcher/registry file for this kind of thing, don't invent one, just "
    "plan the one file the request is actually about. Do not plan a documentation-only "
    "change unless the request explicitly asks to edit documentation. Each real file "
    "appears AT MOST ONCE in your plan -- if a request has several facets that all "
    "land in the same file (e.g. \"edit this function and also add logging to it\"), "
    "that is still ONE target with one combined reason, never the same file_path listed "
    "twice."
)


def _list_repository_files(repository_name: str) -> List[str]:
    """The repo's real, authoritative file list (from already-indexed
    chunk metadata -- no new retrieval/parsing) -- lets planning name a
    dispatcher/enum/registry file by its actual path even though its
    CONTENT would rarely be among the top-5 semantic matches for a
    feature request (glue code reads nothing like the feature itself)."""

    chunks = get_all_chunks(repository_name)
    return sorted({c["path"] for c in chunks})


def _dedupe_targets(targets: List[dict]) -> List[dict]:
    """
    Keep only the first target for each real file_path -- a real,
    confirmed failure mode: the planner can name the SAME file twice
    (e.g. treating "edit the function" and "add logging to it" as two
    separate targets even though they're one coherent change to one
    file), which would otherwise cost a second, wasted generation call
    AND a second write attempt on a file the first write already
    changed -- harmless if that second generation happens to reproduce
    the exact same result, but nothing guarantees that, since it's an
    independent LLM call. A hard, code-level guarantee here is safer
    than trusting the prompt alone to never repeat itself.

    Compares paths with backslashes normalized to forward slashes --
    confirmed as a real gap in the plain-string dedup this replaced: two
    targets differing only in slash direction ('Adobe/x.csv' vs
    'Adobe\\x.csv') survived it as "different" files, even though
    resolve_target() resolves both to the identical file on disk, which
    let a real batch plan two writes at what was actually one target
    (see _apply_write's already_written_this_batch handling for the
    write-time safety net this dedup gap let slip through once).
    """

    seen = set()
    deduped = []

    for t in targets:
        if not isinstance(t, dict):
            continue
        path = (t.get("file_path") or "").strip().lstrip("/\\")
        if not path:
            continue
        normalized = path.replace("\\", "/")
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append({**t, "file_path": path})

    return deduped


def _plan_file_targets(query: str, reference_context: str, repo_files: List[str]) -> List[dict]:
    """
    Returns [{"file_path": str, "reason": str}, ...], or an empty list if
    planning fails outright or returns nothing usable -- the caller
    (act_code_writer) treats an empty plan as "let the generation step
    pick the single file_path itself," i.e. today's original,
    long-proven single-target behavior, unchanged. A planning hiccup
    degrades gracefully instead of breaking the whole request.
    """

    files_block = "\n".join(repo_files[:_MAX_REPO_FILES_LISTED]) if repo_files else "(no indexed file list available)"
    if len(repo_files) > _MAX_REPO_FILES_LISTED:
        files_block += f"\n... and {len(repo_files) - _MAX_REPO_FILES_LISTED} more files not shown"

    user_content = (
        f"{reference_context}\n\n"
        f"Real files that already exist in this repository:\n{files_block}\n\n"
        f"Request:\n{query}"
    )

    try:
        resp = get_client().chat.completions.create(
            model=config.AZURE_CHAT_DEPLOYMENT,
            temperature=0,
            tools=PLAN_SCHEMA,
            tool_choice={"type": "function", "function": {"name": "plan_file_changes"}},
            messages=[
                {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        args = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
        return _dedupe_targets(args.get("targets") or [])[:_MAX_PLAN_TARGETS]
    except Exception as exc:
        log_warning(f"{TOOL_NAME}: file-change planning failed ({exc}); falling back to single-target flow")
        return []


def _gather_target_contents(
    local_path: Path, targets: List[dict], exclude_path: Optional[str] = None,
) -> tuple:
    """
    Read the REAL, current, full content of every EXISTING file named in
    the plan (skipping new-file targets, anything that fails to resolve
    safely inside the repo, and exclude_path -- the one target already
    shown in full elsewhere, see full_symbol_source in act_code_writer --
    so it's never sent twice). Token-budgeted across all of them combined
    (see EXISTING_TARGET_TOKEN_BUDGET), same approach as
    repo_summarizer.py's doc budget.

    Returns (blocks, truncated_paths) -- blocks is [{"path", "content"}]
    for everything that fit; truncated_paths lists any existing target
    that didn't (rare -- the budget is generous for real glue code).
    """

    blocks = []
    truncated = []
    remaining = EXISTING_TARGET_TOKEN_BUDGET

    for t in targets:
        file_path = (t.get("file_path") or "").strip().lstrip("/\\")
        if not file_path or file_path == exclude_path:
            continue

        target = resolve_target(local_path, file_path)
        if target is None or not target.exists() or not target.is_file():
            continue

        text = target.read_text(encoding="utf-8", errors="ignore")
        tokens = _count_tokens(text)

        if tokens > remaining:
            truncated.append(file_path)
            continue

        blocks.append({"path": file_path, "content": text})
        remaining -= tokens

    return blocks, truncated


EXPAND_PLAN_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "expand_plan",
            "description": (
                "Given the REAL content of the already-planned existing files, list any "
                "ADDITIONAL existing files that also need a change for this request to "
                "actually work -- or an empty list if the current plan is already complete."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "additional_targets": {
                        "type": "array",
                        "maxItems": _MAX_PLAN_TARGETS,
                        "items": {
                            "type": "object",
                            "properties": {
                                "file_path": {"type": "string", "description": "Real, exact path of the additional file."},
                                "reason": {"type": "string", "description": "One sentence: why this file also needs to change."},
                            },
                            "required": ["file_path", "reason"],
                        },
                    },
                },
                "required": ["additional_targets"],
            },
        },
    },
]

EXPAND_PLAN_SYSTEM_PROMPT = (
    "You already have a plan naming some files to change, and you're now shown their "
    "REAL, current content. Returning an empty list is the correct, by-far-most-common "
    "answer -- only add a file when the plan's edit would be INCOMPLETE, UNREACHABLE, or "
    "BROKEN without it. The one common real case: a planned file switches/dispatches on "
    "an enum or constant defined in ANOTHER file (visible from an import statement in the "
    "content shown), and the plan is adding a new case for a value that doesn't exist in "
    "that enum's own defining file yet -- without adding it there too, the new case can "
    "never be reached.\n\n"
    "Being merely IMPORTED, REFERENCED, or USED by a planned file is NOT sufficient -- "
    "that is normal in any codebase and essentially never means the referenced file "
    "itself needs to change. Do not add a file because it MIGHT be affected, or because "
    "changing something else COULD have implications for it -- only because the plan's "
    "OWN edit is concretely incomplete without it. If you are not certain, return an "
    "empty list."
)


def _expand_plan_with_dependencies(
    query: str,
    reference_context: str,
    planned_targets: List[dict],
    existing_blocks: List[dict],
    repo_files: List[str],
) -> List[dict]:
    """
    A bounded SECOND planning pass, run only after the first plan's
    existing targets are fetched in full (see _gather_target_contents) --
    catches a real, observed gap the first pass structurally cannot: it
    only sees repo_files' PATHS plus the original top-5 semantic chunks,
    so a file the plan's OWN targets import (e.g. the enum a dispatcher
    switches on) is often invisible until those targets' actual content
    is in hand. Skipped entirely (no LLM call) when there's nothing
    existing to inspect yet -- a request that only creates a new file
    has nothing to discover a dependency from.
    """

    if not existing_blocks:
        return []

    remaining_slots = _MAX_PLAN_TARGETS - len(planned_targets)
    if remaining_slots <= 0:
        return []

    files_block = "\n".join(repo_files[:_MAX_REPO_FILES_LISTED]) if repo_files else "(no indexed file list available)"
    existing_content_block = "\n\n".join(f"--- {b['path']} ---\n{b['content']}" for b in existing_blocks)
    planned_paths = ", ".join(t["file_path"] for t in planned_targets)

    user_content = (
        f"{reference_context}\n\n"
        f"Already-planned files: {planned_paths}\n\n"
        f"Their real, current content:\n{existing_content_block}\n\n"
        f"Real files that already exist in this repository:\n{files_block}\n\n"
        f"Request:\n{query}"
    )

    try:
        resp = get_client().chat.completions.create(
            model=config.AZURE_CHAT_DEPLOYMENT,
            temperature=0,
            tools=EXPAND_PLAN_SCHEMA,
            tool_choice={"type": "function", "function": {"name": "expand_plan"}},
            messages=[
                {"role": "system", "content": EXPAND_PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        args = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
        already_planned = {t["file_path"] for t in planned_targets}
        additional = [
            t for t in _dedupe_targets(args.get("additional_targets") or [])
            if t["file_path"] not in already_planned
        ]
        return additional[:remaining_slots]
    except Exception as exc:
        log_warning(f"{TOOL_NAME}: plan expansion failed ({exc}); continuing with the original plan")
        return []


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
    already_written_this_batch: bool = False,
) -> Optional[dict]:
    """
    Perform the write; returns what actually happened (may differ from
    the requested operation -- e.g. any operation on an existing
    whole-document file (HTML/CSS/JSON/...) replaces its content rather
    than appending). Returns None, having written NOTHING, when
    operation is replace_symbol and the target symbol can't be safely
    located (see _replace_symbol) -- the caller reports this as a clean
    failure rather than guessing.

    `already_written_this_batch` only changes behavior for the plain-data
    formats in NOT_CODE_EXTENSIONS (.csv/.tsv) -- confirmed as a real bug
    when a single batch plans two changes against the SAME such file: the
    second call's "replace the whole file" silently discarded the first
    call's write, even though the reply reported both as applied. For
    those specifically, a repeat write appends instead, since each call's
    payload there is a genuine small fragment, not a full document.

    It deliberately does NOT change behavior for a true whole-document
    format (HTML/CSS/JSON/YAML/TOML/ini, or anything unrecognized) --
    confirmed as a real regression from an earlier version of this same
    fix that treated all non-code formats alike: a repeat write to an
    HTML file appended the model's freshly regenerated COMPLETE document
    after the one already written moments earlier in the same batch,
    doubling the entire file (both `<!DOCTYPE html>`, both `<head>`s,
    everything). These formats always regenerate the whole document on
    every call regardless of repeat status (see the comment below), so
    replacing again -- not appending -- is still the only safe behavior.
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
    suffix = target.suffix.lower()
    # A repeat write only gets to "coexist via append" for the plain-data
    # formats (.csv/.tsv) where a single call's payload is already a small
    # fragment, not a full document -- everything else that isn't code/
    # append-safe-text still always replaces, repeat or not (see the
    # docstring above for why this distinction matters).
    repeat_can_append = already_written_this_batch and suffix in NOT_CODE_EXTENSIONS

    if detect_category(target) != "code" and suffix not in _APPEND_SAFE_TEXT_EXTENSIONS and not repeat_can_append:
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


# ---------------------------------------------------------------------------
# Multi-file atomicity -- a batch either fully succeeds or leaves every file
# it touched exactly as it was. Needed once a request can write more than
# one file: one write can succeed while a later, related one fails or gets
# safely refused (see _replace_symbol), and a half-applied multi-file change
# is worse than not applying it at all (a provider registered in an enum but
# never actually reachable through the dispatcher because THAT write failed).
# ---------------------------------------------------------------------------

def _snapshot(target: Path) -> Optional[str]:
    """The file's real current content, or None if it doesn't exist yet --
    None specifically means _restore should DELETE it (this batch created
    it), not write empty content over something that was never there."""

    return target.read_text(encoding="utf-8", errors="ignore") if target.exists() else None


def _restore(target: Path, snapshot: Optional[str]) -> None:
    if snapshot is None:
        try:
            target.unlink()
        except FileNotFoundError:
            pass
    else:
        target.write_text(snapshot, encoding="utf-8")


# chunker.py's _fixed_size_fallback names every window it produces
# "block_1", "block_2", ... whenever a file has no real tree-sitter
# structure to chunk by (parse failure, or no top-level function/class
# at all -- e.g. the same bare-top-level-dispatcher shape that already
# blocks replace_symbol elsewhere). These are retrieval-only buckets,
# never a real, independently-replaceable declaration.
_FIXED_BLOCK_NAME_RE = re.compile(r"^block_\d+$")


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
        f"    symbol_name: {chunk['chunk_name']} ({chunk['chunk_type']})"
        # chunk_type "imports" is an aggregated summary of a file's import
        # lines, built by chunker.py's _imports_chunk for retrieval
        # purposes only -- it is NOT a real, independently-replaceable
        # declaration, and find_symbol_bounds correctly never finds it
        # (it only walks real tree-sitter structure). Said so directly
        # here, not just in the system prompt: confirmed by testing that
        # a generic "never target 'imports'" instruction lost out to this
        # SAME reference listing's own "use the retrieved symbol_name "
        # verbatim" guidance once the model saw "imports" formatted here
        # identically to any other, real symbol.
        + (
            " -- NOT a real symbol; this is an aggregated summary of import "
            "lines, not something replace_symbol can target. To add a new "
            "import here, use append_to_file instead."
            if chunk["chunk_type"] == "imports" else
            " -- NOT a real symbol; this is an arbitrary fixed-size slice of "
            "the file used only for retrieval (the file has no tree-sitter "
            "structure to chunk by here), not something replace_symbol can "
            "target. Treat this file's content as reference/context only "
            "when deciding how to satisfy the request."
            if _FIXED_BLOCK_NAME_RE.match(chunk["chunk_name"]) else ""
        ) + "\n"
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

    # Everything above is pure CONTEXT (retrieved chunks, confidence signal,
    # the most-likely-existing-target's real content/outline) -- reused
    # verbatim for both the planning call below and the generation call
    # further down, so neither loses any of the existing targeting-accuracy
    # work by being built separately.
    reference_context = f"Reference chunks retrieved from the repository, ranked most relevant first:\n{reference}{confidence_note}{outline_block}{current_impl_block}"

    targeting_instructions = (
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
        "don't repeat it back verbatim in your output."
    )

    # PLAN -- which file(s) does this actually need? See the module
    # docstring's Pipeline section for why this exists: a dispatcher/enum
    # file a new provider must be registered in is glue code, rarely a
    # semantic match for the request itself, so it needs to be named
    # explicitly rather than hoped for from the top-5 chunks above.
    repo_files = _list_repository_files(state["active_repository"])
    planned_targets = _plan_file_targets(query, reference_context, repo_files)

    existing_blocks, truncated_existing = _gather_target_contents(
        local_path, planned_targets, exclude_path=full_target["path"] if full_target else None,
    )

    # SECOND pass -- now that the initially-planned existing files' real
    # content is in hand, check for a dependency the first pass couldn't
    # have seen (e.g. an enum a dispatcher switches on, visible only once
    # the dispatcher's own content is actually read). No-op (no LLM call)
    # when there's nothing existing to inspect yet.
    additional_targets = _expand_plan_with_dependencies(
        query, reference_context, planned_targets, existing_blocks, repo_files,
    )
    if additional_targets:
        planned_targets = planned_targets + additional_targets
        more_blocks, more_truncated = _gather_target_contents(
            local_path, additional_targets, exclude_path=full_target["path"] if full_target else None,
        )
        existing_blocks = existing_blocks + more_blocks
        truncated_existing = truncated_existing + more_truncated

    exclude_path = full_target["path"] if full_target else None

    if truncated_existing:
        truncated_note = (
            "\n\n(NOTE: " + ", ".join(truncated_existing) + " were also planned but too large "
            "to include in full here -- handle them conservatively, e.g. prefer append_to_file "
            "over a guessed replace_symbol for those.)"
        )
    else:
        truncated_note = ""

    # GENERATE -- one call per planned target (or a single call if planning
    # produced nothing, i.e. today's original single-target behavior), NOT
    # one call producing an array for all of them (see WRITE_CODE_SCHEMA's
    # comment for why: that was tried and failed a real test on a large
    # combined payload). Each call sees every OTHER planned file's real
    # content, INCLUDING one already decided earlier in this same loop --
    # using its freshly-generated code, not stale/nonexistent disk content
    # -- so a dispatcher generated after a new provider file can still
    # reference that new file's real, final export shape.
    iteration_targets = planned_targets or [{"file_path": None, "reason": None}]

    # Persona only reframes the human-facing `summary` text and comment
    # style write_code produces -- appended once here, never inside the
    # loop, so it's the exact same instruction for every file in this
    # batch. It must never change WHICH file/operation/code gets chosen;
    # that's still governed entirely by WRITE_CODE_SYSTEM_PROMPT's own
    # rules above, unmodified.
    #
    # Confirmed by direct testing as a REAL failure, not a hypothetical:
    # the student lens's own "explain concepts instead of exactly how it's
    # implemented" framing -- written for a narration prompt, where that's
    # the right instruction -- caused the model to replace a real
    # function's entire body with a placeholder comment ("// Implementation
    # goes here") when reused verbatim for CODE GENERATION. The override
    # below exists specifically to close that failure mode; it is not
    # precautionary text, it is the fix for an observed corruption.
    lens = persona_lens(state.get("persona"))
    if lens:
        write_code_system_prompt = (
            f"{WRITE_CODE_SYSTEM_PROMPT}\n\n{lens}\n\n"
            "The audience lens above changes ONLY how much you explain in "
            "`summary` and how many comments you add in `code` -- it NEVER "
            "changes what the code actually does. `code` must always be the "
            "complete, correct, working implementation this request needs, "
            "with every existing branch and behavior preserved exactly as "
            "the rules above require, for every persona without exception. "
            "Never replace real logic with a placeholder comment (e.g. "
            "'// implementation goes here') or omit a part of the "
            "implementation for the sake of being 'less technical' -- that "
            "is never acceptable, regardless of audience."
        )
    else:
        write_code_system_prompt = WRITE_CODE_SYSTEM_PROMPT

    changes = []
    skipped = []
    generated_so_far = []

    for i, t in enumerate(iteration_targets):
        seen_paths = {g["file_path"] for g in generated_so_far} | {exclude_path} if exclude_path else {g["file_path"] for g in generated_so_far}

        coordination_sections = [
            f"--- {g['file_path']} (DECIDED EARLIER IN THIS SAME REQUEST -- this is its "
            f"real, final content; it does not necessarily exist on disk yet) ---\n{g['code']}"
            for g in generated_so_far if g["file_path"] != exclude_path
        ] + [
            f"--- {b['path']} ---\n{b['content']}"
            for b in existing_blocks if b["path"] not in seen_paths
        ]

        other_files_block = (
            "\n\nOther planned file(s) this change needs to coordinate with:\n\n"
            + "\n\n".join(coordination_sections)
        ) if coordination_sections else ""
        other_files_block += truncated_note

        target_note = (
            f"\n\nYou are producing the change for THIS one planned target: `{t['file_path']}` "
            f"-- {t['reason']}"
            if t.get("file_path") else ""
        )

        user_content = (
            f"{reference_context}"
            f"{other_files_block}"
            f"{target_note}\n\n"
            f"{targeting_instructions}\n\n"
            f"Request:\n{query}"
        )

        if i == 0:
            state["context_used"] = user_content

        try:
            resp = get_client().chat.completions.create(
                model=config.AZURE_CHAT_DEPLOYMENT,
                temperature=0.2,
                tools=WRITE_CODE_SCHEMA,
                tool_choice={"type": "function", "function": {"name": "write_code"}},
                messages=[
                    {"role": "system", "content": write_code_system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            item = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
        except Exception as exc:
            return llm_error(state, TOOL_NAME, exc)

        file_path = (item.get("file_path") or "").strip().lstrip("/\\")
        operation = item.get("operation")
        symbol_name = item.get("symbol_name") or None
        code = item.get("code") or ""
        summary = item.get("summary") or ""

        if not code.strip():
            # An empty code with a real summary is a deliberate, honest
            # skip (see WRITE_CODE_SYSTEM_PROMPT's "no enclosing named
            # symbol" rule) -- not an error, just nothing safe to write
            # for this one file. Continue with any other planned targets.
            if summary.strip():
                skipped.append({"file_path": file_path or t.get("file_path") or "(unspecified)", "summary": summary})
                continue
            state["reply"] = (
                "I couldn't determine a concrete file change from that request. "
                "Try naming the file and describing the change more specifically."
            )
            state["sources"] = []
            state["tool_used"] = TOOL_NAME
            state["error"] = "invalid_plan"
            return state

        if not file_path or operation not in _VALID_OPERATIONS:
            state["reply"] = (
                "I couldn't determine a concrete file change from that request. "
                "Try naming the file and describing the change more specifically."
            )
            state["sources"] = []
            state["tool_used"] = TOOL_NAME
            state["error"] = "invalid_plan"
            return state

        changes.append({
            "file_path": file_path, "operation": operation, "symbol_name": symbol_name,
            "code": code, "summary": summary,
        })
        generated_so_far.append({"file_path": file_path, "code": code})

    if not changes:
        state["reply"] = (
            "I couldn't make any of the planned changes safely:\n\n"
            + "\n".join(f"- `{s['file_path']}`: {s['summary']}" for s in skipped)
        ) if skipped else (
            "I couldn't determine a concrete file change from that request. "
            "Try naming the file and describing the change more specifically."
        )
        state["sources"] = []
        state["tool_used"] = TOOL_NAME
        state["error"] = "invalid_plan"
        return state

    overall_summary = (
        changes[0]["summary"] if len(changes) == 1 and not skipped
        else f"Made {len(changes)} coordinated file change(s)."
    )

    # APPLY -- one file at a time, in order. Any failure rolls back every
    # write already made in THIS batch (see _snapshot/_restore) before
    # reporting, so a multi-file request either fully succeeds or leaves
    # the repo exactly as it was -- never half-wired.
    applied = []
    snapshots = {}
    failure = None

    for change in changes:
        file_path = (change.get("file_path") or "").strip().lstrip("/\\")
        operation = change.get("operation")
        symbol_name = change.get("symbol_name") or None
        code = change.get("code") or ""
        summary = change.get("summary") or ""

        if not file_path or operation not in _VALID_OPERATIONS or not code.strip():
            failure = (
                "I generated an incomplete change (missing file path, operation, or code) "
                "for one of the planned files, so I stopped before writing anything from "
                "this batch."
            )
            break

        target = resolve_target(local_path, file_path)

        if target is None:
            failure = f"Refusing to write outside the repository (`{file_path}` resolves outside it)."
            break

        if file_path not in snapshots:
            snapshots[file_path] = _snapshot(target)

        # Compared via the RESOLVED path, not the raw file_path string --
        # two independent generation calls can name the same real file
        # with different slash directions ('Adobe/x.csv' vs 'Adobe\x.csv'),
        # which resolve_target() resolves identically but a raw string
        # comparison would treat as different targets, missing the exact
        # same-file collision this check exists to catch.
        already_written_this_batch = any(a["target"] == target for a in applied)

        try:
            result = _apply_write(local_path, target, operation, symbol_name, code, already_written_this_batch)
        except Exception as exc:
            log_error(f"{TOOL_NAME} failed to write {target}: {exc}")
            failure = f"I generated the code for `{file_path}` but couldn't write it to disk: {exc}"
            break

        if result is None:
            # replace_symbol targeting a symbol that can't be safely
            # located (see _replace_symbol) -- nothing was written for
            # THIS file, but earlier files in this batch may already
            # have been. Stopping and rolling back below is deliberate,
            # same reasoning as the single-file case: appending the
            # already-generated replacement code instead duplicates
            # whatever surrounding context it re-emitted.
            failure = (
                f"I couldn't safely locate `{symbol_name}` as an independently-replaceable "
                f"symbol in `{file_path}` -- either it doesn't exist under that exact name, "
                "or it's an embedded fragment (e.g. a callback passed into another call) "
                "that can't be replaced in isolation without risking the code around it."
            )
            break

        # A write can succeed as a file operation and still leave invalid
        # syntax behind (e.g. the generated replacement didn't balance a
        # brace, or a fragment only valid inside a switch got appended at
        # top level) -- confirmed as a real failure mode in testing, and
        # previously undetected by anything in this tool, so a broken file
        # could get written AND reported as a successful change. Re-parse
        # what's actually on disk now and revert THIS file's write (not
        # just the ones before it) if it doesn't parse cleanly.
        lang_name = tree_sitter_language_name(target) if detect_category(target) == "code" else None
        if lang_name:
            fresh_content = target.read_text(encoding="utf-8", errors="ignore")
            syntax_errors = check_syntax_errors(lang_name, fresh_content)
            if syntax_errors:
                _restore(target, snapshots[file_path])
                failure = (
                    f"The change generated for `{file_path}` produced invalid syntax "
                    f"({syntax_errors[0]}), so I reverted it rather than leaving broken code "
                    "in place."
                )
                break

        unused_bindings = (
            _unused_appended_bindings(code, fresh_content)
            if operation == "append_to_file" and lang_name else []
        )

        applied.append({
            "file_path": file_path, "target": target, "operation": operation,
            "symbol_name": symbol_name, "code": code, "summary": summary, "result": result,
            "unused_bindings": unused_bindings,
        })

    if failure is not None:
        for done in reversed(applied):
            _restore(done["target"], snapshots[done["file_path"]])

        state["reply"] = failure if not applied else (
            f"{failure}\n\nNothing was changed: {len(applied)} of {len(changes)} planned file(s) "
            "had already been written when this happened, and they've been rolled back so the "
            "repository is left exactly as it was before this request."
        )
        state["sources"] = []
        state["tool_used"] = TOOL_NAME
        state["error"] = "batch_write_failed"
        return state

    # Keep the RAG index live for every file actually written: without
    # this, a file this tool just touched is invisible to retrieve_chunks
    # until the whole repo is reloaded, since indexing otherwise only ever
    # runs once at /api/repositories/ load time. A later request to modify
    # one of THESE files would then silently fall back to whatever
    # unrelated content scores best in the stale index instead.
    for done in applied:
        log_info(f"{TOOL_NAME}: {done['result']['note']} in {done['file_path']}")
        try:
            category = detect_category(done["target"])
            handler = OUTLINE_HANDLERS.get(category)
            fresh_chunks = handler(done["target"], local_path) if handler else []
            reindex_file(state["active_repository"], local_path, done["file_path"], fresh_chunks)
        except Exception as exc:
            log_warning(f"{TOOL_NAME}: wrote {done['file_path']} but failed to re-index it: {exc}")

    if len(applied) == 1:
        # Identical reply shape to the tool's original, long-proven
        # single-file report -- a multi-file-capable batch of exactly
        # one entry (the overwhelmingly common case) looks no different
        # to the user than it always has.
        only = applied[0]
        state["reply"] = (
            f"{overall_summary}\n\n**File:** `{only['file_path']}` -- {only['result']['note']} "
            f"(lines {only['result']['start_line']}-{only['result']['end_line']})\n\n"
            f"```\n{only['code'].rstrip(chr(10))}\n```"
        )
    else:
        file_sections = "\n\n".join(
            f"**File:** `{d['file_path']}` -- {d['result']['note']} "
            f"(lines {d['result']['start_line']}-{d['result']['end_line']})\n"
            f"{d['summary']}\n\n```\n{d['code'].rstrip(chr(10))}\n```"
            for d in applied
        )
        state["reply"] = f"{overall_summary}\n\n{file_sections}"

    if skipped:
        state["reply"] += (
            "\n\n**Note:** couldn't safely auto-edit the following (needs a manual change):\n"
            + "\n".join(f"- `{s['file_path']}`: {s['summary']}" for s in skipped)
        )

    # Checked against the ACTUAL file just written, not trusted from the
    # LLM's own summary -- confirmed by testing that a summary can claim a
    # provider was "registered" while the import it added is never
    # referenced anywhere else in the file (WRITE_CODE_SYSTEM_PROMPT
    # already tells the model to skip in that situation instead of
    # appending a dead import, but doesn't reliably stop it from doing so
    # anyway). Surfaced here so the reply can't overstate what changed
    # even when the model's own summary does.
    unused_notes = [
        f"- `{d['file_path']}`: added `{', '.join(d['unused_bindings'])}` but it isn't "
        "referenced anywhere else in this file -- nothing else was safely wireable here, "
        "so this needs a manual follow-up to actually take effect."
        for d in applied if d.get("unused_bindings")
    ]
    if unused_notes:
        state["reply"] += (
            "\n\n**Note:** the following addition(s) aren't wired up yet:\n"
            + "\n".join(unused_notes)
        )

    state["sources"] = [
        {
            "file": d["file_path"],
            "chunk_name": d["symbol_name"] or d["file_path"],
            "chunk_type": d["operation"],
            "lines": f"{d['result']['start_line']}-{d['result']['end_line']}",
        }
        for d in applied
    ]
    state["tool_used"] = TOOL_NAME

    return state
