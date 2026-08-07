"""
status_check.py

Deterministic Status-Check Fast Path.

Responsibility
--------------
Recognize a narrow, specific class of follow-up -- "did that work?", "is it
deleted?", "check the previous operation" -- asked right after a
code_writer_tool turn, and answer it directly from a real filesystem check
instead of routing it through the LLM classifier into one of the 4 content
tools.

Why this exists
----------------
Confirmed as a real, reproducible failure: none of the 4 declared tools
(repo_summarizer_tool, repo_health_tool, repo_evolution_tool, code_writer_tool)
can answer "does this file currently exist" -- it isn't a summary, a health
score, git history, or a write. Asked this, the router still has to pick one
of the four, landed on repo_evolution_tool, and that tool's own scope
classifier (which also has no "check current existence" option) defaulted to
a generic activity report -- confidently narrating real but entirely
unrelated git data as if it answered the question. Two consecutive queries
did this identically, both marked (inconsistently) verified/unverified,
because the underlying LLM judgment calls involved have no way to know they
don't apply here at all.

The fix is not a better prompt -- there is no focus/tool that correctly
describes "check the live filesystem," so no amount of classifier tuning
closes this. Instead: recognize the narrow, well-defined case deterministically
and answer it with a real Path.exists() check, which is always correct and
needs no LLM judgment at all.

Deliberately narrow by design (see the module's own test coverage for why):
only fires when (a) the message matches one of a small set of curated
"checking on a prior action" phrasings, (b) the immediately preceding turn
was a code_writer_tool call (the only tool with side effects worth checking),
and (c) a concrete file path can be extracted from this message or that
prior turn. Anything else falls through to the normal router/tool pipeline
untouched -- this must never swallow a genuine content question just because
it happens to mention a filename.

This module DOES NOT:
    ✗ Call the LLM
    ✗ Classify intent generally (see router.py) -- this only ever
      intercepts the one narrow "checking on what just happened" case
"""

import re
from pathlib import Path
from typing import Optional

from . import memory
from .tools._shared import resolve_target


TOOL_NAME = "status_check"

# Curated, not exhaustive -- each pattern is a specific, real phrasing this
# case is meant to catch (see the reproduced examples in the module
# docstring). Deliberately not a single broad "does this mention a file"
# check: that would misfire on genuine content questions that happen to
# name a file (e.g. "what does index.html do?").
_STATUS_FOLLOWUP_PATTERNS = [
    re.compile(r"\bis it (actually |really )?(deleted|removed|gone)\b", re.I),
    re.compile(r"\bwas it (actually |really )?(deleted|removed)\b", re.I),
    re.compile(r"\bhas it (actually |really )?been (deleted|removed)\b", re.I),
    re.compile(r"\bcheck (whether|if)\b.{0,20}\b(deleted|removed|done|complete)\b", re.I),
    re.compile(r"\bcheck\b.{0,10}\b(the )?(deletion|removal)\b", re.I),
    re.compile(r"\b(deletion|removal)\b.{0,15}\b(operation)?\b.{0,10}\b(is )?(done|complete|successful|finished)\b", re.I),
    re.compile(r"\bdoes it (still )?exist\b", re.I),
    re.compile(r"\bis it still there\b", re.I),
    re.compile(r"\bdid (that|it) (actually |really )?work\b", re.I),
    re.compile(r"\bis it (done|finished|complete)\b", re.I),
    re.compile(r"\b(check|confirm|verify)\b.{0,15}\b(the )?(previous|last|prior)\b.{0,10}\b(operation|change|edit|request|action)\b", re.I),
]

_BACKTICK_PATH_RE = re.compile(r"`([^`\s]+\.\w{1,8})`")
_BARE_PATH_RE = re.compile(r"\b([\w./\\-]+\.\w{1,8})\b")


def _looks_like_status_followup(message: str) -> bool:
    return any(p.search(message) for p in _STATUS_FOLLOWUP_PATTERNS)


def _extract_file_path(*texts: Optional[str]) -> Optional[str]:
    """First backtick-quoted path found across the given texts (checked in
    order), falling back to a bare-looking filename -- backtick-quoted is
    checked first across ALL texts before ever falling back to a bare
    match, since code_writer_tool's own replies always quote paths in
    backticks and that's a far more reliable signal than a bare regex
    guess picking up an unrelated word that happens to contain a dot."""

    for text in texts:
        if not text:
            continue
        m = _BACKTICK_PATH_RE.search(text)
        if m:
            return m.group(1)

    for text in texts:
        if not text:
            continue
        m = _BARE_PATH_RE.search(text)
        if m:
            return m.group(1)

    return None


# Pre-written per-persona phrasing -- no LLM call happens on this path at
# all (that's the point: a real filesystem check needs no judgment), so
# persona can't be applied by injecting a lens into a narration prompt the
# way the other 4 tools do it. Cheap static variants instead, so this fast
# path doesn't silently become the one reply in the whole system persona
# never reaches. Facts are identical across all three -- only phrasing
# differs, same rule every other tool's persona lens follows.
def _phrase_exists_file(file_path: str, size_bytes: int, persona: Optional[str]) -> str:
    if persona == "student":
        return (
            f"Good news -- `{file_path}` is still there ({size_bytes:,} bytes). I checked the "
            "real file on disk directly just now, rather than guessing from git history or "
            "re-reading the earlier reply, so this is as certain an answer as it gets."
        )
    if persona == "architect":
        return (
            f"`{file_path}` still exists on disk ({size_bytes:,} bytes) -- confirmed by a direct "
            "filesystem check, not derived from history or re-narration. No state change occurred."
        )
    return (
        f"`{file_path}` still exists on disk -- {size_bytes:,} bytes. "
        "This is a direct filesystem check, not inferred from git history or a repeat of the earlier reply."
    )


def _phrase_exists_dir(file_path: str, persona: Optional[str]) -> str:
    if persona == "student":
        return f"`{file_path}` is still there -- it's a directory. Checked directly against the real filesystem, not a guess."
    if persona == "architect":
        return f"`{file_path}` still exists on disk (a directory) -- confirmed via direct filesystem check."
    return f"`{file_path}` still exists on disk (it's a directory). Direct filesystem check, not inferred."


def _phrase_missing(file_path: str, persona: Optional[str]) -> str:
    if persona == "student":
        return f"`{file_path}` is gone -- it no longer exists on disk. I checked the real filesystem directly to be sure."
    if persona == "architect":
        return f"`{file_path}` no longer exists on disk -- confirmed via direct filesystem check; treat any prior reference to it as stale."
    return f"`{file_path}` no longer exists on disk -- checked directly against the real filesystem."


def check_status_followup(message: str, local_path: Optional[Path], persona: Optional[str] = None) -> Optional[dict]:
    """
    Returns a ready-to-use partial state update (reply/sources/tool_used/
    verified/verification_reason/context_used/error) if `message` is a
    status-check follow-up this can answer deterministically, else None --
    the caller falls through to the normal router/tool pipeline unchanged.
    """

    if local_path is None or not message:
        return None

    if not _looks_like_status_followup(message):
        return None

    prev = memory.last_turn()
    if prev is None or prev.get("tool_used") != "code_writer_tool":
        return None

    file_path = _extract_file_path(message, prev.get("assistant"), prev.get("user"))
    if not file_path:
        return None

    target = resolve_target(local_path, file_path)
    if target is None:
        return None

    exists = target.exists()

    if exists and target.is_file():
        reply = _phrase_exists_file(file_path, target.stat().st_size, persona)
    elif exists:
        reply = _phrase_exists_dir(file_path, persona)
    else:
        reply = _phrase_missing(file_path, persona)

    return {
        "reply": reply,
        "sources": [{"file": file_path, "chunk_name": "", "chunk_type": TOOL_NAME, "lines": ""}],
        "tool_used": TOOL_NAME,
        "verified": True,
        "verification_reason": "Deterministic filesystem check -- not an LLM judgment, nothing to verify.",
        "context_used": f"Direct Path.exists() check of {file_path}: {exists}",
        "error": None,
    }
