"""
memory.py

Conversation Memory.

Responsibility
--------------
Give the chat path short-term conversational continuity across turns --
so a follow-up like "what about that file" or "show me its risk too"
resolves against what was actually just said -- without re-sending the
entire chat history forever, which would eventually blow past the LLM's
context window on a long session.

Two things are kept, and together they're what build_context() hands
back for injection into a tool's LLM call:

  1. A raw window of the most recent turns, verbatim.
  2. A single rolling summary of everything OLDER than that window --
     updated incrementally, one aged-out turn at a time, rather than
     regenerated from scratch on a fixed schedule (e.g. "every 10
     turns"). Regenerating from the full history re-pays for content
     already summarized and gets more expensive the longer a session
     runs; folding in one turn at a time keeps the cost constant.

A turn ages out of the raw window on WHICHEVER limit hits first -- a
turn-count ceiling (WINDOW_TURNS) or an estimated-token ceiling
(WINDOW_TOKEN_BUDGET). Tool replies in this app vary wildly in size (a
"hi" versus a full repo_health report), so a pure turn-count window can
still blow the token budget on a single oversized turn; the token
estimate (len(text) // 4 -- a rough approximation, not a real
tokenizer) exists specifically to catch that case. The most recent turn
is never evicted by the token check alone, even if it's oversized on its
own -- only the turn-count ceiling can push the window down to empty.

This module DOES NOT:
    ✗ Persist memory across process restarts (in-memory only, same
      lifetime as active_repo.py's active-repository state)
    ✗ Distinguish between multiple users/sessions -- this app is a
      single-user local tool, same assumption active_repo.py already
      makes
    ✗ Touch code_writer_tool's code-generation calls or the router's
      intent classification -- scoped out deliberately. Code-generation
      correctness must never depend on injected conversational text
      (the persona feature already showed that risk once), and the
      router's classification prompt has been separately, carefully
      tuned; adding history there is a distinct, riskier change than
      this one.
"""

from typing import Optional

from .tools._shared import call_llm


WINDOW_TURNS = 8
WINDOW_TOKEN_BUDGET = 3000

SUMMARY_SYSTEM_PROMPT = (
    "You maintain a running summary of an ongoing conversation between a "
    "user and RepoOps, an assistant for exploring a loaded GitHub "
    "repository. You will be given the existing summary (or told there "
    "isn't one yet) and one new turn that just aged out of the raw "
    "conversation window. Fold the new turn into the summary and return "
    "ONLY the updated summary text -- concise, factual, plain prose. "
    "Preserve anything from the old summary that's still relevant: what "
    "repository/files/topics have been discussed, any concrete facts or "
    "numbers the user was told, any decisions or preferences the user "
    "stated. Drop anything that's now clearly stale or superfluous. Do "
    "not add commentary about this task itself -- return just the "
    "summary, nothing else."
)

_turns: list[dict] = []
_summary: str = ""


def _estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


def _window_token_total() -> int:
    return sum(_estimate_tokens(t["user"]) + _estimate_tokens(t["assistant"]) for t in _turns)


def _fold_into_summary(turn: dict) -> None:
    global _summary

    user_content = (
        "Existing summary of the conversation so far:\n"
        f"{_summary if _summary else '(nothing yet -- this is the first turn being folded in)'}\n\n"
        "New turn to fold in:\n"
        f"User: {turn['user']}\n"
        f"Assistant: {turn['assistant']}"
    )

    try:
        _summary = call_llm(SUMMARY_SYSTEM_PROMPT, user_content)
    except Exception:
        # Summarization is a best-effort enhancement layered on top of the
        # chat path -- if it fails, the aged-out turn is just dropped
        # rather than breaking the (unrelated) chat request in flight.
        pass


def add_turn(user_message: str, assistant_reply: str, tool_used: Optional[str] = None) -> None:
    """Record one completed turn, then evict turns past either window
    limit into the rolling summary, oldest first."""

    _turns.append({"user": user_message, "assistant": assistant_reply, "tool_used": tool_used})

    while len(_turns) > WINDOW_TURNS or (len(_turns) > 1 and _window_token_total() > WINDOW_TOKEN_BUDGET):
        _fold_into_summary(_turns.pop(0))


def build_context() -> str:
    """The text block to prepend to a tool's LLM call -- "" (a true
    no-op) when there's no conversation yet, so the very first message
    in a session sees today's exact prompt, unaffected by this feature."""

    if not _turns and not _summary:
        return ""

    parts = []

    if _summary:
        parts.append(f"--- Summary of earlier conversation ---\n{_summary}")

    if _turns:
        recent = "\n".join(f"User: {t['user']}\nAssistant: {t['assistant']}" for t in _turns)
        parts.append(f"--- Recent conversation (most recent last) ---\n{recent}")

    return "\n\n".join(parts)


def reset() -> None:
    """Clear all conversation memory -- called when the active repository
    switches, since conversation grounded in the previous repo would
    otherwise leak into answers about the new one."""

    global _summary
    _turns.clear()
    _summary = ""
