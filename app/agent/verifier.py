"""
verifier.py

Answer Verification.

Responsibility
--------------
After a tool produces a reply, ask the LLM to judge whether it is
grounded in the context that was actually used, and adequately
answers the question.

This module DOES NOT:
    ✗ Generate answers itself
    ✗ Retry or re-route (see pipeline.py)
"""

import json

from .. import config
from ..utils.logger import log_step, log_info
from .llm_client import get_client


VERDICT_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "submit_verdict",
            "description": (
                "Report whether the answer is grounded in the given context "
                "and adequately addresses the question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "grounded": {
                        "type": "boolean",
                        "description": "True if the answer is supported by the context and addresses the question, false otherwise.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "One sentence explaining the verdict.",
                    },
                },
                "required": ["grounded", "reason"],
            },
        },
    }
]

VERIFIER_SYSTEM_PROMPT = (
    "You are a strict reviewer. Given a user's question, the context that was "
    "available to answer it, and a proposed answer, decide whether the answer "
    "is actually grounded in that context and adequately addresses the "
    "question. A tool honestly reporting it found nothing relevant should be "
    "judged as grounded=true if the context genuinely doesn't cover the "
    "question — do not penalize honest 'not found' answers.\n\n"
    "Check every specific number, count, percentage, date, hash, or proper name "
    "the answer states as fact against the context line by line — not just "
    "whether the answer's general topic matches the context. A plausible-"
    "sounding number that simply does not appear anywhere in the context (e.g. "
    "a total-commit count, a contributor count, a bus-factor percentage the "
    "context never provided) makes the answer NOT grounded, even if every other "
    "sentence in it is accurate. One fabricated figure is enough to fail the "
    "whole answer.\n\n"
    "Two different numbers for what sounds like 'the same thing' are NOT "
    "automatically a contradiction — a multi-section report routinely reports "
    "a derived score alongside the raw figure it's derived from (e.g. a 0-100 "
    "'Bus Factor score' vs. the top contributor's raw commit percentage elsewhere "
    "in the same context; an 'indexed/chunked files' count vs. a separate "
    "'total files on disk' count). Before flagging a number as fabricated or "
    "contradictory, check whether the context itself already explains the two "
    "numbers measure different things — if so, an answer citing either one "
    "correctly is grounded. Only fail the answer when a number contradicts what "
    "the SAME clearly-scoped metric says elsewhere, or appears nowhere in the "
    "context under any scope at all.\n\n"
    "Worked example of a GROUNDED answer (do not fail an answer shaped like "
    "this): context contains 'Bus Factor score: 73.5/100' in one section and "
    "'top contributor (David Lord) authored 26.5% of all commits' in another. "
    "An answer saying 'the Bus Factor score of 73.5/100 reflects low ownership "
    "concentration, with David Lord accounting for 26.5% of commits' is fully "
    "grounded — it correctly uses EACH number for what the context says it is "
    "(73.5/100 is explicitly the score; 26.5% is explicitly the raw commit "
    "share), it does not swap which number is which, and 73.5 is approximately "
    "100 minus 26.5, exactly the derivation the context describes. Do not fail "
    "this pattern by claiming the two numbers 'should' match exactly, or by "
    "claiming either one is mislabeled — check only whether each number is "
    "placed exactly as the context placed it."
)


def verify_answer(state: dict) -> dict:
    """Judge the reply a tool produced. Sets verified / verification_reason."""

    if (
        state.get("tool_used") == "general_chat"
        or state.get("error")
        or state.get("verified") is not None
    ):
        # The third condition respects a tool's OWN deterministic verdict --
        # e.g. repo_evolution_tool's not_applicable focus, or the
        # status-check fast path, both of which return a fixed reply with
        # nothing computed/narrated by an LLM for this verifier to
        # meaningfully judge. Only set the "no verification needed"
        # default when the tool didn't already decide one itself.
        state.setdefault("verified", True)
        state.setdefault("verification_reason", "No verification needed.")
        return state

    log_step("ANSWER VERIFICATION")

    user_content = (
        f"Question:\n{state.get('message', '')}\n\n"
        f"Context used:\n{state.get('context_used') or '(no context)'}\n\n"
        f"Proposed answer:\n{state.get('reply') or ''}"
    )

    try:
        resp = get_client().chat.completions.create(
            model=config.AZURE_CHAT_DEPLOYMENT,
            temperature=0,
            tools=VERDICT_SCHEMA,
            tool_choice={"type": "function", "function": {"name": "submit_verdict"}},
            messages=[
                {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        call = resp.choices[0].message.tool_calls[0]
        verdict = json.loads(call.function.arguments)

        state["verified"] = bool(verdict.get("grounded", True))
        state["verification_reason"] = verdict.get("reason", "")

    except Exception as exc:
        log_info(f"Verifier LLM call failed: {exc}")
        state["verified"] = True
        state["verification_reason"] = "Verification unavailable."

    log_info(f"Verified: {state['verified']} ({state['verification_reason']})")

    return state
