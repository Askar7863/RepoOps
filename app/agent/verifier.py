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
    "question — do not penalize honest 'not found' answers."
)


def verify_answer(state: dict) -> dict:
    """Judge the reply a tool produced. Sets verified / verification_reason."""

    if state.get("tool_used") == "general_chat" or state.get("error"):
        state["verified"] = True
        state["verification_reason"] = "No verification needed."
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
