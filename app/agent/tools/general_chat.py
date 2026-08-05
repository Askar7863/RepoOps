"""
general_chat.py

General Chat Fallback.

Responsibility
--------------
Handle messages the router decided don't need any repository tool -- the
baseline conversational path.

This module DOES NOT:
    ✗ Retrieve chunks or read files
    ✗ Classify or route (see router.py)
"""

from ._shared import call_llm, llm_error


def act_general_chat(state: dict) -> dict:
    tool_name = "general_chat"

    if state.get("classifier_reply"):
        state["reply"] = state["classifier_reply"]
        state["sources"] = []
        state["tool_used"] = tool_name
        return state

    system = "You are RepoOps, an assistant for exploring and modifying a loaded GitHub repository."

    try:
        reply = call_llm(system, state["message"])
    except Exception as exc:
        return llm_error(state, tool_name, exc)

    state["reply"] = reply
    state["sources"] = []
    state["tool_used"] = tool_name
    return state
