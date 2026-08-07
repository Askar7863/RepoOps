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

    # NOTE: state["classifier_reply"] (router.py's own free-text reply,
    # produced as a side effect whenever its tool_choice="auto" call
    # decides no tool is needed) is deliberately NOT used as the final
    # answer here, even though it's already-generated text sitting right
    # there. That call runs at temperature=0 with no persona lens applied
    # (the router's job is choosing a tool, not narrating an answer), so
    # using it directly made every general-chat reply identical across all
    # three personas -- confirmed by direct testing. Always generating the
    # real reply through call_llm() below, with persona applied, is the
    # fix; classifier_reply is discarded.

    system = "You are RepoOps, an assistant for exploring and modifying a loaded GitHub repository."

    try:
        reply = call_llm(
            system,
            state["message"],
            persona=state.get("persona"),
            conversation_context=state.get("conversation_context"),
        )
    except Exception as exc:
        return llm_error(state, tool_name, exc)

    state["reply"] = reply
    state["sources"] = []
    state["tool_used"] = tool_name
    return state
