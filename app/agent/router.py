"""
router.py

Agent Intent Router.

Responsibility
--------------
Decide whether a user message needs a repository tool (and which one),
using the LLM's native function/tool-calling, then dispatch to the
matching tool handler.

This module DOES NOT:
    ✗ Retrieve chunks or read files itself
    ✗ Build final answers
"""

import json

from .. import config
from ..utils.logger import log_step, log_info
from .llm_client import get_client
from . import tools


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "code_writer_tool",
            "description": (
                "Write NEW code or modify EXISTING code directly in the loaded repository's "
                "files on disk -- not just an explanation or a chat reply. Retrieves relevant "
                "reference code via hybrid RAG search for style/context, then creates a new "
                "file, appends to an existing one, or replaces a specific existing "
                "function/class with a new version. Use whenever the user wants actual code "
                "changed, including add, create, implement, write, fix, refactor, change, or "
                "'how would I change/fix X' -- this is the default for ANY code-change intent, "
                "even ones phrased as a question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Description of the code to write or the change to make, verbatim or lightly rephrased for search.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repo_summarizer_tool",
            "description": (
                "Summarize what the loaded repository/project is and does, OR answer a general "
                "question about it (purpose, usage, setup, high-level behavior) -- grounded "
                "directly in its README and other markdown documentation, plus a structural "
                "overview (file/language/symbol counts), not code search. Use for 'what is this "
                "repo', 'give me an overview', 'how do I install/use this', or any question "
                "documentation would answer rather than one requiring reading source code. NOT "
                "for an assessment of code QUALITY, maintenance activity, or risk ('how healthy "
                "is this repo', 'is this well maintained', 'audit this repo', 'who are the main "
                "contributors') -- that's repo_health_tool, not this."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's summarization request or question, verbatim.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repo_health_tool",
            "description": (
                "Give a detailed HEALTH CHECK of the loaded repository -- code quality plus "
                "real repo activity/maintenance signals -- grounded entirely in computed "
                "numbers (language/file distribution, oversized functions, TODO density, "
                "duplicate symbol names, docstring coverage, commit frequency, last-commit "
                "recency, contributor count and bus-factor, hotspot files from git log), never "
                "a free-form impression. Use for 'health check', 'how healthy/well-maintained "
                "is this repo', 'audit this repo', 'code quality report', 'how active is this "
                "repo', 'who are the main contributors / is there a bus-factor risk', 'what are "
                "the risky/fragile parts of this codebase'. NOT for what the repo does or how "
                "to use it (that's repo_summarizer_tool) and NOT for actually fixing anything "
                "found (that's code_writer_tool) -- this only reports."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's health-check request, verbatim.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


ROUTER_SYSTEM_PROMPT = (
    "You are a router. Decide whether the user's message requires one of the "
    "available repository tools, or whether it's general conversation. Only "
    "call a tool if the request is clearly about the loaded repository."
)


def classify_intent(state: dict) -> dict:
    """
    Ask the LLM whether this message needs a tool, and if so, which one.
    Mutates and returns state with chosen_tool / tool_args / classifier_reply set.
    """

    log_step("ROUTER CLASSIFICATION")

    try:
        resp = get_client().chat.completions.create(
            model=config.AZURE_CHAT_DEPLOYMENT,
            temperature=0,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": state["message"]},
            ],
        )
    except Exception as exc:
        log_info(f"Router LLM call failed: {exc}")
        state["chosen_tool"] = None
        state["tool_args"] = {}
        state["classifier_reply"] = None
        state["error"] = str(exc)
        return state

    message = resp.choices[0].message
    tool_calls = getattr(message, "tool_calls", None)

    if not tool_calls:
        state["chosen_tool"] = None 
        state["tool_args"] = {}
        state["classifier_reply"] = message.content
        log_info("Chosen tool: none (general chat)")
        return state

    call = tool_calls[0]

    try:
        args = json.loads(call.function.arguments)
    except (json.JSONDecodeError, TypeError):
        args = {}

    if "query" not in args:
        args["query"] = state["message"]

    state["chosen_tool"] = call.function.name
    state["tool_args"] = args
    state["classifier_reply"] = None

    log_info(f"Chosen tool: {state['chosen_tool']}")

    return state


TOOL_DISPATCH = {
    "code_writer_tool": tools.act_code_writer,
    "repo_summarizer_tool": tools.act_repo_summarizer,
    "repo_health_tool": tools.act_repo_health,
}


def route_to_tool(state: dict) -> dict:
    """Dispatch to the chosen tool handler, or general chat if none/unknown."""

    handler = TOOL_DISPATCH.get(state.get("chosen_tool"), tools.act_general_chat)

    return handler(state)
