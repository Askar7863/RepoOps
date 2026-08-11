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
from .llm_client import get_fast_client
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
                "Give a detailed HEALTH CHECK of the loaded repository -- a POINT-IN-TIME "
                "snapshot of code quality plus repo activity, grounded entirely in computed "
                "numbers (language/file distribution, oversized functions, TODO density, "
                "duplicate symbol names, docstring coverage, a fixed recent-90-day commit "
                "count, one repo-wide contributor count and bus-factor, top hotspot files), "
                "never a free-form impression. Use for 'health check', 'how healthy/well-"
                "maintained is this repo RIGHT NOW', 'audit this repo', 'code quality report'. "
                "NOT for what the repo does or how to use it (that's repo_summarizer_tool), NOT "
                "for actually fixing anything found (that's code_writer_tool), and NOT for a "
                "question about a SPECIFIC time window/file/directory's history, ownership of "
                "one particular area, what KIND of changes have been happening, comparing two "
                "tags/branches, or 'who should I ask about this file' -- those are all "
                "repo_evolution_tool, which analyzes trends and history rather than one fixed "
                "snapshot. In particular, do NOT use this tool for 'which specific FILES are "
                "high-risk/risky/fragile', 'which files are complex AND change a lot AND only "
                "one person understands them', or any question asking to NAME the risky files "
                "rather than score the whole repo -- that per-file compound signal is "
                "repo_evolution_tool's dedicated capability, even though the word 'risk' or "
                "'health' appears in the question. Also NOT for a question PRIMARILY about "
                "ownership/bus-factor itself -- 'who owns this repo', 'is there a bus-factor "
                "risk', 'is ownership concentrated in one person' -- even asked about the whole "
                "repo with no specific file named. Bus-factor here is only one incidental number "
                "inside a much bigger fixed report; repo_evolution_tool's ownership capability "
                "answers that question directly and can also break it down per-directory, which "
                "this tool never does. Only use this tool when the request is for the FULL "
                "health check/audit/quality report, not when ownership/bus-factor is the actual "
                "question being asked."
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
    {
        "type": "function",
        "function": {
            "name": "repo_evolution_tool",
            "description": (
                "Analyze how the loaded repository CHANGED OVER TIME, from its real git "
                "history -- not what it does now (that's repo_summarizer_tool) and not one "
                "fixed point-in-time score (that's repo_health_tool). Use for: activity/commit "
                "trends over a window the user names ('last quarter', 'since January'); the "
                "history of one specific file or directory ('when was X last touched', 'how "
                "often does Y change'); ownership/knowledge-risk for a specific file, directory, "
                "OR the whole repo -- 'who owns this repo', 'who should I ask about this', 'is "
                "there a bus-factor risk', 'is ownership concentrated in one person' -- even with "
                "no specific file named; what KIND of changes have been happening "
                "(bug fixes vs. features vs. refactors); comparing two tags/branches/commits "
                "('what changed between v1 and v2'); or NAMING which specific files are risky -- "
                "'which files are high-risk', 'which files are complicated and change a lot and "
                "only one person understands them', 'any single points of failure in this "
                "codebase' -- a combined signal of complexity + churn + weak ownership that "
                "repo_health_tool's one repo-wide score never breaks down per-file. This tool NEVER "
                "generates or modifies code -- if a request also asks for a fix, route the "
                "history/analysis part here and let code_writer_tool handle any actual change."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's repository-history question, verbatim.",
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

CONTEXT_PREFIX = (
    "The block below is prior conversation context (recent turns and/or a "
    "summary of older ones), included only so you can resolve follow-up "
    "references -- pronouns, \"that file\", \"the other one\", \"and its "
    "risk too\" -- in the CURRENT message, and so a general-conversation "
    "reply you write yourself can stay consistent with what was already "
    "said. It is context, not itself something to classify or act on.\n\n"
)


def classify_intent(state: dict) -> dict:
    """
    Ask the LLM whether this message needs a tool, and if so, which one.
    Mutates and returns state with chosen_tool / tool_args set.
    """

    log_step("ROUTER CLASSIFICATION")

    conversation_context = state.get("conversation_context")
    user_content = (
        f"{CONTEXT_PREFIX}{conversation_context}\n\n--- Current message ---\n{state['message']}"
        if conversation_context else state["message"]
    )

    try:
        resp = get_fast_client().chat.completions.create(
            model=config.FAST_CHAT_MODEL,
            temperature=0,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
    except Exception as exc:
        log_info(f"Router LLM call failed: {exc}")
        state["chosen_tool"] = None
        state["tool_args"] = {}
        state["error"] = str(exc)
        return state

    message = resp.choices[0].message
    tool_calls = getattr(message, "tool_calls", None)

    if not tool_calls:
        # message.content is the router's own free-text reply, generated
        # as a side effect of this tool_choice="auto" call deciding no
        # tool applies -- deliberately discarded, not stored. It runs at
        # temperature=0 with no persona lens, so using it as the final
        # answer (as this used to, via a "classifier_reply" field) made
        # every general_chat reply identical across personas regardless of
        # which one was selected. act_general_chat generates the real,
        # persona-aware reply itself.
        state["chosen_tool"] = None
        state["tool_args"] = {}
        log_info("Chosen tool: none (general chat)")
        return state

    call = tool_calls[0]

    try:
        args = json.loads(call.function.arguments)
    except (json.JSONDecodeError, TypeError):
        args = {}

    chosen_tool = call.function.name

    # repo_summarizer_tool/repo_health_tool/repo_evolution_tool's own schemas
    # all declare "query" as verbatim -- but the model doesn't reliably keep
    # that promise: confirmed in practice, a self-contained follow-up
    # question got its "query" arg rewritten to reference the PREVIOUS
    # turn's file instead of the one actually named in the current message,
    # because conversation_context (included above so the router can resolve
    # pronouns/"that file") bled into the argument it constructed. Since the
    # raw message is already exactly what these three schemas ask for,
    # trust it over whatever the model reconstructed. code_writer_tool's
    # schema is the one exception that explicitly permits "lightly
    # rephrased for search" -- only that tool keeps the model's own query.
    if chosen_tool != "code_writer_tool" or "query" not in args:
        args["query"] = state["message"]

    state["chosen_tool"] = chosen_tool
    state["tool_args"] = args

    log_info(f"Chosen tool: {state['chosen_tool']}")

    return state


TOOL_DISPATCH = {
    "code_writer_tool": tools.act_code_writer,
    "repo_summarizer_tool": tools.act_repo_summarizer,
    "repo_health_tool": tools.act_repo_health,
    "repo_evolution_tool": tools.act_repo_evolution,
}


def route_to_tool(state: dict) -> dict:
    """Dispatch to the chosen tool handler, or general chat if none/unknown."""

    handler = TOOL_DISPATCH.get(state.get("chosen_tool"), tools.act_general_chat)

    return handler(state)
