"""
graph.py

Agent Graph (LangGraph).

Responsibility
--------------
Wire the agent's nodes into an actual LangGraph StateGraph:

    START -> resolve_local_path -> classify_intent -> route_to_tool -> verify_answer
                                                            ^                 |
                                                            |                 v
                                                       mark_retried  <-- failed verification,
                                                                          not yet retried
                                                                                 |
                                                                                 v (verified, or already retried)
                                                                                END

route_to_tool is retried at most once: if verify_answer fails it the
first time, mark_retried flips a flag and loops back to route_to_tool
(same chosen_tool, no re-classification); whatever verify_answer decides
on that second pass is final.

Compiled once at import time -- pipeline.py's run_agent() just invokes
the compiled graph.

This module DOES NOT:
    ✗ Implement classification, tool, or verification logic itself
      (see router.py, tools/, verifier.py)
"""

from langgraph.graph import StateGraph, START, END

from ..github.clone import REPO_STORAGE
from ..utils.logger import log_info
from .state import AgentState
from .router import classify_intent, route_to_tool
from .verifier import verify_answer
from .tools import SIDE_EFFECT_TOOLS


def resolve_local_path(state: dict) -> dict:
    """Set state['local_path'] if the active repository actually exists on disk."""

    repo_name = state.get("active_repository")

    if not repo_name:
        # Path("repos") / "" resolves to Path("repos") itself (pathlib
        # treats joining with an empty string as a no-op) -- and since
        # REPO_STORAGE always exists, that would make an empty/unset
        # active_repository look like a valid loaded repo below. Guard
        # against it explicitly rather than relying on the join to fail.
        state["local_path"] = None
        return state

    candidate = REPO_STORAGE / repo_name
    state["local_path"] = candidate if candidate.exists() else None
    return state


def mark_retried(state: dict) -> dict:
    log_info("Verification failed — retrying tool once")
    state["retried"] = True
    return state


def _after_verify(state: dict) -> str:
    if state.get("verified", True) or state.get("retried", False):
        return "end"

    if state.get("tool_used") in SIDE_EFFECT_TOOLS:
        # Retrying a tool that already mutated real state (wrote a file) is
        # not safe the way retrying a read-only answer is -- it has no
        # memory of its own prior attempt and could duplicate or corrupt
        # what it just wrote. Report the verification concern instead.
        log_info(f"Verification failed for '{state.get('tool_used')}' -- not retrying (has side effects)")
        return "end"

    return "retry"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("resolve_local_path", resolve_local_path)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("route_to_tool", route_to_tool)
    graph.add_node("verify_answer", verify_answer)
    graph.add_node("mark_retried", mark_retried)

    graph.add_edge(START, "resolve_local_path")
    graph.add_edge("resolve_local_path", "classify_intent")
    graph.add_edge("classify_intent", "route_to_tool")
    graph.add_edge("route_to_tool", "verify_answer")
    graph.add_conditional_edges("verify_answer", _after_verify, {"retry": "mark_retried", "end": END})
    graph.add_edge("mark_retried", "route_to_tool")

    return graph.compile()


AGENT_GRAPH = build_graph()
