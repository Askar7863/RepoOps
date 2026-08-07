"""
pipeline.py

Agent Pipeline Entrypoint.

Responsibility
--------------
The single function main.py calls per chat message. Delegates the actual
orchestration -- resolve_local_path -> classify_intent -> route_to_tool ->
verify_answer, retrying route_to_tool once on failed verification -- to
the compiled LangGraph graph (see graph.py), and hand  les the surrounding
logging.

This module DOES NOT:
    ✗ Define the graph's nodes or edges itself (see graph.py)
    ✗ Implement classification or tool logic itself
"""

from ..utils.logger import log_step, log_info
from .graph import AGENT_GRAPH, resolve_local_path
from .status_check import check_status_followup


def run_agent(state: dict) -> dict:
    """Run one message through the agent graph.

    Resolves local_path once, up front, so the deterministic status-check
    fast path (see status_check.py) can use it before ever reaching the
    LLM router -- that fast path exists specifically because none of the 4
    tools the router chooses among can answer "does this file exist right
    now," so no amount of routing/classification would fix it; skipping
    the graph entirely for this narrow case is the actual fix. The graph's
    own resolve_local_path node re-runs this when the fast path doesn't
    match, which is fine -- it's a cheap, pure existence check either way.
    """

    log_step("AGENT ROUTER")

    state = resolve_local_path(state)

    fast_path = check_status_followup(state.get("message", ""), state.get("local_path"), state.get("persona"))
    if fast_path is not None:
        state.update(fast_path)
        log_info(f"Tool used: {state['tool_used']} (deterministic status-check fast path)")
        return state

    state = AGENT_GRAPH.invoke(state)

    log_info(f"Tool used: {state.get('tool_used')}")

    return state
