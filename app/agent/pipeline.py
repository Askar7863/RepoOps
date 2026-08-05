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
from .graph import AGENT_GRAPH


def run_agent(state: dict) -> dict:
    """Run one message through the agent graph."""

    log_step("AGENT ROUTER")

    state = AGENT_GRAPH.invoke(state)

    log_info(f"Tool used: {state.get('tool_used')}")

    return state
