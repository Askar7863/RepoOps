"""
state.py

Agent Graph State Shape.

Responsibility
--------------
Define the single state dict that flows through the agent graph (see
graph.py): resolve_local_path -> classify_intent -> route_to_tool ->
verify_answer, with one retry of route_to_tool -> verify_answer if the
first answer fails verification.

This module DOES NOT:
    ✗ Contain any business logic
    ✗ Call the LLM
    ✗ Retrieve chunks or read files
"""

from pathlib import Path
from typing import TypedDict


class AgentState(TypedDict, total=False):
    # ---- input (set by main.py before entering the graph) ----
    message: str
    active_repository: str
    local_path: Path | None
    persona: str | None  # "student" | "developer" | "architect" -- narration lens only, never affects computed facts
    conversation_context: str | None  # recent turns + rolling summary from memory.py -- narration grounding only, never affects computed facts

    # ---- classifier/router output ----
    chosen_tool: str | None
    tool_args: dict
    classifier_reply: str | None

    # ---- tool execution output ----
    reply: str | None
    sources: list
    tool_used: str | None
    error: str | None
    context_used: str | None  # what the tool actually grounded its answer in, for verify_answer

    # ---- verification output ----
    verified: bool | None
    verification_reason: str | None
    retried: bool  # set once route_to_tool has already been retried, so the graph retries at most once
