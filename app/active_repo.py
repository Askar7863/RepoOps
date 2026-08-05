"""
active_repo.py

Active Repository State.

Responsibility
--------------
Track which repository is currently loaded and queryable, persisted to a
file rather than an in-memory variable.

Why a file and not a module-level global
-----------------------------------------
An in-memory global is per-process. If more than one server process ever
ends up running against the same port -- started twice by accident, a
stale process left over from a previous run, `--reload` spawning a
worker that doesn't fully replace the old one -- each process holds its
own disconnected idea of "the active repo." A `/api/repositories/load`
request handled by one process then has no effect on what a `/api/chat`
request handled by a different process thinks is active: chat can answer
from a stale repo, or from none at all, even though the UI just reported
success. This happened repeatedly in practice. A single file on disk is
one shared source of truth every process reads and writes, so which
process happens to handle a given request stops mattering.

This module DOES NOT:
    ✗ Clone, scan, chunk, or index repositories
    ✗ Delete a repository's stored index (see app/rag/chroma_store.py,
      app/rag/sparse_store.py -- main.py calls those directly)
"""

from pathlib import Path

_STATE_FILE = Path(__file__).resolve().parent.parent / ".active_repository"


def get_active_repository() -> str | None:
    """The currently loaded repository's name, or None if none is loaded."""

    if not _STATE_FILE.exists():
        return None

    name = _STATE_FILE.read_text(encoding="utf-8").strip()

    return name or None


def set_active_repository(name: str) -> None:
    """Record `name` as the currently loaded repository."""

    _STATE_FILE.write_text(name, encoding="utf-8")
