"""
tools/

Agent Tool Handlers.

Responsibility
--------------
One handler per tool name the router can dispatch to: code_writer_tool,
repo_summarizer_tool, and repo_health_tool -- the only three tools this
router declares. general_chat is the conversational fallback for
anything the classifier doesn't route to one of the three.

Note: the router used to declare "ask_repo" as a separate tool alongside
"repo_summarizer" (summarize vs. answer-a-question). Both are now the
same handler (repo_summarizer_tool covers both, deliberately -- see
repo_summarizer.py): two overlapping tool descriptions give the
classifier a fuzzy line to draw wrong, where one tool covering the
whole scope has no such ambiguity.

repo_health_tool is a distinct axis from repo_summarizer_tool: WHAT the
repo is/does/how to use it (repo_summarizer_tool) vs. HOW GOOD/WELL-
MAINTAINED it is -- code quality plus real git-log activity/ownership
signals (repo_health_tool). See router.py for the exact disambiguating
language.

This module DOES NOT:
    ✗ Classify or route (see router.py)
"""

from .code_writer import act_code_writer
from .repo_summarizer import act_repo_summarizer
from .repo_health import act_repo_health
from .general_chat import act_general_chat

# Tools that mutate real state (here: write files to disk) rather than just
# answering. Retrying one of these on a failed verification isn't safe the
# way it is for a read-only tool -- re-running it can duplicate or corrupt
# what it already wrote, since it has no memory of its own prior attempt.
# graph.py checks this set before looping back to route_to_tool.
# repo_summarizer_tool and repo_health_tool are both read-only, so they're
# deliberately NOT in this set -- retrying either on failed verification is
# safe, same as general_chat.
SIDE_EFFECT_TOOLS = {"code_writer_tool"}
