"""
tools/

Agent Tool Handlers.

Responsibility
--------------
One handler per tool name the router can dispatch to: code_writer_tool,
repo_summarizer_tool, repo_health_tool, and repo_evolution_tool -- the
only four tools this router declares. general_chat is the conversational
fallback for anything the classifier doesn't route to one of the four.

Note: the router used to declare "ask_repo" as a separate tool alongside
"repo_summarizer" (summarize vs. answer-a-question). Both are now the
same handler (repo_summarizer_tool covers both, deliberately -- see
repo_summarizer.py): two overlapping tool descriptions give the
classifier a fuzzy line to draw wrong, where one tool covering the
whole scope has no such ambiguity.

Three read-only axes, deliberately kept distinct rather than merged the
way ask_repo was, because each answers a genuinely different question
(see router.py for the exact disambiguating language):
  - repo_summarizer_tool: WHAT the repo is/does/how to use it, from docs + code.
  - repo_health_tool: HOW GOOD/WELL-MAINTAINED it is RIGHT NOW -- one
    point-in-time score (fixed 90-day/12-week window, one repo-wide
    bus-factor number).
  - repo_evolution_tool: HOW it changed, over a window/path/ref-range the
    query itself specifies -- trends, file/directory ownership drilldowns,
    change-intent breakdowns, tag/branch comparisons, and a cross-tool RISK
    view that joins repo_health's own complexity signals against git churn
    and ownership.
code_writer_tool is the only one of the four that writes anything.

This module DOES NOT:
    ✗ Classify or route (see router.py)
"""

from .code_writer import act_code_writer
from .repo_summarizer import act_repo_summarizer
from .repo_health import act_repo_health
from .repo_evolution import act_repo_evolution
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
