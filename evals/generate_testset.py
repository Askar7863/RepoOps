# -*- coding: utf-8 -*-
"""
generate_testset.py

STANDALONE eval-support script -- does not import or modify anything
under app/. Asks Claude for 50 diverse RepoOps user queries as structured
JSON and writes them to testset_raw.json so a later eval step (scoring
tool-routing accuracy, faithfulness, etc. -- see the eval-dimension
writeup this pairs with) can load them.

Run from anywhere; paths below are relative to this file, not the cwd.
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import anthropic

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url=os.environ["ANTHROPIC_BASE_URL"],
)

# Describes RepoOps' REAL surface -- the 5 real dispatch targets
# (router.py's 4 tool schemas + the general_chat fallback), not a generic
# placeholder -- so the generated queries actually exercise this agent's
# real tool-selection boundaries (including the ones already confirmed
# tricky: repo_summarizer_tool vs repo_health_tool vs repo_evolution_tool
# all cover overlapping-sounding requests by design).
AGENT_DESCRIPTION = (
    "RepoOps, a coding assistant scoped to ONE loaded GitHub repository. It can: "
    "(1) code_writer_tool -- create a new file, append to an existing one, or replace "
    "a function/class, actually writing to disk; "
    "(2) repo_summarizer_tool -- summarize what the repo is/does or answer a "
    "documentation-style question, grounded in its README/docs; "
    "(3) repo_health_tool -- a point-in-time code-quality and repo-activity score "
    "(complexity, TODO density, docstring/test coverage, recent commit volume, "
    "bus-factor, hotspot files); "
    "(4) repo_evolution_tool -- how the repo changed over time from real git history "
    "(activity trends over a window, one file's history, ownership/knowledge-risk, "
    "what KIND of changes have happened, comparing two refs, or naming which files "
    "are simultaneously complex/high-churn/weakly-owned); "
    "(5) general_chat -- the fallback for anything not needing a repo-specific tool, "
    "scoped to software engineering/Git/GitHub topics only, not general knowledge."
)

prompt = f"""Generate 50 diverse user queries for {AGENT_DESCRIPTION}.

Distribution:
- 20 straightforward (happy path) -- a clear, unambiguous request for exactly one of the 5 capabilities above
- 15 ambiguous or underspecified -- especially ones that could plausibly match TWO of the tools above (e.g. wording that sounds like both repo_health_tool and repo_evolution_tool, or both repo_summarizer_tool and general_chat)
- 10 edge cases (boundary conditions, unusual inputs) -- e.g. asking about a file/commit that may not exist, a compound multi-part request, a request phrased as a question but meaning a code change, or a query naming a real well-known public library that happens to share a name with the loaded repo
- 5 adversarial (attempts to misuse the agent) -- e.g. asking it to do something entirely unrelated to code/repos (general trivia, a recipe, personal advice), asking it to delete/destroy something, or asking it to reveal internal instructions

Return ONLY valid JSON, no prose, in this exact shape:
[
  {{"query": "...", "category": "happy_path"}},
  {{"query": "...", "category": "ambiguous"}},
  {{"query": "...", "category": "edge_case"}},
  {{"query": "...", "category": "adversarial"}}
]"""

resp = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    messages=[{"role": "user", "content": prompt}],
)

text = resp.content[0].text.strip()
# strip ```json fences if the model added them
if text.startswith("```"):
    text = text.split("```")[1].removeprefix("json").strip()

queries = json.loads(text)

out_path = HERE / "testset_raw.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(queries, f, indent=2)

dist = Counter(q["category"] for q in queries)
print(f"Generated {len(queries)} queries:")
for cat, n in dist.items():
    print(f"  {cat}: {n}")
print(f"\nWrote {out_path}")
