# -*- coding: utf-8 -*-
"""
run_eval.py

STANDALONE eval runner -- imports app.agent.pipeline.run_agent directly
(read-mostly; code_writer_tool items DO write real files into repos/click/,
same as a real user request would) but does not modify any app/ source.

Loads golden_dataset.json (see golden_dataset.py), runs every item through
the REAL agent pipeline against the `click` repo, and writes eval_results.json
with the actual tool_used/verified/reply for each -- for manual scoring
against each item's `expected` criteria using the rubrics already written
for this project (faithfulness / tool_call_accuracy / clarification / safety).

Prints progress and a summary table as it runs so results are visible
without waiting for the whole file to finish.
"""

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.pipeline import run_agent

GOLDEN = json.loads((HERE / "golden_dataset.json").read_text(encoding="utf-8"))
REPO_NAME = "click"

results = []

for i, item in enumerate(GOLDEN, start=1):
    print(f"[{i}/{len(GOLDEN)}] {item['id']} ({item['category']}/{item['dimension']}): {item['input'][:70]}")

    state = {
        "message": item["input"],
        "active_repository": REPO_NAME,
        "persona": None,
        "conversation_context": None,
        "sources": [],
        "tool_used": "general_chat",
        "error": None,
    }

    start = time.time()
    try:
        result = run_agent(state)
        elapsed = round(time.time() - start, 2)
        row = {
            "id": item["id"],
            "input": item["input"],
            "category": item["category"],
            "dimension": item["dimension"],
            "expected": item["expected"],
            "expected_tool": item["expected_tool"],
            "actual_tool": result.get("tool_used"),
            "tool_match": result.get("tool_used") == item["expected_tool"],
            "verified": result.get("verified"),
            "verification_reason": result.get("verification_reason"),
            "reply": result.get("reply"),
            "latency_s": elapsed,
            "error": None,
        }
    except Exception as exc:
        elapsed = round(time.time() - start, 2)
        row = {
            "id": item["id"],
            "input": item["input"],
            "category": item["category"],
            "dimension": item["dimension"],
            "expected": item["expected"],
            "expected_tool": item["expected_tool"],
            "actual_tool": None,
            "tool_match": False,
            "verified": None,
            "verification_reason": None,
            "reply": None,
            "latency_s": elapsed,
            "error": str(exc),
        }

    results.append(row)
    print(f"    -> tool={row['actual_tool']} (expected={row['expected_tool']}, match={row['tool_match']}) "
          f"verified={row['verified']} time={row['latency_s']}s")

    # Write incrementally so results are visible even if a later item hangs/fails.
    (HERE / "eval_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total = len(results)
tool_matches = sum(1 for r in results if r["tool_match"])
verified_true = sum(1 for r in results if r["verified"] is True)
errors = sum(1 for r in results if r["error"])

print("\n" + "=" * 70)
print(f"TOTAL: {total} | tool_match: {tool_matches}/{total} ({tool_matches/total:.0%}) | "
      f"verified=True: {verified_true}/{total} ({verified_true/total:.0%}) | errors: {errors}")

print("\nBy category:")
for cat in ["happy_path", "ambiguous", "edge_case", "adversarial"]:
    cat_rows = [r for r in results if r["category"] == cat]
    if not cat_rows:
        continue
    cat_matches = sum(1 for r in cat_rows if r["tool_match"])
    print(f"  {cat:12s}: {cat_matches}/{len(cat_rows)} tool_match")

print("\nMismatches (actual != expected tool):")
for r in results:
    if not r["tool_match"]:
        print(f"  {r['id']}: expected={r['expected_tool']} actual={r['actual_tool']} -- {r['input'][:60]}")

print(f"\nFull results written to {HERE / 'eval_results.json'}")
