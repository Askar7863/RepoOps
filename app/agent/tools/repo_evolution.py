"""
repo_evolution.py

Repository Evolution Analyzer.

Responsibility
--------------
Answer questions about how the loaded repository got to its current
state -- activity trends, file/module history, ownership and knowledge
risk, the KIND of changes that have been happening, and comparisons
between two points in history -- grounded entirely in real `git log`
data (via app.github.git_history, shared with repo_health_tool so
there's exactly one place that parses git log), never a free-form LLM
impression.

This is a different axis from the other three tools:
  - repo_summarizer_tool answers WHAT the repo is/does, from docs + code.
  - repo_health_tool gives a POINT-IN-TIME snapshot score (one fixed
    90-day / 12-week window, one repo-wide bus-factor number).
  - repo_evolution_tool (this file) answers HOW the repo changed, over
    a window or path the user's OWN query specifies, with per-directory
    ownership drilldowns repo_health_tool doesn't compute, and a
    dedicated capability (RISK) that cross-references repo_health's own
    complexity/oversized-function signals against this tool's churn and
    ownership data -- a compound signal neither tool alone produces.
  - code_writer_tool is the only one of the four that writes anything.
    This tool never generates or modifies code; it only reports.

Pipeline
--------
User query -> extract_scope (forced tool call: which of the 6 focus
areas, which file/dir, which date window or ref pair) -> compute the
one relevant structured block deterministically in Python (never ask
the LLM to invent a number) -> LLM narrates ONLY over that pre-computed
block -> reply + sources.

This module DOES NOT:
    ✗ Write or modify any files
    ✗ Recompute repo_health_tool's own signals -- it imports and reuses
      compute_health() for the RISK capability instead of re-deriving
      complexity/oversized-function data independently
"""

import json
from collections import Counter
from pathlib import Path
from typing import List, Optional

from ... import config
from ...utils.logger import log_warning
from ...github.git_history import (
    log_commits, weekly_activity, hotspot_files, commits_between,
    diff_stat_between, file_touches_with_author, normalize_identity, is_git_repo,
    latest_commit_detail,
)
from ..llm_client import get_client
from ._shared import call_llm, no_repo, llm_error
from .repo_health import compute_health

TOOL_NAME = "repo_evolution_tool"

_TOP_N = 10

# A commit-subject classification batch this size stays well clear of the
# JSON-fragility failure mode confirmed earlier in this project (cramming
# too much into one structured tool-call response produced malformed JSON)
# -- subjects are short, but this cap is deliberately conservative rather
# than pushed to whatever the model might technically fit.
_MAX_SUBJECTS_FOR_CLASSIFICATION = 80


# ---------------------------------------------------------------------------
# Scope extraction -- which of the 6 capabilities, and over what window/path
# ---------------------------------------------------------------------------

EXTRACT_SCOPE_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "extract_scope",
            "description": "Extract the scope of a repository-history question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "enum": ["activity", "last_commit", "file_history", "ownership", "change_intent", "compare", "risk"],
                        "description": (
                            "activity: general commit velocity/contributor TRENDS over a window (the "
                            "default when nothing more specific applies) -- NOT for a question about "
                            "one single commit's own detail. last_commit: detail of the single most "
                            "recent commit itself -- 'when was the last commit', 'what/how many files "
                            "changed in the last commit', 'who made the latest commit', 'what did the "
                            "most recent commit do'. Use this, not activity, whenever the question is "
                            "about ONE specific commit (the latest one) rather than a trend over many. "
                            "file_history: the history of one specific file or directory named in the "
                            "query. ownership: who owns/is responsible for a file, directory, or the "
                            "whole repo -- bus-factor / knowledge-risk questions. change_intent: what "
                            "KIND of changes have been happening (bug fixes vs. features vs. refactors "
                            "vs. docs). compare: what changed between two tags/branches/commits. risk: "
                            "a combined signal -- which files are simultaneously complex, high-churn, "
                            "and weakly-owned."
                        ),
                    },
                    "path": {
                        "type": ["string", "null"],
                        "description": "A specific file or directory WITHIN the repository the query names, or null if it's about the whole repository. Never the repository's own name.",
                    },
                    "since": {
                        "type": ["string", "null"],
                        "description": "The start of the query's time window, if any, else null. For a RELATIVE expression ('last year', 'last 6 months', 'recently'), emit it as a git-relative phrase VERBATIM ('1 year ago', '6 months ago') -- never compute a calendar date yourself. Only emit an absolute date ('2024-01-01') when the query itself names one explicitly.",
                    },
                    "until": {
                        "type": ["string", "null"],
                        "description": "Same as since, but the end of the window, else null. Leave null for an open-ended 'up to now' window -- do not compute today's date yourself.",
                    },
                    "ref_a": {
                        "type": ["string", "null"],
                        "description": "For a compare-focused query only: the earlier/base tag, branch, or commit. Null otherwise.",
                    },
                    "ref_b": {
                        "type": ["string", "null"],
                        "description": "For a compare-focused query only: the later tag/branch/commit -- default to 'HEAD' if the query implies 'up to now'. Null otherwise.",
                    },
                },
                "required": ["focus", "path", "since", "until", "ref_a", "ref_b"],
            },
        },
    }
]

EXTRACT_SCOPE_SYSTEM_PROMPT = (
    "Extract the scope of a repository git-history question into structured fields. "
    "Pick exactly one focus from the enum -- when in doubt between activity and "
    "something more specific, prefer the more specific one only if the query clearly "
    "names a file, an owner/risk question, a kind-of-change question, or two refs to "
    "compare; otherwise default to activity. Never invent a path, date, or ref the "
    "query doesn't name -- leave it null.\n\n"
    "You do not know today's actual date, and confidently guessing one is a real, "
    "confirmed failure mode -- for 'the last year', computing your own absolute "
    "start/end dates has produced a window off by several years, silently making "
    "the whole answer wrong. For ANY relative time expression ('last year', 'last "
    "6 months', 'recently', 'the past few weeks'), emit it as a git-relative phrase "
    "VERBATIM in `since` ('1 year ago', '6 months ago') and leave `until` null -- "
    "git resolves that against the real current time itself when the command runs, "
    "which is exact where your own guess cannot be. Only emit an absolute calendar "
    "date when the query itself states one explicitly (e.g. 'since January 2024')."
)


def _extract_scope(query: str) -> dict:
    try:
        resp = get_client().chat.completions.create(
            model=config.AZURE_CHAT_DEPLOYMENT,
            temperature=0,
            tools=EXTRACT_SCOPE_SCHEMA,
            tool_choice={"type": "function", "function": {"name": "extract_scope"}},
            messages=[
                {"role": "system", "content": EXTRACT_SCOPE_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        return json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
    except Exception as exc:
        log_warning(f"{TOOL_NAME}: scope extraction failed, defaulting to whole-repo activity: {exc}")
        return {"focus": "activity", "path": None, "since": None, "until": None, "ref_a": None, "ref_b": None}


# ---------------------------------------------------------------------------
# Per-focus deterministic computation -- every number here comes straight
# from git_history / compute_health, never invented by the LLM.
# ---------------------------------------------------------------------------

def _bus_factor_for_rows(rows: list) -> dict:
    """rows: [{"path","author","author_email"}, ...] already scoped to one
    group -- contributor breakdown + bus-factor pct for that group, with
    same-person-different-email identities merged via normalize_identity."""

    if not rows:
        return {"touch_count": 0, "contributor_count": 0, "top_contributor": None, "bus_factor_pct": None}

    counts = Counter(normalize_identity(r["author"], r["author_email"]) for r in rows)
    display = {}
    for r in rows:
        key = normalize_identity(r["author"], r["author_email"])
        display.setdefault(key, r["author"])

    top_key, top_count = counts.most_common(1)[0]
    return {
        "touch_count": len(rows),
        "contributor_count": len(counts),
        "top_contributor": display[top_key],
        "top_contributor_touches": top_count,
        "bus_factor_pct": round(100 * top_count / len(rows), 1),
    }


def _compute_activity(local_path: Path, since: Optional[str], until: Optional[str]) -> dict:
    commits = log_commits(local_path, since=since, until=until)
    if commits is None:
        return {"available": False, "reason": "not a git repository"}
    if not commits:
        return {
            "available": True, "window": {"since": since, "until": until},
            "commit_count": 0, "contributor_count": 0, "top_contributors": [], "weekly_activity": [],
        }

    author_counts = Counter(c["author"] for c in commits)
    return {
        "available": True,
        "window": {"since": since, "until": until},
        "commit_count": len(commits),
        "contributor_count": len(author_counts),
        "top_contributors": author_counts.most_common(_TOP_N),
        "weekly_activity": weekly_activity(commits, weeks=12),
        "first_commit_in_window": commits[-1]["date"],
        "last_commit_in_window": commits[0]["date"],
    }


def _compute_last_commit(local_path: Path) -> dict:
    detail = latest_commit_detail(local_path)
    if detail is None:
        return {"available": False, "reason": "not a git repository, or it has no commits"}

    return {
        "available": True,
        "hash": detail["hash"][:10],
        "date": detail["date"],
        "author": detail["author"],
        "subject": detail["subject"],
        "files_changed_count": len(detail["files_changed"]),
        "files_changed": detail["files_changed"],
    }


def _compute_file_history(local_path: Path, path: Optional[str], since: Optional[str], until: Optional[str]) -> dict:
    if not path:
        return {"available": False, "reason": "no file or directory named in the query"}

    commits = log_commits(local_path, since=since, until=until, paths=[path], follow=True)
    if commits is None:
        return {"available": False, "reason": "not a git repository"}
    if not commits:
        return {"available": True, "path": path, "commit_count": 0, "contributors": [], "recent_commits": []}

    author_counts = Counter(c["author"] for c in commits)
    return {
        "available": True,
        "path": path,
        "commit_count": len(commits),
        "first_touched": commits[-1]["date"],
        "last_touched": commits[0]["date"],
        "contributors": author_counts.most_common(_TOP_N),
        "recent_commits": [
            {"date": c["date"], "author": c["author"], "subject": c["subject"]} for c in commits[:15]
        ],
    }


def _compute_ownership(local_path: Path, path: Optional[str], since: Optional[str], until: Optional[str]) -> dict:
    scope_paths = [path] if path else None
    rows = file_touches_with_author(local_path, since=since, until=until, paths=scope_paths)
    if not rows:
        return {"available": False, "reason": "no file touches found for this scope"}

    overall = _bus_factor_for_rows(rows)

    # Per-immediate-subdirectory breakdown under the scoped path (or repo
    # root) -- repo_health_tool only ever reports ONE repo-wide bus-factor
    # number; this drills down to which specific area actually carries
    # the risk.
    prefix_depth = len(Path(path).parts) if path else 0
    groups = {}
    for r in rows:
        parts = Path(r["path"]).parts
        if len(parts) <= prefix_depth:
            continue
        group_key = "/".join(parts[: prefix_depth + 1])
        groups.setdefault(group_key, []).append(r)

    per_directory = sorted(
        ({"directory": g, **_bus_factor_for_rows(gr)} for g, gr in groups.items()),
        key=lambda d: -d["touch_count"],
    )[:_TOP_N]

    return {
        "available": True,
        "scope": path or "(whole repository)",
        "overall": overall,
        "per_directory": per_directory,
    }


CLASSIFY_COMMITS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "classify_commits",
            "description": "Classify each numbered commit subject into exactly one category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "classifications": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "index": {"type": "integer"},
                                "category": {
                                    "type": "string",
                                    "enum": ["feature", "fix", "refactor", "docs", "test", "chore", "other"],
                                },
                            },
                            "required": ["index", "category"],
                        },
                    }
                },
                "required": ["classifications"],
            },
        },
    }
]

CLASSIFY_COMMITS_SYSTEM_PROMPT = (
    "Classify each numbered commit subject line into exactly one category: "
    "feature (new capability), fix (bug fix), refactor (internal restructuring, "
    "no behavior change), docs (documentation only), test (test-only changes), "
    "chore (deps/build/CI/tooling), or other (anything that doesn't clearly fit). "
    "Return exactly one classification per index given, using the ORIGINAL numbering."
)


def _classify_change_intent(commits: list) -> dict:
    if not commits:
        return {"available": True, "commit_count": 0, "categories": {}}

    subjects = commits[:_MAX_SUBJECTS_FOR_CLASSIFICATION]
    numbered = "\n".join(f"{i}: {c['subject']}" for i, c in enumerate(subjects))

    try:
        resp = get_client().chat.completions.create(
            model=config.AZURE_CHAT_DEPLOYMENT,
            temperature=0,
            tools=CLASSIFY_COMMITS_SCHEMA,
            tool_choice={"type": "function", "function": {"name": "classify_commits"}},
            messages=[
                {"role": "system", "content": CLASSIFY_COMMITS_SYSTEM_PROMPT},
                {"role": "user", "content": numbered},
            ],
        )
        data = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
        counts = Counter(item.get("category", "other") for item in data.get("classifications", []))
    except Exception as exc:
        log_warning(f"{TOOL_NAME}: change-intent classification failed: {exc}")
        return {"available": False, "reason": "classification failed"}

    total = sum(counts.values()) or 1
    return {
        "available": True,
        "commit_count": len(subjects),
        # No silent truncation: say plainly when this is a sample, not the whole history.
        "sample_note": (
            f"based on the {len(subjects)} most recent matching commits out of {len(commits)} total"
            if len(commits) > len(subjects) else "based on all matching commits"
        ),
        "categories": {k: {"count": v, "pct": round(100 * v / total, 1)} for k, v in counts.most_common()},
    }


def _compute_change_intent(local_path: Path, path: Optional[str], since: Optional[str], until: Optional[str]) -> dict:
    scope_paths = [path] if path else None
    commits = log_commits(local_path, since=since, until=until, paths=scope_paths)
    if commits is None:
        return {"available": False, "reason": "not a git repository"}
    return _classify_change_intent(commits)


_FIRST_COMMIT_PHRASES = {
    "initial commit", "first commit", "the first commit", "the initial commit",
    "beginning", "the beginning", "start", "the start",
    "start of the repo", "the start of the repo", "when the repo started",
}


def _resolve_ref(local_path: Path, ref: Optional[str]) -> Optional[str]:
    """
    'the initial commit'/'the first commit' isn't a real git revision --
    unlike a date ('1 year ago'), there's no git syntax meaning "the
    earliest commit" that the LLM could pass through verbatim for git to
    resolve itself, so this has to be resolved here instead. Confirmed as
    a real gap: extraction correctly avoided inventing a hash for this
    phrase, but that meant the range failed to resolve at all even though
    the actual first commit's hash is trivially knowable.
    """

    if ref is None or ref.strip().lower() not in _FIRST_COMMIT_PHRASES:
        return ref

    commits = log_commits(local_path)
    return commits[-1]["hash"] if commits else ref


def _compute_compare(local_path: Path, ref_a: Optional[str], ref_b: Optional[str]) -> dict:
    if not ref_a:
        return {"available": False, "reason": "no base ref (tag, branch, or commit) named to compare from"}

    ref_a = _resolve_ref(local_path, ref_a)
    ref_b = _resolve_ref(local_path, ref_b) or "HEAD"

    commits = commits_between(local_path, ref_a, ref_b)
    if commits is None:
        return {"available": False, "reason": f"couldn't resolve the range {ref_a}..{ref_b} -- check the ref names are correct"}

    diff_stat = diff_stat_between(local_path, ref_a, ref_b) or ""
    author_counts = Counter(c["author"] for c in commits)

    return {
        "available": True,
        "ref_a": ref_a,
        "ref_b": ref_b,
        "commit_count": len(commits),
        "contributor_count": len(author_counts),
        "contributors": author_counts.most_common(_TOP_N),
        "commit_subjects_sample": [c["subject"] for c in commits[:25]],
        "diff_stat_tail": "\n".join(diff_stat.strip().splitlines()[-15:]) if diff_stat.strip() else "",
    }


def _compute_risk(repository_name: str, local_path: Path) -> dict:
    """
    The cross-tool signal: which files are simultaneously (a) flagged as
    complex/oversized by repo_health_tool's own already-computed
    analysis, (b) high-churn per git history, and (c) weakly owned (few
    distinct contributors) -- a compound risk view neither repo_health
    (no churn-by-file-vs-complexity join) nor this tool alone (no
    complexity signal -- that's chunker/tree-sitter derived, not git
    history) can produce independently.
    """

    try:
        health = compute_health(repository_name, local_path)
    except Exception as exc:
        return {"available": False, "reason": f"couldn't compute repo_health signals: {exc}"}

    code_health = health.get("code_health", {})
    complexity_info = health.get("complexity_info", {})
    if not code_health.get("available") and not complexity_info.get("available"):
        return {"available": False, "reason": "repo_health signals unavailable (no indexed chunks or no function chunks found)"}

    # `git log`/`git diff` always report paths with forward slashes,
    # regardless of OS -- but code_health/complexity_info come from
    # chunker.py's Path.relative_to(), which stringifies with the native
    # separator (backslash on Windows). Without normalizing, this join
    # silently matches nothing on Windows: confirmed empty every time
    # against a real repo (click) that does have genuinely complex,
    # high-churn files, until this fix.
    high_complexity = {
        path.replace("\\", "/"): (name, score)
        for path, name, score in complexity_info.get("high_complexity_sample", [])
    }
    oversized_paths = {path.replace("\\", "/") for path, _ in code_health.get("oversized_symbols_sample", [])}

    # Unbounded (not top-N) on purpose: this is an internal join key, not a
    # display list -- capping it here would silently zero out the churn
    # count for any complex/oversized file that isn't ALSO in a small
    # top-N slice, understating real risk.
    churn_by_path = dict(hotspot_files(local_path, top_n=100_000))
    rows = file_touches_with_author(local_path)
    touches_by_path = {}
    for r in rows:
        touches_by_path.setdefault(r["path"], []).append(r)

    candidates = []
    for path in set(churn_by_path) | high_complexity.keys() | oversized_paths:
        churn = churn_by_path.get(path, 0)
        flagged_complex = path in high_complexity or path in oversized_paths
        if churn < 5 or not flagged_complex:
            continue
        ownership = _bus_factor_for_rows(touches_by_path.get(path, []))
        if ownership["contributor_count"] and ownership["contributor_count"] > 2:
            continue
        candidates.append({
            "path": path,
            "times_changed": churn,
            "complexity_score": high_complexity[path][1] if path in high_complexity else None,
            "flagged_oversized": path in oversized_paths,
            "contributor_count": ownership["contributor_count"],
            "top_contributor": ownership["top_contributor"],
        })

    candidates.sort(key=lambda c: -c["times_changed"])

    return {
        "available": True,
        "criteria": "changed 5+ times AND flagged complex/oversized by repo_health_tool AND touched by 2 or fewer distinct contributors",
        "flagged_files": candidates[:_TOP_N],
        "flagged_count_total": len(candidates),
    }


# ---------------------------------------------------------------------------
# Report formatting -- deterministic text block; the LLM narrates THIS only.
# ---------------------------------------------------------------------------

def _format_report(focus: str, data: dict) -> str:
    lines = [f"=== FOCUS: {focus.upper()} ===", ""]

    if not data.get("available"):
        lines.append(f"(unavailable: {data.get('reason', 'unknown')})")
        return "\n".join(lines)

    if focus == "activity":
        w = data["window"]
        lines.append(f"Window: since={w['since'] or '(repo start)'}, until={w['until'] or '(now)'}")
        lines.append(f"Commits in window: {data['commit_count']}")
        lines.append(f"Distinct contributors in window: {data['contributor_count']}")
        if data["commit_count"]:
            lines.append(f"First commit in window: {data['first_commit_in_window']}")
            lines.append(f"Last commit in window: {data['last_commit_in_window']}")
            lines.append("Top contributors: " + ", ".join(f"{a} ({n})" for a, n in data["top_contributors"]))
        if data["weekly_activity"]:
            lines.append("Weekly commit counts, oldest to newest: " + ", ".join(
                f"{wk}: {c}" for wk, c in data["weekly_activity"]
            ))

    elif focus == "last_commit":
        lines.append(f"Commit: {data['hash']}")
        lines.append(f"Date: {data['date']}")
        lines.append(f"Author: {data['author']}")
        lines.append(f"Subject: {data['subject']}")
        lines.append(f"Files changed: {data['files_changed_count']}")
        if data["files_changed"]:
            lines.append("File list:")
            for f in data["files_changed"]:
                lines.append(f"  - {f}")

    elif focus == "file_history":
        lines.append(f"Path: {data['path']}")
        lines.append(f"Total commits touching this path: {data['commit_count']}")
        if data["commit_count"]:
            lines.append(f"First touched: {data['first_touched']}")
            lines.append(f"Last touched: {data['last_touched']}")
            lines.append("Contributors: " + ", ".join(f"{a} ({n})" for a, n in data["contributors"]))
            lines.append("Recent commits touching this path:")
            for c in data["recent_commits"]:
                lines.append(f"  {c['date']}  {c['author']}: {c['subject']}")

    elif focus == "ownership":
        lines.append(f"Scope: {data['scope']}")
        o = data["overall"]
        lines.append(
            f"Overall: {o['touch_count']} file-touches, {o['contributor_count']} distinct contributors, "
            f"top contributor {o['top_contributor']} ({o['bus_factor_pct']}% of touches)"
        )
        if data["per_directory"]:
            lines.append("Per-directory breakdown (most-touched first):")
            for d in data["per_directory"]:
                lines.append(
                    f"  {d['directory']}: {d['touch_count']} touches, {d['contributor_count']} contributors, "
                    f"top {d['top_contributor']} ({d['bus_factor_pct']}%)"
                )

    elif focus == "change_intent":
        lines.append(f"Commits analyzed: {data['commit_count']} ({data.get('sample_note', '')})")
        if data["categories"]:
            lines.append("Category breakdown:")
            for cat, v in data["categories"].items():
                lines.append(f"  {cat}: {v['count']} ({v['pct']}%)")
        else:
            lines.append("No commits found for this scope.")

    elif focus == "compare":
        lines.append(f"Range: {data['ref_a']}..{data['ref_b']}")
        lines.append(f"Commits in range: {data['commit_count']}")
        lines.append(f"Distinct contributors in range: {data['contributor_count']}")
        lines.append("Contributors: " + ", ".join(f"{a} ({n})" for a, n in data["contributors"]))
        if data["commit_subjects_sample"]:
            lines.append("Commit subjects (sample):")
            for s in data["commit_subjects_sample"]:
                lines.append(f"  - {s}")
        if data["diff_stat_tail"]:
            lines.append("Diff stat summary (tail):")
            lines.append(data["diff_stat_tail"])

    elif focus == "risk":
        lines.append(f"Criteria: {data['criteria']}")
        lines.append(f"Total files matching all three criteria: {data['flagged_count_total']}")
        if data["flagged_files"]:
            lines.append("Flagged files (highest churn first):")
            for f in data["flagged_files"]:
                lines.append(
                    f"  {f['path']}: changed {f['times_changed']}x, complexity_score={f['complexity_score']}, "
                    f"oversized={f['flagged_oversized']}, {f['contributor_count']} contributor(s) "
                    f"(top: {f['top_contributor']})"
                )
        else:
            lines.append("No files matched all three risk criteria.")

    return "\n".join(lines)


SYSTEM_PROMPT = (
    "You are reporting on a repository's git HISTORY -- how it evolved, not what "
    "its current code does. You are given PRE-COMPUTED, exact numbers below -- use "
    "ONLY these numbers. Do not invent additional statistics, dates, contributor "
    "names, or file paths not present in the data. Where a section says "
    "'(unavailable: ...)', say so plainly instead of guessing. This tool never "
    "generates or modifies code -- if the user's question also asks for a fix or a "
    "change to be made, answer only the history/analysis part and note that making "
    "the actual change is a separate request.\n\n"
    "Write a focused, specific answer (not a generic template) that:\n"
    "1. Directly answers what was asked, citing at least two concrete numbers from "
    "the data below.\n"
    "2. Notes anything genuinely notable (a concentration of ownership, a stale "
    "area, a spike or drop in activity, a risk pattern) ONLY if the data actually "
    "shows it -- don't manufacture a concern to fill space.\n"
    "3. Stays grounded -- every claim must trace back to a number in the data."
)


def _sources_from_focus(focus: str, path: Optional[str], data: dict) -> list:
    if not data.get("available"):
        return []

    if focus == "last_commit":
        return [
            {"file": f, "chunk_name": "", "chunk_type": "last_commit_file", "lines": ""}
            for f in data.get("files_changed", [])
        ]
    if focus == "file_history" and data.get("path"):
        return [{"file": data["path"], "chunk_name": "", "chunk_type": "file_history", "lines": ""}]
    if focus == "ownership":
        return [
            {"file": d["directory"], "chunk_name": "", "chunk_type": "ownership", "lines": f"{d['touch_count']} touches"}
            for d in data.get("per_directory", [])
        ]
    if focus == "risk":
        return [
            {"file": f["path"], "chunk_name": "", "chunk_type": "risk_flag", "lines": f"changed {f['times_changed']}x"}
            for f in data.get("flagged_files", [])
        ]
    return []


def act_repo_evolution(state: dict) -> dict:
    if state.get("local_path") is None:
        return no_repo(state, TOOL_NAME)

    local_path: Path = state["local_path"]
    query = state["tool_args"].get("query", state["message"])

    if not is_git_repo(local_path):
        state["reply"] = "This repository has no git history available (not a git clone), so I can't analyze its evolution."
        state["sources"] = []
        state["tool_used"] = TOOL_NAME
        return state

    scope = _extract_scope(query)
    focus = scope.get("focus") or "activity"
    path = scope.get("path")
    since, until = scope.get("since"), scope.get("until")

    if focus == "activity":
        data = _compute_activity(local_path, since, until)
    elif focus == "last_commit":
        data = _compute_last_commit(local_path)
    elif focus == "file_history":
        data = _compute_file_history(local_path, path, since, until)
    elif focus == "ownership":
        data = _compute_ownership(local_path, path, since, until)
    elif focus == "change_intent":
        data = _compute_change_intent(local_path, path, since, until)
    elif focus == "compare":
        data = _compute_compare(local_path, scope.get("ref_a"), scope.get("ref_b"))
    elif focus == "risk":
        data = _compute_risk(state["active_repository"], local_path)
    else:
        focus = "activity"
        data = _compute_activity(local_path, since, until)

    report_text = _format_report(focus, data)
    state["context_used"] = report_text

    try:
        reply = call_llm(
            SYSTEM_PROMPT,
            report_text,
            persona=state.get("persona"),
            conversation_context=state.get("conversation_context"),
        )
    except Exception as exc:
        return llm_error(state, TOOL_NAME, exc)

    state["reply"] = reply
    state["sources"] = _sources_from_focus(focus, path, data)
    state["tool_used"] = TOOL_NAME

    return state
