"""
git_history.py

Shared Git History Primitives.

Responsibility
--------------
The one place that shells out to `git log` and turns its output into
structured Python data. repo_health_tool (a point-in-time snapshot: total
commits, contributor count, bus factor, hotspot files, a fixed 12-week
activity series) and repo_evolution_tool (query-driven trends, ownership
drilldowns, and comparisons over an arbitrary window) both build on these
same primitives -- neither re-implements git-log parsing independently, so
there is exactly one place that can get the Windows-encoding handling, or
the commit-record shape, wrong.

`basic_history()` is kept byte-identical in output shape to repo_health.py's
original, private `_git_history()` -- extracting it here must not change
that tool's report.

This module DOES NOT:
    ✗ Call the LLM
    ✗ Know about any specific tool's report format
"""

import subprocess
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional


def is_git_repo(local_path: Path) -> bool:
    return (local_path / ".git").exists()


def run_git(local_path: Path, args: list) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=str(local_path),
            capture_output=True,
            # Explicit UTF-8 rather than text=True's locale default: commit
            # author names/messages routinely contain non-ASCII bytes that
            # aren't valid under Windows' cp1252 default, which crashes the
            # decode entirely. errors="replace" so one odd byte in one commit
            # message can't take down the whole history read.
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None

    return result.stdout


def log_commits(
    local_path: Path,
    since: Optional[str] = None,
    until: Optional[str] = None,
    paths: Optional[List[str]] = None,
    follow: bool = False,
) -> Optional[List[dict]]:
    """
    [{"hash", "date" (ISO), "author", "author_email", "subject"}, ...],
    newest first. None if this isn't a git repo or `git log` fails
    outright; an empty list means it IS a repo with zero matching commits
    (a real, different answer from "can't read history at all").

    `since`/`until` accept anything `git log --since/--until` understands
    (an absolute date, or a relative form like "90 days ago"). `paths`
    scopes the log to those files/directories. `follow` adds --follow to
    track a single file across renames -- only meaningful with exactly
    one path.
    """

    if not is_git_repo(local_path):
        return None

    args = ["log", "--format=%H|%aI|%an|%ae|%s", "--date=iso-strict"]
    if since:
        args.append(f"--since={since}")
    if until:
        args.append(f"--until={until}")
    if follow and paths and len(paths) == 1:
        args.append("--follow")
    if paths:
        args.append("--")
        args.extend(paths)

    output = run_git(local_path, args)
    if output is None:
        return None

    commits = []
    for line in output.splitlines():
        parts = line.split("|", 4)
        if len(parts) == 5:
            commits.append({
                "hash": parts[0], "date": parts[1],
                "author": parts[2], "author_email": parts[3], "subject": parts[4],
            })
    return commits


def weekly_activity(commits: list, weeks: int = 12) -> list:
    """
    [(week_start_iso_date, commit_count), ...] oldest to newest, for a
    simple recent-activity histogram. Buckets by each commit's own
    recorded timezone rather than normalizing to UTC; at week granularity
    the odd commit landing in an adjacent bucket near a boundary doesn't
    matter.
    """

    if not commits:
        return []

    buckets = Counter()
    for c in commits:
        try:
            dt = datetime.fromisoformat(c["date"])
        except ValueError:
            continue
        monday = (dt - timedelta(days=dt.weekday())).date()
        buckets[monday] += 1

    if not buckets:
        return []

    latest_monday = max(buckets)
    return [
        ((latest_monday - timedelta(weeks=i)).isoformat(), buckets.get(latest_monday - timedelta(weeks=i), 0))
        for i in range(weeks - 1, -1, -1)
    ]


def hotspot_files(
    local_path: Path, since: Optional[str] = None, until: Optional[str] = None, top_n: int = 10,
) -> list:
    """[(path, times_changed), ...], most-changed first, over an optional window."""

    args = ["log", "--format=format:", "--name-only"]
    if since:
        args.append(f"--since={since}")
    if until:
        args.append(f"--until={until}")

    output = run_git(local_path, args)
    counts = Counter(line for line in (output.splitlines() if output else []) if line.strip())
    return counts.most_common(top_n)


def commits_between(
    local_path: Path, ref_a: str, ref_b: str, paths: Optional[List[str]] = None,
) -> Optional[List[dict]]:
    """
    Commits reachable from ref_b but not ref_a (`git log ref_a..ref_b`) --
    the standard "what's new since X" range, for comparing two tags,
    branches, or commits. None if this isn't a git repo, either ref
    fails to resolve, or the command otherwise fails.
    """

    if not is_git_repo(local_path):
        return None

    args = ["log", f"{ref_a}..{ref_b}", "--format=%H|%aI|%an|%ae|%s", "--date=iso-strict"]
    if paths:
        args.append("--")
        args.extend(paths)

    output = run_git(local_path, args)
    if output is None:
        return None

    commits = []
    for line in output.splitlines():
        parts = line.split("|", 4)
        if len(parts) == 5:
            commits.append({
                "hash": parts[0], "date": parts[1],
                "author": parts[2], "author_email": parts[3], "subject": parts[4],
            })
    return commits


def diff_stat_between(local_path: Path, ref_a: str, ref_b: str) -> Optional[str]:
    """Raw `git diff --stat ref_a..ref_b` text -- files changed, insertions/deletions."""

    return run_git(local_path, ["diff", "--stat", f"{ref_a}..{ref_b}"])


def file_touches_with_author(
    local_path: Path,
    since: Optional[str] = None,
    until: Optional[str] = None,
    paths: Optional[List[str]] = None,
) -> List[dict]:
    """
    [{"path", "author", "author_email"}, ...], one row per (commit, file)
    pair touched -- the raw join needed for per-file / per-directory
    ownership and bus-factor drilldowns. `git log --name-only` alone
    lists files per commit but doesn't hand you a flat (file, author)
    pairing, so this parses that per-commit grouping explicitly rather
    than assuming every changed file in the repo shares one author.
    """

    args = ["log", "--format=COMMIT|%an|%ae", "--name-only"]
    if since:
        args.append(f"--since={since}")
    if until:
        args.append(f"--until={until}")
    if paths:
        args.append("--")
        args.extend(paths)

    output = run_git(local_path, args)
    if output is None:
        return []

    rows = []
    author = email = None
    for line in output.splitlines():
        if line.startswith("COMMIT|"):
            _, author, email = line.split("|", 2)
        elif line.strip():
            rows.append({"path": line.strip(), "author": author, "author_email": email})
    return rows


def latest_commit_detail(local_path: Path) -> Optional[dict]:
    """
    The single most recent commit's hash, date, author, subject, and the
    list of files it touched. A different question from activity/trend
    data over a window -- "when/what was the last commit" needs the
    detail of ONE commit, not an aggregate, and an aggregate's own
    last-commit-date field is not the same as "what files did it touch."
    None if this isn't a git repo or it has no commits.
    """

    if not is_git_repo(local_path):
        return None

    header = run_git(local_path, ["log", "-1", "--format=%H|%aI|%an|%ae|%s"])
    if not header or not header.strip():
        return None

    parts = header.strip().split("|", 4)
    if len(parts) != 5:
        return None

    files_output = run_git(local_path, ["show", "--name-only", "--format=", "HEAD"])
    files_changed = [f for f in (files_output.splitlines() if files_output else []) if f.strip()]

    return {
        "hash": parts[0], "date": parts[1], "author": parts[2], "author_email": parts[3],
        "subject": parts[4], "files_changed": files_changed,
    }


def normalize_identity(author: str, email: str) -> str:
    """
    Best-effort identity key for contributor counting / bus-factor / file
    ownership -- the same person committing under two email addresses (a
    work + personal account, or a name change) would otherwise silently
    inflate contributor counts and understate real ownership
    concentration. No .mailmap file is assumed to exist; this only merges
    entries by exact (case-insensitive) email, or falls back to exact
    display name when no email is available. Deliberately conservative --
    treating two different people as one is worse for a risk signal than
    missing a real duplicate, so this never does fuzzy/partial name
    matching.
    """

    return (email or author or "unknown").strip().lower()


def basic_history(local_path: Path, top_n: int = 10) -> dict:
    """
    The point-in-time git-activity snapshot repo_health_tool has always
    reported: total commits, first/last commit date, commits in the last
    90 days, contributor count, top contributors, bus factor, hotspot
    files, a 12-week weekly activity series. Kept byte-identical to the
    original private _git_history() this was extracted from.
    """

    commits = log_commits(local_path)
    if commits is None:
        return {"available": False, "reason": "not a git repository"}
    if not commits:
        return {"available": False, "reason": "repository has no commits"}

    total_commits = len(commits)
    last_commit_date = commits[0]["date"]
    first_commit_date = commits[-1]["date"]

    commits_last_90d = log_commits(local_path, since="90 days ago")
    commits_last_90d_count = len(commits_last_90d) if commits_last_90d else 0

    author_counts = Counter(c["author"] for c in commits)
    top_author, top_author_commits = author_counts.most_common(1)[0]
    bus_factor_pct = round(100 * top_author_commits / total_commits, 1)

    return {
        "available": True,
        "total_commits": total_commits,
        "first_commit_date": first_commit_date,
        "last_commit_date": last_commit_date,
        "commits_last_90_days": commits_last_90d_count,
        "contributor_count": len(author_counts),
        "top_contributors": author_counts.most_common(top_n),
        "bus_factor_top_author": top_author,
        "bus_factor_pct": bus_factor_pct,
        "hotspot_files": hotspot_files(local_path, top_n=top_n),
        "weekly_activity": weekly_activity(commits),
    }
