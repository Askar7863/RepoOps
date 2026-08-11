"""
repo_limits.py

Pre-Clone Repository Size Guard.

Responsibility
--------------
Reject a repository that's too large or has too many files BEFORE
cloning it, using GitHub's own REST API to check its size and file
count first -- so an oversized repo fails instantly with zero wasted
clone bandwidth, disk, or indexing time, instead of only being
discovered after a multi-minute clone (or one that blows past disk on
a storage-constrained deployment).

Why two separate limits, and why these numbers
-------------------------------------------------
Size alone misses a pathological many-tiny-files repo: modest total
bytes, but every file still gets scanned, filtered, tree-sitter-parsed,
and embedded, so indexing time and per-object memory overhead track
file COUNT more than raw bytes. File count alone misses a repo with a
huge .git history or large committed binaries, which GitHub's own
packed "size" figure catches directly. Neither one alone is a reliable
proxy for the other.

The thresholds were checked against real GitHub data, not guessed:
widely-used, actively-maintained libraries (pallets/click, pallets/
flask, psf/requests, expressjs/express) sit at 5-14 MB and 130-236
files. MAX_REPO_SIZE_KB / MAX_REPO_FILE_COUNT below are ~20-50x that --
generous enough that no realistically-scoped repo trips them, while
still rejecting genuine outliers (a repo the size of microsoft/vscode
at ~1.3GB, or a kernel/monorepo-scale project at multiple GB) that were
never going to give a coherent single-repo chat/summary experience
anyway, limit or not.

This module DOES NOT:
    ✗ Clone, scan, or index anything (see clone.py)
    ✗ Enforce anything AFTER a clone already happened -- this is a
      pre-flight check only, meant to run before clone_repository()
    ✗ Treat "GitHub's API is unreachable/rate-limited" as a rejection --
      see check_repo_limits's own docstring for why this fails open
"""

import json
import re
import urllib.error
import urllib.request

from ..utils.logger import log_warning


MAX_REPO_SIZE_KB = 250 * 1024  # ~250 MB, GitHub's own packed "size" field
MAX_REPO_FILE_COUNT = 5000

GITHUB_API = "https://api.github.com"
_REQUEST_TIMEOUT_SECONDS = 10


def _parse_owner_repo(repo_url: str):
    """
    Pull (owner, repo) out of a github.com HTTPS URL, or None if it
    doesn't look like one. None is treated by check_repo_limits as
    "couldn't check" (skip), not as a rejection -- validating what
    counts as a real GitHub URL at all is main.py's job, already done
    before this runs; this function only needs to handle URLs that
    have already passed that gate.
    """

    match = re.search(r"github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", repo_url.strip())
    if not match:
        return None

    return match.group(1), match.group(2)


def _get_json(url: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "RepoOps"},
    )
    with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
        return json.loads(response.read())


def check_repo_limits(repo_url: str) -> None:
    """
    Raise an Exception with a clear, specific message if the repository
    exceeds MAX_REPO_SIZE_KB or MAX_REPO_FILE_COUNT -- checked via
    GitHub's REST API, before anything is cloned.

    Fails OPEN on anything other than a genuine over-limit verdict: a
    network error, a rate limit (GitHub's unauthenticated API allows
    only 60 requests/hour per IP, and this check spends up to 2 of
    them), a URL that doesn't parse, or an unexpected response shape
    all just log a warning and let the load proceed to the real clone
    step -- whose own error handling (see clone.py's
    _explain_clone_failure) already correctly covers "doesn't exist" /
    "private" / "network unreachable". This check's only job is to
    catch a genuinely oversized PUBLIC repo before wasting a clone on
    it; it must never be the reason an otherwise-normal load fails just
    because GitHub's API had a bad moment.
    """

    parsed = _parse_owner_repo(repo_url)
    if parsed is None:
        return

    owner, repo = parsed

    try:
        repo_info = _get_json(f"{GITHUB_API}/repos/{owner}/{repo}")
    except Exception as exc:
        log_warning(f"repo_limits: could not fetch repo metadata ({exc}); skipping pre-clone size check")
        return

    size_kb = repo_info.get("size")
    if isinstance(size_kb, int) and size_kb > MAX_REPO_SIZE_KB:
        raise Exception(
            f"'{repo_url}' is about {size_kb / 1024:.0f} MB, which is over this "
            f"deployment's {MAX_REPO_SIZE_KB / 1024:.0f} MB repository size limit. "
            "Try a smaller repository."
        )

    default_branch = repo_info.get("default_branch")
    if not default_branch:
        return

    try:
        tree_info = _get_json(
            f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1"
        )
    except Exception as exc:
        log_warning(f"repo_limits: could not fetch file tree ({exc}); skipping pre-clone file-count check")
        return

    # GitHub only truncates this listing well past our own ceiling, so
    # a truncated response itself already means "far over the limit" --
    # trusting the partial count it did return would understate it.
    if tree_info.get("truncated"):
        raise Exception(
            f"'{repo_url}' has too many files to even list in full (GitHub truncated "
            f"the listing), which means it's well over this deployment's "
            f"{MAX_REPO_FILE_COUNT:,}-file limit. Try a smaller repository."
        )

    file_count = sum(1 for entry in tree_info.get("tree", []) if entry.get("type") == "blob")
    if file_count > MAX_REPO_FILE_COUNT:
        raise Exception(
            f"'{repo_url}' has {file_count:,} files, which is over this deployment's "
            f"{MAX_REPO_FILE_COUNT:,}-file limit. Try a smaller repository."
        )
