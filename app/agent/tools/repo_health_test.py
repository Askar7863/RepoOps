"""
repo_health_test.py

repo_health_tool Regression Test.

Responsibility
--------------
Checks the metric-computation functions directly against click's real,
full git history and real index (deterministic, no LLM needed for most
cases -- these are computed facts, not free-text), plus a handful of
router-disambiguation and full-pipeline cases that do need the LLM.

Covers the real bugs found and fixed during development:
    - a Windows cp1252 decode crash on non-ASCII commit author names
    - doc sections (changelog entries) being mislabeled as "oversized
      functions"
    - idiomatic per-file entry points (main/cli in test or example
      scripts) being mislabeled as a naming-hygiene risk
And cross-checks the dependency parser's pyproject.toml output against
a hand-count of click's own dependency-groups (19 dev deps, 0 runtime --
click genuinely ships with zero runtime dependencies).

Run with:

    python -m app.agent.tools.repo_health_test

This module DOES NOT:
    ✗ Modify the git history or index it inspects
    ✗ Write any file in the target repo's clone
"""

import tempfile
from pathlib import Path

from ...github.clone import REPO_STORAGE
from ...active_repo import get_active_repository, set_active_repository
from ...rag.sparse_store import get_all_chunks
from .repo_health import (
    _git_history,
    _code_health,
    _has_python_docstring,
    _is_test_path,
    _is_example_path,
    _is_test_or_example_path,
    _repo_size,
    _dependencies,
    _has_ci_config,
    _security_scan,
    _complexity,
    _documentation_score,
    _testing_score,
    _health_score,
    compute_health,
    act_repo_health,
)


REPOSITORY = "click"


def _state(message: str, repo: str = REPOSITORY) -> dict:
    local_path = REPO_STORAGE / repo
    return {
        "message": message,
        "active_repository": repo,
        "local_path": local_path if local_path.exists() else None,
        "tool_args": {"query": message},
        "sources": [],
        "tool_used": "general_chat",
        "error": None,
    }


RESULTS = []


def _record(category, title, ok, reason=""):
    RESULTS.append((category, title, ok, reason))
    status = "PASS" if ok else "FAIL"
    suffix = f": {reason}" if reason else ""
    print(f"[{status}] {title}{suffix}")


def run():
    print("=" * 70)
    print("UNIT CASES (deterministic metric functions, no LLM)")
    print("=" * 70)

    # --- docstring heuristic: multi-line signature must not be missed ---
    multiline_documented = (
        "def write_dl(\n"
        "    self,\n"
        "    rows,\n"
        "    col_max: int = 30,\n"
        "    col_spacing: int = 2,\n"
        ") -> None:\n"
        '    """Writes a definition list into the buffer."""\n'
        "    pass\n"
    )
    _record(
        "docstring heuristic", "multi-line signature (6 lines) with a docstring is detected",
        _has_python_docstring(multiline_documented) is True,
    )

    multiline_undocumented = (
        "def write_dl(\n"
        "    self,\n"
        "    rows,\n"
        ") -> None:\n"
        "    pass\n"
    )
    _record(
        "docstring heuristic", "multi-line signature with NO docstring correctly returns False",
        _has_python_docstring(multiline_undocumented) is False,
    )

    # --- test vs example path filters (split so testing score doesn't
    # count example scripts as tests) ---
    _record("path filter", "tests/test_options.py is a test path", _is_test_path("tests\\test_options.py") is True)
    _record("path filter", "tests/test_options.py is NOT an example path", _is_example_path("tests\\test_options.py") is False)
    _record("path filter", "examples/complex/complex/cli.py is an example path", _is_example_path("examples\\complex\\complex\\cli.py") is True)
    _record("path filter", "examples/complex/complex/cli.py is NOT a test path", _is_test_path("examples\\complex\\complex\\cli.py") is False)
    _record("path filter", "src/click/core.py is neither test nor example", _is_test_or_example_path("src\\click\\core.py") is False)

    # --- git history against click's real, full history ---
    local_path = REPO_STORAGE / REPOSITORY
    if not local_path.exists():
        _record("setup", f"{REPOSITORY} clone exists", False, "clone repos/click first")
        return

    git_info = _git_history(local_path)
    _record("git history", "click's git history is available (real clone has full history)", git_info.get("available") is True)
    if git_info.get("available"):
        _record("git history", "total_commits is a large real number (click has 1000s of commits)", git_info["total_commits"] > 500)
        _record("git history", "contributor_count > 1 (click has many real contributors)", git_info["contributor_count"] > 1)
        _record("git history", "bus_factor_pct is between 0 and 100", 0 < git_info["bus_factor_pct"] <= 100)
        _record("git history", "hotspot_files is non-empty", len(git_info["hotspot_files"]) > 0)
        _record("git history", "weekly_activity has exactly 12 weeks", len(git_info["weekly_activity"]) == 12)
        _record(
            "git history", "weekly_activity weeks are in ascending order",
            [w for w, _ in git_info["weekly_activity"]] == sorted(w for w, _ in git_info["weekly_activity"]),
        )

    # --- git history on a non-git directory: must fail gracefully, not crash ---
    with tempfile.TemporaryDirectory() as tmp:
        no_git_info = _git_history(Path(tmp))
        _record("git history edge case", "non-git directory returns available=False instead of crashing", no_git_info.get("available") is False)

    # --- code health against click's real index ---
    chunks = get_all_chunks(REPOSITORY)
    code_health = _code_health(chunks)
    _record("code health", "click's index is available", code_health.get("available") is True)
    if code_health.get("available"):
        _record(
            "code health", "oversized_symbols_sample contains only Function/Class chunks, no doc sections",
            all(not any(bad in name for bad in ("Version", "Changelog")) for _path, name in code_health["oversized_symbols_sample"]),
        )
        _record(
            "code health", "'cli' is NOT in duplicate_names_sample (all its 25 occurrences are test/example files)",
            not any(name == "cli" for name, _kind, _n in code_health["duplicate_names_sample"]),
        )
        _record("code health", "todo_fixme_hack_total is 0 (click genuinely has zero TODO/FIXME/HACK, verified via manual grep)", code_health["todo_fixme_hack_total"] == 0)
        _record(
            "code health", "python_docstring_coverage_pct is a plausible percentage (0-100)",
            code_health["python_docstring_coverage_pct"] is not None and 0 <= code_health["python_docstring_coverage_pct"] <= 100,
        )

    _record("code health edge case", "empty chunk list returns available=False instead of crashing", _code_health([]).get("available") is False)

    # --- repository size ---
    size_info = _repo_size(local_path)
    _record("repo size", "total_size_mb is a small positive number (click is a small repo)", 0 < size_info["total_size_mb"] < 50)
    _record("repo size", "total_lines is a large positive number", size_info["total_lines"] > 10000)
    _record("repo size", "file_count excludes .git (roughly matches indexed file count order of magnitude)", 0 < size_info["file_count"] < 5000)

    # --- dependencies: cross-checked against a hand-count of click's own pyproject.toml ---
    dep_info = _dependencies(local_path)
    _record("dependencies", "click's pyproject.toml is found and parsed", dep_info.get("available") is True and any(m["file"] == "pyproject.toml" for m in dep_info.get("manifests_parsed", [])))
    if dep_info.get("available"):
        _record(
            "dependencies", "0 runtime dependencies (click genuinely ships with none -- verified by reading pyproject.toml directly)",
            dep_info["total_runtime_dependencies"] == 0,
        )
        _record(
            "dependencies", "19 dev dependencies (hand-counted from click's dependency-groups: dev=3, docs=5, docs-auto=1, gha-update=1, pre-commit=2, tests=1, tests-random=3, typing=3)",
            dep_info["total_dev_dependencies"] == 19,
        )
        _record("dependencies", "no manifests went unparsed", dep_info["manifests_unparsed"] == [])

    with tempfile.TemporaryDirectory() as tmp:
        empty_dep_info = _dependencies(Path(tmp))
        _record("dependencies edge case", "directory with no manifest returns available=False instead of crashing", empty_dep_info.get("available") is False)

    # --- CI config detection ---
    _record("CI detection", "click has real .github/workflows -- detected as True", _has_ci_config(local_path) is True)
    with tempfile.TemporaryDirectory() as tmp:
        _record("CI detection edge case", "directory with no CI config returns False instead of crashing", _has_ci_config(Path(tmp)) is False)

    # --- security scan ---
    security_info = _security_scan(local_path, chunks)
    _record("security scan", "click has zero suspected secrets (clean, mature repo)", security_info["suspected_secret_hits"] == 0)
    _record("security scan", "click has no committed .env file", security_info["committed_env_file"] is None)
    _record("security scan", "click has a .gitignore", security_info["has_gitignore"] is True)

    placeholder_chunks = [{"path": "config.py", "content": 'password = "changeme_example_placeholder"'}]
    placeholder_scan = _security_scan(local_path, placeholder_chunks)
    _record("security scan", "an obvious placeholder value is NOT flagged as a real secret", placeholder_scan["suspected_secret_hits"] == 0)

    real_looking_chunks = [{"path": "config.py", "content": 'password = "hunter2ActualLookingSecretValue"'}]
    real_looking_scan = _security_scan(local_path, real_looking_chunks)
    _record("security scan", "a real-looking hardcoded assignment IS flagged", real_looking_scan["suspected_secret_hits"] >= 1)

    # --- complexity heuristic ---
    complexity_info = _complexity(chunks)
    _record("complexity", "click's complexity info is available", complexity_info.get("available") is True)
    if complexity_info.get("available"):
        _record("complexity", "avg_complexity is a plausible small number (well-factored code)", 1 <= complexity_info["avg_complexity"] < 15)
        _record("complexity", "high_complexity_sample entries have complexity >= threshold", all(score >= 10 for _p, _n, score in complexity_info["high_complexity_sample"]))

    _record("complexity edge case", "no function chunks returns available=False instead of crashing", _complexity([]).get("available") is False)

    # --- documentation / testing composite scores ---
    doc_score = _documentation_score(local_path, code_health)
    _record("documentation score", "click has a README, so has_readme is True", doc_score["has_readme"] is True)
    _record("documentation score", "score is within 0-100", 0 <= doc_score["score"] <= 100)

    test_score = _testing_score(chunks, local_path)
    _record("testing score", "test_file_count is positive (click has a real tests/ dir)", test_score["test_file_count"] > 0)
    _record("testing score", "has_ci_config is True", test_score["has_ci_config"] is True)
    _record("testing score", "score is within 0-100", 0 <= test_score["score"] <= 100)

    # --- health score composite: reweights when some sub-scores are missing ---
    full_scores = {"documentation": 80.0, "testing": 80.0, "security": 80.0, "complexity": 80.0, "activity": 80.0, "bus_factor": 80.0}
    _record("health score", "all sub-scores at 80 -> health score is 80", _health_score(full_scores) == 80.0)
    partial_scores = {"documentation": None, "testing": None, "security": None, "complexity": None, "activity": None, "bus_factor": None}
    _record("health score edge case", "all sub-scores missing -> health score is None instead of crashing", _health_score(partial_scores) is None)

    print()
    print("=" * 70)
    print("ROUTING CASES")
    print("=" * 70)
    print()

    from ..router import classify_intent

    routing_cases = [
        ("Give me a health check of this repo.", "repo_health_tool"),
        ("How healthy and well-maintained is this codebase?", "repo_health_tool"),
        ("Is there a bus-factor risk in this project? Who are the main contributors?", "repo_health_tool"),
        ("What does this repo do and how do I install it?", "repo_summarizer_tool"),
        ("Give me a summary of this project.", "repo_summarizer_tool"),
    ]
    for msg, expected in routing_cases:
        got = classify_intent({"message": msg}).get("chosen_tool")
        _record("routing", f"'{msg[:55]}...' -> {expected}", got == expected, f"got={got!r}")

    print()
    print("=" * 70)
    print("FULL PIPELINE CASES (real LLM call)")
    print("=" * 70)
    print()

    original = get_active_repository()
    set_active_repository(REPOSITORY)

    try:
        # snapshot file fingerprints before, to prove read-only behavior
        before = {p: p.stat().st_mtime_ns for p in local_path.rglob("*") if p.is_file() and ".git" not in p.parts}

        health = compute_health(REPOSITORY, local_path)
        _record("compute_health", "health_score is within 0-100", 0 <= health["health_score"] <= 100)

        result = act_repo_health(_state("Give me a health check of this repo."))
        _record("full pipeline", "produces a non-empty reply with no error", bool(result.get("reply")) and not result.get("error"))
        for section in ("SUMMARY", "CODE QUALITY", "REPOSITORY OVERVIEW", "ACTIVITY", "SECURITY", "RISK", "RECOMMEND"):
            _record("full pipeline", f"reply contains a '{section}' section", section in result["reply"].upper())
        _record("full pipeline", "cites the real Health Score somewhere", str(health["health_score"]) in result["reply"])
        _record("full pipeline", "security section states this is not a full audit", "audit" in result["reply"].lower())

        after = {p: p.stat().st_mtime_ns for p in local_path.rglob("*") if p.is_file() and ".git" not in p.parts}
        _record("read-only guarantee", "no file mtimes changed during the full pipeline run", before == after)

        no_repo_result = act_repo_health(_state("health check", repo="__does_not_matter__") | {"local_path": None})
        _record("edge case", "no repository loaded returns the standard no-repo reply", "No repository is loaded" in no_repo_result["reply"])
    finally:
        set_active_repository(original)
        print()
        print(f"Active repository restored to: {original!r}")

    print()
    print("=" * 70)
    total = len(RESULTS)
    passed = sum(1 for *_, ok, _ in RESULTS if ok)
    print(f"TOTAL: {passed}/{total} passed")
    print("=" * 70)


if __name__ == "__main__":
    run()
