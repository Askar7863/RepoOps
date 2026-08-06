"""
repo_health.py

Repository Health Checker.

Responsibility
--------------
Give a grounded, detailed health check of the loaded repository, using
only data actually computed from the repo -- never an LLM's free
impression of "how good" it seems. Covers:

  1. CODE HEALTH -- from the already-built index: language/file
     distribution, oversized functions, TODO/FIXME/HACK density,
     duplicate symbol names across files, Python docstring coverage.
  2. REPO ACTIVITY & MAINTENANCE -- from `git log` on the local clone:
     commit frequency, last-commit recency, contributor count and
     bus-factor, hotspot files, a 12-week commit-activity series.
  3. REPOSITORY SIZE -- on-disk bytes and line count (excluding .git).
  4. DEPENDENCIES -- parsed from whatever manifests are actually
     present (requirements.txt, package.json, pyproject.toml, Pipfile,
     Cargo.toml; go.mod is detected but not parsed -- see _dependencies).
  5. SECURITY -- a STATIC pattern scan only (hardcoded-secret-shaped
     strings, a committed .env file, .gitignore presence). This is NOT
     a vulnerability/CVE audit and makes no network calls; it's labeled
     as a heuristic everywhere it's surfaced, same discipline as the
     docstring-coverage heuristic below.
  6. CODE COMPLEXITY -- an approximate per-function complexity via
     control-flow keyword counting (not a real AST-based cyclomatic
     complexity -- see _complexity for why, and its real limitations).

Six composite 0-100 scores (documentation, testing, security,
complexity, activity, bus-factor) combine into one overall Health
Score. Every sub-score is computed by an explicit, disclosed formula in
this file -- never asked of the LLM to "guess a vibe" -- so the number
is reproducible and the LLM's job stays narrative synthesis, not
invention. This is deliberately the same discipline as
repo_summarizer_tool's structural overview -- a "154 total files" vs.
"Python: 2259 files" contradiction was a real, previously-fixed bug
from letting an LLM eyeball raw counts instead of handing it
pre-aggregated ones.

This module DOES NOT:
    ✗ Write or modify any files
    ✗ Make any network calls (everything here is the local clone + git log)
    ✗ Claim a real vulnerability/CVE audit, or a real AST-based complexity
      metric -- both are disclosed, static heuristics only
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from ...github.git_history import basic_history
from ...rag.sparse_store import get_all_chunks
from ._shared import call_llm, no_repo, llm_error

try:
    import tomllib as _toml
except ModuleNotFoundError:
    try:
        import tomli as _toml
    except ModuleNotFoundError:
        _toml = None


TOOL_NAME = "repo_health_tool"

# How many oversized functions / hotspot files / contributors / high-
# complexity functions to name individually in the report -- enough to
# be concrete, not a full dump.
TOP_N = 10

_TODO_RE = re.compile(r"\b(TODO|FIXME|HACK)\b")

_TEST_SEGMENTS = {"test", "tests", "spec", "specs", "__tests__", "fixture", "fixtures"}
_EXAMPLE_SEGMENTS = {"example", "examples"}


def _is_test_path(path: str) -> bool:
    segments = re.split(r"[\\/]", path.lower())
    return any(seg in _TEST_SEGMENTS for seg in segments)


def _is_example_path(path: str) -> bool:
    segments = re.split(r"[\\/]", path.lower())
    return any(seg in _EXAMPLE_SEGMENTS for seg in segments)


def _is_test_or_example_path(path: str) -> bool:
    return _is_test_path(path) or _is_example_path(path)


def _git_history(local_path: Path) -> dict:
    """
    Commit activity, contributors, hotspot files, and a recent-activity
    series -- entirely from `git log` against the local clone, via the
    shared git_history module (also used by repo_evolution_tool, so
    there's exactly one place that parses `git log`). Returns
    {"available": False, "reason": ...} if this isn't a git repo or git
    isn't on PATH, so callers can report that honestly instead of
    crashing or guessing.
    """

    return basic_history(local_path, top_n=TOP_N)


def _has_python_docstring(content: str) -> bool:
    """
    True if the first non-blank line after the def/class signature opens
    a triple-quoted string. Tracks bracket depth line-by-line rather than
    scanning a fixed window -- a signature with several typed parameters
    routinely spans well past 5 lines (e.g. one parameter per line), and a
    fixed window would never reach the real closing ':', undercounting
    coverage for exactly the heavily-typed, multi-line-signature style
    modern Python favors.
    """

    lines = content.splitlines()
    depth = 0

    for i, line in enumerate(lines):
        depth += line.count("(") + line.count("[") + line.count("{")
        depth -= line.count(")") + line.count("]") + line.count("}")

        if depth <= 0 and line.rstrip().endswith(":"):
            for rest in lines[i + 1 :]:
                if not rest.strip():
                    continue
                return rest.strip().startswith(('"""', "'''"))
            return False

    return False


def _code_health(chunks: list) -> dict:
    if not chunks:
        return {"available": False, "reason": "repository has no indexed chunks"}

    file_languages = {c["path"]: c["language"] for c in chunks}
    language_counts = Counter(file_languages.values())

    todo_total = 0
    todo_files = set()
    for c in chunks:
        matches = len(_TODO_RE.findall(c["content"]))
        if matches:
            todo_total += matches
            todo_files.add(c["path"])

    # Oversized functions/classes: chunker splits any chunk whose token count
    # would truncate the embedder, naming continuations "name#2", "name#3" --
    # a chunk with a "#" suffix is a real, direct size signal, not a
    # heuristic guess. Restricted to actual code symbols (Function/Class) --
    # the same split-on-oversize marker also applies to long doc sections
    # (e.g. a changelog's per-version entries), which aren't a code-quality
    # signal and would be mislabeled as "oversized functions" otherwise.
    oversized_symbols = set()
    for c in chunks:
        if c["chunk_type"] in ("Function", "Class") and "#" in c["chunk_name"]:
            oversized_symbols.add((c["path"], c["chunk_name"].split("#")[0]))
    oversized_list = sorted(oversized_symbols)[:TOP_N]

    # Duplicate symbol names across DIFFERENT files -- restricted to
    # non-dunder Function/Class names appearing in 3+ distinct files, to
    # avoid flagging ordinary, expected repeats (every class's own
    # __init__/__repr__) as if they were a real naming collision. Also
    # excludes test/example paths -- confirmed against click that a name
    # like "cli" showing up in 25 files is entirely test/example scripts
    # each defining their own standalone entry point (idiomatic, zero
    # actual collision risk), not a real naming-hygiene problem; without
    # this filter the report flagged it as a maintenance risk, which was
    # simply wrong.
    name_to_files = {}
    for c in chunks:
        base_name = c["chunk_name"].split("#")[0]
        if c["chunk_type"] not in ("Function", "Class") or base_name.startswith("__"):
            continue
        if _is_test_or_example_path(c["path"]):
            continue
        name_to_files.setdefault((base_name, c["chunk_type"]), set()).add(c["path"])
    duplicates = sorted(
        ((name, kind, len(paths)) for (name, kind), paths in name_to_files.items() if len(paths) >= 3),
        key=lambda x: -x[2],
    )[:TOP_N]

    python_chunks = [c for c in chunks if c["language"] == "python" and c["chunk_type"] in ("Function", "Class")]
    docstring_pct = None
    if python_chunks:
        documented = sum(1 for c in python_chunks if _has_python_docstring(c["content"]))
        docstring_pct = round(100 * documented / len(python_chunks), 1)

    return {
        "available": True,
        "total_files": len(file_languages),
        "total_symbols": len(chunks),
        "language_counts": language_counts.most_common(),
        "todo_fixme_hack_total": todo_total,
        "todo_fixme_hack_files": len(todo_files),
        "oversized_symbol_count": len(oversized_symbols),
        "oversized_symbols_sample": oversized_list,
        "duplicate_names_sample": duplicates,
        "python_docstring_coverage_pct": docstring_pct,
        "python_symbol_count": len(python_chunks),
    }


# ---------------------------------------------------------------------------
# Repository size
# ---------------------------------------------------------------------------

# Skip line-counting files above this size (generated/minified/data blobs) --
# still counted toward total byte size, just not read line-by-line.
_LINE_COUNT_MAX_BYTES = 2_000_000


def _repo_size(local_path: Path) -> dict:
    total_bytes = 0
    total_lines = 0
    file_count = 0

    for p in local_path.rglob("*"):
        if not p.is_file() or ".git" in p.parts:
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue

        total_bytes += size
        file_count += 1

        if size <= _LINE_COUNT_MAX_BYTES:
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    total_lines += sum(1 for _ in f)
            except OSError:
                pass

    return {
        "total_size_bytes": total_bytes,
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        "total_lines": total_lines,
        "file_count": file_count,
    }


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

_MANIFEST_FILENAMES = {"requirements.txt", "pyproject.toml", "package.json", "Pipfile", "Cargo.toml", "go.mod"}
_SKIP_DIR_NAMES = {"node_modules", ".git", "venv", ".venv", "__pycache__", "dist", "build", ".tox", "vendor"}
_VERSION_OPERATOR_RE = re.compile(r"[=<>~!^]")


def _find_manifests(local_path: Path) -> list:
    """
    Root + immediate subdirectories only (covers the common backend/
    package.json, frontend/package.json split-layout seen in real repos
    in this project) -- deliberately bounded, not a full recursive walk,
    to keep this fast and avoid picking up a vendored/committed
    dependency's own manifest deep in a subtree.
    """

    found = []
    try:
        depth1_dirs = [d for d in local_path.iterdir() if d.is_dir() and d.name not in _SKIP_DIR_NAMES and not d.name.startswith(".")]
    except OSError:
        depth1_dirs = []

    for root in [local_path] + depth1_dirs:
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        for f in entries:
            if f.is_file() and f.name in _MANIFEST_FILENAMES:
                found.append(f)

    # requirements/*.txt -- click's own layout (requirements/dev.txt, ...)
    req_dir = local_path / "requirements"
    if req_dir.is_dir():
        try:
            found.extend(p for p in req_dir.glob("*.txt") if p.is_file())
        except OSError:
            pass

    return found


def _parse_requirements_txt(path: Path) -> Optional[dict]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    deps = [line.split("#", 1)[0].strip() for line in text.splitlines()]
    deps = [d for d in deps if d and not d.startswith("-")]
    pinned = sum(1 for d in deps if _VERSION_OPERATOR_RE.search(d))

    return {"runtime_count": len(deps), "dev_count": 0, "pinned_count": pinned}


def _parse_package_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, json.JSONDecodeError):
        return None

    deps = data.get("dependencies") or {}
    dev_deps = data.get("devDependencies") or {}
    pinned = sum(1 for v in deps.values() if isinstance(v, str) and v[:1].isdigit())

    return {"runtime_count": len(deps), "dev_count": len(dev_deps), "pinned_count": pinned}


def _load_toml(path: Path) -> Optional[dict]:
    if _toml is None:
        return None
    try:
        with open(path, "rb") as f:
            return _toml.load(f)
    except Exception:
        return None


def _parse_pyproject_toml(path: Path) -> Optional[dict]:
    data = _load_toml(path)
    if data is None:
        return None

    # PEP 621 runtime deps, plus legacy Poetry-style (pre PEP 621 adoption).
    runtime = list(data.get("project", {}).get("dependencies") or [])
    poetry_deps = data.get("tool", {}).get("poetry", {}).get("dependencies") or {}
    runtime.extend(k for k in poetry_deps if k.lower() != "python")

    # PEP 735 dependency-groups, plus legacy Poetry dev-dependencies.
    dev = []
    for group_deps in (data.get("dependency-groups") or {}).values():
        if isinstance(group_deps, list):
            dev.extend(str(d) for d in group_deps)
    dev.extend((data.get("tool", {}).get("poetry", {}).get("dev-dependencies") or {}).keys())

    # No solid, universal "pinned" concept across PEP 508 strings, poetry
    # version specs, and dependency-group extras -- left unclaimed rather
    # than guessed.
    return {"runtime_count": len(runtime), "dev_count": len(dev), "pinned_count": None}


def _parse_pipfile(path: Path) -> Optional[dict]:
    data = _load_toml(path)  # Pipfile is TOML-syntax under the hood
    if data is None:
        return None

    runtime = list((data.get("packages") or {}).keys())
    dev = list((data.get("dev-packages") or {}).keys())

    return {"runtime_count": len(runtime), "dev_count": len(dev), "pinned_count": None}


def _parse_cargo_toml(path: Path) -> Optional[dict]:
    data = _load_toml(path)
    if data is None:
        return None

    runtime = list((data.get("dependencies") or {}).keys())
    dev = list((data.get("dev-dependencies") or {}).keys())

    return {"runtime_count": len(runtime), "dev_count": len(dev), "pinned_count": None}


_DEPENDENCY_PARSERS = {
    "requirements.txt": _parse_requirements_txt,
    "pyproject.toml": _parse_pyproject_toml,
    "package.json": _parse_package_json,
    "Pipfile": _parse_pipfile,
    "Cargo.toml": _parse_cargo_toml,
}


def _dependencies(local_path: Path) -> dict:
    """
    Parses whatever dependency manifests are actually present. go.mod is
    deliberately detected but NOT parsed for a count -- go.mod has both a
    single-line `require x v1.2.3` form and a `require (...)` block form,
    and without a real go.mod in this project's test fixtures to verify
    against, reporting a possibly-wrong count would be worse than
    honestly saying "found, not parsed."
    """

    manifests = _find_manifests(local_path)
    if not manifests:
        return {"available": False, "reason": "no recognized dependency manifest found"}

    parsed = []
    unparsed = []
    total_runtime = 0
    total_dev = 0

    for path in manifests:
        rel = str(path.relative_to(local_path))
        if path.name == "go.mod":
            unparsed.append(rel)
            continue
        result = _DEPENDENCY_PARSERS[path.name](path)
        if result is None:
            unparsed.append(rel)
            continue
        parsed.append({"file": rel, **result})
        total_runtime += result["runtime_count"]
        total_dev += result["dev_count"]

    return {
        "available": True,
        "manifests_parsed": parsed,
        "manifests_unparsed": unparsed,
        "total_runtime_dependencies": total_runtime,
        "total_dev_dependencies": total_dev,
    }


# ---------------------------------------------------------------------------
# CI config detection
# ---------------------------------------------------------------------------

_CI_CONFIG_PATHS = [".github/workflows", ".gitlab-ci.yml", ".travis.yml", "azure-pipelines.yml", ".circleci/config.yml", "Jenkinsfile"]


def _has_ci_config(local_path: Path) -> bool:
    for rel in _CI_CONFIG_PATHS:
        candidate = local_path / rel
        if candidate.is_file():
            return True
        if candidate.is_dir():
            try:
                if any(candidate.iterdir()):
                    return True
            except OSError:
                pass
    return False


# ---------------------------------------------------------------------------
# Security (STATIC pattern scan only -- not a vulnerability/CVE audit)
# ---------------------------------------------------------------------------

_SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id shape
    re.compile(r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),  # private key material
    re.compile(r"(?i)\b(password|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"),
]

# Common placeholder/env-lookup shapes near a "secret-looking" assignment --
# excluding these is what keeps this from flagging every `os.environ["API_KEY"]`
# or `password = "changeme"` in the codebase as a real leaked credential.
_PLACEHOLDER_HINTS_RE = re.compile(r"(?i)(your[_-]|example|placeholder|xxxx|changeme|<.*>|\{\{|\$\{|env\.|os\.environ|getenv)")


def _security_scan(local_path: Path, chunks: list) -> dict:
    hits = 0
    hit_files = set()

    for c in chunks:
        content = c["content"]
        for pattern in _SECRET_PATTERNS:
            for m in pattern.finditer(content):
                if not _PLACEHOLDER_HINTS_RE.search(m.group(0)):
                    hits += 1
                    hit_files.add(c["path"])

    env_file = None
    try:
        for p in local_path.rglob(".env"):
            if ".git" not in p.parts:
                env_file = str(p.relative_to(local_path))
                break
    except OSError:
        pass

    return {
        "available": True,
        "suspected_secret_hits": hits,
        "suspected_secret_files": len(hit_files),
        "committed_env_file": env_file,
        "has_gitignore": (local_path / ".gitignore").is_file(),
    }


# ---------------------------------------------------------------------------
# Code complexity (approximate -- see docstring)
# ---------------------------------------------------------------------------

# NOT a real AST-based cyclomatic complexity -- that would need a per-
# language parser for every one of the ~300 languages the chunker
# supports. This counts control-flow-shaped keywords/operators in a
# function's own text as a decision-point proxy (+1 baseline path).
# Known, disclosed imprecision: "and"/"or"/"if" can appear in prose
# inside a docstring or comment within the same chunk content and would
# be counted as if they were real branches. Treat this as a rough
# signal for "which functions are worth a closer look," not an exact count.
_COMPLEXITY_RE = re.compile(r"\b(if|elif|for|foreach|while|case|catch|except|and|or)\b|(&&|\|\|)")

_HIGH_COMPLEXITY_THRESHOLD = 10


def _complexity(chunks: list) -> dict:
    function_chunks = [c for c in chunks if c["chunk_type"] == "Function"]
    if not function_chunks:
        return {"available": False, "reason": "no function chunks indexed"}

    scored = [
        (c["path"], c["chunk_name"].split("#")[0], len(_COMPLEXITY_RE.findall(c["content"])) + 1)
        for c in function_chunks
    ]

    avg_complexity = round(sum(s[2] for s in scored) / len(scored), 2)
    high = sorted((s for s in scored if s[2] >= _HIGH_COMPLEXITY_THRESHOLD), key=lambda s: -s[2])[:TOP_N]

    return {
        "available": True,
        "function_count": len(scored),
        "avg_complexity": avg_complexity,
        "high_complexity_count": sum(1 for s in scored if s[2] >= _HIGH_COMPLEXITY_THRESHOLD),
        "high_complexity_sample": high,
    }


# ---------------------------------------------------------------------------
# Composite scores -- every formula here is explicit and disclosed; none
# of these numbers are ever asked of the LLM.
# ---------------------------------------------------------------------------

_HEALTH_SCORE_WEIGHTS = {
    "documentation": 0.15,
    "testing": 0.20,
    "security": 0.20,
    "complexity": 0.15,
    "activity": 0.15,
    "bus_factor": 0.15,
}


def _documentation_score(local_path: Path, code_health: dict) -> dict:
    has_readme = any((local_path / name).is_file() for name in ("README.md", "README.rst", "README.txt", "README"))
    docstring_pct = code_health.get("python_docstring_coverage_pct") if code_health.get("available") else None

    total_files = code_health.get("total_files", 0) if code_health.get("available") else 0
    doc_files = sum(
        count for lang, count in code_health.get("language_counts", []) if lang in ("markdown", "restructuredtext")
    ) if code_health.get("available") else 0
    doc_ratio = (doc_files / total_files) if total_files else 0.0
    ratio_component = min(1.0, doc_ratio * 5)  # a 1-in-5 doc-to-code-file ratio already maxes this out

    # Disclosed weights: README presence 40, docstring coverage 40, doc-file
    # ratio 20 -- redistributed proportionally onto the other two when
    # docstring coverage isn't computable (no Python indexed at all).
    if docstring_pct is not None:
        score = 40 * has_readme + 0.4 * docstring_pct + 20 * ratio_component
    else:
        score = (200 / 3) * has_readme + (100 / 3) * ratio_component

    return {
        "score": round(min(100.0, score), 1),
        "has_readme": has_readme,
        "docstring_coverage_pct": docstring_pct,
        "doc_file_ratio_pct": round(doc_ratio * 100, 1),
    }


def _testing_score(chunks: list, local_path: Path) -> dict:
    file_paths = {c["path"] for c in chunks}
    test_files = {p for p in file_paths if _is_test_path(p)}
    source_files = file_paths - test_files
    has_ci = _has_ci_config(local_path)

    test_ratio = (len(test_files) / len(source_files)) if source_files else 0.0
    # Disclosed formula: test-to-source ratio worth up to 60 pts (maxing out
    # at a 1:2 test-to-source ratio -- not requiring 1:1), CI configured
    # worth a flat 40 pts.
    ratio_component = min(1.0, test_ratio * 2)
    score = 60 * ratio_component + 40 * has_ci

    return {
        "score": round(score, 1),
        "test_file_count": len(test_files),
        "source_file_count": len(source_files),
        "has_ci_config": has_ci,
    }


def _security_score(security_info: dict) -> Optional[float]:
    if not security_info.get("available"):
        return None

    score = 100.0
    score -= min(60, security_info["suspected_secret_hits"] * 15)
    if security_info["committed_env_file"]:
        score -= 30
    if not security_info["has_gitignore"]:
        score -= 10

    return round(max(0.0, score), 1)


def _complexity_score(complexity_info: dict) -> Optional[float]:
    if not complexity_info.get("available"):
        return None

    # Disclosed formula: average complexity of 1 (no branching at all)
    # scores 100; each point above that costs 10, floored at 0 by ~avg 11.
    avg = complexity_info["avg_complexity"]
    return round(max(0.0, min(100.0, 100 - (avg - 1) * 10)), 1)


def _activity_score(git_info: dict) -> Optional[float]:
    if not git_info.get("available"):
        return None

    # Disclosed formula: 90-day commit volume, capped at 20 commits/90d = 100.
    return round(min(100.0, git_info["commits_last_90_days"] * 5), 1)


def _bus_factor_score(git_info: dict) -> Optional[float]:
    if not git_info.get("available"):
        return None

    # Inverse of ownership concentration -- lower top-author share scores higher.
    return round(max(0.0, 100 - git_info["bus_factor_pct"]), 1)


def _health_score(sub_scores: dict) -> Optional[float]:
    """
    Weighted average of whichever sub-scores are actually available --
    reweighted proportionally among just those, so a repo missing one
    category of data (e.g. no Python, so no complexity score) isn't
    penalized for the absence itself.
    """

    available = {k: v for k, v in sub_scores.items() if v is not None}
    if not available:
        return None

    total_weight = sum(_HEALTH_SCORE_WEIGHTS[k] for k in available)
    weighted = sum(available[k] * _HEALTH_SCORE_WEIGHTS[k] for k in available)

    return round(weighted / total_weight, 1)


def _format_report(repo_name: str, health: dict) -> str:
    code_health = health["code_health"]
    git_info = health["git_info"]
    size_info = health["size_info"]
    dependency_info = health["dependency_info"]
    security_info = health["security_info"]
    complexity_info = health["complexity_info"]
    doc_score = health["documentation_score"]
    test_score = health["testing_score"]

    lines = [f"Repository: {repo_name}", ""]

    lines.append("=== HEALTH SCORE ===")
    if health["health_score"] is not None:
        lines.append(f"Overall Health Score: {health['health_score']}/100 (weighted average of the sub-scores below)")
        for key, weight in _HEALTH_SCORE_WEIGHTS.items():
            val = health["sub_scores"].get(key)
            if val is not None:
                lines.append(f"  - {key.replace('_', ' ').title()} score: {val}/100 (weight {int(weight * 100)}%)")
    else:
        lines.append("(unavailable: no sub-scores could be computed)")

    lines.append("")
    lines.append("=== CODE HEALTH ===")
    if not code_health.get("available"):
        lines.append(f"(unavailable: {code_health.get('reason')})")
    else:
        lines.append(f"Total files: {code_health['total_files']}")
        lines.append(f"Total indexed symbols (functions/classes/sections): {code_health['total_symbols']}")
        lines.append("Language distribution (by file count, one language per file): " + ", ".join(
            f"{lang}: {count}" for lang, count in code_health["language_counts"]
        ))
        lines.append(
            f"Oversized functions/classes (large enough that the chunker had to "
            f"split them): {code_health['oversized_symbol_count']}"
        )
        if code_health["oversized_symbols_sample"]:
            lines.append("  Sample: " + ", ".join(f"{path}:{name}" for path, name in code_health["oversized_symbols_sample"]))
        lines.append(f"TODO/FIXME/HACK comments: {code_health['todo_fixme_hack_total']} across {code_health['todo_fixme_hack_files']} files")
        if code_health["duplicate_names_sample"]:
            lines.append(
                "Function/class names reused (same name+kind) across 3+ different files: "
                + ", ".join(f"{name} ({kind}, in {n} files)" for name, kind, n in code_health["duplicate_names_sample"])
            )
        else:
            lines.append("No function/class name reused across 3+ different files.")
        if code_health["python_docstring_coverage_pct"] is not None:
            lines.append(
                f"Python docstring coverage (heuristic, {code_health['python_symbol_count']} "
                f"Python functions/classes checked): {code_health['python_docstring_coverage_pct']}%"
            )
        else:
            lines.append("Python docstring coverage: not computed (no Python symbols indexed).")

    lines.append("")
    lines.append("=== REPOSITORY SIZE ===")
    lines.append(f"On-disk size: {size_info['total_size_mb']} MB across {size_info['file_count']} files (excluding .git)")
    lines.append(f"Total lines (text files only): {size_info['total_lines']:,}")

    lines.append("")
    lines.append("=== DEPENDENCIES ===")
    if not dependency_info.get("available"):
        lines.append(f"(unavailable: {dependency_info.get('reason')})")
    else:
        lines.append(f"Manifests found and parsed: {', '.join(m['file'] for m in dependency_info['manifests_parsed']) or 'none'}")
        if dependency_info["manifests_unparsed"]:
            lines.append(f"Manifests found but NOT parsed (format not confidently supported): {', '.join(dependency_info['manifests_unparsed'])}")
        lines.append(f"Total runtime dependencies declared: {dependency_info['total_runtime_dependencies']}")
        lines.append(f"Total dev/test dependencies declared: {dependency_info['total_dev_dependencies']}")
        for m in dependency_info["manifests_parsed"]:
            pin_note = f", {m['pinned_count']} version-pinned" if m["pinned_count"] is not None else ""
            lines.append(f"  {m['file']}: {m['runtime_count']} runtime, {m['dev_count']} dev{pin_note}")

    lines.append("")
    lines.append("=== DOCUMENTATION SCORE ===")
    lines.append(f"Score: {doc_score['score']}/100 -- README present: {doc_score['has_readme']}, "
                  f"docstring coverage: {doc_score['docstring_coverage_pct']}%, "
                  f"doc-file ratio: {doc_score['doc_file_ratio_pct']}% of files")

    lines.append("")
    lines.append("=== TESTING SCORE ===")
    lines.append(f"Score: {test_score['score']}/100 -- {test_score['test_file_count']} test files vs. "
                  f"{test_score['source_file_count']} source files, CI configured: {test_score['has_ci_config']}")

    lines.append("")
    lines.append("=== SECURITY (static pattern scan only -- NOT a vulnerability/CVE audit) ===")
    if not security_info.get("available"):
        lines.append(f"(unavailable: {security_info.get('reason')})")
    else:
        lines.append(f"Suspected hardcoded-secret-shaped strings: {security_info['suspected_secret_hits']} across {security_info['suspected_secret_files']} files (heuristic pattern match -- may include false positives/negatives)")
        lines.append(f"Committed .env file: {security_info['committed_env_file'] or 'none found'}")
        lines.append(f".gitignore present: {security_info['has_gitignore']}")

    lines.append("")
    lines.append("=== CODE COMPLEXITY (approximate -- control-flow keyword count per function, not real AST cyclomatic complexity) ===")
    if not complexity_info.get("available"):
        lines.append(f"(unavailable: {complexity_info.get('reason')})")
    else:
        lines.append(f"Average approximate complexity across {complexity_info['function_count']} functions: {complexity_info['avg_complexity']}")
        lines.append(f"Functions at or above the high-complexity threshold ({_HIGH_COMPLEXITY_THRESHOLD}): {complexity_info['high_complexity_count']}")
        if complexity_info["high_complexity_sample"]:
            lines.append("  Sample: " + ", ".join(f"{path}:{name} (~{score})" for path, name, score in complexity_info["high_complexity_sample"]))

    lines.append("")
    lines.append("=== REPO ACTIVITY & MAINTENANCE (from git log) ===")
    if not git_info.get("available"):
        lines.append(f"(unavailable: {git_info.get('reason')})")
    else:
        lines.append(f"Total commits: {git_info['total_commits']}")
        lines.append(f"First commit: {git_info['first_commit_date']}")
        lines.append(f"Last commit: {git_info['last_commit_date']}")
        lines.append(f"Commits in the last 90 days: {git_info['commits_last_90_days']}")
        lines.append(f"Contributors: {git_info['contributor_count']}")
        lines.append(
            f"Bus factor: top contributor ({git_info['bus_factor_top_author']}) "
            f"authored {git_info['bus_factor_pct']}% of all commits"
        )
        lines.append("Top contributors: " + ", ".join(f"{a} ({n})" for a, n in git_info["top_contributors"]))
        if git_info["hotspot_files"]:
            lines.append("Hotspot files (changed most often): " + ", ".join(f"{f} ({n} commits)" for f, n in git_info["hotspot_files"]))
        if git_info["weekly_activity"]:
            lines.append("Weekly commit activity, oldest to newest (week starting / commits): " + ", ".join(
                f"{week}: {count}" for week, count in git_info["weekly_activity"]
            ))

    return "\n".join(lines)


SYSTEM_PROMPT = (
    "You are producing a repository health report. You are given PRE-COMPUTED, "
    "exact numbers and scores below -- use ONLY these numbers, including the "
    "Health Score and every sub-score. Do not invent additional statistics, do "
    "not estimate anything not given to you, and never compute or restate a "
    "score differently than given. Where a section says '(unavailable: ...)', "
    "say so plainly instead of guessing. The security section is a STATIC "
    "PATTERN SCAN ONLY, not a vulnerability/CVE audit -- never claim or imply "
    "the repo has been verified free of vulnerabilities; only speak to what "
    "the pattern scan actually found. The complexity section is an "
    "APPROXIMATION, not real cyclomatic complexity -- frame it accordingly.\n\n"
    "Produce a report with exactly these sections, in this order. Format EVERY "
    "section heading as a markdown '##' header (e.g. '## 1. OVERALL SUMMARY') -- "
    "the UI's renderer specifically looks for that markdown syntax, and a "
    "differently-styled heading (plain numbered text, bold text with no '##') "
    "won't render as a heading at all.\n\n"
    "1. OVERALL SUMMARY -- 2-4 sentences giving a genuine, specific read on "
    "this repo's health, opening with the Health Score and citing at least "
    "two other actual numbers below. Not a generic 'this repo looks fine' -- "
    "ground it in what's actually notable.\n"
    "2. CODE QUALITY -- walk through the code-health numbers: language spread, "
    "oversized functions, TODO density, duplicate names, docstring coverage, "
    "code complexity. Note on duplicate names: a common, conventional entry-"
    "point name (main, cli, run, setup, and similar) repeated across many "
    "files is usually each file being its own independent standalone script/"
    "example/entry point -- that's normal, not a naming-hygiene problem. Only "
    "call a repeated name a real concern if it's a more distinctive, non-"
    "conventional name.\n"
    "3. REPOSITORY OVERVIEW -- size, dependencies (runtime vs. dev, and "
    "whether any manifest went unparsed), documentation score, testing score.\n"
    "4. REPO ACTIVITY & MAINTENANCE -- walk through the git numbers: is this "
    "actively maintained or stale, how concentrated is ownership (bus "
    "factor), which files churn the most, what the weekly activity trend "
    "looks like (picking up, steady, or tailing off).\n"
    "5. SECURITY -- report exactly what the static scan found (or didn't), "
    "with the explicit caveat that this isn't a full audit.\n"
    "6. RISKS / RED FLAGS -- ONLY flag something if the actual numbers support "
    "it (e.g. a high bus factor, a stale last-commit date, a low security or "
    "testing score, a large oversized-function count). If nothing stands out "
    "as a real risk, say so -- do not manufacture a risk to fill this section.\n"
    "7. RECOMMENDATIONS -- concrete, tied directly to the risks/scores above, "
    "not generic best-practice advice unrelated to what was actually found."
)


def compute_health(repository_name: str, local_path: Path) -> dict:
    """
    Pure computation, no LLM call -- returns every raw structured number
    and score plus the formatted text block the LLM narrative is built
    from. Split out from act_repo_health so a direct API endpoint (see
    main.py's health-check button) can get the structured dicts straight
    -- e.g. to render real stat tiles -- without needing to go through
    the chat pipeline's state dict shape or re-run the (slower)
    narrative LLM call to get at them.
    """

    chunks = get_all_chunks(repository_name)

    code_health = _code_health(chunks)
    git_info = _git_history(local_path)
    size_info = _repo_size(local_path)
    dependency_info = _dependencies(local_path)
    security_info = _security_scan(local_path, chunks)
    complexity_info = _complexity(chunks)
    documentation_score = _documentation_score(local_path, code_health)
    testing_score = _testing_score(chunks, local_path)

    sub_scores = {
        "documentation": documentation_score["score"],
        "testing": testing_score["score"],
        "security": _security_score(security_info),
        "complexity": _complexity_score(complexity_info),
        "activity": _activity_score(git_info),
        "bus_factor": _bus_factor_score(git_info),
    }

    health = {
        "code_health": code_health,
        "git_info": git_info,
        "size_info": size_info,
        "dependency_info": dependency_info,
        "security_info": security_info,
        "complexity_info": complexity_info,
        "documentation_score": documentation_score,
        "testing_score": testing_score,
        "sub_scores": sub_scores,
        "health_score": _health_score(sub_scores),
    }

    health["report_text"] = _format_report(repository_name, health)

    return health


def narrate_health(
    report_text: str, persona: Optional[str] = None, conversation_context: Optional[str] = None,
) -> str:
    """The LLM narrative pass over an already-computed report_text block.
    `persona` only changes how this gets explained -- report_text itself
    (every number) is already fixed before this is ever called.
    conversation_context defaults to None because the Health Check
    button's direct endpoint calls this with no chat history at all."""

    return call_llm(SYSTEM_PROMPT, report_text, persona=persona, conversation_context=conversation_context)


def _sources_from_health(health: dict) -> list:
    sources = []
    code_health, git_info = health["code_health"], health["git_info"]
    complexity_info = health["complexity_info"]

    if code_health.get("available"):
        for path, name in code_health["oversized_symbols_sample"]:
            sources.append({"file": path, "chunk_name": name, "chunk_type": "oversized_symbol", "lines": ""})
    if git_info.get("available"):
        for path, count in git_info["hotspot_files"]:
            sources.append({"file": path, "chunk_name": "", "chunk_type": "hotspot_file", "lines": f"{count} commits"})
    if complexity_info.get("available"):
        for path, name, score in complexity_info["high_complexity_sample"]:
            sources.append({"file": path, "chunk_name": name, "chunk_type": "high_complexity_function", "lines": f"~{score} complexity"})

    return sources


def act_repo_health(state: dict) -> dict:
    if state.get("local_path") is None:
        return no_repo(state, TOOL_NAME)

    local_path: Path = state["local_path"]
    repo_name = state["active_repository"]

    health = compute_health(repo_name, local_path)
    state["context_used"] = health["report_text"]

    try:
        reply = narrate_health(
            health["report_text"],
            persona=state.get("persona"),
            conversation_context=state.get("conversation_context"),
        )
    except Exception as exc:
        return llm_error(state, TOOL_NAME, exc)

    state["reply"] = reply
    state["sources"] = _sources_from_health(health)
    state["tool_used"] = TOOL_NAME

    return state
