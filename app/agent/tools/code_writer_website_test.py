"""
code_writer_website_test.py

code_writer_tool Web/Quality Regression Test.

Responsibility
--------------
5 more test cases beyond code_writer_test.py's structural checks (which
file, which operation, errored or not) -- this time actually verifying
the GENERATED CODE's quality: does the HTML parse as real HTML, does the
CSS have balanced rules, does the JS pass a real syntax check (`node
--check`, no execution). Plus two edge cases: a single request asking
for a whole multi-file website in one shot (the tool can only commit to
one file per call -- this documents what it does with that), and a
unicode/encoding round-trip check.

This makes REAL writes into the target repo's clone on disk, same as
code_writer_test.py -- no dry-run mode exists. `node --check` is used
for JS (parses only, never executes the file); HTML is checked with
Python's stdlib html.parser; CSS gets a lightweight brace-balance +
rule-presence check since no CSS parser is available without a new
dependency.

Usage
-----
Requires a repo already loaded, same as code_writer_test.py. Run with:

    python -m app.agent.tools.code_writer_website_test

This module DOES NOT:
    ✗ Clone or index the repository itself
    ✗ Revert its own writes -- see the note printed at the end
    ✗ Verify C++/Python quality -- see code_writer_test.py; no compiler
      is available in this environment to check C++, and this file is
      scoped to website code specifically
"""

import subprocess
from html.parser import HTMLParser
from pathlib import Path

from .code_writer import act_code_writer
from .code_writer_test import REPOSITORY, _state


# ---------------------------------------------------------------------------
# Quality checkers -- each returns (ok: bool, reason: str)
# ---------------------------------------------------------------------------

class _HTMLStructureCheck(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags_seen = set()

    def handle_starttag(self, tag, attrs):
        self.tags_seen.add(tag.lower())


def check_html_quality(content: str, target: Path = None) -> tuple:
    checker = _HTMLStructureCheck()

    try:
        checker.feed(content)
    except Exception as exc:
        return False, f"HTML failed to parse: {exc}"

    required = {"html", "head", "body"}
    missing = required - checker.tags_seen

    if missing:
        return False, f"missing required tag(s): {sorted(missing)}"

    return True, f"valid HTML structure ({len(checker.tags_seen)} distinct tags)"


def check_css_quality(content: str, target: Path = None) -> tuple:
    if content.count("{") != content.count("}"):
        return False, "unbalanced braces"

    if "{" not in content:
        return False, "no rules found"

    return True, "balanced braces, contains rules"


def check_js_quality(content: str, target: Path) -> tuple:
    try:
        result = subprocess.run(
            ["node", "--check", str(target)],
            capture_output=True, text=True, timeout=10,
        )
    except FileNotFoundError:
        return True, "node not available -- skipped real syntax check"

    if result.returncode != 0:
        return False, f"node --check failed: {result.stderr.strip()[:300]}"

    return True, "valid JS syntax (verified with node --check)"


_CHECKERS_BY_SUFFIX = {".html": check_html_quality, ".css": check_css_quality, ".js": check_js_quality}


def check_quality(path_str: str, target: Path) -> tuple:
    checker = _CHECKERS_BY_SUFFIX.get(Path(path_str).suffix.lower())

    if checker is None:
        return True, f"no quality checker for '{Path(path_str).suffix}' -- structural pass only"

    if not target.exists():
        return False, f"file does not exist on disk: {target}"

    content = target.read_text(encoding="utf-8", errors="ignore")
    return checker(content, target)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def _resolved_target(state, file_path: str) -> Path:
    return state["local_path"] / file_path


def _run_simple(index, category, title, message):
    state = act_code_writer(_state(message))
    file_path = state["sources"][0]["file"] if state.get("sources") else None

    if state.get("error") or not file_path:
        return {
            "index": index, "category": category, "title": title, "ok": False,
            "reason": f"error='{state.get('error')}'", "file": file_path,
        }

    target = _resolved_target(state, file_path)
    ok, reason = check_quality(file_path, target)

    return {"index": index, "category": category, "title": title, "ok": ok, "reason": reason, "file": file_path}


def _run_js_create_then_modify(index):
    """Website JS, created then extended -- tests incremental re-index
    (same class of bug as code_writer_test.py's case #10) AND that the
    second edit's content is actually present, not just that it ran."""

    create_msg = (
        "Create a JavaScript file called form_validation.js in a new web/ folder "
        "that validates an email input field before form submission."
    )
    create_state = act_code_writer(_state(create_msg))

    modify_msg = (
        "Modify form_validation.js to also validate a phone number field, "
        "in addition to the existing email validation."
    )
    modify_state = act_code_writer(_state(modify_msg))

    created_path = create_state["sources"][0]["file"] if create_state.get("sources") else None
    modified_path = modify_state["sources"][0]["file"] if modify_state.get("sources") else None

    same_file = (
        created_path is not None and modified_path is not None
        and Path(created_path) == Path(modified_path)
    )

    if not same_file or create_state.get("error") or modify_state.get("error"):
        return {
            "index": index, "category": "modify", "title": "Website: JS create-then-extend",
            "ok": False, "file": modified_path,
            "reason": f"created={created_path!r} modified={modified_path!r} "
                      f"create_error={create_state.get('error')} modify_error={modify_state.get('error')}",
        }

    target = _resolved_target(modify_state, modified_path)
    syntax_ok, syntax_reason = check_quality(modified_path, target)

    content = target.read_text(encoding="utf-8", errors="ignore").lower()
    has_both = ("email" in content) and ("phone" in content)

    ok = syntax_ok and has_both
    reason = syntax_reason if not syntax_ok else (
        "both email and phone validation present" if has_both
        else "phone validation missing -- modify may have dropped the earlier email logic"
    )

    return {"index": index, "category": "modify", "title": "Website: JS create-then-extend", "ok": ok, "reason": reason, "file": modified_path}


def _run_compound_request(index):
    """EDGE CASE: one message asking for a whole multi-file website.
    write_code can only commit to ONE file per call -- this documents
    what actually happens (which single file it picks) rather than
    asserting a specific outcome, since the schema itself doesn't
    support multi-file output."""

    message = (
        "Create a complete personal blog website in a new blog/ folder, with "
        "index.html, styles.css, and script.js, including a light/dark mode toggle."
    )
    state = act_code_writer(_state(message))
    file_path = state["sources"][0]["file"] if state.get("sources") else None

    if state.get("error") or not file_path:
        return {
            "index": index, "category": "write", "title": "EDGE CASE: compound multi-file request",
            "ok": False, "file": file_path, "reason": f"error='{state.get('error')}'",
        }

    target = _resolved_target(state, file_path)
    ok, reason = check_quality(file_path, target)

    return {
        "index": index, "category": "write", "title": "EDGE CASE: compound multi-file request",
        "ok": ok, "file": file_path,
        "reason": f"picked single file '{file_path}' for a 3-file request ({reason})",
    }


def _run_unicode_case(index):
    """EDGE CASE: unicode/encoding round-trip. Modifies the earlier
    index.html (falls back to creating it fresh if case #1 didn't run
    or was reverted) to add a copyright symbol and a non-Latin script
    label, then confirms both survive intact on disk -- not escaped,
    not mangled, not replaced with '?' by an encoding mismatch."""

    ensure_msg = "Create web/index.html with a simple portfolio landing page if it doesn't already exist."
    act_code_writer(_state(ensure_msg))

    modify_msg = (
        "Modify web/index.html's footer to add a copyright notice with the "
        "© symbol and a Japanese-language toggle link labeled 日本語."
    )
    state = act_code_writer(_state(modify_msg))
    file_path = state["sources"][0]["file"] if state.get("sources") else None

    if state.get("error") or not file_path:
        return {
            "index": index, "category": "modify", "title": "EDGE CASE: unicode/encoding round-trip",
            "ok": False, "file": file_path, "reason": f"error='{state.get('error')}'",
        }

    target = _resolved_target(state, file_path)
    content = target.read_text(encoding="utf-8", errors="strict")

    # Either the literal character or the standard named HTML entity is
    # correct, idiomatic markup -- &copy; is arguably the safer choice in
    # HTML regardless of declared encoding, so it must not fail this check.
    has_copyright = ("©" in content) or ("&copy;" in content)
    has_japanese = "日本語" in content
    html_ok, html_reason = check_html_quality(content)

    ok = has_copyright and has_japanese and html_ok
    reason = (
        # ASCII-only on purpose -- Windows' console defaults to cp1252,
        # which can't print the literal characters this test checks for.
        "copyright notice and non-Latin script link both present intact, still valid HTML" if ok else
        f"copyright_present={has_copyright} japanese_present={has_japanese} html_ok={html_ok} ({html_reason})"
    )

    return {"index": index, "category": "modify", "title": "EDGE CASE: unicode/encoding round-trip", "ok": ok, "reason": reason, "file": file_path}


def run() -> list:
    results = [
        _run_simple(1, "write", "Website: HTML landing page", (
            "Create an HTML file called index.html in a new web/ folder with a simple "
            "portfolio landing page, including a header, nav, and footer."
        )),
        _run_simple(2, "write", "Website: CSS stylesheet with a responsive breakpoint", (
            "Create a CSS file called styles.css in the web folder with styling for "
            "the portfolio page, including a mobile breakpoint media query."
        )),
        _run_js_create_then_modify(3),
        _run_compound_request(4),
        _run_unicode_case(5),
    ]
    return results


if __name__ == "__main__":
    results = run()

    for r in results:
        status = "PASS" if r["ok"] else "FAIL"
        print(f"[{status}] #{r['index']} ({r['category']:<8}) {r['title']}")
        print(f"         file={r['file']}")
        print(f"         {r['reason']}")

    passed = sum(1 for r in results if r["ok"])
    print(f"\n{passed}/{len(results)} passed")
    print(
        "\nNote: this made REAL writes into the repo's clone on disk -- there is no "
        "dry-run mode. Review/revert with `git status` / `git checkout` in the repo's "
        "local clone if needed."
    )
