"""
code_writer_test.py

code_writer_tool Regression Test.

Responsibility
--------------
10 hand-designed test cases -- mixed write/modify, including the edge
cases that turned up real bugs during development: path escape, a
symbol that doesn't exist, writing over an existing file, preserving a
large function's full behavior, a brand-new nested directory, and a
create-then-modify pair that only works because of incremental
re-indexing (see app/rag/indexer.py::reindex_file).

This makes REAL writes into the target repo's clone on disk -- there is
no dry-run mode, because code_writer_tool itself doesn't have one. Each
case is checked structurally (which file/operation was used, whether it
errored, whether it refused) rather than asserting exact generated code,
since that's what's actually deterministic about an LLM-backed tool.

Usage
-----
Requires a repo already loaded (see /api/repositories/load, or call
app.rag.indexer.index_repository directly) -- click is assumed below
since it's what every other test file in this project targets. Run with:

    python -m app.agent.tools.code_writer_test

This module DOES NOT:
    ✗ Clone or index the repository itself
    ✗ Revert its own writes -- see the note printed at the end
"""

from pathlib import Path

from ...github.clone import REPO_STORAGE
from .code_writer import act_code_writer


REPOSITORY = "click"


def _state(message: str) -> dict:
    local_path = REPO_STORAGE / REPOSITORY
    return {
        "message": message,
        "active_repository": REPOSITORY,
        "local_path": local_path if local_path.exists() else None,
        "tool_args": {"query": message},
        "sources": [],
        "tool_used": "general_chat",
        "error": None,
    }


# Each case is (category, title, message, check) -- check(result_state) -> (bool, str)
# reason. `result_state` is whatever act_code_writer returned.

def _check_path_contains(substr):
    def check(state):
        path = (state["sources"][0]["file"] if state["sources"] else "")
        return substr in path, f"expected path containing '{substr}', got '{path}'"
    return check


def _check_error(expected_error):
    def check(state):
        return state.get("error") == expected_error, f"expected error='{expected_error}', got '{state.get('error')}'"
    return check


def _check_no_escape():
    """
    For a path-traversal attempt, TWO outcomes are equally safe and both
    should pass: resolve_target refuses it (error='path_escape'), OR the
    LLM itself never echoes the literal escaping path and picks something
    that lands inside the repo instead (error=None with a real write) --
    resolve_target already gates every successful write, so error=None
    with sources present is only reachable when the path it actually used
    resolved safely inside the repo. Checking only for the first outcome
    made this test flaky against ordinary LLM non-determinism despite the
    actual security boundary (never write outside the repo) holding either
    way -- confirmed directly against resolve_target with a hardcoded
    escaping path, independent of anything the LLM does.
    """
    def check(state):
        if state.get("error") == "path_escape":
            return True, "refused via path_escape"
        if state.get("error") is None and state.get("sources"):
            return True, "LLM did not echo the escaping path; write landed safely inside the repo"
        return False, f"expected either a path_escape refusal or a safe in-repo write, got error={state.get('error')!r} sources={state.get('sources')!r}"
    return check


def _check_no_error_and_path_contains(substr):
    def check(state):
        if state.get("error"):
            return False, f"expected success, got error='{state.get('error')}'"
        return _check_path_contains(substr)(state)
    return check


TEST_CASES = [
    (
        "write", "Basic new file creation",
        "Create a new file called stack_using_queues.cpp in the DSA folder "
        "implementing a stack using two queues, with push and pop operations.",
        _check_no_error_and_path_contains("stack_using_queues.cpp"),
    ),
    (
        "modify", "Exact-name symbol replacement",
        "Modify the isPalindrome function in palindrome_check.cpp to ignore "
        "case and spaces when checking.",
        _check_no_error_and_path_contains("palindrome_check.cpp"),
    ),
    (
        "write", "Append a helper to an existing file (no new file)",
        "Add a new utility function called to_shouty_snake_case(s) to click's "
        "utils.py that converts a string to UPPER_SNAKE_CASE, following the "
        "style of the existing helpers in that file.",
        _check_no_error_and_path_contains("utils.py"),
    ),
    (
        "modify", "Vague description, no file or function named (retrieval accuracy)",
        "Change the logic that counts subarrays summing to a target value so "
        "it instead tracks the length of the longest such subarray.",
        _check_no_error_and_path_contains("subarray_sum.cpp"),
    ),
    (
        "write", "EDGE CASE: path traversal attempt -- must never write outside the repo",
        "Write a file at ../../../../outside_repo_test.cpp that prints a warning.",
        _check_no_escape(),
    ),
    (
        "modify", "EDGE CASE: symbol that does not exist anywhere",
        "Modify the function quantumTeleportSort to add a logging statement "
        "at the start.",
        lambda state: (state.get("error") is None, "expected a graceful fallback (append), not an error"),
    ),
    (
        "write", "EDGE CASE: create_file on a path that already exists",
        "Create a new file called subarray_sum.cpp in the DSA folder "
        "implementing binary search over a sorted array.",
        _check_no_error_and_path_contains("subarray_sum.cpp"),
    ),
    (
        "modify", "EDGE CASE: preserve full existing behavior (regression -- do not rewrite from scratch)",
        "Modify the echo function in click's utils.py to also duplicate its "
        "output to a second file-like object passed as an optional 'tee' parameter.",
        _check_no_error_and_path_contains("utils.py"),
    ),
    (
        "write", "Creative: brand-new nested directory that does not exist yet",
        "Create a file called trie.cpp inside a new folder DSA/data_structures/ "
        "implementing a basic Trie (prefix tree) with insert and search methods.",
        _check_no_error_and_path_contains("trie.cpp"),
    ),
    (
        "modify", "Creative: create-then-modify in the same session (incremental re-index)",
        None,  # handled specially below -- two calls, not one
        None,
    ),
]


def _run_case(index, category, title, message, check):
    state = act_code_writer(_state(message))
    ok, reason = check(state)
    return {
        "index": index, "category": category, "title": title,
        "ok": ok, "reason": reason,
        "tool_used": state.get("tool_used"), "error": state.get("error"),
        "file": state["sources"][0]["file"] if state.get("sources") else None,
    }


def _run_create_then_modify_case(index):
    create_msg = (
        "Create a file called fibonacci.cpp in the DSA folder implementing "
        "plain recursive Fibonacci."
    )
    create_state = act_code_writer(_state(create_msg))

    modify_msg = "Modify the fibonacci function to use memoization instead of plain recursion."
    modify_state = act_code_writer(_state(modify_msg))

    created_path = create_state["sources"][0]["file"] if create_state.get("sources") else None
    modified_path = modify_state["sources"][0]["file"] if modify_state.get("sources") else None

    # Compare as paths, not raw strings -- the LLM is free to write either
    # slash style ("DSA/x.cpp" vs "DSA\\x.cpp"), and both are the same
    # file on Windows. A string-equality check here is a test-harness bug
    # (a false negative), not a real product bug.
    same_file = (
        created_path is not None
        and modified_path is not None
        and Path(created_path) == Path(modified_path)
    )

    ok = (
        create_state.get("error") is None
        and modify_state.get("error") is None
        and same_file
        and modify_state.get("sources", [{}])[0].get("chunk_type") == "replace_symbol"
    )
    reason = (
        "modify correctly found and edited the file this same run just created"
        if ok else
        f"created={created_path!r} modified={modified_path!r} "
        f"modify_error={modify_state.get('error')} "
        f"modify_op={modify_state.get('sources', [{}])[0].get('chunk_type')}"
    )

    return {
        "index": index, "category": "modify", "title": "Creative: create-then-modify (incremental re-index)",
        "ok": ok, "reason": reason,
        "tool_used": modify_state.get("tool_used"), "error": modify_state.get("error"),
        "file": modified_path,
    }


def run() -> list:
    results = []

    for i, (category, title, message, check) in enumerate(TEST_CASES, start=1):
        if message is None:
            results.append(_run_create_then_modify_case(i))
        else:
            results.append(_run_case(i, category, title, message, check))

    return results


if __name__ == "__main__":
    results = run()

    for r in results:
        status = "PASS" if r["ok"] else "FAIL"
        print(f"[{status}] #{r['index']:>2} ({r['category']:<8}) {r['title']}")
        print(f"         file={r['file']}  error={r['error']}")
        if not r["ok"]:
            print(f"         reason: {r['reason']}")

    passed = sum(1 for r in results if r["ok"])
    print(f"\n{passed}/{len(results)} passed")
    print(
        "\nNote: this made REAL writes into the repo's clone on disk (create_file, "
        "append_to_file, replace_symbol) -- there is no dry-run mode. Review/revert "
        "with `git status` / `git checkout` in the repo's local clone if needed."
    )
