"""
chunker.py

Repository Code Chunker.

Responsibilities
----------------
- Detect a file's category: code / doc / config (see language.py)
- Chunk it at semantically meaningful boundaries for that category
- Never let one bad file abort the whole repository
- Generate metadata for RAG

Strategy per category (dispatched by chunk_repository)
--------------------------------------------------------
code   -- tree-sitter, chunked at function/class/method node boundaries.
          A class/struct/interface's own chunk holds only the lines it
          directly owns (signature, docstring, fields) -- not its methods'
          bodies, which are chunked separately. This is what keeps a large
          class from becoming one multi-thousand-line chunk that gets
          silently truncated at the embedder's token limit.
          @typing.overload-style stub bodies (`...` / `pass` /
          `raise NotImplementedError`) are skipped -- they duplicate the
          real implementation's name with no retrievable content.
          Falls back to fixed-size chunking when there's no grammar for the
          language, parsing throws, or parsing finds zero structure.
doc    -- markdown, split on heading boundaries. Whole-file if short and
          headless. Fixed-size fallback otherwise.
config -- yaml/json/toml/ini, whole-file if under the token budget,
          otherwise split by top-level key. Fixed-size fallback on anything
          that doesn't look like clean top-level keys.

Every one of those falls back to the same fixed-size, token-bounded,
line-aligned splitter (_fixed_size_chunks) so nothing is ever silently
dropped -- and it's also what re-splits any single oversized node (a huge
class header, a huge doc paragraph) that exceeds MAX_CHUNK_TOKENS.

This module DOES NOT
--------------------
- Generate embeddings
- Store vectors
- Query ChromaDB
"""

from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import tree_sitter_language_pack as tlp

from .. import config
from ..utils.logger import log_warning
from ..rag.embeddings import count_embedding_tokens as _count_tokens
from .language import detect_category, tree_sitter_language_name

# Token counting (shared by the fixed-size fallback + oversized-chunk split)
# is borrowed from the embedding model itself (see the import above) -- the
# whole point is to catch anything that WOULD get silently truncated at
# embed time, so it has to use that model's own tokenizer, not a generic one.


# ---------------------------------------------------------------------------
# Chunk ID
# ---------------------------------------------------------------------------

def create_chunk_id(
    file_path: str,
    chunk_name: str,
    start_line: int,
    end_line: int,
) -> str:
    """
    Generate a unique chunk ID.

    Example:
        app_py_login_20_45
        models_py_User_10_80
    """

    safe_path = (
        file_path
        .replace("\\", "_")
        .replace("/", "_")
        .replace(".", "_")
    )

    return f"{safe_path}_{chunk_name}_{start_line}_{end_line}"


def _make_chunk(
    repo_name: str,
    relative_path: str,
    language: str,
    chunk_type: str,
    chunk_name: str,
    start_line: int,
    end_line: int,
    content: str,
    replaceable: bool = True,
) -> Dict:
    return {
        "repo": repo_name,
        "path": relative_path,
        "language": language,
        "chunk_type": chunk_type,
        "chunk_name": chunk_name,
        "chunk_id": create_chunk_id(relative_path, chunk_name, start_line, end_line),
        "start_line": start_line,
        "end_line": end_line,
        "content": content,
        # False only for a code chunk whose true boundaries share a line
        # with a sibling's (a chained call -- see _corrected_bounds) --
        # find_symbol_bounds refuses to resolve replace_symbol to one of
        # these, since whole-line replacement would corrupt that shared
        # line. Always True for non-code chunks (doc/config/fixed-size),
        # which replace_symbol never targets anyway.
        "replaceable": replaceable,
    }


# ---------------------------------------------------------------------------
# Fixed-size fallback chunker (token-bounded, line-aligned, with overlap)
# ---------------------------------------------------------------------------

def _fixed_size_chunks(
    lines: List[str],
    chunk_tokens: int = config.CHUNK_TOKEN_SIZE,
    overlap_tokens: int = config.CHUNK_TOKEN_OVERLAP,
) -> List[tuple]:
    """
    Split lines into token-bounded windows, each carrying `overlap_tokens`
    worth of trailing context into the next window.

    Returns a list of (start_line, end_line, content) — 1-based, inclusive.
    """

    if not lines:
        return []

    line_tokens = [_count_tokens(line) + 1 for line in lines]  # +1 for the newline
    n = len(lines)
    windows = []

    start = 0
    while start < n:
        total = 0
        end = start
        while end < n and (total + line_tokens[end] <= chunk_tokens or end == start):
            total += line_tokens[end]
            end += 1

        windows.append((start + 1, end, "\n".join(lines[start:end])))

        if end >= n:
            break

        # Step back `overlap_tokens` worth of lines so the next window
        # carries trailing context forward.
        back_tokens = 0
        step_end = end
        while step_end > start and back_tokens < overlap_tokens:
            step_end -= 1
            back_tokens += line_tokens[step_end]

        start = max(step_end, start + 1)

    return windows


def _fixed_size_fallback(
    repo_name: str,
    relative_path: str,
    language: str,
    content: str,
    chunk_type: str = "fixed_block",
) -> List[Dict]:
    lines = content.splitlines()
    windows = _fixed_size_chunks(lines)

    return [
        _make_chunk(
            repo_name, relative_path, language, chunk_type,
            chunk_name=f"block_{i}",
            start_line=start, end_line=end, content=text,
        )
        for i, (start, end, text) in enumerate(windows, start=1)
    ]


# ---------------------------------------------------------------------------
# Code chunking (tree-sitter)
# ---------------------------------------------------------------------------

_STUB_TAIL_RE = re.compile(r"(\.\.\.|pass|raise\s+NotImplementedError\b[^\n]*);?\s*$")

# Presence of any of these means the block has real control flow, so a
# trailing `pass`/`...` is a legitimate nested statement (e.g. an
# `except: pass` deep inside a real implementation) -- not an empty
# placeholder body.
_CONTROL_FLOW_RE = re.compile(r"\b(if|elif|else|for|while|try|except|finally|with|switch|case|catch|match)\b")


def _is_stub(content: str) -> bool:
    """
    True for a body whose ONLY statement is `...`, `pass`, or
    `raise NotImplementedError` (a @typing.overload stub, or an
    interface/abstract placeholder) -- these add no retrievable content
    and duplicate the real implementation's name. Checking just the last
    line isn't enough: a real implementation can legitimately end in a
    nested `except: pass`, so this only counts as a stub when there's no
    control flow anywhere in the body to make that `pass` a nested one.
    """

    tail = content.strip()

    if not tail:
        return True

    last_line = tail.splitlines()[-1].strip()

    looks_like_marker = bool(_STUB_TAIL_RE.search(last_line)) or (
        last_line.endswith(";") and "{" not in tail  # e.g. `void foo();`
    )

    if not looks_like_marker:
        return False

    return not _CONTROL_FLOW_RE.search(tail)


def _split_if_oversized(chunk: Dict) -> List[Dict]:
    """
    Re-split a single chunk that exceeds MAX_CHUNK_TOKENS using the
    fixed-size splitter, so one huge node can't blow past the embedder's
    token window and get silently truncated.
    """

    if _count_tokens(chunk["content"]) <= config.MAX_CHUNK_TOKENS:
        return [chunk]

    lines = chunk["content"].splitlines()
    windows = _fixed_size_chunks(lines, chunk_tokens=config.MAX_CHUNK_TOKENS)

    if len(windows) <= 1:
        return [chunk]

    base_start = chunk["start_line"]
    split_chunks = []

    for i, (local_start, local_end, text) in enumerate(windows, start=1):
        name = chunk["chunk_name"] if i == 1 else f"{chunk['chunk_name']}#{i}"
        split_chunks.append(_make_chunk(
            chunk["repo"], chunk["path"], chunk["language"], chunk["chunk_type"],
            chunk_name=name,
            start_line=base_start + local_start - 1,
            end_line=base_start + local_end - 1,
            content=text,
        ))

    return split_chunks


_DECORATOR_LINE_RE = re.compile(r"^(@|#\[|\[.+\]$)")

# Confirms a candidate line/column is really where a function LITERAL
# begins (an arrow function, a bare `function` expression) -- used to
# off-by-one-correct an ANONYMOUS item's start_line the same way a named
# item's start_line is corrected by searching for its name (see
# _locate_declaration_line): there's no name to search for, but
# span.start_column still says exactly where the literal starts, so this
# just confirms real function-literal syntax actually begins there.
# The last two alternatives cover a parameter list that continues past
# this line (e.g. `async (\n  a: T,\n  b: U,\n) => {` -- confirmed real,
# not a rare case: multi-param arrow/async functions routinely wrap) --
# checked against a SINGLE line's text, so a closed `()...=>` has to
# appear in full on this one line, but an unclosed `(` with no `)`
# anywhere else on the line is accepted as "the list continues below".
_FUNC_LITERAL_START_RE = re.compile(
    r"^("
    r"async\s+function\b"
    r"|function\b"
    r"|async\s*\([^()]*\)\s*(:[^=]+)?=>"
    r"|\([^()]*\)\s*(:[^=]+)?=>"
    r"|async\s+[A-Za-z_$][\w$]*\s*=>"
    r"|[A-Za-z_$][\w$]*\s*=>"
    r"|async\s*\([^()]*$"
    r"|\([^()]*$"
    r")"
)


def _locate_literal_line(lines: List[str], reported_start: int, start_column: Optional[int]) -> int:
    """
    Off-by-one-correct an ANONYMOUS item's start_line: there's no name to
    search for, but span.start_column still says exactly where the
    function literal begins, so this confirms real function-literal
    syntax actually starts there on each nearby candidate line (see
    _FUNC_LITERAL_START_RE) instead.

    Factored out of _locate_declaration_line so _infer_symbol_name can
    call it directly: naming runs (via _ensure_named) BEFORE
    _corrected_bounds does its official correction pass, so if naming
    only had the raw, uncorrected line to read, it would be reading the
    wrong line too (confirmed: this is what made the very first version
    of this fix still misname a `.then()`/`Promise.all()` callback after
    its enclosing call -- the prefix text was being sliced from one line
    too early).
    """

    if start_column is None:
        return reported_start

    for candidate in (reported_start, reported_start + 1, reported_start - 1):
        idx = candidate - 1
        if 0 <= idx < len(lines) and start_column <= len(lines[idx]) \
                and _FUNC_LITERAL_START_RE.match(lines[idx][start_column:]):
            return candidate

    return reported_start


def _locate_declaration_line(
    lines: List[str],
    name: Optional[str],
    reported_start: int,
    start_column: Optional[int] = None,
) -> int:
    """
    tree-sitter-language-pack's structure spans have been observed to be
    off by one for some node/language combinations (verified: a plain
    Python class with no decorator reports start_line one less than its
    actual `class ...:` line). Rather than trust the raw number, confirm
    it against the source: look for the line that actually contains the
    item's name within a small window around the reported start, and use
    that as the true declaration line.

    Falls back to the column-based literal check (_locate_literal_line)
    when there's no name, OR when a given name simply isn't found nearby
    -- which happens for a callback's synthetic, made-up name (e.g.
    "then_callback_L71"), since that string was never really in the
    source to begin with and a text search for it can never succeed.
    """

    if name:
        name_re = re.compile(rf"\b{re.escape(name)}\b")
        for candidate in (reported_start, reported_start + 1, reported_start - 1):
            idx = candidate - 1
            if 0 <= idx < len(lines) and name_re.search(lines[idx]):
                return candidate

    return _locate_literal_line(lines, reported_start, start_column)


def _extend_over_decorators(lines: List[str], declaration_line: int) -> int:
    """Walk upward over contiguous decorator/attribute/annotation lines
    (`@foo`, `#[attr]`, `[Attribute]`) directly above the declaration."""

    idx = declaration_line - 1
    j = idx - 1

    while j >= 0:
        stripped = lines[j].strip()
        if stripped and _DECORATOR_LINE_RE.match(stripped):
            idx = j
            j -= 1
            continue
        break

    return idx + 1


def _corrected_bounds(items, lines: List[str]) -> List[tuple]:
    """
    Compute a verified (item, start_line, end_line, replaceable) tuple
    for every item in a sibling list. Start is confirmed against the
    source directly; end is derived from the library's (also off-by-one)
    end_line and then clamped so it can never run into the next
    sibling's confirmed start.

    `replaceable` is False when this item's own true closing line and
    the next sibling's true opening line are the SAME physical line --
    a chained call like `}).finally(() => {` puts one callback's real
    closing `}` and the next callback's opening on one line. Whole-line
    replacement (see code_writer.py's replace_symbol) can only ever
    replace complete lines, so it cannot isolate either callback there
    without destroying the other's half of that shared line. Clamping
    the end down to avoid the overlap -- the previous behavior -- just
    moves the corruption instead of preventing it: it silently drops
    this item's own real closing token, so a replace leaves an orphaned/
    duplicate closing paren behind (confirmed: this is what actually
    corrupted a real file). Flagging both sides unsafe instead lets
    find_symbol_bounds refuse the match and fall back to a safe append.
    """

    starts = [
        _extend_over_decorators(
            lines,
            _locate_declaration_line(lines, item.name, item.span.start_line, item.span.start_column),
        )
        for item in items
    ]

    n = len(items)
    raw_ends = [max(item.span.end_line + 1, starts[i]) for i, item in enumerate(items)]
    conflicts = [False] * n

    for i in range(n - 1):
        if raw_ends[i] >= starts[i + 1]:
            conflicts[i] = True
            conflicts[i + 1] = True

    bounds = []
    for i, item in enumerate(items):
        if conflicts[i] or i + 1 >= n:
            end = raw_ends[i]
        else:
            end = min(raw_ends[i], starts[i + 1] - 1)
        bounds.append((item, starts[i], max(end, starts[i]), not conflicts[i]))

    return bounds


# ---------------------------------------------------------------------------
# Name recovery (tree-sitter-language-pack has been observed to leave
# `.name` as None for some languages -- confirmed: every C++ function and
# class -- even though `.span`/`.kind` are populated correctly. Silently
# filtering those out (the original behavior) drops real code entirely,
# and doesn't trigger the "zero structure" fallback either, since
# result.structure itself isn't empty -- it just produces zero *named*
# items. A C++ file with two real top-level functions was chunked to
# nothing by this before the fix below.
# ---------------------------------------------------------------------------

class _NamedItem:
    """A tree-sitter structure item, guaranteed to have a name."""

    __slots__ = ("name", "kind", "span", "children")

    def __init__(self, name, kind, span, children):
        self.name = name
        self.kind = kind
        self.span = span
        self.children = children


_FUNC_NAME_RE = re.compile(r"([A-Za-z_]\w*)\s*\(")
_TYPE_NAME_RE = re.compile(r"\b(?:class|struct|interface|enum|union|trait)\s+([A-Za-z_]\w*)")

# The three ways an anonymous function LITERAL's own start_column prefix
# (everything on its true declaration line before the literal begins)
# says what it actually is -- checked in this order, most-certain first:
#   1. a real assignment:      const NAME = (...) => {...}
#   2. an argument to a call:  someCall(..., (...) => {...})
#   3. an object/class member: NAME: (...) => {...}   /   NAME = (...) => {...}
_ASSIGN_NAME_RE = re.compile(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=]*)?=\s*$")
_CALL_ARG_NAME_RE = re.compile(
    r"([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)(?:<[^<>()]*>)?\s*\(\s*(?:[^()]*,\s*)?$"
)
_PROPERTY_NAME_RE = re.compile(r"([A-Za-z_$][\w$]*)\s*[:=]\s*$")


def _infer_symbol_name(
    kind: str,
    lines: List[str],
    start_line: int,
    end_line: int,
    start_column: Optional[int] = None,
) -> str:
    """
    Recover a plausible name for an item tree-sitter left unnamed.

    An anonymous function LITERAL (an arrow function, a bare `function`
    expression) is checked first, using start_column: the text
    immediately BEFORE its own start position on its true declaration
    line says, unambiguously, whether it's a real assignment, a call
    argument, or an object/class member (see the regexes above) --
    because that text is exactly everything before this literal and
    nothing after it.

    The previous approach -- searching a multi-line window starting at
    the item for the first `identifier(` pattern -- routinely found the
    wrong identifier for exactly this case: in `.then(() => {...})`, that
    window contains `.then(`, so the callback got named "then" (its
    ENCLOSING call's name, not its own); `Promise.all(...)` produced
    "all" the same way. Both were confirmed, real mistargeting bugs in
    code_writer_tool's replace_symbol, which trusts an exact name match
    against the real file (see find_symbol_bounds) -- a callback
    misnamed after its enclosing call collides with, or gets confused
    for, that call's own name.

    Still falls back to the old header-window regex, then a positional
    name, for anything the column-based checks don't recognize (e.g. a
    plain named declaration in a language where tree-sitter simply left
    `.name` empty -- see the note above _NamedItem) so a found chunk is
    never dropped outright just because a name couldn't be determined.
    """

    if start_column is not None:
        # This runs (via _ensure_named) BEFORE _corrected_bounds' official
        # correction pass, so start_line is still the raw, possibly
        # off-by-one line straight from tree-sitter -- correct it the
        # same way _corrected_bounds itself will, or the prefix below
        # gets sliced from the wrong line entirely.
        true_start = _locate_literal_line(lines, start_line, start_column)

        if 0 <= true_start - 1 < len(lines):
            decl_line = lines[true_start - 1]
            if 0 <= start_column <= len(decl_line):
                prefix = decl_line[:start_column]

                if not prefix.strip():
                    # The literal sits at the very start of its own
                    # line -- e.g. a callback whose `(params) =>` is on
                    # the line right after the call that takes it,
                    # `someCall(\n  (params) => {...}\n)`. There's
                    # nothing useful on this line before it, so look at
                    # a small, bounded window of the lines just above
                    # instead of giving up outright.
                    lookback = min(3, true_start - 1)
                    prefix = " ".join(lines[true_start - 1 - lookback: true_start - 1] + [prefix])

                assign_match = _ASSIGN_NAME_RE.search(prefix)
                if assign_match:
                    return assign_match.group(1)

                call_match = _CALL_ARG_NAME_RE.search(prefix)
                if call_match:
                    callee = call_match.group(1).rsplit(".", 1)[-1]
                    # Line-suffixed, unlike the other two matches above:
                    # this name is made up (the literal has no real
                    # identity of its own), so nothing else guarantees a
                    # repo with several `.then(`/`.map(` callbacks won't
                    # produce two chunks sharing one inferred name --
                    # true_start always does.
                    return f"{callee}_callback_L{true_start}"

                prop_match = _PROPERTY_NAME_RE.search(prefix)
                if prop_match:
                    return prop_match.group(1)

    header_end = min(start_line + 2, end_line)
    header = "\n".join(lines[start_line - 1: header_end])

    type_match = _TYPE_NAME_RE.search(header)
    if type_match:
        return type_match.group(1)

    func_match = _FUNC_NAME_RE.search(header)
    if func_match:
        return func_match.group(1)

    return f"{kind.lower()}_L{start_line}"


def _ensure_named(item, lines: List[str]):
    """Recursively wrap a structure item (and its children) so every item
    with a real span has a usable name, real or inferred."""

    children = [_ensure_named(c, lines) for c in item.children if c.span is not None]

    if item.span is None:
        return _NamedItem(item.name, item.kind, item.span, children)

    name = item.name or _infer_symbol_name(
        str(item.kind), lines, item.span.start_line, item.span.end_line, item.span.start_column,
    )

    return _NamedItem(name, item.kind, item.span, children)


def _flatten_structure(
    items,
    lines: List[str],
    repo_name: str,
    relative_path: str,
    language: str,
) -> List[Dict]:
    """
    Recursively turn a tree-sitter structure tree into one chunk per node:
    a leaf (function/method/...) gets its full span; a container
    (class/struct/interface/...) gets only the lines it directly owns
    (its signature/docstring/fields), not its children's bodies -- those
    are chunked separately by the recursive call.
    """

    chunks: List[Dict] = []

    named_items = [item for item in items if item.name and item.span is not None]

    for item, start, end, replaceable in _corrected_bounds(named_items, lines):

        children = [c for c in item.children if c.name and c.span is not None]
        kind = str(item.kind)

        if children:
            child_bounds = _corrected_bounds(children, lines)
            first_child_start = min(cs for _, cs, _, _ in child_bounds)
            last_child_end = max(ce for _, _, ce, _ in child_bounds)

            header_lines = lines[start - 1: first_child_start - 1]
            trailer_lines = lines[last_child_end: end]
            header_content = "\n".join(header_lines + trailer_lines).strip("\n")

            if header_content.strip():
                chunks.append(_make_chunk(
                    repo_name, relative_path, language, kind,
                    chunk_name=item.name, start_line=start, end_line=end,
                    content=header_content, replaceable=replaceable,
                ))

            chunks.extend(_flatten_structure(children, lines, repo_name, relative_path, language))

        else:
            content = "\n".join(lines[start - 1: end])

            if _is_stub(content):
                continue

            chunks.append(_make_chunk(
                repo_name, relative_path, language, kind,
                chunk_name=item.name, start_line=start, end_line=end,
                replaceable=replaceable,
                content=content,
            ))

    return chunks


def _imports_chunk(result, lines: List[str], repo_name: str, relative_path: str, language: str) -> Optional[Dict]:
    """
    tree-sitter-language-pack's `imports` list includes EVERY import in the
    file, including a lazy/local one nested inside a function body (e.g.
    `from transformers import ...` inside a `from_pretrained` classmethod,
    only reached conditionally). Folding one of those into this chunk would
    stretch its start_line/end_line across the whole file -- a hundreds-of-
    lines "imports" citation for what's actually a handful of short lines.
    Keep only genuine top-level imports: no leading indentation.
    """

    def _is_top_level(imp) -> bool:
        # Import spans have the same off-by-one quirk as structure spans
        # (see _locate_declaration_line) -- confirm against the source
        # text itself rather than trusting the reported line directly,
        # or an indented import one line off from its "if" guard reads
        # as unindented and slips through as if it were top-level.
        source = (imp.source or "").strip()
        for candidate in (imp.span.start_line, imp.span.start_line + 1, imp.span.start_line - 1):
            idx = candidate - 1
            if 0 <= idx < len(lines) and lines[idx].strip() == source:
                return not lines[idx][:1].isspace()
        return not lines[imp.span.start_line - 1][:1].isspace()

    imports = [
        imp for imp in (result.imports or [])
        if imp.span is not None
        and 0 <= imp.span.start_line - 1 < len(lines)
        and _is_top_level(imp)
    ]

    if not imports:
        return None

    imports.sort(key=lambda imp: imp.span.start_line)
    import_lines = [imp.source for imp in imports if imp.source]

    if not import_lines:
        return None

    start = imports[0].span.start_line
    end = imports[-1].span.end_line

    return _make_chunk(
        repo_name, relative_path, language, "imports",
        chunk_name="imports", start_line=start, end_line=end,
        content="\n".join(import_lines),
    )


def check_syntax_errors(language: str, content: str) -> List[str]:
    """
    Real, tree-sitter-backed syntax validation for a piece of generated
    code -- used by code_writer.py to confirm a write didn't produce
    invalid syntax before reporting success. This only catches parse-level
    errors (an unbalanced brace, a `case` clause with no enclosing switch)
    -- it's the same grammar chunk_code_file's own parse relies on, not a
    real type checker, so an undefined name or wrong type is invisible to
    it. Returns an empty list on any internal failure (unsupported
    language, parser error) rather than raising -- a failed syntax check
    must never be the reason a legitimate write gets blocked.
    """
    try:
        result = tlp.process(content, tlp.ProcessConfig(
            language=language, structure=False, imports=False, exports=False,
            diagnostics=True,
        ))
    except Exception:
        return []

    return [
        d.message for d in result.diagnostics
        if d.severity == tlp.DiagnosticSeverity.ERROR
    ]


def chunk_code_file(file_path: Path, repo_root: Path) -> List[Dict]:
    """
    Split a source file into semantic chunks using tree-sitter.
    """

    relative_path = str(file_path.relative_to(repo_root))
    repo_name = repo_root.name
    lang_name = tree_sitter_language_name(file_path)

    content = file_path.read_text(encoding="utf-8", errors="ignore")

    if lang_name is None:
        return _fixed_size_fallback(repo_name, relative_path, "unknown", content)

    try:
        result = tlp.process(content, tlp.ProcessConfig(
            language=lang_name, structure=True, imports=True,
        ))
    except Exception as exc:
        log_warning(f"tree-sitter parse failed for {relative_path} ({exc}); using fixed-size fallback")
        return _fixed_size_fallback(repo_name, relative_path, lang_name, content)

    if not result.structure:
        return _fixed_size_fallback(repo_name, relative_path, lang_name, content)

    lines = content.splitlines()
    chunks = []

    imports_chunk = _imports_chunk(result, lines, repo_name, relative_path, lang_name)
    if imports_chunk is not None:
        chunks.append(imports_chunk)

    structure = [_ensure_named(item, lines) for item in result.structure if item.span is not None]

    chunks.extend(_flatten_structure(structure, lines, repo_name, relative_path, lang_name))

    # Re-split anything that still exceeds the embedder's effective window.
    final_chunks = []
    for chunk in chunks:
        final_chunks.extend(_split_if_oversized(chunk))

    return final_chunks


# ---------------------------------------------------------------------------
# Doc chunking (markdown headings)
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+\S")


def chunk_doc_file(file_path: Path, repo_root: Path) -> List[Dict]:
    relative_path = str(file_path.relative_to(repo_root))
    repo_name = repo_root.name

    content = file_path.read_text(encoding="utf-8", errors="ignore")
    lines = content.splitlines()

    if not lines:
        return []

    heading_starts = [i for i, line in enumerate(lines) if _HEADING_RE.match(line)]

    if not heading_starts:
        if _count_tokens(content) <= config.MAX_CHUNK_TOKENS:
            return [_make_chunk(
                repo_name, relative_path, "markdown", "doc_section",
                chunk_name=file_path.stem, start_line=1, end_line=len(lines),
                content=content,
            )]
        return _fixed_size_fallback(repo_name, relative_path, "markdown", content, chunk_type="doc_section")

    chunks = []

    # Anything before the first heading (title/intro) is its own section.
    if heading_starts[0] > 0:
        intro = "\n".join(lines[0:heading_starts[0]])
        if intro.strip():
            chunks.append(_make_chunk(
                repo_name, relative_path, "markdown", "doc_section",
                chunk_name="intro", start_line=1, end_line=heading_starts[0],
                content=intro,
            ))

    for i, start in enumerate(heading_starts):
        end = heading_starts[i + 1] if i + 1 < len(heading_starts) else len(lines)
        heading_text = lines[start].lstrip("#").strip() or f"section_{i + 1}"
        section = "\n".join(lines[start:end])

        chunk = _make_chunk(
            repo_name, relative_path, "markdown", "doc_section",
            chunk_name=heading_text, start_line=start + 1, end_line=end,
            content=section,
        )

        if _count_tokens(section) <= config.MAX_CHUNK_TOKENS:
            chunks.append(chunk)
        else:
            chunks.extend(_split_if_oversized(chunk))

    return chunks


# ---------------------------------------------------------------------------
# Config chunking (yaml/json/toml/ini — split by top-level key)
# ---------------------------------------------------------------------------

_TOP_LEVEL_KEY_RE = re.compile(r"^([^\s#;/\[][^\s:=]{0,120})\s*[:=]")
_TOP_LEVEL_TABLE_RE = re.compile(r"^\[[^\]]+\]\s*$")


def chunk_config_file(file_path: Path, repo_root: Path) -> List[Dict]:
    relative_path = str(file_path.relative_to(repo_root))
    repo_name = repo_root.name
    lang = file_path.suffix.lstrip(".").lower() or "config"

    content = file_path.read_text(encoding="utf-8", errors="ignore")
    lines = content.splitlines()

    if not lines:
        return []

    if _count_tokens(content) <= config.MAX_CHUNK_TOKENS:
        return [_make_chunk(
            repo_name, relative_path, lang, "config_section",
            chunk_name=file_path.name, start_line=1, end_line=len(lines),
            content=content,
        )]

    boundaries = [
        i for i, line in enumerate(lines)
        if not line.startswith((" ", "\t")) and (_TOP_LEVEL_KEY_RE.match(line) or _TOP_LEVEL_TABLE_RE.match(line))
    ]

    if not boundaries:
        return _fixed_size_fallback(repo_name, relative_path, lang, content, chunk_type="config_section")

    chunks = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(lines)
        key_match = _TOP_LEVEL_KEY_RE.match(lines[start])
        name = key_match.group(1) if key_match else lines[start].strip("[] \t")
        section = "\n".join(lines[start:end])

        chunk = _make_chunk(
            repo_name, relative_path, lang, "config_section",
            chunk_name=name, start_line=start + 1, end_line=end,
            content=section,
        )
        chunks.extend(_split_if_oversized(chunk))

    return chunks


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_CATEGORY_HANDLERS = {
    "code": chunk_code_file,
    "doc": chunk_doc_file,
    "config": chunk_config_file,
}


def chunk_repository(
    filtered_files: List[Path],
    repo_root: Path,
) -> List[Dict]:
    """
    Chunk every file in a repository that has a known category. One bad
    file logs a warning and is skipped -- it never aborts the whole run.
    """

    repository_chunks: List[Dict] = []

    for file_path in filtered_files:

        category = detect_category(file_path)
        handler = _CATEGORY_HANDLERS.get(category)

        if handler is None:
            continue

        try:
            repository_chunks.extend(handler(file_path, repo_root))
        except Exception as exc:
            log_warning(f"Skipping {file_path.relative_to(repo_root)}: {exc}")
            continue

    return repository_chunks


# ---------------------------------------------------------------------------
# Symbol lookup (used by the code writer tool to safely replace a function
# or class in place, rather than guessing at its boundaries separately)
# ---------------------------------------------------------------------------

def find_symbol_bounds(
    file_path: Path,
    repo_root: Path,
    symbol_name: str,
) -> Optional[Tuple[int, int]]:
    """
    Locate a top-level or nested function/class/method by exact name in a
    source file and return its true (start_line, end_line), 1-based
    inclusive -- reusing the same tree-sitter parse and off-by-one
    correction (_corrected_bounds) that chunk_code_file relies on, so a
    caller replacing this symbol's source lines gets the same boundaries
    the index was built from rather than a second, independently-guessed
    set that could disagree with it.

    Returns None if the file has no parser for its language, parsing
    fails, no symbol with that exact name is found, OR the only match(es)
    by that name are flagged unsafe by _corrected_bounds (their true
    boundaries share a physical line with a sibling's -- see its
    docstring). That last case matters specifically for replace_symbol:
    an unsafe match is real, but whole-line replacement cannot isolate
    it without corrupting the sibling on the shared line, so returning
    None here is what lets the caller fall back to a safe append instead
    (confirmed: this exact scenario -- a `Promise.all(...)` argument
    misidentified as symbol "all" -- previously caused a real file to be
    written with a broken, orphaned closing paren).
    """

    lang_name = tree_sitter_language_name(file_path)
    if lang_name is None:
        return None

    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        result = tlp.process(content, tlp.ProcessConfig(language=lang_name, structure=True))
    except Exception:
        return None

    if not result.structure:
        return None

    lines = content.splitlines()

    def search(items) -> Optional[Tuple[int, int]]:
        named_items = [item for item in items if item.name and item.span is not None]

        for item, start, end, replaceable in _corrected_bounds(named_items, lines):
            if item.name == symbol_name and replaceable:
                return start, end

            found = search(item.children)
            if found is not None:
                return found

        return None

    structure = [_ensure_named(item, lines) for item in result.structure if item.span is not None]

    return search(structure)
