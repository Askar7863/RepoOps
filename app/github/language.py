"""
language.py

Programming Language & Content-Category Detector for RepoOps.

Responsibility
--------------
Determine the programming language of each repository file, and classify
it into one of three chunking categories used by chunker.py:

    "code"   — has a tree-sitter grammar, chunked at function/class
               boundaries (see app/github/chunker.py::chunk_code_file)
    "doc"    — markdown/text, chunked by heading (chunk_doc_file)
    "config" — yaml/json/toml/ini, chunked by top-level key (chunk_config_file)

A file whose extension isn't recognized in either map returns
category=None — filter.py may still keep it as "useful", but chunker.py
skips it rather than guessing.

This module DOES NOT:
- Read file contents
- Chunk source code
- Generate embeddings
"""

from pathlib import Path

from tree_sitter_language_pack import detect_language_from_path


# ---------------------------------------------------------------------------
# Human-readable display names (unrelated to tree-sitter's internal grammar
# names, which are lowercase identifiers like "csharp" or "cpp").
# ---------------------------------------------------------------------------

DISPLAY_NAMES = {

    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript React",
    ".jsx": "JavaScript React",
    ".java": "Java",
    ".cpp": "C++",
    ".c": "C",
    ".h": "C/C++ Header",
    ".cs": "C#",
    ".go": "Go",
    ".rs": "Rust",
    ".php": "PHP",
    ".rb": "Ruby",
    ".kt": "Kotlin",
    ".swift": "Swift",

    ".html": "HTML",
    ".css": "CSS",
    ".scss": "SCSS",

    ".json": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".toml": "TOML",
    ".ini": "INI",

    ".md": "Markdown",
    ".txt": "Text",
}

SPECIAL_FILES = {
    "Dockerfile": "Docker",
    "Makefile": "Make",
    "README": "Markdown",
    "LICENSE": "Text",
}


# ---------------------------------------------------------------------------
# Chunking categories
# ---------------------------------------------------------------------------

DOC_EXTENSIONS = {".md", ".txt"}

CONFIG_EXTENSIONS = {".json", ".yaml", ".yml", ".toml", ".ini"}

# Everything else with a tree-sitter grammar (checked via detect_language_from_path)
# is treated as "code". .html/.css/.scss have grammars too but no meaningful
# function/class structure for our purposes — treat them as docs so they're
# still indexed whole/paragraph-split rather than silently dropped.
DOC_LIKE_MARKUP_EXTENSIONS = {".html", ".css", ".scss"}

# tree-sitter-language-pack has a grammar for these (for editor syntax
# highlighting), so detect_language_from_path below would otherwise call
# them "code" -- but a grammar existing doesn't mean the format has
# independently-appendable function/class symbols the way real source
# code does. Confirmed as a real, harmful consequence, not theoretical:
# code_writer_tool trusts category=="code" to mean "safe to append a
# generated code block to," and appended a Python function directly into
# a real .csv file's rows, corrupting it. These are plain tabular data,
# not code, regardless of what tree-sitter can parse.
NOT_CODE_EXTENSIONS = {".csv", ".tsv"}


def detect_language(file: Path) -> str:
    """
    Human-readable language name for display/metadata purposes.
    """

    if file.name in SPECIAL_FILES:
        return SPECIAL_FILES[file.name]

    return DISPLAY_NAMES.get(file.suffix.lower(), "Unknown")


def detect_category(file: Path) -> str | None:
    """
    Classify a file into a chunking category: "code", "doc", "config", or
    None if chunker.py has no strategy for it.
    """

    suffix = file.suffix.lower()

    if file.name in SPECIAL_FILES:
        return "doc" if SPECIAL_FILES[file.name] in {"Markdown", "Text"} else "code"

    if suffix in CONFIG_EXTENSIONS:
        return "config"

    if suffix in DOC_EXTENSIONS or suffix in DOC_LIKE_MARKUP_EXTENSIONS:
        return "doc"

    if suffix in NOT_CODE_EXTENSIONS:
        return None

    # tree-sitter-language-pack knows ~300 grammars by extension; anything it
    # recognizes gets AST-aware code chunking.
    if detect_language_from_path(str(file)) is not None:
        return "code"

    return None


def tree_sitter_language_name(file: Path) -> str | None:
    """
    The tree-sitter grammar name for this file (e.g. "python", "csharp"),
    or None if unsupported.
    """

    return detect_language_from_path(str(file))
