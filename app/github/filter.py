"""
filter.py

Repository File Filter for RepoOps.

Responsibility
--------------
Filter repository files and keep only useful source code and
documentation files for the RAG indexing pipeline.

This module DOES NOT:
- Read file contents
- Detect programming language
- Chunk source code
- Generate embeddings
"""

from pathlib import Path


# --------------------------------------------------
# Directories to ignore completely
# --------------------------------------------------

IGNORED_DIRECTORIES = {
    ".git",
    ".github",
    ".idea",
    ".vscode",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
    "env",
    "dist",
    "build",
    "target",
    "coverage",
    ".cache",
    ".next",
    "out",
}


# --------------------------------------------------
# Allowed source file extensions
# --------------------------------------------------

ALLOWED_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".cs",
    ".go",
    ".rs",
    ".php",
    ".rb",
    ".kt",
    ".swift",

    ".html",
    ".css",
    ".scss",

    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",

    ".md",
    ".txt",
}


# Files without extensions that we still want
ALLOWED_FILENAMES = {
    "Dockerfile",
    "Makefile",
    "README",
    "LICENSE",
}


def filter_repository_files(files: list[Path]) -> list[Path]:
    """
    Filter repository files.

    Parameters
    ----------
    files : list[Path]
        Files returned from scanner.py

    Returns
    -------
    list[Path]
        Only useful files for indexing.
    """

    filtered_files = []

    for file in files:

        # Ignore files inside unwanted directories
        if any(part in IGNORED_DIRECTORIES for part in file.parts):
            continue

        # Keep files with supported extensions
        if file.suffix.lower() in ALLOWED_EXTENSIONS:
            filtered_files.append(file)
            continue

        # Keep important extension-less files
        if file.name in ALLOWED_FILENAMES:
            filtered_files.append(file)

    return filtered_files