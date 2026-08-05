"""
scanner.py

Repository Scanner for RepoOps.

Responsibility:
---------------
Traverse a cloned GitHub repository and return every file found.

This module DOES NOT:
- Filter files
- Read file contents
- Detect programming languages
- Chunk source code
- Generate embeddings

Those responsibilities belong to later stages of the RAG pipeline.
"""

from pathlib import Path


def scan_repository(repository_path: Path) -> list[Path]:
    """
    Scan an entire repository and return all files.

    Parameters
    ----------
    repository_path : Path
        Local path of the cloned repository.

    Returns
    -------
    list[Path]
        List containing every file inside the repository.

    Raises
    ------
    FileNotFoundError
        If the repository path does not exist.

    NotADirectoryError
        If the supplied path is not a directory.
    """

    # ----------------------------
    # Validate repository
    # ----------------------------
    if not repository_path.exists():
        raise FileNotFoundError(
            f"Repository not found: {repository_path}"
        )

    if not repository_path.is_dir():
        raise NotADirectoryError(
            f"{repository_path} is not a directory."
        )

    # ----------------------------
    # Scan repository
    # ----------------------------
    files = []

    for path in repository_path.rglob("*"):

        if path.is_file():
            files.append(path)

    # Sort paths for deterministic output
    files.sort()

    return files