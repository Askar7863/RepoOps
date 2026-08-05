"""
logger.py

Simple logging utilities for RepoOps.
"""

from datetime import datetime


def log_step(title: str):
    """
    Print a section heading.
    """

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def log_info(message: str):
    """
    Print an informational message.
    """

    current_time = datetime.now().strftime("%H:%M:%S")

    print(f"[INFO {current_time}] {message}")


def log_success(message: str):
    """
    Print a success message.
    """

    current_time = datetime.now().strftime("%H:%M:%S")

    print(f"[SUCCESS {current_time}] {message}")


def log_error(message: str):
    """
    Print an error message.
    """

    current_time = datetime.now().strftime("%H:%M:%S")

    print(f"[ERROR {current_time}] {message}")


def log_warning(message: str):
    """
    Print a warning message (e.g. a fallback path was taken).
    """

    current_time = datetime.now().strftime("%H:%M:%S")

    print(f"[WARN {current_time}] {message}")


def log_count(title: str, value: int):
    """
    Print numeric statistics.
    """

    print(f"{title:<30}: {value}")


def log_list(title: str, items, limit: int = 10):
    """
    Print first few items from a list.
    """

    print(f"\n{title}")

    for item in items[:limit]:
        print(f"  • {item}")