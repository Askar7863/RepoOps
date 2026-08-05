import os
import stat
import subprocess
import shutil
from pathlib import Path

#################################################
# CONFIG
#################################################

REPO_STORAGE = Path(__file__).resolve().parents[2] / "repos"
REPO_STORAGE.mkdir(parents=True, exist_ok=True)


#################################################
# UTILITIES
#################################################

def extract_repo_name(repo_url: str):

    repo_name = repo_url.rstrip("/").split("/")[-1]

    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    return repo_name


def _clear_readonly_and_retry(func, path, exc_info):
    """
    shutil.rmtree error handler: git marks its packed object files
    (.git/objects/pack/*.idx, *.pack) read-only, which raises
    PermissionError/WinError 5 on Windows during a plain rmtree. Clear the
    read-only bit and retry the same operation once.
    """

    os.chmod(path, stat.S_IWRITE)
    func(path)


def clone_repository(repo_url: str):

    repo_name = extract_repo_name(repo_url)

    clone_path = REPO_STORAGE / repo_name

    if clone_path.exists():
        shutil.rmtree(clone_path, onerror=_clear_readonly_and_retry)

    command = [
        "git",
        "clone",
        repo_url,
        str(clone_path)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=120,
    )

    # If git reports an error AND the repository wasn't created,
    # treat it as a real failure.
    if result.returncode != 0 and not clone_path.exists():
        raise Exception(result.stderr)

    # A valid Git repository should contain a .git directory.
    if not (clone_path / ".git").exists():
        raise Exception(
            "Repository checkout did not complete successfully."
        )

    return clone_path
