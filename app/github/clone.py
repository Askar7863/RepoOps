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


def _reject_unsafe_clone_path(repo_name: str, clone_path: Path) -> None:
    """
    repo_name comes straight from the last '/'-delimited segment of a
    user-controlled repo_url and is about to be shutil.rmtree'd (to clear
    a stale clone) BEFORE anything is cloned -- so any way for it to
    resolve outside REPO_STORAGE turns a repo-load request into an
    arbitrary-directory-delete. Confirmed as real, not hypothetical:
    a repo_url ending in '/..' makes repo_name == '..', which resolves
    clone_path to the project root itself (deleting app/, venv/, .git/,
    everything); a Windows-backslash-smuggled segment ('..\\..\\..\\x')
    escapes arbitrarily far beyond that, since pathlib treats '\\' as a
    separator on this OS too even though repo_name was only ever split
    on '/'. Checked two ways on purpose: an early reject on the obvious
    patterns for a clear error message, and an independent resolved-path
    containment check as the real safety net, so a future change to
    extract_repo_name can't silently reopen this.
    """

    if not repo_name or repo_name in {".", ".."} or "/" in repo_name or "\\" in repo_name or ":" in repo_name:
        raise ValueError("Could not derive a safe local directory name from this URL.")

    storage_root = REPO_STORAGE.resolve()
    resolved = clone_path.resolve()
    if resolved != storage_root and storage_root not in resolved.parents:
        raise ValueError("Could not derive a safe local directory name from this URL.")


def _explain_clone_failure(repo_url: str, stderr: str) -> str:
    """
    Turn git's raw stderr into a specific, actionable reason instead of
    dumping the technical text verbatim. 'Repository not found' is the
    single message GitHub returns for BOTH a nonexistent repo AND a
    private one accessed with no credentials -- confirmed against a real
    nonexistent repo, and this is GitHub's own documented, deliberate
    behavior (it never reveals that a private repo exists to someone
    without access to it) -- so this app genuinely cannot tell those two
    cases apart from git's output, and the message says so honestly
    rather than guessing at only one of them.
    """

    text = stderr or ""
    lowered = text.lower()

    if "repository not found" in lowered or "not found" in lowered:
        return (
            f"'{repo_url}' could not be found. This means one of: the URL is "
            "mistyped, the repository doesn't exist, or it's private -- this "
            "app only clones public repositories over HTTPS with no "
            "credentials, and GitHub deliberately returns the exact same "
            "'not found' response for a private repo as for one that "
            "doesn't exist at all, so there's no way to tell those two "
            "apart from here. Double-check the URL, or make the repository "
            "public if it's yours."
        )

    if "could not read username" in lowered or "terminal prompts disabled" in lowered or "authentication failed" in lowered:
        return (
            f"'{repo_url}' requires authentication to access, which this app "
            "does not support -- it only clones public repositories with no "
            "credentials. Make the repository public, or use a public repo."
        )

    if "could not resolve host" in lowered or "failed to connect" in lowered or "connection timed out" in lowered:
        return (
            "Could not reach GitHub over the network. Check the server's "
            f"internet connection and try again. (git said: {text.strip()})"
        )

    # Anything else genuinely unrecognized -- surface git's own text
    # rather than hide it behind a generic message.
    return f"git could not clone '{repo_url}': {text.strip() or 'no error output was produced.'}"


def clone_repository(repo_url: str):

    repo_name = extract_repo_name(repo_url)

    clone_path = REPO_STORAGE / repo_name

    _reject_unsafe_clone_path(repo_name, clone_path)

    if clone_path.exists():
        shutil.rmtree(clone_path, onerror=_clear_readonly_and_retry)

    command = [
        "git",
        "clone",
        repo_url,
        str(clone_path)
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        raise Exception(
            f"Cloning '{repo_url}' took longer than 120 seconds and was "
            "aborted. This usually means the repository is unusually large, "
            "or the server's network connection to GitHub is slow/unstable. "
            "Try again, or use a smaller repository."
        ) from exc

    # If git reports an error AND the repository wasn't created,
    # treat it as a real failure.
    if result.returncode != 0 and not clone_path.exists():
        raise Exception(_explain_clone_failure(repo_url, result.stderr))

    # A valid Git repository should contain a .git directory.
    if not (clone_path / ".git").exists():
        raise Exception(
            f"Repository checkout of '{repo_url}' did not complete "
            "successfully (no .git directory was produced). This is "
            "unexpected -- git reported success but the clone looks "
            "incomplete; try again."
        )

    return clone_path


def delete_repository_clone(repo_name: str) -> None:
    """
    Remove a previously cloned repository's directory from disk. Reuses
    the same safety check and readonly-retry handling clone_repository
    itself uses -- this deletes real files on a storage-constrained
    deployment, so it must be exactly as careful about what path it's
    allowed to touch as the clone path already is.

    A repo that was never actually cloned (directory doesn't exist) is
    not an error here -- nothing to remove.
    """

    clone_path = REPO_STORAGE / repo_name

    _reject_unsafe_clone_path(repo_name, clone_path)

    if clone_path.exists():
        shutil.rmtree(clone_path, onerror=_clear_readonly_and_retry)
