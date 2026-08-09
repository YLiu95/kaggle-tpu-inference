from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from ktpu.errors import CheckpointError


@dataclass(frozen=True)
class Checkpoint:
    repo: Path
    commit: str
    remote: str


def _git(repo: Path, *args: str, timeout: int = 60) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        message = getattr(exc, "stderr", "") or str(exc)
        raise CheckpointError(f"Git checkpoint check failed: {message.strip()}") from exc
    return result.stdout.strip()


def find_repo_root(start: Path | None = None) -> Path | None:
    path = (start or Path.cwd()).resolve()
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return Path(output)


def run_lightweight_tests(repo: Path) -> None:
    env = dict(os.environ)
    source = str(repo / "src")
    env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-q",
            ],
            cwd=repo,
            env=env,
            check=True,
            timeout=180,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CheckpointError(
            "Lightweight tests failed; risky operation was not started."
        ) from exc


def _reject_credentialed_remote(remote: str) -> None:
    if remote.startswith(("http://", "https://")):
        parsed = urlsplit(remote)
        if parsed.username or parsed.password:
            raise CheckpointError(
                "The Git remote contains credentials. Replace it with a clean URL "
                "before continuing."
            )


def verify_checkpoint(
    start: Path | None = None,
    *,
    run_tests: bool = True,
) -> Checkpoint:
    repo = find_repo_root(start)
    if repo is None:
        raise CheckpointError(
            "No Git repository was found. Risky operations require a pushed checkpoint."
        )
    if run_tests:
        run_lightweight_tests(repo)
    status = _git(repo, "status", "--porcelain", "--untracked-files=normal")
    if status:
        raise CheckpointError(
            "Working tree is not clean. Commit and push every change before continuing."
        )
    branch = _git(repo, "branch", "--show-current")
    if branch != "main":
        raise CheckpointError(
            f"Risky operations must run from main; current branch is {branch!r}."
        )
    remote = _git(repo, "remote", "get-url", "origin")
    _reject_credentialed_remote(remote)
    local = _git(repo, "rev-parse", "HEAD")
    remote_line = _git(repo, "ls-remote", "--exit-code", "origin", "refs/heads/main")
    remote_sha = remote_line.split()[0] if remote_line else ""
    if local != remote_sha:
        raise CheckpointError(
            f"Local HEAD {local} does not match origin/main {remote_sha or 'missing'}."
        )
    return Checkpoint(repo=repo, commit=local, remote=remote)

