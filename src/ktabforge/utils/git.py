from __future__ import annotations

import subprocess
from pathlib import Path


def collect_git_info(cwd: str | Path | None = None) -> dict[str, str | None]:
    root = Path(cwd) if cwd is not None else Path.cwd()
    return {
        "commit": _git(["rev-parse", "HEAD"], root),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], root),
        "is_dirty": _git(["status", "--porcelain"], root) not in (None, ""),
    }


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()
