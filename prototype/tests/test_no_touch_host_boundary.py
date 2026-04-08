"""No-touch host-boundary compliance test for adapter directory."""

from __future__ import annotations

import subprocess
from shutil import which
from pathlib import Path

import pytest


def _repo_root_from_test_file() -> Path:
    # prototype/tests/test_no_touch_host_boundary.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def test_host_adapter_tree_has_no_local_modifications() -> None:
    if which("git") is None:
        pytest.skip("git is unavailable; cannot run no-touch host diff check.")
    if not (_repo_root_from_test_file() / ".git").exists():
        pytest.skip("Not a git repository; cannot run no-touch host diff check.")

    cmd = [
        "git",
        "status",
        "--porcelain",
        "--",
        "prototype/adapter/WACV2026-Oral-ITSELF",
    ]
    result = subprocess.run(
        cmd,
        cwd=_repo_root_from_test_file(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            "Failed to run git status for host-boundary verification: "
            f"returncode={result.returncode}, stderr={result.stderr.strip()!r}"
        )

    modified = result.stdout.strip()
    assert modified == "", (
        "Host adapter tree must remain untouched, found modifications:\n"
        f"{modified}"
    )
