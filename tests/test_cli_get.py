"""CLI tests for kdb get subcommand."""

from __future__ import annotations

import subprocess
import sys


def test_cli_get_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "knowledge", "get", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "hash_prefix" in result.stdout


def test_cli_get_no_index(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "knowledge", "get", "a1b2c3d4e5", "-c", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
