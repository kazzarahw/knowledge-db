"""CLI smoke tests — subprocess-based, no model loading."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_cli_help() -> None:
    """--help exits 0 and prints description."""
    result = subprocess.run(
        [sys.executable, "-m", "knowledge", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "Pentest knowledge database" in result.stdout


def test_cli_search_help() -> None:
    """search --help exits 0 and shows search options."""
    result = subprocess.run(
        [sys.executable, "-m", "knowledge", "search", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--top-k" in result.stdout


def test_cli_index_help() -> None:
    """index --help exits 0 and shows index options."""
    result = subprocess.run(
        [sys.executable, "-m", "knowledge", "index", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--force" in result.stdout


def test_cli_fetch_help() -> None:
    """fetch --help exits 0 and shows fetch options."""
    result = subprocess.run(
        [sys.executable, "-m", "knowledge", "fetch", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--only" in result.stdout


def test_cli_list_sources_bad_yaml(tmp_path: Path) -> None:
    """list-sources with invalid YAML exits 1."""
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "sources.yaml").write_text("invalid: [yaml: bad")
    result = subprocess.run(
        [sys.executable, "-m", "knowledge", "-c", str(cfg), "list-sources"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Config error" in result.stdout


def test_cli_search_missing_query() -> None:
    """search with no query exits 1."""
    result = subprocess.run(
        [sys.executable, "-m", "knowledge", "search"], capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "search query is required" in result.stdout


def test_cli_search_bad_top_k() -> None:
    """search with --top-k 0 exits 1."""
    result = subprocess.run(
        [sys.executable, "-m", "knowledge", "search", "foo", "--top-k", "0"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "--top-k must be" in result.stderr or "--top-k must be" in result.stdout
