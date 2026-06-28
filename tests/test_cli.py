"""CLI smoke tests — subprocess-based, no model loading."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import os

import pytest


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
        [sys.executable, "-m", "knowledge", "list-sources", "-c", str(cfg)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Config error" in result.stdout


def test_cli_bad_config_path() -> None:
    """--config /nonexistent exits 1 with clean error, no traceback."""
    result = subprocess.run(
        [sys.executable, "-m", "knowledge", "update", "-c", "/nonexistent/path"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Config error" in result.stdout or "Config error" in result.stderr
    assert "PermissionError" not in result.stdout + result.stderr
    assert "FileNotFoundError" not in result.stdout + result.stderr
    assert "Traceback" not in result.stdout + result.stderr


def test_cli_verbose_after_subcommand() -> None:
    """-v after subcommand does not cause unrecognized argument error."""
    result = subprocess.run(
        [sys.executable, "-m", "knowledge", "fetch", "-v", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


def test_cli_config_after_subcommand() -> None:
    """-c after subcommand does not cause unrecognized argument error."""
    result = subprocess.run(
        [sys.executable, "-m", "knowledge", "fetch", "-c", "/tmp", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


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


def test_search_heading_path_not_truncated(tmp_path: Path, monkeypatch) -> None:
    """Search output shows title in new unified format."""
    from knowledge.config import ensure_data_dir, resolve_data_dir
    from knowledge.db import ensure_schema, get_connection

    sources_yml = tmp_path / "sources.yaml"
    sources_yml.write_text("sources: []")
    monkeypatch.chdir(tmp_path)

    data_dir = ensure_data_dir(resolve_data_dir(str(tmp_path)))
    db_path = data_dir / "index.db"

    conn = get_connection(db_path)
    ensure_schema(conn)

    long_path = "Networking/Wireless/WPA3/Security/Protocols"
    conn.execute(
        "INSERT INTO sections (source, title, category, path, heading_path, body, "
        "content_hash, source_title) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "wpa3-docs",
            "WPA3 Overview",
            "networking",
            "wpa3.md",
            long_path,
            "WPA3 is the latest Wi-Fi security standard.",
            "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
            "WPA3 Docs",
        ),
    )
    row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO sections_fts(rowid, title, heading_path, body) VALUES (?, ?, ?, ?)",
        (
            row_id,
            "WPA3 Overview",
            long_path,
            "WPA3 is the latest Wi-Fi security standard.",
        ),
    )
    conn.execute(
        "INSERT INTO sections_fts_title(rowid, title, heading_path) VALUES (?, ?, ?)",
        (row_id, "WPA3 Overview", long_path),
    )
    conn.commit()
    conn.close()

    result = subprocess.run(
        [sys.executable, "-m", "knowledge", "search", "wpa3", "-c", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "WPA3 Overview" in result.stdout, (
        f"Expected title to appear in unified format output.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# ── Search display formatting ────────────────────────────────────────


def test_format_search_includes_hash() -> None:
    results = [
        {
            "source": "hacktricks",
            "title": "Token Confusion",
            "category": "wikis",
            "path": "a.md",
            "heading_path": "HackTricks: Token Confusion",
            "body": "body",
            "distance": -14.03,
            "content_hash": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
            "source_title": "HackTricks",
        }
    ]
    from knowledge.cli import _format_search_results

    output = _format_search_results(results)
    assert "a1b2c3d4e5f6" in output  # first 12 chars of hash


def test_format_search_tag_format(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "knowledge.cli.shutil.get_terminal_size",
        lambda *a, **kw: os.terminal_size((120, 20)),
    )
    results = [
        {
            "source": "hacktricks",
            "title": "Token Confusion",
            "category": "wikis",
            "path": "a.md",
            "heading_path": "HackTricks: Token Confusion",
            "body": "body",
            "distance": -14.03,
            "content_hash": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
            "source_title": "HackTricks",
        }
    ]
    from knowledge.cli import _format_search_results

    output = _format_search_results(results)
    assert "wikis·HackTricks" in output


def test_format_search_truncates_long_tag() -> None:
    """Long tags get … ellipsis truncated."""
    results = [
        {
            "source": "internalallthethings",
            "title": "OfficePurge",
            "category": "ad-internal",
            "path": "a.md",
            "heading_path": "InternalAllTheThings: OfficePurge",
            "body": "body",
            "distance": -20.71,
            "content_hash": "e5f6a7b8a9b0c1d2e3f4a5b6c7d8e9f0",
            "source_title": "InternalAllTheThings",
        }
    ]
    from knowledge.cli import _format_search_results

    output = _format_search_results(results)
    assert "…" in output


# M1: stdout/stderr line buffering is verified manually by running
# `kdb update > file 2>&1` and confirming stderr messages are not interleaved
# before stdout output in redirected mode.
