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


def test_cli_get_formatted_output(tmp_path) -> None:
    import sqlite3

    db_path = tmp_path / "index.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_hash TEXT,
            rank_bias REAL DEFAULT 1.0,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'general',
            path TEXT NOT NULL,
            heading_path TEXT NOT NULL DEFAULT '',
            body TEXT NOT NULL DEFAULT '',
            source_title TEXT DEFAULT ''
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts USING fts5(
            title, heading_path, body, content='sections', content_rowid='id'
        );
        CREATE TABLE IF NOT EXISTS index_meta(
            key TEXT PRIMARY KEY,
            value TEXT
        );
        INSERT INTO sections (content_hash, rank_bias, source, title, category, path, heading_path, body, source_title)
        VALUES ('a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0', 1.0, 'test-source', 'Test Section', 'general', '/path/to/doc.md', '', 'This is the body text of the section.', 'Test Source');
    """)
    conn.commit()
    conn.close()

    result = subprocess.run(
        [sys.executable, "-m", "knowledge", "get", "a1b2c3d4e5", "-c", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert "a1b2c3d4e5" in result.stdout
    assert "This is the body text" in result.stdout
    assert "Test Section" in result.stdout
