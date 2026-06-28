"""Tests for knowledge.db — schema migration (content_hash, rank_bias columns)."""

from __future__ import annotations

import sqlite3

from knowledge.db import _migrate_schema, get_connection


def test_migrate_adds_missing_columns(tmp_path) -> None:
    """_migrate_schema adds content_hash, rank_bias, and source_title to old schema."""
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sections (
            id INTEGER PRIMARY KEY,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            category TEXT NOT NULL,
            path TEXT NOT NULL,
            heading_path TEXT,
            body TEXT NOT NULL,
            UNIQUE(source, path, heading_path)
        );
    """)

    cols_before = {row[1] for row in conn.execute("PRAGMA table_info(sections)")}
    assert "content_hash" not in cols_before
    assert "rank_bias" not in cols_before
    assert "source_title" not in cols_before

    msgs = _migrate_schema(conn)
    assert "added content_hash column" in msgs
    assert "added rank_bias column" in msgs
    assert "added source_title column" in msgs

    cols_after = {row[1] for row in conn.execute("PRAGMA table_info(sections)")}
    assert "content_hash" in cols_after
    assert "rank_bias" in cols_after
    assert "source_title" in cols_after
    conn.close()


def test_migrate_idempotent(tmp_path) -> None:
    """Second call returns empty list (no migrations needed)."""
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sections (
            id INTEGER PRIMARY KEY,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            category TEXT NOT NULL,
            path TEXT NOT NULL,
            heading_path TEXT,
            body TEXT NOT NULL,
            UNIQUE(source, path, heading_path)
        );
    """)
    _migrate_schema(conn)
    msgs = _migrate_schema(conn)
    assert msgs == []
    conn.close()


def test_content_hash_nullable(tmp_path) -> None:
    """content_hash column allows NULL (existing rows survive migration)."""
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sections (
            id INTEGER PRIMARY KEY,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            category TEXT NOT NULL,
            path TEXT NOT NULL,
            heading_path TEXT,
            body TEXT NOT NULL,
            UNIQUE(source, path, heading_path)
        );
    """)
    conn.execute(
        "INSERT INTO sections (source, title, category, path, heading_path, body) "
        "VALUES ('src', 't', 'cat', 'p', 'hp', 'b')"
    )
    conn.commit()
    _migrate_schema(conn)
    row = conn.execute(
        "SELECT content_hash, rank_bias FROM sections WHERE source='src'"
    ).fetchone()
    assert row["content_hash"] is None
    assert row["rank_bias"] == 1.0
    conn.close()
