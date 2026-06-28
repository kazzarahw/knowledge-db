"""SQLite connection and schema — FTS5 tables replace vec0."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Open a WAL-mode SQLite connection.

    FTS5 is built into SQLite — no extension loading needed.
    sqlite-vec extension loading has been removed.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Configured connection with Row factory, foreign keys, WAL mode.
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create all required tables if they don't exist.

    Creates sections, sections_fts (porter), sections_fts_title (trigram),
    source_state, and index_meta tables. Idempotent — safe to call repeatedly.

    Args:
        conn: Open database connection.
    """
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sections (
            id INTEGER PRIMARY KEY,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            category TEXT NOT NULL,
            path TEXT NOT NULL,
            heading_path TEXT,
            body TEXT NOT NULL,
            content_hash TEXT,
            rank_bias REAL NOT NULL DEFAULT 1.0,
            source_title TEXT NOT NULL DEFAULT '',
            UNIQUE(source, path, heading_path)
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts USING fts5(
            title, heading_path, body,
            content=sections,
            tokenize='porter unicode61'
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts_title USING fts5(
            title, heading_path,
            content=sections,
            tokenize='trigram'
        );

        CREATE TABLE IF NOT EXISTS source_state (
            name TEXT PRIMARY KEY,
            git_head TEXT,
            indexed_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS index_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)


def _migrate_schema(conn: sqlite3.Connection) -> list[str]:
    """Add missing columns to existing sections table.

    Idempotent — safe to call repeatedly. Returns list of migration
    messages (empty if none needed).

    Args:
        conn: Open database connection.

    Returns:
        List of description strings for applied migrations.
    """
    existing = {row[1] for row in conn.execute("PRAGMA table_info(sections)")}
    migrations: list[str] = []
    if "content_hash" not in existing:
        conn.execute("ALTER TABLE sections ADD COLUMN content_hash TEXT")
        migrations.append("added content_hash column")
    if "rank_bias" not in existing:
        conn.execute(
            "ALTER TABLE sections ADD COLUMN rank_bias REAL NOT NULL DEFAULT 1.0"
        )
        migrations.append("added rank_bias column")
    if "source_title" not in existing:
        conn.execute(
            "ALTER TABLE sections ADD COLUMN source_title TEXT NOT NULL DEFAULT ''"
        )
        migrations.append("added source_title column")
    return migrations
