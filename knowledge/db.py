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
