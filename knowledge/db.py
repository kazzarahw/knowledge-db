from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec


def get_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection, dim: int) -> None:
    conn.executescript(f"""
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

        CREATE VIRTUAL TABLE IF NOT EXISTS section_vectors USING vec0(
            section_id INTEGER PRIMARY KEY,
            source TEXT PARTITION KEY,
            embedding FLOAT[{dim}] DISTANCE_METRIC=COSINE
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
