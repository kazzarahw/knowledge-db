"""Tests for content hashing in indexer."""

from __future__ import annotations

import hashlib
import sqlite3

from knowledge.db import ensure_schema, get_connection
from knowledge.indexer import _fts5_sync_sections
from knowledge.chunk import Section


def test_content_hash_computed_during_sync(tmp_path) -> None:
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    ensure_schema(conn)

    sections = [
        Section(
            source="test",
            title="Test Section",
            category="wikis",
            path="test.md",
            heading_path="Test Section",
            body="# Hello World",
        )
    ]
    _fts5_sync_sections(conn, "test", sections, rank_bias=0.7)

    row = conn.execute(
        "SELECT content_hash, rank_bias FROM sections WHERE source='test'"
    ).fetchone()
    expected_hash = hashlib.sha256(b"# Hello World").hexdigest()
    assert row["content_hash"] == expected_hash
    assert row["rank_bias"] == 0.7
    conn.close()


def test_dedup_first_source_wins(tmp_path) -> None:
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    ensure_schema(conn)

    body = "# Hello World"
    h = hashlib.sha256(body.encode()).hexdigest()

    s1 = Section(
        source="src1",
        title="S1",
        category="wikis",
        path="a.md",
        heading_path="S1",
        body=body,
    )
    s2 = Section(
        source="src2",
        title="S2",
        category="wikis",
        path="b.md",
        heading_path="S2",
        body=body,
    )

    _fts5_sync_sections(conn, "src1", [s1], rank_bias=0.7)
    _fts5_sync_sections(conn, "src2", [s2], rank_bias=0.7, content_hashes_seen={h})

    rows = conn.execute("SELECT source, title FROM sections ORDER BY id").fetchall()
    assert len(rows) == 1
    assert rows[0]["source"] == "src1"
    conn.close()
