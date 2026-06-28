"""Tests for knowledge.indexer — indexing pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from knowledge.chunk import Section


def test_index_source_inserts_sections(tmp_path: Path) -> None:
    """_index_source inserts sections and FTS5 entries."""
    from knowledge.db import get_connection, ensure_schema
    from knowledge.indexer import _index_source
    from knowledge.sources import Source

    source = Source(
        name="test",
        source_type="git",
        url="https://example.com/repo.git",
        category="docs",
        title="Test",
    )
    conn = get_connection(tmp_path / "test.db")
    ensure_schema(conn)

    data_dir = tmp_path
    (data_dir / "sources" / "test").mkdir(parents=True)

    with (
        patch(
            "knowledge.indexer._walk_files",
            return_value=[data_dir / "sources" / "test" / "a.md"],
        ),
        patch("knowledge.indexer.chunk_file") as mock_chunk,
        patch("knowledge.indexer.load_config") as mock_cfg,
    ):
        sections = [
            Section(
                source="test",
                title="Heading",
                category="docs",
                path="a.md",
                heading_path="",
                body="Body text.",
            ),
        ]
        mock_chunk.side_effect = lambda fpath, src, cat, rel_path, **kw: sections
        mock_cfg.return_value = MagicMock()

        count = _index_source(source, conn, tmp_path, verbose=False)

        assert count == 1
        # Verify sections were inserted
        rows = conn.execute("SELECT title, body FROM sections ORDER BY id").fetchall()
        assert len(rows) == 1
        assert rows[0]["title"] == "Heading"
        assert rows[0]["body"] == "Body text."

        # Verify FTS5 index was populated
        fts_rows = conn.execute("SELECT count(*) FROM sections_fts").fetchone()[0]
        assert fts_rows == 1

        fts_title_rows = conn.execute(
            "SELECT count(*) FROM sections_fts_title"
        ).fetchone()[0]
        assert fts_title_rows == 1
    conn.close()
