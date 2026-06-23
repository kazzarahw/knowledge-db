"""Integration tests: end-to-end chunking, indexing, and search."""

from __future__ import annotations

from pathlib import Path

from knowledge.chunk import chunk_text


def test_chunk_search_roundtrip() -> None:
    md = "# Test\nBody.\n## Sub\nMore body."
    sections = chunk_text(md, "test", "e2e", "test.md")
    assert len(sections) == 2
    assert sections[0].title == "Test"
    assert sections[1].title == "Sub"


def test_no_headings_fallback() -> None:
    md = "Just some plain text.\n\nMore text."
    sections = chunk_text(md, "test", "e2e", "plain.txt")
    assert len(sections) == 1


def test_frontmatter_stripped() -> None:
    md = "---\ntitle: Test\n---\n# Heading\nBody."
    sections = chunk_text(md, "test", "e2e", "fm.md")
    assert len(sections) == 1
    assert "title: Test" not in sections[0].body


def test_setext_headings() -> None:
    md = "H1\n===\nBody1.\n\nH2\n---\nBody2."
    sections = chunk_text(md, "test", "e2e", "setext.md")
    assert len(sections) == 2
    assert sections[0].title == "H1"
    assert sections[1].title == "H2"


def test_walk_files(tmp_path: Path) -> None:
    from knowledge.indexer import _walk_files
    from knowledge.sources import Source

    source = Source(
        name="test",
        source_type="local",
        path=str(tmp_path),
        title="Test",
        category="e2e",
    )
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "secret.md").write_text("# secret")
    (tmp_path / "visible.md").write_text("# visible")

    files = _walk_files(tmp_path, source)
    assert len(files) == 1
    assert "visible.md" in files[0].name


def test_source_signature(tmp_path: Path) -> None:
    from knowledge.indexer import _source_signature

    sig1 = _source_signature(tmp_path)
    f = tmp_path / "new.md"
    f.write_text("# new")
    sig2 = _source_signature(tmp_path)
    assert sig1 != sig2


def test_db_schema_creates_tables(tmp_path: Path) -> None:
    from knowledge.db import get_connection, ensure_schema

    conn = get_connection(tmp_path / "test.db")
    ensure_schema(conn, dim=384)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "sections" in tables
    assert "section_vectors" in tables
    assert "source_state" in tables
    assert "index_meta" in tables
    conn.close()


def test_search_dimension_validation(tmp_path: Path) -> None:
    from knowledge.config import resolve_data_dir, ensure_data_dir
    from knowledge.db import get_connection, ensure_schema
    from knowledge.search import cmd_search

    data_dir = ensure_data_dir(resolve_data_dir(str(tmp_path)))
    db_path = data_dir / "index.db"

    conn = get_connection(db_path)
    ensure_schema(conn, dim=768)
    conn.execute(
        "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('embedding_dim', '9999')"
    )
    conn.commit()
    conn.close()

    results = cmd_search("test", config_dir=str(tmp_path))
    assert results == []
