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


def test_index_source_single_file(tmp_path: Path) -> None:
    """_index_source indexes a single file and returns section count."""
    from knowledge.db import get_connection, ensure_schema
    from knowledge.indexer import _index_source
    from knowledge.sources import Source

    source = Source(
        name="test-index",
        source_type="git",
        url="https://github.com/user/repo.git",
    )
    source_dir = tmp_path / "sources" / source.name
    source_dir.mkdir(parents=True)
    (source_dir / "doc.md").write_text("# Heading\nBody text.\n\n## Sub\nMore.")
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    ensure_schema(conn)
    count = _index_source(source, conn, tmp_path, verbose=False)
    assert count == 2
    rows = conn.execute("SELECT title, body FROM sections ORDER BY id").fetchall()
    assert rows[0]["title"] == "Heading"
    assert rows[0]["body"] == "Body text."
    assert rows[1]["title"] == "Sub"
    conn.close()


def test_cmd_index_orphan_cleanup(tmp_path: Path, monkeypatch) -> None:
    """cmd_index removes sources no longer in sources.yaml."""
    from knowledge.config import resolve_data_dir, ensure_data_dir
    from knowledge.db import get_connection, ensure_schema
    from knowledge.indexer import cmd_index

    sources_yml = tmp_path / "sources.yaml"
    sources_yml.write_text(
        "sources:\n  - name: orphaned\n    type: git\n    url: https://github.com/user/repo.git\n"
    )
    monkeypatch.chdir(tmp_path)

    data_dir = ensure_data_dir(resolve_data_dir(str(tmp_path)))
    db_path = data_dir / "index.db"
    conn = get_connection(db_path)
    ensure_schema(conn)

    conn.execute(
        "INSERT INTO sections (source, title, category, path, heading_path, body) VALUES (?,?,?,?,?,?)",
        ("old-source", "orphan", "", "p", "", "body"),
    )
    row_id = conn.execute(
        "SELECT id FROM sections WHERE source = 'old-source'"
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO sections_fts(rowid, title, heading_path, body) VALUES (?, ?, ?, ?)",
        (row_id, "orphan", "", "body"),
    )
    conn.execute(
        "INSERT INTO sections_fts_title(rowid, title, heading_path) VALUES (?, ?, ?)",
        (row_id, "orphan", ""),
    )
    conn.execute(
        "INSERT INTO source_state (name, git_head) VALUES (?, ?)",
        ("old-source", "abc123"),
    )
    conn.commit()
    conn.close()

    cmd_index(config_dir=str(tmp_path), force=False, verbose=False)

    conn2 = get_connection(db_path)
    remaining = conn2.execute("SELECT source FROM sections").fetchall()
    assert len(remaining) == 0
    fts_count = conn2.execute("SELECT count(*) FROM sections_fts").fetchone()[0]
    assert fts_count == 0
    conn2.close()


def test_cmd_index_force_rebuild(tmp_path: Path, monkeypatch) -> None:
    """--force drops and recreates all tables, then indexes fresh."""
    from knowledge.config import resolve_data_dir, ensure_data_dir
    from knowledge.db import get_connection, ensure_schema
    from knowledge.indexer import cmd_index

    sources_yml = tmp_path / "sources.yaml"
    sources_yml.write_text(
        "sources:\n  - name: test-force\n    type: local\n    path: /tmp/nonexistent\n"
    )
    monkeypatch.chdir(tmp_path)

    data_dir = ensure_data_dir(resolve_data_dir(str(tmp_path)))
    db_path = data_dir / "index.db"
    conn = get_connection(db_path)
    ensure_schema(conn)
    conn.execute("INSERT INTO index_meta (key, value) VALUES ('indexed_at', 'stale')")
    conn.commit()
    conn.close()

    cmd_index(config_dir=str(tmp_path), force=True, verbose=False)

    conn2 = get_connection(db_path)
    tables = {
        row[0]
        for row in conn2.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "sections_fts" in tables
    assert "sections_fts_title" in tables
    stale = conn2.execute(
        "SELECT value FROM index_meta WHERE key = 'indexed_at'"
    ).fetchone()
    assert stale is not None
    assert stale[0] != "stale"
    conn2.close()
