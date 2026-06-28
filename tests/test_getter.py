"""Tests for knowledge.getter — hash-prefix section retrieval."""

from __future__ import annotations

from knowledge.db import ensure_schema, get_connection


def test_get_by_full_hash(tmp_path) -> None:
    from knowledge.getter import cmd_get

    db_path = tmp_path / "data" / "index.db"
    db_path.parent.mkdir()
    conn = get_connection(db_path)
    ensure_schema(conn)
    conn.execute(
        "INSERT INTO sections (source, title, category, path, heading_path, body, content_hash, rank_bias) "
        "VALUES ('src', 'Title', 'cat', 'p', 'hp', 'body text', 'a1b2c3d4e5f6a7b8c9d0', 1.0)"
    )
    conn.commit()
    conn.close()

    result = cmd_get("a1b2c3d4e5f6a7b8c9d0", config_dir=str(tmp_path / "data"))
    assert result is not None
    assert result["source"] == "src"
    assert result["title"] == "Title"
    assert result["body"] == "body text"


def test_get_by_prefix(tmp_path) -> None:
    from knowledge.getter import cmd_get

    db_path = tmp_path / "data" / "index.db"
    db_path.parent.mkdir()
    conn = get_connection(db_path)
    ensure_schema(conn)
    conn.execute(
        "INSERT INTO sections (source, title, category, path, heading_path, body, content_hash, rank_bias) "
        "VALUES ('src', 'Title', 'cat', 'p', 'hp', 'body', 'a1b2c3d4e5f6a7b8c9d0', 1.0)"
    )
    conn.commit()
    conn.close()

    result = cmd_get("a1b2c3d4e5", config_dir=str(tmp_path / "data"))
    assert result is not None
    assert result["body"] == "body"


def test_get_no_match(tmp_path) -> None:
    from knowledge.getter import cmd_get

    db_path = tmp_path / "data" / "index.db"
    db_path.parent.mkdir()
    conn = get_connection(db_path)
    ensure_schema(conn)
    conn.commit()
    conn.close()

    result = cmd_get("ffffffffffff", config_dir=str(tmp_path / "data"))
    assert result is None


def test_get_ambiguous_prefix(tmp_path) -> None:
    from knowledge.getter import cmd_get

    db_path = tmp_path / "data" / "index.db"
    db_path.parent.mkdir()
    conn = get_connection(db_path)
    ensure_schema(conn)
    conn.execute(
        "INSERT INTO sections (source, title, category, path, heading_path, body, content_hash, rank_bias) "
        "VALUES ('a', 'T1', 'cat', 'a.md', 'hp', 'body1', 'a1b2c3d4e5f6a7b8c9d0', 1.0)"
    )
    conn.execute(
        "INSERT INTO sections (source, title, category, path, heading_path, body, content_hash, rank_bias) "
        "VALUES ('b', 'T2', 'cat', 'b.md', 'hp', 'body2', 'a1b2c3d4e5f6a7b8c9d1', 1.0)"
    )
    conn.commit()
    conn.close()

    result = cmd_get("a1b2c3d4", config_dir=str(tmp_path / "data"))
    assert result is None  # ambiguous -> returns None, stderr prints matches


def test_get_non_hex_prefix(tmp_path) -> None:
    from knowledge.getter import cmd_get

    result = cmd_get("xyz12345", config_dir=str(tmp_path / "data"))
    assert result is None  # invalid -> returns None
