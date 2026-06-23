import sqlite3

from knowledge.db import get_connection, ensure_schema


def test_get_connection_wal(tmp_path):
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    cursor = conn.execute("PRAGMA journal_mode")
    assert cursor.fetchone()[0] == "wal"
    cur2 = conn.execute("PRAGMA foreign_keys")
    assert cur2.fetchone()[0] == 1
    assert conn.row_factory is sqlite3.Row


def test_ensure_schema_creates_tables(tmp_path):
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    ensure_schema(conn, dim=1024)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    names = {row[0] for row in tables}
    assert "sections" in names
    assert "section_vectors" in names
    assert "source_state" in names
    assert "index_meta" in names


def test_ensure_schema_vec0_dim(tmp_path):
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    ensure_schema(conn, dim=768)
    sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='section_vectors' AND type='table'"
    ).fetchone()[0]
    assert "FLOAT[768]" in sql


def test_ensure_schema_idempotent(tmp_path):
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    ensure_schema(conn, dim=1024)
    ensure_schema(conn, dim=1024)


def test_column_definitions(tmp_path):
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    ensure_schema(conn, dim=1024)
    cols = {
        row[1]: row for row in conn.execute("PRAGMA table_info(sections)").fetchall()
    }
    assert cols["id"][2] == "INTEGER"
    assert cols["id"][5] == 1
    assert cols["source"][2] == "TEXT"
    assert cols["source"][3] == 1
    assert cols["title"][2] == "TEXT"
    assert cols["title"][3] == 1
    assert cols["category"][2] == "TEXT"
    assert cols["category"][3] == 1
    assert cols["path"][2] == "TEXT"
    assert cols["path"][3] == 1
    assert cols["heading_path"][2] == "TEXT"
    assert cols["heading_path"][3] == 0
    assert cols["body"][2] == "TEXT"
    assert cols["body"][3] == 1
    cols_ss = {
        row[1]: row
        for row in conn.execute("PRAGMA table_info(source_state)").fetchall()
    }
    assert cols_ss["name"][2] == "TEXT"
    assert cols_ss["name"][5] == 1
    assert cols_ss["git_head"][2] == "TEXT"
    assert cols_ss["git_head"][3] == 0
    assert cols_ss["indexed_at"][2] == "TEXT"
    assert cols_ss["indexed_at"][3] == 1
    assert "datetime" in cols_ss["indexed_at"][4]
