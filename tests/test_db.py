from knowledge.db import get_connection, ensure_schema


def test_get_connection_wal(tmp_path):
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    cursor = conn.execute("PRAGMA journal_mode")
    assert cursor.fetchone()[0] == "wal"


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
