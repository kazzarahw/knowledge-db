"""Tests for knowledge.search — query router and FTS5 search."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from knowledge.search import (
    QueryTier,
    SearchResult,
    _classify_query,
    cmd_search,
)


class TestQueryRouter:
    """Query router classification tests."""

    def test_classify_exact_cve(self) -> None:
        assert _classify_query("CVE-2025-31161") == QueryTier.EXACT

    def test_classify_exact_krb_err(self) -> None:
        assert _classify_query("KRB_AP_ERR_SKEW") == QueryTier.EXACT

    def test_classify_exact_ntstatus(self) -> None:
        assert _classify_query("NTSTATUS 0xC0000022") == QueryTier.EXACT

    def test_classify_tool_command(self) -> None:
        assert _classify_query("responder -I eth0 -w") == QueryTier.TOOL_COMMAND

    def test_classify_tool_command_long_flag(self) -> None:
        assert _classify_query("sqlmap --os-shell --batch") == QueryTier.TOOL_COMMAND

    def test_classify_path_unix(self) -> None:
        assert _classify_query("/etc/nginx/nginx.conf") == QueryTier.PATH

    def test_classify_path_domain(self) -> None:
        assert _classify_query("example.com/wp-admin") == QueryTier.PATH

    def test_classify_conceptual(self) -> None:
        assert _classify_query("bypass av using unhooking") == QueryTier.CONCEPTUAL

    def test_classify_conceptual_with_dash(self) -> None:
        """cross-site scripting — but '--' without preceding tool name is conceptual."""
        assert (
            _classify_query("bypass sql injection using -- techniques")
            == QueryTier.CONCEPTUAL
        )

    def test_classify_conceptual_with_dot_sentence(self) -> None:
        """Dot in a sentence (not a path) is conceptual."""
        assert (
            _classify_query("lateral movement through sql server")
            == QueryTier.CONCEPTUAL
        )

    def test_classify_conceptual_version_number(self) -> None:
        """Version numbers in multi-word queries should not be PATH."""
        assert _classify_query("install python 3.12") == QueryTier.CONCEPTUAL
        assert _classify_query("upgrade to version 2.0") == QueryTier.CONCEPTUAL

    def test_classify_exact_bare_hex(self) -> None:
        """Standalone hex error code routes as EXACT."""
        assert _classify_query("0xDEADBEEF") == QueryTier.EXACT
        assert _classify_query("0xC0000022") == QueryTier.EXACT

    def test_classify_priority_exact_over_command(self) -> None:
        """CVE with flags should route as EXACT (highest priority)."""
        assert _classify_query("CVE-2024-1234 --exploit") == QueryTier.EXACT

    def test_classify_priority_exact_over_path(self) -> None:
        """CVE with / in it should route as EXACT."""
        assert _classify_query("CVE-2024-1234/exploit") == QueryTier.EXACT

    # ── Edge cases: FTS5 keywords & special chars ────────────────────

    def test_classify_conceptual_with_fts5_keyword(self) -> None:
        """FTS5 keywords (AND, OR, NOT) should not crash MATCH."""
        assert _classify_query("NOT") == QueryTier.CONCEPTUAL
        assert _classify_query("AND") == QueryTier.CONCEPTUAL
        assert _classify_query("or") == QueryTier.CONCEPTUAL

    def test_classify_tool_command_fts5_keyword(self) -> None:
        """Tool command starting with FTS5 keyword routes TOOL_COMMAND."""
        assert _classify_query("or --flag value") == QueryTier.TOOL_COMMAND
        assert _classify_query("AND -x") == QueryTier.TOOL_COMMAND

    def test_classify_hex_incidental_in_sentence(self) -> None:
        """Hex in a multi-word query should stay CONCEPTUAL, not EXACT."""
        assert (
            _classify_query("guide to 0xDEADBEEF canary values") == QueryTier.CONCEPTUAL
        )

    def test_classify_exact_ntstatus_no_0x_prefix_falls_through(self) -> None:
        """NTSTATUS C0000022 (bare hex) is CONCEPTUAL — acceptable trade-off."""
        assert _classify_query("NTSTATUS C0000022") == QueryTier.CONCEPTUAL

    def test_classify_dashes_handling(self) -> None:
        """Queries consisting only of dashes should not crash."""
        assert _classify_query("---") == QueryTier.CONCEPTUAL
        assert _classify_query(" - - ") == QueryTier.CONCEPTUAL


class TestSearchExecution:
    """Search result structure, edge cases, and crash-safety tests."""

    def test_search_result_has_distance_field(self) -> None:
        """SearchResult includes distance field for JSON backward compat."""
        r = SearchResult(
            source="test",
            title="Test",
            category="e2e",
            path="t.md",
            heading_path="",
            body="body",
            distance=0.0,
        )
        assert "distance" in r
        assert isinstance(r["distance"], float)

    def test_empty_index_returns_empty(self, tmp_path: Path) -> None:
        """Search on an empty (no tables) index prints error, returns []."""
        from knowledge.config import resolve_data_dir, ensure_data_dir

        data_dir = ensure_data_dir(resolve_data_dir(str(tmp_path)))
        with patch("sys.stderr"):
            results = cmd_search("test", config_dir=str(tmp_path))
            assert results == []

    def test_search_with_special_chars_does_not_crash(self, tmp_path: Path) -> None:
        """Special FTS5 chars/AND/NOT should not cause MATCH syntax errors."""
        from knowledge.db import get_connection, ensure_schema
        from knowledge.config import resolve_data_dir, ensure_data_dir

        data_dir = ensure_data_dir(resolve_data_dir(str(tmp_path)))
        db_path = data_dir / "index.db"
        conn = get_connection(db_path)
        ensure_schema(conn)
        conn.close()

        # These would crash if FTS5 escaping is broken
        with patch("sys.stderr"):
            results = cmd_search("bypass +av", config_dir=str(tmp_path))
            assert results == []  # empty index → no results, no crash

            results = cmd_search("NOT something", config_dir=str(tmp_path))
            assert results == []

            results = cmd_search("tool AND flag", config_dir=str(tmp_path))
            assert results == []

    def test_no_matches_returns_empty(self, tmp_path: Path) -> None:
        """Search with no matching rows returns []."""
        from knowledge.db import get_connection, ensure_schema
        from knowledge.config import resolve_data_dir, ensure_data_dir

        data_dir = ensure_data_dir(resolve_data_dir(str(tmp_path)))
        db_path = data_dir / "index.db"
        conn = get_connection(db_path)
        ensure_schema(conn)
        conn.close()

        results = cmd_search("zzzzzznonexistentqueryzzzzzz", config_dir=str(tmp_path))
        assert results == []

    def test_bm25_ranking_title_over_body(self, tmp_path: Path) -> None:
        """Title match ranks higher than body-only match."""
        from knowledge.db import get_connection, ensure_schema
        from knowledge.config import resolve_data_dir, ensure_data_dir

        data_dir = ensure_data_dir(resolve_data_dir(str(tmp_path)))
        db_path = data_dir / "index.db"
        conn = get_connection(db_path)
        ensure_schema(conn)

        # Insert two sections: one with "pivoting" in title, one in body
        conn.execute(
            "INSERT INTO sections (source, title, category, path, heading_path, body) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "test",
                "Pivoting through firewalls",
                "e2e",
                "a.md",
                "",
                "Body text here.",
            ),
        )
        conn.execute(
            "INSERT INTO sections (source, title, category, path, heading_path, body) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "test",
                "General",
                "e2e",
                "b.md",
                "",
                "Learn about pivoting and tunneling.",
            ),
        )
        # Sync FTS5 tables
        rows = conn.execute(
            "SELECT id, source, title, heading_path, body FROM sections ORDER BY id"
        ).fetchall()
        for row in rows:
            conn.execute(
                "INSERT INTO sections_fts(rowid, title, heading_path, body) VALUES (?, ?, ?, ?)",
                (row["id"], row["title"], row["heading_path"], row["body"]),
            )
            conn.execute(
                "INSERT INTO sections_fts_title(rowid, title, heading_path) VALUES (?, ?, ?)",
                (row["id"], row["title"], row["heading_path"]),
            )
        conn.commit()
        conn.close()

        results = cmd_search("pivoting", config_dir=str(tmp_path))
        assert len(results) >= 2
        # Title match should have lower BM25 score (better)
        assert results[0]["distance"] <= results[1]["distance"]

    def test_source_filter(self, tmp_path: Path) -> None:
        """--source filter restricts results to named source."""
        from knowledge.db import get_connection, ensure_schema
        from knowledge.config import resolve_data_dir, ensure_data_dir

        data_dir = ensure_data_dir(resolve_data_dir(str(tmp_path)))
        db_path = data_dir / "index.db"
        conn = get_connection(db_path)
        ensure_schema(conn)

        conn.execute(
            "INSERT INTO sections (source, title, category, path, heading_path, body) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("src-a", "Alpha Topic", "e2e", "a.md", "", "Content about alpha."),
        )
        conn.execute(
            "INSERT INTO sections (source, title, category, path, heading_path, body) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("src-b", "Beta Topic", "e2e", "b.md", "", "Content about beta."),
        )
        rows = conn.execute(
            "SELECT id, source, title, heading_path, body FROM sections ORDER BY id"
        ).fetchall()
        for row in rows:
            conn.execute(
                "INSERT INTO sections_fts(rowid, title, heading_path, body) VALUES (?, ?, ?, ?)",
                (row["id"], row["title"], row["heading_path"], row["body"]),
            )
            conn.execute(
                "INSERT INTO sections_fts_title(rowid, title, heading_path) VALUES (?, ?, ?)",
                (row["id"], row["title"], row["heading_path"]),
            )
        conn.commit()
        conn.close()

        results = cmd_search("topic", source="src-a", config_dir=str(tmp_path))
        assert all(r["source"] == "src-a" for r in results)

    # ── End-to-end data tests per tier ──────────────────────────────

    def test_exact_tier_retrieval(self, tmp_path: Path) -> None:
        """EXACT tier finds exact CVE match via phrase search."""
        from knowledge.db import get_connection, ensure_schema
        from knowledge.config import resolve_data_dir, ensure_data_dir

        data_dir = ensure_data_dir(resolve_data_dir(str(tmp_path)))
        db_path = data_dir / "index.db"
        conn = get_connection(db_path)
        ensure_schema(conn)

        conn.execute(
            "INSERT INTO sections (source, title, category, path, heading_path, body) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "test",
                "CVE-2024-1234: critical vuln",
                "e2e",
                "a.md",
                "",
                "Description here.",
            ),
        )
        conn.execute(
            "INSERT INTO sections (source, title, category, path, heading_path, body) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("test", "General Topic", "e2e", "b.md", "", "body"),
        )
        rows = conn.execute(
            "SELECT id, source, title, heading_path, body FROM sections ORDER BY id"
        ).fetchall()
        for row in rows:
            conn.execute(
                "INSERT INTO sections_fts(rowid, title, heading_path, body) VALUES (?, ?, ?, ?)",
                (row["id"], row["title"], row["heading_path"], row["body"]),
            )
            conn.execute(
                "INSERT INTO sections_fts_title(rowid, title, heading_path) VALUES (?, ?, ?)",
                (row["id"], row["title"], row["heading_path"]),
            )
        conn.commit()
        conn.close()

        results = cmd_search("CVE-2024-1234", config_dir=str(tmp_path))
        assert len(results) >= 1

    def test_tool_command_tier_prefix_matching(self, tmp_path: Path) -> None:
        """TOOL_COMMAND tier finds docs matching tool name."""
        from knowledge.db import get_connection, ensure_schema
        from knowledge.config import resolve_data_dir, ensure_data_dir

        data_dir = ensure_data_dir(resolve_data_dir(str(tmp_path)))
        db_path = data_dir / "index.db"
        conn = get_connection(db_path)
        ensure_schema(conn)

        conn.execute(
            "INSERT INTO sections (source, title, category, path, heading_path, body) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "test",
                "Responder Guide",
                "e2e",
                "r.md",
                "",
                "Using responder to capture hashes.",
            ),
        )
        conn.execute(
            "INSERT INTO sections (source, title, category, path, heading_path, body) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("test", "Other", "e2e", "o.md", "", "No match."),
        )
        rows = conn.execute(
            "SELECT id, source, title, heading_path, body FROM sections ORDER BY id"
        ).fetchall()
        for row in rows:
            conn.execute(
                "INSERT INTO sections_fts(rowid, title, heading_path, body) VALUES (?, ?, ?, ?)",
                (row["id"], row["title"], row["heading_path"], row["body"]),
            )
            conn.execute(
                "INSERT INTO sections_fts_title(rowid, title, heading_path) VALUES (?, ?, ?)",
                (row["id"], row["title"], row["heading_path"]),
            )
        conn.commit()
        conn.close()

        results = cmd_search("responder -I eth0 -w", config_dir=str(tmp_path))
        # Should find at least the responder doc via prefix match
        assert len(results) >= 1
        assert results[0]["source"] == "test"

    def test_path_tier_trigram_matching(self, tmp_path: Path) -> None:
        """PATH tier finds path-like queries via trigram FTS5 table."""
        from knowledge.db import get_connection, ensure_schema
        from knowledge.config import resolve_data_dir, ensure_data_dir

        data_dir = ensure_data_dir(resolve_data_dir(str(tmp_path)))
        db_path = data_dir / "index.db"
        conn = get_connection(db_path)
        ensure_schema(conn)

        conn.execute(
            "INSERT INTO sections (source, title, category, path, heading_path, body) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "test",
                "Nginx Config",
                "e2e",
                "n.md",
                "",
                "/etc/nginx/nginx.conf config.",
            ),
        )
        rows = conn.execute(
            "SELECT id, source, title, heading_path, body FROM sections ORDER BY id"
        ).fetchall()
        for row in rows:
            conn.execute(
                "INSERT INTO sections_fts(rowid, title, heading_path, body) VALUES (?, ?, ?, ?)",
                (row["id"], row["title"], row["heading_path"], row["body"]),
            )
            conn.execute(
                "INSERT INTO sections_fts_title(rowid, title, heading_path) VALUES (?, ?, ?)",
                (row["id"], row["title"], row["heading_path"]),
            )
        conn.commit()
        conn.close()

        results = cmd_search("/etc/nginx/", config_dir=str(tmp_path))
        assert len(results) >= 1
        assert results[0]["source"] == "test"
