"""FTS5 full-text search with query router and column-weighted BM25."""

from __future__ import annotations

import re
import sqlite3
import sys
from enum import StrEnum
from typing import TypedDict

from knowledge.config import resolve_data_dir
from knowledge.db import get_connection


class QueryTier(StrEnum):
    """Query classification tier for routing to optimal FTS5 strategy."""

    EXACT = "exact"
    TOOL_COMMAND = "tool"
    PATH = "path"
    CONCEPTUAL = "conceptual"


_QUERY_ROUTERS: list[tuple[re.Pattern[str], QueryTier]] = [
    (
        re.compile(
            r"^(?:"
            r"CVE-\d{4}-\d{4,}"
            r"|"
            r"[A-Z]+_[A-Z]+_\w+(?:\s+(?:0x)?[0-9A-Fa-f]+)?"
            r"|"
            r"[A-Z]+\s+0x[0-9A-Fa-f]{8,}"
            r"|"
            r"0x[0-9A-Fa-f]{8,}"
            r")"
        ),
        QueryTier.EXACT,
    ),
    (
        re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*\s+-{1,2}\w+"),
        QueryTier.TOOL_COMMAND,
    ),
    (
        re.compile(r"/"),
        QueryTier.PATH,
    ),
]


def _classify_query(query: str) -> QueryTier:
    """Classify a search query into a routing tier.

    First-match-wins priority: EXACT > TOOL_COMMAND > PATH > CONCEPTUAL.

    Args:
        query: Raw user query string.

    Returns:
        QueryTier enum value determining the search strategy.
    """
    stripped = query.strip()
    for pattern, tier in _QUERY_ROUTERS:
        if pattern.search(stripped):
            return tier
    return QueryTier.CONCEPTUAL


def _escape_fts5_value(term: str) -> str:
    """Escape a single FTS5 term to prevent operator/keyword injection.

    Always wraps in double quotes to neutralise FTS5 operators
    (NOT, AND, OR) and special characters (``^  +  -  *  (  )  ~  ``).
    Embedded quotes are doubled per FTS5 escaping rules.
    """
    escaped = term.replace('"', '""')
    return f'"{escaped}"'


def _build_fts5_query(query: str, tier: QueryTier) -> str:
    """Build an FTS5 MATCH expression from the query and tier.

    Args:
        query: Raw user query string.
        tier: Classified query tier.

    Returns:
        FTS5 MATCH expression string.
    """
    match tier:
        case QueryTier.EXACT:
            return _escape_fts5_value(query.strip())
        case QueryTier.TOOL_COMMAND:
            first = query.strip().split()[0]
            if not first or first.startswith("-"):
                return ""
            return f"{_escape_fts5_value(first)}*"
        case QueryTier.PATH:
            return _escape_fts5_value(query.strip())
        case QueryTier.CONCEPTUAL:
            tokens = query.strip().split()
            if not tokens:
                return ""
            return " AND ".join(_escape_fts5_value(t) for t in tokens)


class SearchResult(TypedDict):
    """Single knowledge-base search result row.

    ``distance`` field name preserved from vec0 era for JSON backward
    compatibility. Now contains a BM25 relevance score (lower = more
    relevant). BM25 scores are not comparable across different queries.
    """

    source: str
    source_title: str
    title: str
    category: str
    path: str
    heading_path: str
    body: str
    distance: float
    content_hash: str


def _select_fts_table(tier: QueryTier) -> str:
    """Select which FTS5 table to query based on tier.

    Each tier routes to exactly one table — scores are never merged
    across different tokenizers.

    Args:
        tier: Classified query tier.

    Returns:
        FTS5 table name.
    """
    if tier == QueryTier.EXACT:
        return "sections_fts_title"
    return "sections_fts"


def _get_bm25_order(tier: QueryTier) -> tuple[str, str]:
    """Return (select_expr, order_expr) with consistent column-weighted BM25.

    Args:
        tier: Classified query tier.

    Returns:
        (select_expr, order_expr) SQL fragments using the same bm25() call.
    """
    if tier == QueryTier.EXACT:
        bm25 = "bm25(sections_fts_title, 5.0, 3.0)"
    else:
        bm25 = "bm25(sections_fts, 5.0, 3.0, 1.0)"
    return (bm25, bm25)


def cmd_search(
    query: str,
    top_k: int = 10,
    source: str | None = None,
    config_dir: str | None = None,
) -> list[SearchResult]:
    """Search the FTS5 index.

    Classifies the query via the query router, dispatches to the
    appropriate FTS5 table, ranks by column-weighted BM25, and
    returns results ordered by relevance.

    Args:
        query: Search query string.
        top_k: Maximum number of results to return.
        source: Optional source name filter.
        config_dir: Override config directory path.

    Returns:
        List of SearchResult dicts ordered by relevance (best first).
    """
    data_dir = resolve_data_dir(config_dir)
    db_path = data_dir / "index.db"

    if not db_path.exists():
        print("Error: No index found. Run 'kdb index' first.", file=sys.stderr)
        return []

    if not query.strip():
        print("Error: empty search query", file=sys.stderr)
        return []

    conn = None
    try:
        conn = get_connection(db_path)

        has_fts = (
            conn.execute(
                "SELECT COUNT(*) FROM sqlite_master "
                "WHERE type='table' AND name='sections_fts'"
            ).fetchone()[0]
            > 0
        )
        has_sections = (
            conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='sections'"
            ).fetchone()[0]
            > 0
        )

        if not has_fts and has_sections:
            print(
                "Error: Index needs rebuild. "
                "Run 'kdb index --force' to migrate from old format.",
                file=sys.stderr,
            )
            return []
        if not has_sections:
            print("Error: No index found. Run 'kdb index' first.", file=sys.stderr)
            return []

        tier = _classify_query(query)
        fts_table = _select_fts_table(tier)
        fts_query = _build_fts5_query(query, tier)
        if not fts_query:
            print("Error: empty search query after processing", file=sys.stderr)
            return []
        bm25_select, bm25_order = _get_bm25_order(tier)

        source_filter = ""
        source_params: list[str] = []
        if source:
            source_filter = "AND s.source = ?"
            source_params = [source]

        sql = f"""
            SELECT s.source, s.title, s.category, s.path,
                   s.heading_path, s.body, s.content_hash, s.source_title,
                   {bm25_select} as rank
            FROM {fts_table} f
            JOIN sections s ON s.id = f.rowid
            WHERE {fts_table} MATCH ?
              {source_filter}
            -- Division (not multiplication): FTS5 bm25() can return negative values.
            -- Dividing by rank_bias means boosted sources (lower rank_bias) produce
            -- more negative scores → sort first with default ASC ordering.
            ORDER BY {bm25_order} / s.rank_bias
            LIMIT ?
        """
        try:
            rows = conn.execute(sql, [fts_query, *source_params, top_k]).fetchall()
        except sqlite3.OperationalError as e:
            print(f"Error: FTS5 query syntax error: {e}", file=sys.stderr)
            return []

        if not rows:
            print("No results found.", file=sys.stderr)
            return []

        results: list[SearchResult] = []
        for row in rows:
            results.append(
                SearchResult(
                    source=row["source"],
                    source_title=row["source_title"],
                    title=row["title"],
                    category=row["category"],
                    path=row["path"],
                    heading_path=row["heading_path"],
                    body=row["body"],
                    distance=float(row["rank"]),
                    content_hash=row["content_hash"],
                )
            )

        return results
    finally:
        if conn:
            conn.close()
