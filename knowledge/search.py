"""Vector search via sqlite-vec CTE pattern."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from knowledge.config import resolve_data_dir
from knowledge.db import get_connection
from knowledge.embed import get_embedder


def cmd_search(
    query: str,
    top_k: int = 10,
    source: str | None = None,
    json_output: bool = False,
    config_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Search the index. Returns list of result dicts ordered by relevance.

    Validates that the embedding model dimension matches the stored index
    to prevent silent garbage results from model mismatch.
    """
    data_dir = resolve_data_dir(config_dir)
    db_path = data_dir / "index.db"

    if not db_path.exists():
        print("Error: No index found. Run 'kdb index' first.")
        return []

    conn = get_connection(db_path)
    has_sections = (
        conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='sections'"
        ).fetchone()[0]
        > 0
    )
    if not has_sections:
        print("Error: No index found. Run 'kdb index' first.")
        conn.close()
        return []

    # Validate model dimension matches stored index
    stored_dim = conn.execute(
        "SELECT value FROM index_meta WHERE key = 'embedding_dim'"
    ).fetchone()
    embedder = get_embedder(config_dir=config_dir)
    if stored_dim and embedder.dim != int(stored_dim[0]):
        print(
            f"Error: Model dimension ({embedder.dim}) differs from stored index ({stored_dim[0]}). Rebuild with 'kdb index --force'."
        )
        conn.close()
        return []

    query_vec = embedder.embed_query(query)
    serialized = query_vec.tobytes()

    source_filter = ""
    source_params: list[str] = []
    if source:
        source_filter = "AND section_vectors.source = ?"
        source_params = [source]

    rows = conn.execute(
        """
        WITH knn_matches AS (
            SELECT section_id, distance
            FROM section_vectors
            WHERE embedding MATCH ?
              AND k = ?
              {source_filter}
        )
        SELECT s.source, s.title, s.category, s.path,
               s.heading_path, s.body, m.distance
        FROM knn_matches m
        JOIN sections s ON s.id = m.section_id
        ORDER BY m.distance
        """.format(source_filter=source_filter),
        [serialized, top_k, *source_params],
    ).fetchall()

    results = []
    for row in rows:
        results.append(
            {
                "source": row["source"],
                "title": row["title"],
                "category": row["category"],
                "path": row["path"],
                "heading_path": row["heading_path"],
                "body": row["body"],
                "distance": row["distance"],
            }
        )

    conn.close()
    return results
