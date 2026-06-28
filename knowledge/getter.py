"""Hash-prefix section retrieval for ``kdb get``."""

from __future__ import annotations

import sys

from knowledge.config import resolve_data_dir
from knowledge.db import get_connection


def cmd_get(
    hash_prefix: str,
    config_dir: str | None = None,
) -> dict | None:
    """Retrieve a section by content hash prefix.

    Args:
        hash_prefix: At least 10 hex characters. Lowercased automatically.
        config_dir: Override config directory path.

    Returns:
        Section dict with keys hash, source, title, category, path,
        heading_path, body, or None if no/ambiguous match.
    """
    hash_prefix = hash_prefix.lower().strip()
    # Duplicate of cli._validate_hex_prefix — serves as defense-in-depth
    # for programmatic callers that bypass argparse.
    if not all(c in "0123456789abcdef" for c in hash_prefix):
        print("Error: hash prefix must be hex characters only", file=sys.stderr)
        return None
    if len(hash_prefix) < 10:
        print("Error: hash prefix must be at least 10 hex characters", file=sys.stderr)
        return None

    data_dir = resolve_data_dir(config_dir)
    db_path = data_dir / "index.db"

    if not db_path.exists():
        print("Error: No index found. Run 'kdb index' first.", file=sys.stderr)
        return None

    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT content_hash, source, title, category, path, heading_path, body "
            "FROM sections WHERE content_hash LIKE ?",
            (hash_prefix + "%",),
        ).fetchall()

        if not rows:
            print(f"No section with hash prefix '{hash_prefix}'", file=sys.stderr)
            return None

        if len(rows) > 1:
            print(
                f"Ambiguous hash prefix '{hash_prefix}' matches {len(rows)} sections:\n"
                + "\n".join(
                    f"  {r['content_hash']}  {r['source']}: {r['title']}" for r in rows
                ),
                file=sys.stderr,
            )
            return None

        row = rows[0]
        return {
            "content_hash": row["content_hash"],
            "source": row["source"],
            "title": row["title"],
            "category": row["category"],
            "path": row["path"],
            "heading_path": row["heading_path"],
            "body": row["body"],
        }
    finally:
        conn.close()
