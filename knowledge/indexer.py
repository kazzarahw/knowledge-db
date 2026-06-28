"""Index pipeline orchestrator: chunk -> store (FTS5)."""

from __future__ import annotations

import hashlib
import logging
import signal
import sqlite3
import sys
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType

from knowledge.chunk import Section, chunk_file
from knowledge.config import (
    Config,
    load_config,
    resolve_data_dir,
    ensure_data_dir,
    resolve_sources_yaml,
)
from knowledge.db import get_connection, ensure_schema
from knowledge.fetch import get_git_head
from knowledge.sources import Source, load_sources

logger = logging.getLogger(__name__)


def _source_signature(source_dir: Path) -> str | None:
    """Compute a content-based hash for local source change detection."""
    if not source_dir.exists():
        return None
    h = hashlib.sha256()
    for fpath in sorted(source_dir.rglob("*")):
        if fpath.is_file():
            h.update(str(fpath.relative_to(source_dir)).encode())
            try:
                with open(fpath, "rb") as f:
                    h.update(f.read(4096))
                h.update(str(fpath.stat().st_size).encode())
            except OSError:
                pass
    return h.hexdigest()[:32]


def _walk_files(
    source_dir: Path, source: Source, cfg: Config | None = None
) -> list[Path]:
    """Walk source directory for indexable files matching doc_extensions."""
    if cfg is None:
        cfg = load_config()

    base = source_dir
    if source.docs_dir:
        base = source_dir / source.docs_dir
        if not base.exists():
            base = source_dir

    extra_exts = set(source.index_ext or ())
    valid_exts = set(cfg.index.doc_extensions) | extra_exts

    files: list[Path] = []
    for fpath in base.rglob("*"):
        if fpath.is_file() and fpath.suffix.lower() in valid_exts:
            if any(
                part.startswith(".") for part in fpath.relative_to(source_dir).parts
            ):
                continue
            files.append(fpath)
    return sorted(files)


def _deduplicate_sections(sections: list[Section]) -> list[Section]:
    """Merge sections with duplicate (source, path, heading_path) keys.

    Bodies are joined with ``\\n\\n---\\n\\n``. Keeps the first occurrence's
    title and other metadata. Preserves original order.

    Args:
        sections: List of Section instances to deduplicate.

    Returns:
        Deduplicated list preserving insertion order.
    """
    seen: dict[tuple[str, str, str], Section] = {}
    merged: dict[tuple[str, str, str], list[str]] = {}
    order: list[tuple[str, str, str]] = []

    for s in sections:
        key = (s.source, s.path, s.heading_path)
        if key not in seen:
            seen[key] = s
            merged[key] = [s.body]
            order.append(key)
        else:
            merged[key].append(s.body)

    result: list[Section] = []
    for key in order:
        s = seen[key]
        result.append(
            Section(
                source=s.source,
                title=s.title,
                category=s.category,
                path=s.path,
                heading_path=s.heading_path,
                body="\n\n---\n\n".join(merged[key]),
            )
        )
    return result


_RANK_BIAS_MAP: dict[str, float] = {
    "wikis": 0.7,
    "ad-internal": 0.8,
    "web-api": 0.8,
    "dfir": 0.9,
    "wifi": 0.9,
    "bluetooth": 0.9,
    "c2": 1.0,
    "hardware-iot": 1.0,
    "mobile": 1.0,
    "lotl": 1.0,
    "re-books": 1.0,
    "re-tools": 1.0,
    "re-indexes": 1.0,
    "osint": 1.0,
    "glitching": 1.0,
    "sdr": 1.0,
    "firmware": 1.1,
    "compliance": 1.1,
}


def _lookup_rank_bias(category: str) -> float:
    """Return rank_bias for a category, defaulting to 1.0."""
    if not category:
        logger.debug("Empty category, using default rank_bias=1.0")
        return 1.0
    return _RANK_BIAS_MAP.get(category, 1.0)


def _fts5_sync_sections(
    conn: sqlite3.Connection,
    source_name: str,
    sections: list[Section],
    rank_bias: float | None = None,
    content_hashes_seen: set[str] | None = None,
    source_title: str = "",
) -> int:
    """Insert sections with content hashing and source-quality bias.

    Args:
        conn: Open database connection (in transaction).
        source_name: Source name to re-index.
        sections: List of Section dataclass instances to insert.
        rank_bias: Pre-computed rank bias. Falls back to
            _lookup_rank_bias(sections[0].category) if None.
        content_hashes_seen: Set of content hashes already inserted
            (cross-source dedup). Passed by caller for cumulative tracking.
        source_title: Human-readable source title (shared by all sections).
    """
    if content_hashes_seen is None:
        content_hashes_seen = set()

    conn.execute(
        "DELETE FROM sections_fts WHERE rowid IN "
        "(SELECT id FROM sections WHERE source = ?)",
        (source_name,),
    )
    conn.execute(
        "DELETE FROM sections_fts_title WHERE rowid IN "
        "(SELECT id FROM sections WHERE source = ?)",
        (source_name,),
    )
    conn.execute("DELETE FROM sections WHERE source = ?", (source_name,))

    if not sections:
        return 0

    sections = _deduplicate_sections(sections)

    if rank_bias is None:
        rank_bias = _lookup_rank_bias(sections[0].category)

    _SecRow = namedtuple(
        "_SecRow",
        "content_hash rank_bias source title category path heading_path body source_title",
    )
    to_insert: list[_SecRow] = []
    for s in sections:
        h = hashlib.sha256(s.body.encode()).hexdigest()
        if h in content_hashes_seen:
            continue
        content_hashes_seen.add(h)
        to_insert.append(
            _SecRow(
                content_hash=h,
                rank_bias=rank_bias,
                source=s.source,
                title=s.title,
                category=s.category,
                path=s.path,
                heading_path=s.heading_path,
                body=s.body,
                source_title=source_title,
            )
        )

    if not to_insert:
        return 0

    conn.executemany(
        "INSERT INTO sections "
        "(content_hash, rank_bias, source, title, category, path, "
        "heading_path, body, source_title) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        to_insert,
    )

    sec_ids = conn.execute(
        "SELECT id FROM sections WHERE source = ? ORDER BY id",
        (source_name,),
    ).fetchall()

    fts_tuples = [
        (sec_ids[i][0], t.title, t.heading_path, t.body)
        for i, t in enumerate(to_insert)
    ]
    conn.executemany(
        "INSERT INTO sections_fts(rowid, title, heading_path, body) "
        "VALUES (?, ?, ?, ?)",
        fts_tuples,
    )
    conn.executemany(
        "INSERT INTO sections_fts_title(rowid, title, heading_path) VALUES (?, ?, ?)",
        [(rowid, title, hpath) for rowid, title, hpath, _ in fts_tuples],
    )

    return len(to_insert)


def _index_source(
    source: Source,
    conn: sqlite3.Connection,
    data_dir: Path,
    verbose: bool,
    current_head: str | None = None,
    cfg: Config | None = None,
    content_hashes_seen: set[str] | None = None,
) -> int:
    """Index a single source: walk, chunk, and store via FTS5.

    Args:
        source: Source configuration dataclass.
        conn: Open database connection (caller manages transaction).
        data_dir: Root data directory.
        verbose: Print per-file progress.
        current_head: Git HEAD or source signature for state tracking.
        cfg: Application configuration.

    Returns:
        Number of sections indexed.
    """
    if cfg is None:
        cfg = load_config()

    source_dir = data_dir / "sources" / source.name
    if not source_dir.exists():
        print(f"  Warning: source directory for '{source.name}' not found -- skipping")
        return 0

    files = _walk_files(source_dir, source, cfg)
    if verbose:
        print(f"  Walking {len(files)} files in {source.name}")

    all_sections: list[Section] = []
    for fpath in files:
        try:
            rel = str(fpath.relative_to(source_dir))
            sections = chunk_file(
                fpath,
                source.name,
                source.category,
                rel_path=rel,
                source_title=source.title or source.name,
            )
            all_sections.extend(sections)
        except Exception as e:
            if verbose:
                print(f"    Warning: error processing {fpath}: {e}")
            continue

    rb = _lookup_rank_bias(source.category)
    inserted = _fts5_sync_sections(
        conn,
        source.name,
        all_sections,
        rank_bias=rb,
        content_hashes_seen=content_hashes_seen,
        source_title=source.title or source.name,
    )

    if not all_sections:
        if current_head is not None:
            conn.execute(
                "INSERT OR REPLACE INTO source_state (name, git_head, indexed_at) "
                "VALUES (?, ?, datetime('now'))",
                (source.name, current_head),
            )
        return 0

    if current_head is None:
        current_head = (
            get_git_head(source_dir)
            if source.source_type == "git"
            else _source_signature(source_dir)
        )
    conn.execute(
        "INSERT OR REPLACE INTO source_state (name, git_head, indexed_at) "
        "VALUES (?, ?, datetime('now'))",
        (source.name, current_head),
    )

    return inserted


def cmd_index(
    config_dir: str | None = None,
    force: bool = False,
    verbose: bool = False,
) -> None:
    """Index all configured sources: walk, chunk, store in FTS5.

    Supports ``--force`` for full rebuild and SIGINT for graceful interruption.
    Cleans up orphan entries for sources no longer configured.
    Creates FTS5 tables unconditionally (idempotent CREATE IF NOT EXISTS).

    Args:
        config_dir: Override config directory path.
        force: Drop and recreate the entire index.
        verbose: Print per-file and per-source progress.
    """
    cfg = load_config(config_dir)
    data_dir = ensure_data_dir(resolve_data_dir(config_dir))
    sources = load_sources(resolve_sources_yaml(config_dir))
    db_path = data_dir / "index.db"
    conn = get_connection(db_path)

    ensure_schema(conn)

    from knowledge.db import _migrate_schema

    msgs = _migrate_schema(conn)
    for m in msgs:
        print(f"  Schema migration: {m}")

    content_hashes_seen: set[str] = set()

    if not force:
        null_hashes = conn.execute(
            "SELECT COUNT(*) FROM sections WHERE content_hash IS NULL"
        ).fetchone()[0]
        if null_hashes > 0:
            print(
                f"  {null_hashes} sections missing content_hash — rebuilding all sources"
            )
            conn.execute("BEGIN")
            conn.execute("DELETE FROM source_state")
            conn.execute("DELETE FROM sections_fts")
            conn.execute("DELETE FROM sections_fts_title")
            conn.execute("DELETE FROM sections")
            conn.commit()

    if force:
        conn.executescript("DROP TABLE IF EXISTS sections_fts_title")
        conn.executescript("DROP TABLE IF EXISTS sections_fts")
        conn.executescript("DROP TABLE IF EXISTS sections")
        conn.executescript("DROP TABLE IF EXISTS source_state")
        conn.executescript("DROP TABLE IF EXISTS index_meta")
        ensure_schema(conn)
        conn.execute("VACUUM")

    def _on_sigint(signum: int, frame: FrameType | None) -> None:
        print("\nInterrupted. Index is partial. Run 'kdb index' to resume.")
        try:
            conn.execute(
                "INSERT OR REPLACE INTO index_meta (key, value) VALUES "
                "('index_status', 'interrupted')"
            )
            conn.commit()
        except Exception:
            conn.rollback()
        conn.close()
        sys.exit(130)

    old_handler = signal.signal(signal.SIGINT, _on_sigint)
    try:
        source_failures = 0
        for source in sources:
            source_dir = data_dir / "sources" / source.name

            if source.source_type == "git" and not source_dir.exists():
                print(f"Skipping {source.name} -- not cloned. Run 'kdb fetch' first.")
                continue

            if force:
                needs_index = True
                current_head: str | None = None
            else:
                if source.source_type == "git":
                    current_head = get_git_head(source_dir)
                else:
                    current_head = (
                        None
                        if not source_dir.exists()
                        else _source_signature(source_dir)
                    )
                stored = conn.execute(
                    "SELECT git_head FROM source_state WHERE name = ?",
                    (source.name,),
                ).fetchone()
                needs_index = stored is None or stored[0] != current_head

            if not needs_index:
                if verbose:
                    print(f"  {source.name}: unchanged, skipping")
                continue

            print(f"Indexing {source.name}...")
            try:
                conn.execute("BEGIN")
                num_sections = _index_source(
                    source,
                    conn,
                    data_dir,
                    verbose,
                    current_head=current_head,
                    cfg=cfg,
                    content_hashes_seen=content_hashes_seen,
                )
                conn.commit()
                if verbose:
                    print(f"  {source.name}: {num_sections} sections indexed")
            except Exception as e:
                source_failures += 1
                conn.rollback()
                print(f"  Error indexing {source.name}: {e}", file=sys.stderr)
                continue

        configured_names = [s.name for s in sources]
        if configured_names:
            conn.execute("BEGIN")
            placeholders = ",".join("?" * len(configured_names))
            conn.execute(
                f"DELETE FROM sections_fts WHERE rowid IN "
                f"(SELECT id FROM sections WHERE source NOT IN ({placeholders}))",
                configured_names,
            )
            conn.execute(
                f"DELETE FROM sections_fts_title WHERE rowid IN "
                f"(SELECT id FROM sections WHERE source NOT IN ({placeholders}))",
                configured_names,
            )
            conn.execute(
                f"DELETE FROM sections WHERE source NOT IN ({placeholders})",
                configured_names,
            )
            conn.execute(
                f"DELETE FROM source_state WHERE name NOT IN ({placeholders})",
                configured_names,
            )

        conn.execute(
            "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('indexed_at', ?)",
            (datetime.now(timezone.utc).isoformat(),),
        )

        if source_failures == 0:
            conn.commit()
            conn.close()
            print("Index complete.")
        else:
            conn.commit()
            conn.close()
            print(
                f"Index complete with {source_failures} source(s) failed.",
                file=sys.stderr,
            )

        if source_failures > 0:
            sys.exit(1)
    finally:
        signal.signal(signal.SIGINT, old_handler)
