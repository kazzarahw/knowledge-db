"""Index pipeline orchestrator: chunk -> store (FTS5)."""

from __future__ import annotations

import hashlib
import signal
import sqlite3
import sys
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
from knowledge.fetch import fetch_sources, get_git_head
from knowledge.sources import Source, load_sources


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


def _fts5_sync_sections(
    conn: sqlite3.Connection,
    source_name: str,
    sections: list[Section],
) -> None:
    """Insert sections and sync FTS5 indexes within an active transaction.

    Phase 1: Delete old FTS5 entries for this source (must precede section delete).
    Phase 2: Delete old sections for this source.
    Phase 3: Insert new sections.
    Phase 4: Insert into FTS5 tables with matching rowids.

    Args:
        conn: Open database connection (in transaction).
        source_name: Source name to re-index.
        sections: List of Section dataclass instances to insert.
    """
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
        return

    section_tuples = [
        (s.source, s.title, s.category, s.path, s.heading_path, s.body)
        for s in sections
    ]
    conn.executemany(
        "INSERT INTO sections (source, title, category, path, heading_path, body) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        section_tuples,
    )

    sec_ids = conn.execute(
        "SELECT id FROM sections WHERE source = ? ORDER BY id",
        (source_name,),
    ).fetchall()

    fts_tuples = [
        (row_id[0], s.title, s.heading_path, s.body)
        for row_id, s in zip(sec_ids, sections)
    ]
    conn.executemany(
        "INSERT INTO sections_fts(rowid, title, heading_path, body) "
        "VALUES (?, ?, ?, ?)",
        fts_tuples,
    )
    fts_title_tuples = [
        (row_id[0], s.title, s.heading_path) for row_id, s in zip(sec_ids, sections)
    ]
    conn.executemany(
        "INSERT INTO sections_fts_title(rowid, title, heading_path) VALUES (?, ?, ?)",
        fts_title_tuples,
    )


def _index_source(
    source: Source,
    conn: sqlite3.Connection,
    data_dir: Path,
    verbose: bool,
    current_head: str | None = None,
    cfg: Config | None = None,
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
            sections = chunk_file(fpath, source.name, source.category, rel_path=rel)
            all_sections.extend(sections)
        except Exception as e:
            if verbose:
                print(f"    Warning: error processing {fpath}: {e}")
            continue

    _fts5_sync_sections(conn, source.name, all_sections)

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

    return len(all_sections)


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

    if force:
        conn.executescript("DROP TABLE IF EXISTS sections_fts_title")
        conn.executescript("DROP TABLE IF EXISTS sections_fts")
        conn.executescript("DROP TABLE IF EXISTS sections")
        conn.executescript("DROP TABLE IF EXISTS source_state")
        conn.executescript("DROP TABLE IF EXISTS index_meta")
        ensure_schema(conn)
        conn.execute("VACUUM")
    else:
        tables_exist = (
            conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='sections'"
            ).fetchone()[0]
            > 0
        )
        if not tables_exist:
            ensure_schema(conn)

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

    signal.signal(signal.SIGINT, _on_sigint)

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
                    None if not source_dir.exists() else _source_signature(source_dir)
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
