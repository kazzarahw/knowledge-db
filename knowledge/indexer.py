"""Index pipeline orchestrator: chunk -> embed -> store."""

from __future__ import annotations

import hashlib
import signal
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType

import numpy as np

from knowledge.chunk import Section, chunk_file
from knowledge.config import resolve_data_dir, ensure_data_dir, resolve_sources_yaml
from knowledge.db import get_connection, ensure_schema
from knowledge.embed import SentenceTransformerEmbedder, get_embedder
from knowledge.fetch import fetch_sources, get_git_head
from knowledge.sources import Source, load_sources

DOC_EXTENSIONS: set[str] = {
    ".md",
    ".markdown",
    ".mdx",
    ".rst",
    ".txt",
    ".yml",
    ".yaml",
    ".ipynb",
}

BATCH_SIZE = 32


def _source_signature(source_dir: Path) -> str | None:
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


def _walk_files(source_dir: Path, source: Source) -> list[Path]:
    base = source_dir
    if source.docs_dir:
        base = source_dir / source.docs_dir
        if not base.exists():
            base = source_dir

    extra_exts = set(source.index_ext or ())
    valid_exts = DOC_EXTENSIONS | extra_exts

    files: list[Path] = []
    for fpath in base.rglob("*"):
        if fpath.is_file() and fpath.suffix.lower() in valid_exts:
            if any(
                part.startswith(".") for part in fpath.relative_to(source_dir).parts
            ):
                continue
            files.append(fpath)
    return sorted(files)


def _index_source(
    source: Source,
    embedder: SentenceTransformerEmbedder,
    conn: sqlite3.Connection,
    data_dir: Path,
    verbose: bool,
    current_head: str | None = None,
) -> int:
    source_dir = data_dir / "sources" / source.name
    if not source_dir.exists():
        print(f"  Warning: source directory for '{source.name}' not found -- skipping")
        return 0

    files = _walk_files(source_dir, source)
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

    if not all_sections:
        return 0

    texts = [s.body for s in all_sections]
    all_embeddings: list[np.ndarray] = []

    iterator = range(0, len(texts), BATCH_SIZE)
    try:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc=f"  Embedding {source.name}", leave=False)
    except ImportError:
        pass

    for batch_start in iterator:
        batch_texts = texts[batch_start : batch_start + BATCH_SIZE]
        batch_embeddings = embedder.embed(batch_texts)
        all_embeddings.append(batch_embeddings)

    if not all_embeddings:
        return 0
    embedding_matrix = np.vstack(all_embeddings)

    conn.execute("DELETE FROM sections WHERE source = ?", (source.name,))
    conn.execute("DELETE FROM section_vectors WHERE source = ?", (source.name,))

    section_rows = [
        (s.source, s.title, s.category, s.path, s.heading_path, s.body)
        for s in all_sections
    ]
    conn.executemany(
        "INSERT INTO sections (source, title, category, path, heading_path, body) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        section_rows,
    )

    sec_ids = conn.execute(
        "SELECT id FROM sections WHERE source = ? ORDER BY id",
        (source.name,),
    ).fetchall()

    vec_rows = [
        (sid[0], source.name, vec.tobytes())
        for sid, vec in zip(sec_ids, embedding_matrix)
    ]
    conn.executemany(
        "INSERT INTO section_vectors (section_id, source, embedding) VALUES (?, ?, ?)",
        vec_rows,
    )

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
    """Index all configured sources: chunk → embed → store.

    Validates embedding dimension against stored index metadata.
    Supports ``--force`` for full rebuild and SIGINT for graceful interruption.
    Cleans up orphan entries for sources no longer configured.

    Args:
        config_dir: Override config directory path.
        force: Drop and recreate the entire index.
        verbose: Print per-file and per-source progress.
    """
    data_dir = ensure_data_dir(resolve_data_dir(config_dir))
    sources = load_sources(resolve_sources_yaml(config_dir))
    db_path = data_dir / "index.db"
    conn = get_connection(db_path)

    try:
        embedder = get_embedder(config_dir=config_dir)
    except BaseException:
        conn.close()
        raise
    dim = embedder.dim
    model_name = embedder.model_name

    # Check if database has existing tables (safe before ensure_schema)
    tables_exist = (
        conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='sections'"
        ).fetchone()[0]
        > 0
    )

    if tables_exist and not force:
        has_meta = conn.execute("SELECT COUNT(*) FROM index_meta").fetchone()[0] > 0
        if not has_meta:
            conn.close()
            print(
                "Warning: Index metadata missing -- index may be corrupt. "
                "Run 'kdb index --force' to rebuild."
            )
            sys.exit(1)
        existing_dim = conn.execute(
            "SELECT value FROM index_meta WHERE key = 'embedding_dim'"
        ).fetchone()
        if existing_dim:
            try:
                stored_dim = int(existing_dim[0])
            except (ValueError, TypeError):
                conn.close()
                print(
                    f"Warning: corrupt index metadata (embedding_dim='{existing_dim[0]}'). "
                    "Run 'kdb index --force' to rebuild."
                )
                sys.exit(1)
            if stored_dim != dim:
                conn.close()
                print(
                    f"Error: Model dimension ({dim}) differs from stored index ({stored_dim}). "
                    "Run 'kdb index --force' to rebuild."
                )
                sys.exit(1)

    if force:
        conn.executescript("DROP TABLE IF EXISTS sections")
        conn.executescript("DROP TABLE IF EXISTS section_vectors")
        conn.executescript("DROP TABLE IF EXISTS source_state")
        conn.executescript("DROP TABLE IF EXISTS index_meta")
        ensure_schema(conn, dim)
        conn.execute("VACUUM")
    elif not tables_exist:
        ensure_schema(conn, dim)

    def _on_sigint(signum: int, frame: FrameType | None) -> None:
        print("\nInterrupted. Index is partial. Run 'kdb index' to resume.")
        try:
            conn.execute(
                "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('index_status', 'interrupted')"
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
                source, embedder, conn, data_dir, verbose, current_head=current_head
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
            f"DELETE FROM sections WHERE source NOT IN ({placeholders})",
            configured_names,
        )
        conn.execute(
            f"DELETE FROM section_vectors WHERE source NOT IN ({placeholders})",
            configured_names,
        )
        conn.execute(
            f"DELETE FROM source_state WHERE name NOT IN ({placeholders})",
            configured_names,
        )

    conn.execute(
        "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('embedding_model', ?)",
        (model_name,),
    )
    conn.execute(
        "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('embedding_dim', ?)",
        (str(dim),),
    )
    if source_failures == 0:
        conn.execute(
            "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('indexed_at', ?)",
            (datetime.now(timezone.utc).isoformat(),),
        )

    conn.commit()
    conn.close()
    if source_failures:
        print(
            f"Index complete with {source_failures} source(s) failed.", file=sys.stderr
        )
    else:
        print("Index complete.")
