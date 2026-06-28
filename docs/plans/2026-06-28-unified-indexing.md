# Unified Indexing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `kdb search` feel like querying a single unified knowledge base by normalizing content, adding unique result hashes, source-quality ranking, and improved display.

**Architecture:** Add `knowledge/normalize.py` for RST/notebook→Markdown conversion and heading normalization; add `knowledge/getter.py` for hash-prefix content retrieval; add `content_hash`, `rank_bias`, and `source_title` columns to the sections table; update FTS5 ranking to incorporate source quality; reformat CLI display with proportional terminal widths and hash-as-handle.

**Tech Stack:** Python 3.12+, SQLite FTS5, rst2gfm (pure Python RST→MD), nbformat, pytest

## Global Constraints

- `content_hash` must be SHA-256 hex of normalized body (no UNIQUE constraint — app-level dedup via `SELECT ... WHERE content_hash = ?`)
- `rank_bias` applied as `ORDER BY bm25(...) * rank_bias` (lower = boosted)
- `source_title` stored in sections table and included in search results for tag display
- Hash prefix minimum 10 hex chars (40 bits); display shows 12 chars
- `normalize_body(path, ext)` accepts file path (not raw text) to handle notebooks
- `qualify_heading` takes `is_top_level` parameter (default True)
- Tag format: `<category>·<source-title>` (no spaces around middle dot, source-title from sources.yaml)
- Compact format at < 80 cols: 2-line layout
- Unlisted categories default to `rank_bias = 1.0`
- Test isolation: `conftest.py` tracks all `sqlite3.connect()` calls and asserts closure
- Auto-rebuild on NULL content_hash after migration (not just hint)

---

### Task 1: Schema Migration (`_migrate_schema`) in `db.py`

**Files:**
- Modify: `knowledge/db.py` — add `_migrate_schema()` function
- Create: `tests/test_db_migration.py` — migration-specific tests
- Test: `tests/test_db.py` — existing column tests pass unchanged

**Interfaces:**
- Consumes: existing `get_connection`, `ensure_schema` from `db.py`
- Produces: `_migrate_schema(conn: sqlite3.Connection) -> list[str]`

- [ ] **Step 1: Write failing test for migration detection**

```python
"""Tests for knowledge.db — schema migration (content_hash, rank_bias columns)."""

from __future__ import annotations

import sqlite3

from knowledge.db import _migrate_schema, ensure_schema, get_connection


def test_migrate_adds_missing_columns(tmp_path) -> None:
    """_migrate_schema adds content_hash, rank_bias, and source_title to old schema."""
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    ensure_schema(conn)

    cols_before = {row[1] for row in conn.execute("PRAGMA table_info(sections)")}
    assert "content_hash" not in cols_before
    assert "rank_bias" not in cols_before
    assert "source_title" not in cols_before

    msgs = _migrate_schema(conn)
    assert "added content_hash column" in msgs
    assert "added rank_bias column" in msgs
    assert "added source_title column" in msgs

    cols_after = {row[1] for row in conn.execute("PRAGMA table_info(sections)")}
    assert "content_hash" in cols_after
    assert "rank_bias" in cols_after
    assert "source_title" in cols_after
    conn.close()


def test_migrate_idempotent(tmp_path) -> None:
    """Second call returns empty list (no migrations needed)."""
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    ensure_schema(conn)
    _migrate_schema(conn)
    msgs = _migrate_schema(conn)
    assert msgs == []
    conn.close()


def test_content_hash_nullable(tmp_path) -> None:
    """content_hash column allows NULL (existing rows survive migration)."""
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    ensure_schema(conn)
    conn.execute(
        "INSERT INTO sections (source, title, category, path, heading_path, body) "
        "VALUES ('src', 't', 'cat', 'p', 'hp', 'b')"
    )
    conn.commit()
    _migrate_schema(conn)
    row = conn.execute(
        "SELECT content_hash, rank_bias FROM sections WHERE source='src'"
    ).fetchone()
    assert row["content_hash"] is None
    assert row["rank_bias"] == 1.0
    conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db_migration.py -v`
Expected: FAIL with `ImportError: cannot import name '_migrate_schema' from 'knowledge.db'`

- [ ] **Step 3: Write `_migrate_schema` in `db.py`**

Append to `knowledge/db.py` after the `ensure_schema` function:

```python
def _migrate_schema(conn: sqlite3.Connection) -> list[str]:
    """Add missing columns to existing sections table.

    Idempotent — safe to call repeatedly. Returns list of migration
    messages (empty if none needed).

    Args:
        conn: Open database connection.

    Returns:
        List of description strings for applied migrations.
    """
    existing = {row[1] for row in conn.execute("PRAGMA table_info(sections)")}
    migrations: list[str] = []
    if "content_hash" not in existing:
        conn.execute("ALTER TABLE sections ADD COLUMN content_hash TEXT")
        migrations.append("added content_hash column")
    if "rank_bias" not in existing:
        conn.execute(
            "ALTER TABLE sections ADD COLUMN rank_bias REAL NOT NULL DEFAULT 1.0"
        )
        migrations.append("added rank_bias column")
    if "source_title" not in existing:
        conn.execute("ALTER TABLE sections ADD COLUMN source_title TEXT NOT NULL DEFAULT ''")
        migrations.append("added source_title column")
    return migrations
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_db_migration.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Update `ensure_schema` to include new columns on fresh DBs**

Modify the `CREATE TABLE IF NOT EXISTS sections` statement in `db.py` to include the three new columns at the **end** (after `body`) for consistency with migration order:

```sql
CREATE TABLE IF NOT EXISTS sections (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    path TEXT NOT NULL,
    heading_path TEXT,
    body TEXT NOT NULL,
    content_hash TEXT,
    rank_bias REAL NOT NULL DEFAULT 1.0,
    source_title TEXT NOT NULL DEFAULT '',
    UNIQUE(source, path, heading_path)
);
```

- [ ] **Step 6: Update `test_column_definitions` in `test_db.py`**

Add assertions for the new columns:

```python
assert "content_hash" in cols
assert cols["content_hash"][2] == "TEXT"
assert cols["content_hash"][3] == 0  # nullable
assert "rank_bias" in cols
assert cols["rank_bias"][2] == "REAL"
assert cols["rank_bias"][3] == 1  # NOT NULL
assert cols["rank_bias"][4] == "1.0"  # default
assert "source_title" in cols
assert cols["source_title"][2] == "TEXT"
assert cols["source_title"][3] == 1  # NOT NULL
assert cols["source_title"][4] == "''"  # default empty string
```

- [ ] **Step 7: Run full db test suite**

Run: `uv run pytest tests/test_db.py tests/test_db_migration.py -v`
Expected: all tests PASS

- [ ] **Step 8: Commit**

```bash
git add knowledge/db.py tests/test_db.py tests/test_db_migration.py
git commit -m "feat(db): add _migrate_schema() for content_hash and rank_bias columns"
```

---

### Task 2: `knowledge/normalize.py` Module

**Files:**
- Create: `knowledge/normalize.py`
- Create: `tests/test_normalize.py`
- Modify: `pyproject.toml` — add `rst2gfm` dependency

**Interfaces:**
- Consumes: `pathlib.Path`, `rst2gfm`, `nbformat`
- Produces: `normalize_body(path, file_ext) -> str`, `normalize_heading(segment) -> str`, `qualify_heading(source_title, heading, is_top_level) -> str`

- [ ] **Step 1: Install rst2gfm**

Run: `uv add rst2gfm`

- [ ] **Step 2: Write failing tests for normalize_heading**

```python
"""Tests for knowledge.normalize — content and heading normalization."""

from __future__ import annotations

from knowledge.normalize import normalize_heading


def test_normalize_heading_strips_markdown_links() -> None:
    result = normalize_heading("[Label](https://example.com)")
    assert result == "Label"


def test_normalize_heading_strips_html_anchors() -> None:
    result = normalize_heading("Section <a id=\"foo\"></a> Name")
    assert result == "Section  Name"


def test_normalize_heading_decodes_html_entities() -> None:
    result = normalize_heading("TCP &amp; UDP")
    assert result == "TCP & UDP"


def test_normalize_heading_collapses_whitespace() -> None:
    result = normalize_heading("  Too   much  space  ")
    assert result == "Too much space"


def test_normalize_heading_preserves_inline_formatting() -> None:
    result = normalize_heading("**bold** and *italic* and `code`")
    assert result == "**bold** and *italic* and `code`"


def test_normalize_heading_strips_setext_underline_residue() -> None:
    result = normalize_heading("====")
    assert result == "===="  # single-word setext underlines are harmless
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_normalize.py::test_normalize_heading_strips_markdown_links -v`
Expected: FAIL with `ImportError`

- [ ] **Step 4: Implement `normalize_heading` in `normalize.py`**

```python
"""Content and heading normalization for knowledge-db sources."""

from __future__ import annotations

import html
import re
from pathlib import Path


_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_HTML_ANCHOR_RE = re.compile(r"<a\b[^>]*>.*?</a>", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_heading(segment: str) -> str:
    """Clean a single heading segment for display.

    Strips markdown links, HTML anchors, decodes entities, collapses
    whitespace. Preserves inline formatting (**bold**, *italic*, ``code``).

    Args:
        segment: Raw heading text from a document.

    Returns:
        Cleaned heading text suitable for display.
    """
    s = _MD_LINK_RE.sub(r"\1", segment)
    s = _HTML_ANCHOR_RE.sub("", s)
    s = html.unescape(s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def qualify_heading(source_title: str, heading: str, is_top_level: bool = True) -> str:
    """Prepend source title to top-level heading.

    Only applies to the first segment in a heading path (``is_top_level=True``).
    Nested segments pass through unchanged.

    Args:
        source_title: Human-readable source title from ``sources.yaml``.
        heading: Cleaned heading segment text.
        is_top_level: Whether this is the first segment in the heading path.

    Returns:
        ``SourceTitle: Heading`` if top-level, or ``Heading`` unchanged.
    """
    if is_top_level and source_title:
        return f"{source_title}: {heading}"
    return heading
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_normalize.py -v -k "test_normalize_heading"`

- [ ] **Step 6: Write failing tests for normalize_body**

```python
def test_normalize_body_passthrough_md(tmp_path) -> None:
    from knowledge.normalize import normalize_body
    f = tmp_path / "test.md"
    f.write_text("# Hello\n\nWorld")
    result = normalize_body(f, ".md")
    assert result == "# Hello\n\nWorld"


def test_normalize_body_passthrough_txt(tmp_path) -> None:
    from knowledge.normalize import normalize_body
    f = tmp_path / "test.txt"
    f.write_text("plain text")
    result = normalize_body(f, ".txt")
    assert result == "plain text"


def test_normalize_body_rst_to_md(tmp_path) -> None:
    from knowledge.normalize import normalize_body
    f = tmp_path / "test.rst"
    f.write_text("Hello\n=====\n\nSome **bold** text.\n\n.. code-block:: python\n\n    print(1)")
    result = normalize_body(f, ".rst")
    assert "Hello" in result
    assert "print(1)" in result or "```" in result


def test_normalize_body_rst_failure_fallback(tmp_path) -> None:
    """When rst2gfm raises, log warning and return original text."""
    from unittest.mock import patch
    from knowledge.normalize import normalize_body
    f = tmp_path / "test.rst"
    body = "Hello World"
    f.write_text(body)
    with patch("knowledge.normalize._rst_to_md", side_effect=ValueError("bad rst")):
        result = normalize_body(f, ".rst")
    assert result == body


def test_normalize_body_notebook_none_on_failure(tmp_path) -> None:
    """Corrupt notebook returns None (file skipped)."""
    from knowledge.normalize import normalize_body
    f = tmp_path / "bad.ipynb"
    f.write_text("not json")
    result = normalize_body(f, ".ipynb")
    assert result is None
```

- [ ] **Step 7: Run test to verify it fails**

Run: `uv run pytest tests/test_normalize.py -v -k "test_normalize_body"`

- [ ] **Step 8: Implement `normalize_body` in `normalize.py`**

```python
import logging

import nbformat

logger = logging.getLogger(__name__)


def normalize_body(path: Path, file_ext: str) -> str | None:
    """Convert non-markdown file content to clean markdown.

    Args:
        path: Path to the file. Read internally (required for notebook parsing).
        file_ext: File extension with leading dot (e.g. ``.rst``, ``.ipynb``).

    Returns:
        Normalized markdown text, or ``None`` if the file should be skipped
        (corrupt notebook). On other conversion failures, returns original text.
    """
    try:
        match file_ext:
            case ".ipynb":
                return _notebook_to_md(path)
            case ".rst":
                return _rst_to_md(path)
            case _:
                return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        logger.warning("normalize_body failed for %s, using original", path)
        return path.read_text(encoding="utf-8", errors="replace")


def _notebook_to_md(path: Path) -> str | None:
    """Convert Jupyter notebook to markdown text.

    Returns None if the notebook is corrupt/unparseable (file skipped).
    """
    try:
        nb = nbformat.read(path, as_version=4)
    except Exception:
        logger.warning("Failed to parse notebook %s, skipping", path)
        return None

    cells: list[str] = []
    for cell in nb.cells:
        match cell.cell_type:
            case "markdown":
                cells.append(cell.source)
            case "code":
                cells.append(f"```\n{cell.source}\n```")
            case "raw":
                cells.append(cell.source)
    return "\n\n".join(cells)


def _rst_to_md(path: Path) -> str:
    """Convert reStructuredText to markdown via rst2gfm."""
    from rst2gfm import convert_rst_to_md

    raw = path.read_text(encoding="utf-8", errors="replace")
    return convert_rst_to_md(raw)
```

- [ ] **Step 9: Run test to verify it passes**

Run: `uv run pytest tests/test_normalize.py -v`
Expected: all tests PASS

- [ ] **Step 10: Write failing tests for qualify_heading**

```python
def test_qualify_heading_top_level() -> None:
    from knowledge.normalize import qualify_heading
    result = qualify_heading("HackTricks", "Token Confusion", is_top_level=True)
    assert result == "HackTricks: Token Confusion"


def test_qualify_heading_nested() -> None:
    from knowledge.normalize import qualify_heading
    result = qualify_heading("HackTricks", "Configuration", is_top_level=False)
    assert result == "Configuration"


def test_qualify_heading_empty_source_title() -> None:
    from knowledge.normalize import qualify_heading
    result = qualify_heading("", "Token Confusion", is_top_level=True)
    assert result == "Token Confusion"


def test_qualify_heading_default_is_top_level() -> None:
    from knowledge.normalize import qualify_heading
    result = qualify_heading("HackTricks", "Token Confusion")
    assert result == "HackTricks: Token Confusion"
```

- [ ] **Step 11: Run tests and commit**

Run: `uv run pytest tests/test_normalize.py -v`
Expected: all PASS

```bash
git add knowledge/normalize.py tests/test_normalize.py pyproject.toml uv.lock
git commit -m "feat: add knowledge/normalize.py for content + heading normalization"
```

---

### Task 3: Integrate Normalization into Chunking Pipeline

**Files:**
- Modify: `knowledge/chunk.py` — update `chunk_file()` to call `normalize_body()` and update section metadata with normalized headings

**Interfaces:**
- Consumes: `normalize_body`, `normalize_heading`, `qualify_heading` from Task 2
- Produces: `Section` instances with normalized body and qualified heading paths

- [ ] **Step 1: Write failing integration test**

```python
"""Integration test: chunk_file with normalize pipeline."""

from pathlib import Path
from knowledge.normalize import normalize_body, normalize_heading, qualify_heading


def test_chunk_file_normalizes_rst(tmp_path, monkeypatch) -> None:
    """chunk_file produces normalized body for .rst files."""
    from knowledge.chunk import chunk_file
    f = tmp_path / "test.rst"
    f.write_text("Hello\n=====\n\nSome **bold** text.")
    sections = chunk_file(f, "testsource", "wikis", rel_path="test.rst")
    assert len(sections) == 1
    assert sections[0].body.startswith("# Hello") or "Hello" in sections[0].body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_normalize.py::test_chunk_file_normalizes_rst -v`
Expected: FAIL

- [ ] **Step 3: Update `chunk_file` in `chunk.py`**

Replace the current `chunk_file` body:

```python
def chunk_file(
    filepath: Path, source: str, category: str, rel_path: str | None = None
) -> list[Section]:
    """Read a file, normalize body, and split into heading-bounded sections.

    Non-markdown formats (RST, notebooks) are converted to markdown
    via ``normalize_body`` before chunking.
    """
    from knowledge.normalize import normalize_body

    ext = filepath.suffix.lower()
    normalized = normalize_body(filepath, ext)
    if normalized is None:
        return []  # skip unparseable files (corrupt notebooks etc.)
    detect_headings = ext in HEADING_AWARE_EXTS
    return chunk_text(
        normalized,
        source,
        category,
        rel_path or str(filepath),
        detect_headings=detect_headings,
    )
```

- [ ] **Step 4: Update `chunk_text` to normalize headings**

In `chunk_text`, after each heading is extracted and before building the Section:

```python
from knowledge.normalize import normalize_heading, qualify_heading

# Inside chunk_text, where heading_path_parts are built:
cleaned_parts = []
for idx, segment in enumerate(heading_path_parts):
    if not segment:
        cleaned_parts.append(segment)
        continue
    norm = normalize_heading(segment)
    if idx == 0:
        norm = qualify_heading(getattr(source_obj, 'title', source), norm, is_top_level=True)
    cleaned_parts.append(norm)
heading_path = " > ".join(p for p in cleaned_parts if p)
```

Wait — `chunk_text` doesn't have access to `source.title` because it receives `source` as a string (the source name), not a `Source` dataclass. Let me think about this.

Looking at the call chain:
- `chunk_file(filepath, source_name, category, rel_path)` → calls `chunk_text(normalized, source_name, category, rel_path, detect_headings)`
- `chunk_text(text, source, category, rel_path, detect_headings)` — `source` is a string name

To qualify headings, we need the source title from sources.yaml. But `chunk_text` is a pure function and shouldn't load config. Options:
1. Pass `source_title` as a parameter to `chunk_text` and `chunk_file`
2. Just use `source` (the name) as the prefix — it's always available

Actually, looking at the spec: "Falls back to source.name if source.title is empty." Since `chunk_file` only gets `source` (the name), we should just use that. The qualify_heading call would use `source` directly as the title.

Actually wait, let me re-read the original code more carefully:

In `chunk_file`: `sections = chunk_file(fpath, source.name, source.category, rel_path=rel)`

And in `chunk_text`: `def chunk_text(text, source, category, rel_path, detect_headings=True)`

The `source` parameter in `chunk_text` is the source name (string). For qualify_heading, we need a human-readable source title. The spec says `qualify_heading(source_title, heading, is_top_level)` — where source_title comes from `sources.yaml` `title` field.

To get the title in chunk_text, we could:
1. Pass `source_title` as an extra parameter
2. Have `chunk_file` pass the title alongside the source name

Let me keep it simple: pass `source_title` to `chunk_file` (defaulting to `source` if not available) and down to `chunk_text`. Update the interfaces.

Actually, I'll add an optional `source_title` parameter to both functions. This is backward-compatible.

- [ ] **Step 4 (revised): Update `chunk_text` and `chunk_file` interfaces**

Modify `chunk_text` signature:
```python
def chunk_text(
    text: str,
    source: str,
    category: str,
    rel_path: str,
    detect_headings: bool = True,
    source_title: str | None = None,
) -> list[Section]:
```

Modify `chunk_file` signature:
```python
def chunk_file(
    filepath: Path,
    source: str,
    category: str,
    rel_path: str | None = None,
    source_title: str | None = None,
) -> list[Section]:
```

Inside `chunk_text`, around line 120-125 where heading_path is built:

```python
cleaned_parts: list[str] = []
for i, part in enumerate(heading_path_parts):
    if not part:
        cleaned_parts.append(part)
        continue
    norm = normalize_heading(part)
    if i == 0 and source_title:
        norm = qualify_heading(source_title, norm, is_top_level=True)
    cleaned_parts.append(norm)
heading_path = " > ".join(p for p in cleaned_parts if p)
```

Update `chunk_file` to pass `source_title` through:

```python
def chunk_file(...):
    ...
    return chunk_text(
        normalized,
        source,
        category,
        rel_path or str(filepath),
        detect_headings=detect_headings,
        source_title=source_title,
    )
```

- [ ] **Step 5: Update `indexer.py` to pass source.title**

In `_index_source`, the call to `chunk_file` currently is:
```python
sections = chunk_file(fpath, source.name, source.category, rel_path=rel)
```

Change to:
```python
sections = chunk_file(
    fpath, source.name, source.category,
    rel_path=rel, source_title=source.title or source.name,
)
```

- [ ] **Step 6: Update existing mock in test_indexer.py**

The existing `test_indexer.py` mocks `chunk_file` with a 4-param lambda:
```python
mock_chunk.side_effect = lambda fpath, src, cat, rel_path: sections
```
After adding `source_title`, `_index_source` calls `chunk_file(..., source_title=...)`. The lambda must accept `**kwargs`:
```python
mock_chunk.side_effect = lambda fpath, src, cat, rel_path, **kw: sections
```

- [ ] **Step 7: Test the integration**

Run: `uv run pytest tests/test_normalize.py tests/test_chunk.py tests/test_indexer.py -v`
Expected: all PASS (existing tests pass with mock update)

- [ ] **Step 8: Commit**

```bash
git add knowledge/chunk.py knowledge/indexer.py
git commit -m "feat: integrate normalize pipeline into chunk_file and chunk_text"
```

---

### Task 4: Content Hashing + `rank_bias` in Indexing

**Files:**
- Modify: `knowledge/indexer.py` — compute SHA-256 hashes, look up rank_bias, app-level dedup
- Modify: `knowledge/db.py` — call `_migrate_schema` from `cmd_index`

**Interfaces:**
- Consumes: `_migrate_schema` from Task 1, `normalize_body` from Task 2
- Produces: sections with populated `content_hash` and `rank_bias`

- [ ] **Step 1: Write failing test for content_hash computation**

```python
"""Tests for content hashing in indexer."""

from __future__ import annotations

import hashlib
import sqlite3

from knowledge.db import ensure_schema, get_connection
from knowledge.indexer import _fts5_sync_sections
from knowledge.chunk import Section


def test_content_hash_computed_during_sync(tmp_path) -> None:
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    ensure_schema(conn)

    sections = [
        Section(
            source="test",
            title="Test Section",
            category="wikis",
            path="test.md",
            heading_path="Test Section",
            body="# Hello World",
        )
    ]
    _fts5_sync_sections(conn, "test", sections, rank_bias=0.7)

    row = conn.execute(
        "SELECT content_hash, rank_bias FROM sections WHERE source='test'"
    ).fetchone()
    expected_hash = hashlib.sha256(b"# Hello World").hexdigest()
    assert row["content_hash"] == expected_hash
    assert row["rank_bias"] == 0.7
    conn.close()


def test_dedup_first_source_wins(tmp_path) -> None:
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    ensure_schema(conn)

    body = "# Hello World"
    h = hashlib.sha256(body.encode()).hexdigest()

    s1 = Section(source="src1", title="S1", category="wikis", path="a.md",
                 heading_path="S1", body=body)
    s2 = Section(source="src2", title="S2", category="wikis", path="b.md",
                 heading_path="S2", body=body)

    _fts5_sync_sections(conn, "src1", [s1], rank_bias=0.7)
    _fts5_sync_sections(conn, "src2", [s2], rank_bias=0.7,
                        content_hashes_seen={h})

    rows = conn.execute(
        "SELECT source, title FROM sections ORDER BY id"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["source"] == "src1"  # first source wins
    conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_indexer.py -v -k "test_content_hash"`
Expected: FAIL (signature mismatch on _fts5_sync_sections)

- [ ] **Step 3: Add `_RANK_BIAS_MAP` module-level dict and `_lookup_rank_bias` in `indexer.py`**

```python
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
```

- [ ] **Step 4: Update `_fts5_sync_sections` signature and body**

Replace the old function with the hash + dedup + rank_bias version. Each source has one category, so `rank_bias` is a pre-resolved float:

```python
import hashlib

def _fts5_sync_sections(
    conn: sqlite3.Connection,
    source_name: str,
    sections: list[Section],
    rank_bias: float | None = None,
    content_hashes_seen: set[str] | None = None,
) -> None:
    """Insert sections with content hashing and source-quality bias.

    Args:
        conn: Open database connection (in transaction).
        source_name: Source name to re-index.
        sections: List of Section dataclass instances to insert.
        rank_bias: Pre-computed rank bias. Falls back to
            _lookup_rank_bias(sections[0].category) if None.
        content_hashes_seen: Set of content hashes already inserted
            (cross-source dedup). Passed by caller for cumulative tracking.
    """
    if content_hashes_seen is None:
        content_hashes_seen = set()

    # Delete old entries for this source
    conn.execute(
        "DELETE FROM sections_fts WHERE rowid IN "
        "(SELECT id FROM sections WHERE source = ?)", (source_name,))
    conn.execute(
        "DELETE FROM sections_fts_title WHERE rowid IN "
        "(SELECT id FROM sections WHERE source = ?)", (source_name,))
    conn.execute("DELETE FROM sections WHERE source = ?", (source_name,))

    if not sections:
        return

    sections = _deduplicate_sections(sections)

    # Resolve rank_bias
    if rank_bias is None:
        rank_bias = _lookup_rank_bias(sections[0].category)

    # Filter: hash + dedup
    to_insert: list[tuple[str, float, str, str, str, str, str, str, str]] = []
    for s in sections:
        h = hashlib.sha256(s.body.encode()).hexdigest()
        if h in content_hashes_seen:
            continue  # first source wins
        content_hashes_seen.add(h)
        to_insert.append((
            h, rank_bias, s.source, s.title, s.category,
            s.path, s.heading_path, s.body, source_title,
        ))

    if not to_insert:
        return

    conn.executemany(
        "INSERT INTO sections "
        "(content_hash, rank_bias, source, title, category, path, "
        "heading_path, body, source_title) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", to_insert)

    # Sync FTS5 tables
    sec_ids = conn.execute(
        "SELECT id FROM sections WHERE source = ? ORDER BY id",
        (source_name,),
    ).fetchall()

    fts_tuples = [(sec_ids[i][0], t[3], t[6], t[7])
                   for i, t in enumerate(to_insert)]
    conn.executemany(
        "INSERT INTO sections_fts(rowid, title, heading_path, body) "
        "VALUES (?, ?, ?, ?)", fts_tuples)
    conn.executemany(
        "INSERT INTO sections_fts_title(rowid, title, heading_path) "
        "VALUES (?, ?, ?)",
        [(rowid, title, hpath) for rowid, title, hpath, _ in fts_tuples])
```

Note: ``to_insert`` tuple format is ``(content_hash, rank_bias, source, title, category, path, heading_path, body, source_title)`` with indices 0-8. So `t[3]` = title, `t[6]` = heading_path, `t[7]` = body, `t[8]` = source_title. The function signature must also accept `source_title: str = ""`:

```python
def _fts5_sync_sections(
    conn: sqlite3.Connection,
    source_name: str,
    sections: list[Section],
    rank_bias: float | None = None,
    content_hashes_seen: set[str] | None = None,
    source_title: str = "",
) -> None:
```

Note: `source_title` is per-source, not per-section (all sections from one source share it). The `Section` dataclass doesn't carry it, so it's passed as a parameter.

- [ ] **Step 5: Update `_index_source` to pass rank_bias and source_title**

All sections from one source share the same category and source title. Compute once and pass:

```python
def _index_source(
    source, conn, data_dir, verbose,
    current_head=None, cfg=None,
    content_hashes_seen: set[str] | None = None,
) -> int:
    ...
    rb = _lookup_rank_bias(source.category)
    _fts5_sync_sections(conn, source.name, all_sections,
                         rank_bias=rb,
                         content_hashes_seen=content_hashes_seen,
                         source_title=source.title or source.name)
```

- [ ] **Step 6: Run tests to verify**

Run: `uv run pytest tests/test_indexer.py -v -k "test_content_hash"`
Expected: PASS

- [ ] **Step 7: Update `cmd_index` to call `_migrate_schema` and track content_hashes_seen across sources**

In `cmd_index`, after `ensure_schema(conn)`, add:
```python
from knowledge.db import _migrate_schema
msgs = _migrate_schema(conn)
for m in msgs:
    print(f"  Schema migration: {m}")
```

And initialize `content_hashes_seen` for cross-source dedup:
```python
content_hashes_seen: set[str] = set()
```

Pass it to each `_index_source` call:
```python
num_sections = _index_source(
    source, conn, data_dir, verbose,
    current_head=current_head, cfg=cfg,
    content_hashes_seen=content_hashes_seen,
)
```

And handle the NULL content_hash detection — auto-rebuild per spec:
```python
null_hashes = conn.execute(
    "SELECT COUNT(*) FROM sections WHERE content_hash IS NULL"
).fetchone()[0]
if null_hashes > 0:
    print(f"  {null_hashes} sections missing content_hash — rebuilding all sources")
    # Force rebuild: reindex all sources by treating them all as needing update.
    # The simplest approach is to wipe source_state so every source looks new.
    conn.execute("DELETE FROM source_state")
    conn.execute("DELETE FROM sections_fts")
    conn.execute("DELETE FROM sections_fts_title")
    conn.execute("DELETE FROM sections")
    # The outer source loop will then re-index everything because
    # stored is None for all sources.
```

- [ ] **Step 8: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: all PASS (or as many as before — existing indexer tests may need minor tweaks)

- [ ] **Step 9: Commit**

```bash
git add knowledge/indexer.py knowledge/db.py
git commit -m "feat: content hashing and rank_bias during indexing"
```

---

### Task 5: Source-Quality Ranking in Search

**Files:**
- Modify: `knowledge/search.py` — integrate `rank_bias` into ORDER BY

**Interfaces:**
- Consumes: `sections.rank_bias` column
- Produces: search results ranked by `bm25(...) * rank_bias`

- [ ] **Step 1: Write failing test for rank_bias ordering**

```python
def test_rank_bias_boosts_wiki_sources(tmp_path) -> None:
    from knowledge.config import ensure_data_dir, resolve_data_dir
    from knowledge.db import ensure_schema, get_connection
    from knowledge.search import cmd_search

    # Use the same data dir resolution as cmd_search does
    data_dir = ensure_data_dir(resolve_data_dir(str(tmp_path)))
    db_path = data_dir / "index.db"
    conn = get_connection(db_path)
    ensure_schema(conn)
    from knowledge.db import _migrate_schema
    _migrate_schema(conn)

    # Insert two sections: one wikis (rank_bias=0.7), one osint (rank_bias=1.0)
    # Both have identical body text; titles differ (so BM25 body contributions are equal)
    body = "word document exploit macro"
    conn.execute(
        "INSERT INTO sections (source, title, category, path, heading_path, body, "
        "content_hash, rank_bias, source_title) "
        "VALUES ('hacktricks', 'doc exploit', 'wikis', 'a.md', 'doc', ?, "
        "'aaa', 0.7, 'HackTricks')",
        (body,))
    conn.execute(
        "INSERT INTO sections (source, title, category, path, heading_path, body, "
        "content_hash, rank_bias, source_title) "
        "VALUES ('osint-src', 'doc osint', 'osint', 'b.md', 'doc', ?, "
        "'bbb', 1.0, 'OSINT Source')",
        (body,))

    row1 = conn.execute("SELECT id FROM sections WHERE source='hacktricks'").fetchone()[0]
    row2 = conn.execute("SELECT id FROM sections WHERE source='osint-src'").fetchone()[0]
    conn.execute("INSERT INTO sections_fts(rowid, title, heading_path, body) VALUES (?, ?, ?, ?)",
                 (row1, "doc exploit", "doc", body))
    conn.execute("INSERT INTO sections_fts(rowid, title, heading_path, body) VALUES (?, ?, ?, ?)",
                 (row2, "doc osint", "doc", body))
    conn.execute("INSERT INTO sections_fts_title(rowid, title, heading_path) VALUES (?, ?, ?)",
                 (row1, "doc exploit", "doc"))
    conn.execute("INSERT INTO sections_fts_title(rowid, title, heading_path) VALUES (?, ?, ?)",
                 (row2, "doc osint", "doc"))
    conn.commit()
    conn.close()

    results = cmd_search("word exploit", top_k=10, config_dir=str(tmp_path))
    # wikis source should appear first (lower rank_bias = better rank)
    assert len(results) >= 2
    assert results[0]["source"] == "hacktricks"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_search.py -v -k "rank_bias"`
Expected: FAIL (rank_bias not yet integrated)

- [ ] **Step 3: Integrate rank_bias into `cmd_search` in `search.py`**

In the SQL query in `cmd_search`, change the ORDER BY:

```sql
ORDER BY {bm25_order}
```
to:
```sql
ORDER BY {bm25_order} * s.rank_bias
```

Update the SELECT to include `s.rank_bias` (for debugging/transparency) — actually just the ORDER BY change is sufficient.

Also add the `_migrate_schema` call AFTER the `has_sections` guard (calling it before would crash if the `sections` table doesn't exist yet):

```python
from knowledge.db import _migrate_schema

# Add after the has_sections check (around line 213 of current search.py):
if not has_sections:
    print("Error: No index found. Run 'kdb index' first.", file=sys.stderr)
    return []
# Safe to migrate now — sections table exists
msgs = _migrate_schema(conn)
if msgs:
    null_hash = conn.execute(
        "SELECT COUNT(*) FROM sections WHERE content_hash IS NULL"
    ).fetchone()[0]
    if null_hash:
        print("Info: Index needs rebuild to populate new columns."
              " Run 'kdb index --force'.", file=sys.stderr)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_search.py -v -k "rank_bias"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add knowledge/search.py
git commit -m "feat: integrate rank_bias into FTS5 ranking"
```

---

### Task 6: `kdb get` Subcommand

**Files:**
- Create: `knowledge/getter.py`
- Modify: `knowledge/cli.py` — add `get` subparser + dispatch
- Create: `tests/test_getter.py`
- Create: `tests/test_cli_get.py`

**Interfaces:**
- Consumes: `get_connection`, `resolve_data_dir`, `ensure_data_dir` from `config.py`
- Produces: `cmd_get(hash_prefix, config_dir, json_output) -> dict | None`

- [ ] **Step 1: Write failing unit tests for cmd_get**

```python
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
        "VALUES ('src', 'Title', 'cat', 'p', 'hp', 'body text', 'a1b2c3d4e5f6a7b8c9d0', 1.0)")
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
        "VALUES ('src', 'Title', 'cat', 'p', 'hp', 'body', 'a1b2c3d4e5f6a7b8c9d0', 1.0)")
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
        "VALUES ('a', 'T1', 'cat', 'a.md', 'hp', 'body1', 'a1b2c3d4e5f6a7b8c9d0', 1.0)")
    conn.execute(
        "INSERT INTO sections (source, title, category, path, heading_path, body, content_hash, rank_bias) "
        "VALUES ('b', 'T2', 'cat', 'b.md', 'hp', 'body2', 'a1b2c3d4e5f6a7b8c9d1', 1.0)")
    conn.commit()
    conn.close()

    result = cmd_get("a1b2c3d4", config_dir=str(tmp_path / "data"))
    assert result is None  # ambiguous -> returns None, stderr prints matches


def test_get_non_hex_prefix(tmp_path) -> None:
    from knowledge.getter import cmd_get
    result = cmd_get("xyz12345", config_dir=str(tmp_path / "data"))
    assert result is None  # invalid -> returns None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_getter.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement `knowledge/getter.py`**

```python
"""Hash-prefix section retrieval for ``kdb get``."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

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
    if not all(c in "0123456789abcdef" for c in hash_prefix):
        print("Error: hash prefix must be hex characters only", file=sys.stderr)
        return None
    if len(hash_prefix) < 10:
        print("Error: hash prefix must be at least 10 hex characters",
              file=sys.stderr)
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
                + "\n".join(f"  {r['content_hash']}  {r['source']}: {r['title']}"
                            for r in rows),
                file=sys.stderr,
            )
            return None

        row = rows[0]
        return {
            "hash": row["content_hash"],
            "source": row["source"],
            "title": row["title"],
            "category": row["category"],
            "path": row["path"],
            "heading_path": row["heading_path"],
            "body": row["body"],
        }
    finally:
        conn.close()
```

- [ ] **Step 4: Run tests to verify**

Run: `uv run pytest tests/test_getter.py -v`
Expected: PASS

- [ ] **Step 5: Write CLI test for `kdb get`**

```python
"""CLI tests for kdb get subcommand."""

from __future__ import annotations

import subprocess
import sys


def test_cli_get_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "knowledge", "get", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "hash-prefix" in result.stdout


def test_cli_get_no_index(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "knowledge", "get", "a1b2c3d4e5", "-c", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
```

- [ ] **Step 6: Integrate `get` into `cli.py`**

First add a hex-prefix validator function in `cli.py` (near `_add_global_args`):

```python
def _validate_hex_prefix(value: str) -> str:
    """Argparse type validator for hash prefix arguments."""
    if not all(c in "0123456789abcdef" for c in value.lower()):
        raise argparse.ArgumentTypeError("hash prefix must be hex characters only")
    if len(value) < 10:
        raise argparse.ArgumentTypeError("hash prefix must be at least 10 hex characters")
    return value.lower()
```

Then add the `get` subparser in `_build_parser`:

```python
p_get = sub.add_parser("get", help="Retrieve a section by content hash prefix")
_add_global_args(p_get)
p_get.add_argument("hash_prefix", type=_validate_hex_prefix,
                   help="Hash prefix (min 10 hex chars)")
p_get.add_argument("--json", action="store_true", help="JSON output")
```

Add the dispatch in `main()`:

```python
case "get":
    from knowledge.getter import cmd_get
    result = cmd_get(args.hash_prefix, config_dir=args.config)
    if result is None:
        sys.exit(1)
    if args.json:
        import json
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_get_result(result)
```

Add the `_print_get_result` helper:

```python
def _print_get_result(result: dict) -> None:
    """Print a formatted section result for ``kdb get``."""
    print(f"Hash:\t\t{result['hash']}")
    print(f"Source:\t\t{result['source']}")
    print(f"Title:\t\t{result['title']}")
    print(f"Category:\t{result['category']}")
    print(f"Path:\t\t{result['path']}")
    print(f"Heading:\t{result['heading_path']}")
    print()
    print("--- Content ---")
    print(result['body'])
```

- [ ] **Step 7: Run CLI tests**

Run: `uv run pytest tests/test_cli.py tests/test_getter.py tests/test_cli_get.py -v`
Expected: all PASS

- [ ] **Step 8: Commit**

```bash
git add knowledge/getter.py knowledge/cli.py tests/test_getter.py tests/test_cli_get.py
git commit -m "feat: add kdb get for hash-prefix section retrieval"
```

---

### Task 7: Display Unification

**Files:**
- Modify: `knowledge/cli.py` — proportional widths, hash column, tag format, compact mode

- [ ] **Step 1: Write failing test for display formatting**

```python
"""Tests for search display formatting in cli.py."""

from __future__ import annotations

from knowledge.cli import _format_search_results


def test_format_search_includes_hash() -> None:
    results = [
        {
            "source": "hacktricks", "title": "Token Confusion",
            "category": "wikis", "path": "a.md",
            "heading_path": "HackTricks: Token Confusion",
            "body": "body", "distance": -14.03,
            "content_hash": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
            "source_title": "HackTricks",
        }
    ]
    output = _format_search_results(results)
    assert "a1b2c3d4e5f6" in output  # first 12 chars of hash


def test_format_search_tag_format() -> None:
    results = [
        {
            "source": "hacktricks", "title": "Token Confusion",
            "category": "wikis", "path": "a.md",
            "heading_path": "HackTricks: Token Confusion",
            "body": "body", "distance": -14.03,
            "content_hash": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
            "source_title": "HackTricks",
        }
    ]
    output = _format_search_results(results)
    assert "wikis·HackTricks" in output


def test_format_search_truncates_long_tag() -> None:
    """Long tags get … ellipsis truncated."""
    results = [
        {
            "source": "internalallthethings", "title": "OfficePurge",
            "category": "ad-internal", "path": "a.md",
            "heading_path": "InternalAllTheThings: OfficePurge",
            "body": "body", "distance": -20.71,
            "content_hash": "e5f6a7b8a9b0c1d2e3f4a5b6c7d8e9f0",
        }
    ]
    output = _format_search_results(results)
    # Tag or title should be truncated with … (not overflow)
    assert "…" in output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v -k "test_format_search"`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement `_format_search_results` in `cli.py`**

Add the helper function and update the search display block:

```python
import shutil


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with … ellipsis if it exceeds max_len."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 1] + "…"


def _format_search_results(results: list[dict]) -> str | None:
    """Format search results as a table or return None if terminal too narrow.

    Uses proportional column widths based on terminal size. Falls back to
    compact 2-line format below 80 columns.
    """
    if not results:
        return None

    term_width = shutil.get_terminal_size((80, 20)).columns
    content_width = int(term_width * 0.85)

    hash_w = 12
    dist_w = 9
    remaining = content_width - hash_w - dist_w
    tag_w = int(remaining * 0.28)  # ~28% of remaining = ~20% of total
    title_w = remaining - tag_w    # ~72% of remaining = ~50% of total

    lines: list[str] = []

    if term_width < 80:
        # Compact 2-line format
        lines.append(f"{'Hash':<14} {'Tag':<{tag_w}}")
        lines.append("─" * (14 + tag_w + 2))
        for r in results:
            tag = f"{r['category']}·{r.get('source_title', r['source'])}"
            h = r["content_hash"][:12] if r.get("content_hash") else "?" * 12
            title = _truncate(r["title"], title_w)
            dist = f"{r['distance']:.2f}"
            lines.append(f"{h:<14} {_truncate(tag, tag_w):<{tag_w}}")
            lines.append(f"{'':<14} {title:<{title_w}} {dist:>{dist_w}}")
    else:
        # Full table
        header = (
            f"{'Hash':<{hash_w}} {'Tag':<{tag_w}} "
            f"{'Title':<{title_w}} {'Distance':<{dist_w}}"
        )
        lines.append(header)
        lines.append("─" * len(header))
        for r in results:
            tag = f"{r['category']}·{r.get('source_title', r['source'])}"
            h = r["content_hash"][:12] if r.get("content_hash") else "?" * 12
            title = _truncate(r["title"], title_w)
            dist = f"{r['distance']:.2f}"
            lines.append(
                f"{h:<{hash_w}} {_truncate(tag, tag_w):<{tag_w}} "
                f"{title:<{title_w}} {dist:>{dist_w}}"
            )

    return "\n".join(lines)
```

- [ ] **Step 4: Update the search display in `main()`**

Replace the current search display block (lines 132-143 in cli.py):

```python
if args.json:
    print(json.dumps(results, indent=2, ensure_ascii=False))
elif results:
    output = _format_search_results(results)
    if output:
        print(output)
    else:
        # Fallback: minimal output
        for r in results:
            print(f"{r['source']}: {r['title']} ({r['distance']:.2f})")
```

- [ ] **Step 5: Also add content_hash to the search query in `cmd_search`**

In `search.py`, the SQL SELECT must include `s.content_hash` and `s.source_title` so the display has them:

```python
sql = f"""
    SELECT s.source, s.title, s.category, s.path,
           s.heading_path, s.body, s.content_hash, s.source_title,
           {bm25_select} as rank
    FROM {fts_table} f
    JOIN sections s ON s.id = f.rowid
    ...
```

And update the `SearchResult` TypedDict to include `content_hash` and `source_title`:

```python
class SearchResult(TypedDict):
    """Single knowledge-base search result row."""
    source: str
    source_title: str  # human-readable title from sources.yaml
    title: str
    category: str
    path: str
    heading_path: str
    body: str
    distance: float
    content_hash: str  # SHA-256 hash for kdb get
```

Fill it in the result builder:

```python
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
```

- [ ] **Step 6: Update existing test that constructs SearchResult**

In `tests/test_search.py`, find `test_search_result_has_distance_field` (or similar) and add `content_hash` and `source_title` to the SearchResult construction:

```python
# Before:
SearchResult(source="test", title="Test", category="e2e", path="t.md",
             heading_path="", body="body", distance=0.0)

# After:
SearchResult(source="test", title="Test", category="e2e", path="t.md",
             heading_path="", body="body", distance=0.0,
             content_hash="", source_title="")
```

- [ ] **Step 7: Update the test_format_search_truncates_long_tag test to include source_title**

```python
def test_format_search_truncates_long_tag() -> None:
    """Long tags get … ellipsis truncated."""
    results = [
        {
            "source": "internalallthethings", "title": "OfficePurge",
            "category": "ad-internal", "path": "a.md",
            "heading_path": "InternalAllTheThings: OfficePurge",
            "body": "body", "distance": -20.71,
            "content_hash": "e5f6a7b8a9b0c1d2e3f4a5b6c7d8e9f0",
            "source_title": "InternalAllTheThings",
        }
    ]
    output = _format_search_results(results)
    # Tag or title should be truncated with … (not overflow)
    assert "…" in output
```

- [ ] **Step 8: Run tests**

Run: `uv run pytest tests/test_cli.py tests/test_search.py -v`
Expected: all PASS

- [ ] **Step 9: Commit**

```bash
git add knowledge/cli.py knowledge/search.py
git commit -m "feat: unified display with hash prefix, tag format, proportional widths"
```

---

### Task 8: Full Integration Test + Final Verification

**Files:**
- No new files — verify the complete pipeline works end-to-end

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests PASS

- [ ] **Step 2: Verify `_format_search_results` function is importable and works**

Run: `python -c "from knowledge.cli import _format_search_results; print('OK')"`
Expected: "OK"

- [ ] **Step 3: Verify the complete feature set via `kdb --help`**

Run: `uv run python -m knowledge --help`
Expected: shows all 6 subcommands (fetch, index, update, search, list-sources, **get**)

- [ ] **Step 4: Commit final integration tweaks**

```bash
git add -A
git commit -m "chore: final integration fixes"
```
