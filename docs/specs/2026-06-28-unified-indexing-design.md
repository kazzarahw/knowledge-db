# Unified Indexing — Design Spec

**Date:** 2026-06-28
**Status:** Draft

## Problem

`kdb search` returns results whose source names, titles, heading paths, and body formatting
are each wiki's raw output — truncated, inconsistent, source-specific. No actionable handle
exists to retrieve a result's full content. Ranking mixes authoritative and peripheral sources
evenly.

## Scope

Single cycle targeting:

1. **Content normalization** — convert RST/notebook bodies to consistent markdown; normalize
   heading text (strip markup, decode entities, source-qualify top-level headings)
2. **Source-quality ranking** — category-derived boost factor on BM25 scores
3. **Content hashing + `kdb get`** — SHA-256 content-addressed section identity; new
   subcommand for hash-prefix retrieval
4. **Display unification** — proportional terminal widths, hash-as-handle, compact
   source/category tag

## Schema Changes (`db.py`)

`sections` table gains two columns:

| Column | Type | Constraint | Purpose |
|--------|------|------------|---------|
| `content_hash` | `TEXT` | (none — app-level dedup) | SHA-256 hex of normalized body |
| `rank_bias` | `REAL` | `NOT NULL DEFAULT 1.0` | Category-derived BM25 multiplier |

`rank_bias`: applied as `ORDER BY bm25(...) * rank_bias`. Lower BM25 = better, so
values below 1.0 boost rank (reduce effective score) and values above 1.0 debuff.
Named `rank_bias` (not `source_quality` or `boost_factor`) because the semantic is
direction-agnostic — it biases the rank, and the docstring explains the math.

### Migration path

`ensure_schema()` cannot add columns via `CREATE TABLE IF NOT EXISTS`. Instead,
`cmd_index()` and `cmd_search()` call a new `_migrate_schema()` that detects
missing columns via `PRAGMA table_info(sections)` and runs `ALTER TABLE`:

```python
def _migrate_schema(conn: sqlite3.Connection) -> list[str]:
    \"\"\"Add missing columns to existing sections table.

    Returns list of migration messages (empty if none needed).
    Idempotent — safe to call repeatedly.
    \"\"\"
    existing = {row[1] for row in conn.execute("PRAGMA table_info(sections)")}
    migrations: list[str] = []
    if "content_hash" not in existing:
        conn.execute(
            "ALTER TABLE sections ADD COLUMN content_hash TEXT"
        )
        migrations.append("added content_hash column")
    if "rank_bias" not in existing:
        conn.execute(
            "ALTER TABLE sections ADD COLUMN rank_bias REAL"
            " NOT NULL DEFAULT 1.0"
        )
        migrations.append("added rank_bias column")
    return migrations
```

On first run after upgrade, `cmd_index()` auto-migrates and re-indexes all
sources (since existing rows lack hashes). `cmd_search()` migrates if needed
and prints "Run `kdb index --force` to populate new columns" if rows have
NULL content_hash.

## New Module: `knowledge/normalize.py`

**Responsibility:** Transform raw file text and extracted headings into uniform representations
before chunking.

### Exported interface

```python
def normalize_body(path: Path, file_ext: str) -> str:
    """Convert non-markdown formats to markdown.

    Accepts file path (not text) because .ipynb requires structured parsing
    via nbformat. Reads the file internally. ``file_ext`` includes leading
    dot (e.g. ``.rst``, ``.ipynb``, ``.md``).

    - .rst → rst2gfm convert (pure Python)
    - .ipynb → nbformat read → clean markdown output (replacing current
      ad-hoc notebook conversion in chunk.py); on nbformat parse failure
      → log warning, skip file (not indexed)
    - everything else → read text, pass through
    - on failure → log warning, return original text
    """

def normalize_heading(segment: str) -> str:
    """Clean a single heading segment for display.
    - Strip markdown link syntax ([Label](url) → Label)
    - Strip HTML anchor tags
    - Decode HTML entities (&amp; → &)
    - Collapse whitespace
    - Preserve inline formatting (**bold**, *italic*, `code`) — these are
      meaningful visual cues in pentest headings
    """

def qualify_heading(source_title: str, heading: str, is_top_level: bool = True) -> str:
    """Prepend source title to top-level heading.

    Only applies to the first segment in a heading path (is_top_level=True).
    Nested segments pass through unchanged — avoids duplicating the prefix
    at every hierarchy level.

    E.g. 'InternalAllTheThings: OfficePurge' for top-level,
    'Configuration' for nested (no prefix).
    Falls back to source.name if source.title is empty.
    """
```

### Data flow

```
chunk_file():
  path, ext → normalize_body(path, ext) → normalized text
  normalized text → chunk_text(...) → list[Section]

Within chunk_text():
  heading segments → normalize_heading() per segment
  first segment only → qualify_heading(source_title, seg, is_top_level=True)
  subsequent segments → pass through (already normalized)
```

### Dependency

Add `rst2gfm` (pure Python, docutils-based, ~100KB, MIT).

## Source-Quality Ranking (`search.py`)

Category-to-bias mapping (defined as a module-level dict). Any category not listed
defaults to 1.0 (neutral). Sources with empty `category` string also default to 1.0
with a debug-level log.

| Category | `rank_bias` | Effect | Rationale |
|----------|-------------|--------|-----------|
| wikis | 0.7 | Strong boost | Primary pentest reference content |
| ad-internal | 0.8 | Moderate boost | Specialised AD/internal content |
| web-api | 0.8 | Moderate boost | OWASP references |
| dfir | 0.9 | Mild boost | Forensics documentation |
| wifi, bluetooth | 0.9 | Mild boost | Protocol-specific authoritative docs |
| c2, hardware-iot | 1.0 | Neutral | |
| mobile | 1.0 | Neutral | |
| lotl | 1.0 | Neutral | Living-off-the-land binaries |
| re-books, re-tools, re-indexes | 1.0 | Neutral | |
| osint | 1.0 | Neutral | |
| glitching, sdr | 1.0 | Neutral | |
| firmware | 1.1 | Mild debuff | Peripheral to core pentest queries |
| compliance | 1.1 | Mild debuff | Peripheral to core pentest queries |

Applied in ORDER BY as:

```sql
ORDER BY bm25(...) * s.rank_bias
```

Lower BM25 = better rank, so lower `rank_bias` = boosted. This is a **coarse
heuristic** — its effect on rank position varies with BM25 score distribution.
For queries where BM25 scores are tightly clustered, the bias dominates. For
queries with wide BM25 spread, the bias is subtle. This inconsistency is
documented in the `rank_bias` column docstring; exact tuning happens after
observing real query behavior.

The multiplier is applied identically to both FTS5 tables (`sections_fts` and
`sections_fts_title`). Since BM25 values differ in scale between the two
tokenizers, the bias effect varies by query tier — acceptable for v1 but
noted for future refinement.

## Content Hashing + Dedup (`db.py`, `indexer.py`)

- Hash computed as `hashlib.sha256(normalized_body.encode()).hexdigest()` during
  `_fts5_sync_sections`
- `content_hash` column has **no UNIQUE constraint** (SQLite forbids `ALTER TABLE
  ADD COLUMN` with `UNIQUE`). Dedup enforced at application level:

  ```python
  existing = conn.execute(
      "SELECT 1 FROM sections WHERE content_hash = ?", (content_hash,)
  ).fetchone()
  if not existing:
      # proceed with INSERT
  ```

  First source indexed wins on collision. The existing `UNIQUE(source, path,
  heading_path)` constraint is separate and unaffected.
- **Hash stability:** Depends on `normalize_body()` output. Any change to
  normalization logic invalidates existing hashes. Run `kdb index --force` after
  updating normalization, the same as after an embedding model change.

## `kdb get` Subcommand (`cli.py`, new `knowledge/getter.py`)

### Interface

```
kdb get <hash-prefix>        # unique prefix match (min 10 hex chars)
kdb get <hash-prefix> --json  # JSON output
```

### Behavior

- Query against `sections.content_hash LIKE '<prefix>%'`
- 0 matches → error to stderr, exit 1
- 1 match → print full section: metadata header + body (body can be piped to `less`)
- 2+ matches → list all with full hashes (one per line), error "ambiguous prefix"
- Hash prefix minimum: **10 hex chars** (40 bits). At 100K sections, birthday
  paradox collision probability ≈ 0.0045 — negligible.
- Validate in argparse: hex chars only + minimum length.
- Prefix is lowercased before querying (hash stored lowercase).

### Output: `--json` format

```json
{
  "hash": "a1b2c3d4e5f67890abcdef1234567890abcdef1234567890abcdef1234567890",
  "source": "hacktricks",
  "title": "Token Confusion",
  "category": "wikis",
  "path": "docs/pentest/token-confusion.md",
  "heading_path": "InternalAllTheThings: OfficePurge",
  "body": "<full body text>"
}
```

### Output format (text)

```
Hash:      a1b2c3d4e5f6...
Source:    hacktricks
Title:     Token Confusion
Category:  wikis
Path:      docs/pentest/token-confusion.md
Heading:   InternalAllTheThings: OfficePurge

--- Content ---
<body text>
```

## Display Changes (`cli.py`)

### Terminal-aware proportional widths

- Use `shutil.get_terminal_size()` to detect terminal width
- Columns: hash(12) | tag(category·source) | title | distance
  - 85% of width distributed to content columns; 15% for separators + gutters
  - Tag gets ~20%, title gets ~50%, hash fixed at 12 chars, distance fixed at 9 chars
  - **Truncation:** Tag and title columns truncated with `…` ellipsis when content
    exceeds allocated width. This keeps the tabular layout intact at any terminal size.
  - If terminal < 80 cols → fall back to compact 2-line format (no tabular layout needed)
- Hash prefix shows **first 12 hex chars** (48 bits) — no collision risk, and
  `kdb get` only needs 10+ so this works as a copy-paste handle

### Tag format

`<category>·<source-title-prefix>` — e.g. `wikis·HackTricks`. Source-title-prefix
uses `sources.yaml` `title` field. No spaces around the middle dot (keeps it compact).

### Search output layout

```
  Hash           Tag                    Title                              Distance
  ─────          ───                    ─────                              ────────
  a1b2c3d4e5f6   wikis·InternalAllTh   OfficePurge                       -20.71
  e5f6a7b8a9b0   wikis·HackTricks      Token Confusion                   -14.03
```

If headings contain source-specific nav prefixes (e.g., "Office - Attacks > Word >"),
the chunker's `qualify_heading` replaces the top-level segment with a clean
`SourceName: Title` form, so nesting crumbs from the original wiki hierarchy are
dropped.

### Compact format (< 80 cols)

```
  a1b2c3d4e5f6  wikis·InternalAllThings
                OfficePurge  -20.71
```

## Out of Scope (Future)

- RST→MD for complex directives (tables, custom roles) — `rst2gfm` covers the common subset;
  edge cases fall through as plain text
- Vector/hybrid search re-addition
- Result grouping ("group by source")
- `kdb get` paging (can pipe to `less`)

## Integration Notes

### `_fts5_sync_sections()` must include new columns

The INSERT statement in `_fts5_sync_sections` adds `content_hash` and `rank_bias`
to the column list. Hash is computed from normalized body; `rank_bias` is looked
up from the category→bias mapping. Both are set during section insertion.

### Incremental indexing after upgrade

Adding columns via `ALTER TABLE` leaves existing rows with `content_hash = NULL`.
To detect this: `cmd_index()` checks `SELECT COUNT(*) FROM sections WHERE
content_hash IS NULL` after migration. If any rows are null, force a full re-index
of all sources (same as `--force` but only for data population, not schema).

## Testing

- `test_normalize.py` — unit tests for `normalize_body` (.rst→MD round-trip,
  .ipynb→MD, passthrough for .md, warning on failure), `normalize_heading`
  (link stripping, entity decoding, whitespace collapse), `qualify_heading`
  (top-level vs nested, missing source title fallback)
- Test `rst2gfm` failure behavior — mock the import to verify warning+fallback
- `test_getter.py` — `kdb get` with exact hash, prefix match, ambiguous prefix,
  0 results, non-hex input, `--json` output format
- All new tests follow per-module naming: `test_X.py` for `knowledge/X.py`

## Rejection Handling

- **RST→MD conversion fails a specific file:** log warning, use original text, continue
- **Hash collision on prefix:** error listing full hashes (one per line); user provides more chars
- **`--force` rebuild not run:** old schema without hash/quality columns → `kdb search` detects
  missing columns and prints migration hint
