# Codebase Cleanup — Lint, Type, Naming, Stale Config

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all identified issues from `ruff check`, `pyright`, and naming/doc rules audit: 7 lint errors, 3 type errors, 1 formatting issue, 1 typing gap, 1 stale config, 1 stale test param, 3 naming violations.

**Architecture:** Each task is an independent file-level fix. No structural changes, no new code. Tasks can be applied in any order.

**Tech Stack:** Python 3.12+, pytest, ruff, pyright, FTS5 SQLite

## Global Constraints

- All existing tests must continue to pass (125 total).
- `ruff check .` must produce 0 errors after all tasks.
- `pyright .` must produce 0 errors after all tasks.
- `ruff format --check .` must produce no differences after all tasks.
- No behavioral changes to any function — only type annotations, naming, imports, and dead code.
- Function renames update ALL cross-file references (definition + every call site + test imports + test qualname strings).

---

### Task 1: Fix `tests/conftest.py` — Type-annotate `tracking_connect`

**Files:**
- Modify: `tests/conftest.py`

**Problem:** `tracking_connect(*args, **kwargs)` has no parameter types — violates type annotation rules. Two `# type: ignore[assignment]` lines lack justification comments.

**Design:** Use `Callable[..., sqlite3.Connection]` (not `Any`) for `original_connect` to preserve type information. The `# type: ignore[assignment]` on `sqlite3.connect = tracking_connect` is still needed because `tracking_connect` uses `*args: Any, **kwargs: Any` while `sqlite3.connect` expects a more specific signature.

- [ ] **Step 1: Read current file**

- [ ] **Step 2: Apply the full corrected fixture**

Replace the entire file content with:

```python
"""pytest configuration for knowledge-db tests."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Generator
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _track_connections() -> Generator[None, None, None]:
    """Track sqlite3 connections opened during each test; assert all closed after."""
    opened: list[sqlite3.Connection] = []
    original_connect: Callable[..., sqlite3.Connection] = sqlite3.connect

    def tracking_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = original_connect(*args, **kwargs)
        opened.append(conn)
        return conn

    sqlite3.connect = tracking_connect  # type: ignore[assignment]  # monkey-patch: tracking_connect has broader signature than sqlite3.connect
    yield
    sqlite3.connect = original_connect  # type: ignore[assignment]  # restore original after monkey-patch
    for conn in opened:
        try:
            conn.execute("SELECT 1")
            pytest.fail(f"sqlite3.Connection at {hex(id(conn))} was never closed")
        except sqlite3.ProgrammingError:
            pass  # closed — connection raises when you try to use it
```

Key changes from current:
- Added `Callable` to imports
- `original_connect` typed as `Callable[..., sqlite3.Connection]` instead of bare variable
- `tracking_connect` typed with `*args: Any, **kwargs: Any` and `-> sqlite3.Connection`
- Both `# type: ignore[assignment]` lines have justification comments

- [ ] **Step 3: Verify no pyright errors on conftest.py**

Run: `uv run pyright tests/conftest.py`
Expected: 0 errors

- [ ] **Step 4: Verify tests still pass**

Run: `uv run pytest tests/ -v --tb=short 2>&1 | tail -20`
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py
git commit -m "fix: type-annotate tracking_connect and justify type: ignore"
```

---

### Task 2: Remove stale `embed:` section from `config.yaml`

**Files:**
- Modify: `config.yaml`

**Problem:** The `embed:` section references `sentence-transformers` model, device, batch_size, trust_remote_code, dtype — all from the vec0/sentence-transformers era. The codebase migrated to FTS5. No `Config` dataclass field references these.

- [ ] **Step 1: Read current `config.yaml`**

- [ ] **Step 2: Remove the embed section (lines starting with `# ── Embedding` through the blank line after the `dtype:` line)**

Delete the `embed:` block (everything between the `# ── Embedding` comment header and the blank line before `# ── Git fetch settings`). After removal:

```yaml
# knowledge-db configuration
# Place alongside sources.yaml in your config directory.
# All fields have sensible defaults; uncomment to override.

# ── Git fetch settings ─────────────────────────────────────────────
# fetch:
...
```

- [ ] **Step 3: Verify file is valid YAML**

Run: `python3 -c "import yaml; yaml.safe_load(open('config.yaml')); print('OK')"`
Expected: prints "OK"

- [ ] **Step 4: Commit**

```bash
git add config.yaml
git commit -m "fix: remove stale embed section from config.yaml (FTS5 migration)"
```

---

### Task 3: Fix `tests/test_types.py` — stale params, pyright safety, formatting

**Files:**
- Modify: `tests/test_types.py`

**Problem:** Three issues in one file:
1. 6 `should_pass_before=False` entries (not including `_clone`/`_pull` which Task 7 handles) should be `True` — all functions already have return annotations
2. `hints.get("return")` can return `None` — add guard for pyright safety
3. Multi-line `assert hasattr(...)` can collapse to one line per ruff format

- [ ] **Step 1: Read current `tests/test_types.py`**

- [ ] **Step 2: Flip `should_pass_before` from `False` to `True` for 6 entries**

The `_clone` and `_pull` entries are handled by Task 7. These 6 are independent:

```python
("knowledge.chunk", "_convert_notebook", True),   # was False — already has -> str
("knowledge.cli", "_build_parser", True),          # was False — already has -> argparse.ArgumentParser
("knowledge.indexer", "_source_signature", True),   # was False — already has -> str | None
("knowledge.indexer", "_walk_files", True),         # was False — already has -> list[Path]
("knowledge.indexer", "_index_source", True),       # was False — already has -> int
("knowledge.fetch", "_fetch_git_source", True),     # was False — already has -> bool
```

- [ ] **Step 3: Add `None` guard for pyright in `test_search_result_is_typeddict`**

```python
    hints = typing.get_type_hints(cmd_search)
    ret = hints.get("return")
    if ret is None:
        pytest.fail("cmd_search must have a return annotation")
    # Should be list[SearchResult] where SearchResult is a TypedDict
    assert hasattr(ret, "__origin__"), f"return type must be generic, got {ret}"
    assert ret.__origin__ is list, f"must be list[...], got {ret.__origin__}"
    elem = ret.__args__[0]
    assert hasattr(elem, "__annotations__"), f"element type {elem} is not a TypedDict"
```

- [ ] **Step 4: Fix formatting — collapse multi-line assert**

The current code has:
```python
    assert hasattr(elem, "__annotations__"), (
        f"element type {elem} is not a TypedDict"
    )
```

`ruff format --check` reports this should be a single line. Change to:
```python
    assert hasattr(elem, "__annotations__"), f"element type {elem} is not a TypedDict"
```

- [ ] **Step 5: Verify ruff + pyright + tests**

```bash
uv run ruff check tests/test_types.py
uv run pyright tests/test_types.py
uv run pytest tests/test_types.py -v
```
Expected: 0 errors, tests pass

- [ ] **Step 6: Commit**

```bash
git add tests/test_types.py
git commit -m "fix: update test_types.py stale params, pyright safety, formatting"
```

---

### Task 4: Remove unused imports (ruff F401 — 5 files)

**Files:**
- Modify: `knowledge/indexer.py` (remove `fetch_sources` from import)
- Modify: `knowledge/search.py` (remove `Path` from import)
- Modify: `tests/test_chunk.py` (remove `Path` and `Section` from imports)
- Modify: `tests/test_search.py` (remove `pytest` from import)
- Modify: `tests/test_sources.py` (remove `Path` from import)

**Problem:** `ruff check .` reports 6 F401 unused-import errors across 5 files.

- [ ] **Step 1: Fix `knowledge/indexer.py:22`**

```python
# Before:
from knowledge.fetch import fetch_sources, get_git_head
# After:
from knowledge.fetch import get_git_head
```

- [ ] **Step 2: Fix `knowledge/search.py:9`**

```python
# Before:
from pathlib import Path
from typing import TypedDict
# After:
from typing import TypedDict
```

Also verify with pyright after removal — `Path` is only used on the import line. If importing `resolve_data_dir` from `config.py` returns `Path` objects, that's fine (the type is imported at the source).

- [ ] **Step 3: Fix `tests/test_chunk.py:5,7`**

```python
# Before:
from pathlib import Path

from knowledge.chunk import Section, chunk_file, chunk_text
# After:
from knowledge.chunk import chunk_file, chunk_text
```

- [ ] **Step 4: Fix `tests/test_search.py:8`**

```python
# Before:
import pytest

from knowledge.search import (
# After:
from knowledge.search import (
```

- [ ] **Step 5: Fix `tests/test_sources.py:5`**

```python
# Before:
from pathlib import Path

import pytest
# After:
import pytest
```

- [ ] **Step 6: Run ruff check to verify**

Run: `uv run ruff check .`
Expected: 0 remaining F401 errors

- [ ] **Step 7: Run pyright to verify search.py has no new type errors**

Run: `uv run pyright knowledge/search.py`
Expected: 0 errors (confirms removing `Path` import doesn't break type resolution)

- [ ] **Step 8: Run tests to verify no regressions**

Run: `uv run pytest tests/ -v --tb=short 2>&1 | tail -10`
Expected: all tests pass

- [ ] **Step 9: Commit**

```bash
git add knowledge/indexer.py knowledge/search.py tests/test_chunk.py tests/test_search.py tests/test_sources.py
git commit -m "fix: remove unused imports across 5 files"
```

---

### Task 5: Remove unused variable (ruff F841)

**Files:**
- Modify: `tests/test_search.py` (remove `data_dir` assignment + its local imports)

**Problem:** `data_dir` is assigned but never used in `test_empty_index_returns_empty`.

- [ ] **Step 1: Read current `tests/test_search.py` around `test_empty_index_returns_empty`**

- [ ] **Step 2: Remove the unused assignment and its supporting imports**

```python
# Before:
def test_empty_index_returns_empty(self, tmp_path: Path) -> None:
    """Search on an empty (no tables) index prints error, returns []."""
    from knowledge.config import resolve_data_dir, ensure_data_dir

    data_dir = ensure_data_dir(resolve_data_dir(str(tmp_path)))
    with patch("sys.stderr"):
        results = cmd_search("test", config_dir=str(tmp_path))
        assert results == []

# After:
def test_empty_index_returns_empty(self, tmp_path: Path) -> None:
    """Search on an empty (no tables) index prints error, returns []."""
    with patch("sys.stderr"):
        results = cmd_search("test", config_dir=str(tmp_path))
        assert results == []
```

- [ ] **Step 3: Verify ruff + tests**

```bash
uv run ruff check tests/test_search.py
uv run pytest tests/test_search.py -v
```
Expected: 0 errors, tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_search.py
git commit -m "fix: remove unused variable data_dir in test_empty_index_returns_empty"
```

---

### Task 6: Fix `_validate_git_url` naming — predicate prefix

**Files:**
- Modify: `knowledge/sources.py`

**Problem:** `_validate_git_url` returns `bool` but uses no predicate prefix. Per naming-conventions.md § Predicates, bool-returning functions should use `is_`/`has_`/`can_` prefix.

**Change:** `_validate_git_url` → `_is_valid_git_url`

- [ ] **Step 1: Rename function definition and update docstring**

```python
# Before:
def _validate_git_url(url: str) -> bool:
    """Basic git URL validation."""
    return bool(_GIT_URL_RE.match(url))

# After:
def _is_valid_git_url(url: str) -> bool:
    """Whether url matches _GIT_URL_RE pattern."""
    return bool(_GIT_URL_RE.match(url))
```

- [ ] **Step 2: Update the call site in `Source.__post_init__`**

```python
# Before (line 49):
if not _validate_git_url(self.url):
# After:
if not _is_valid_git_url(self.url):
```

- [ ] **Step 3: Verify**

Run: `uv run pytest tests/test_sources.py -v`
Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add knowledge/sources.py
git commit -m "fix: rename _validate_git_url to _is_valid_git_url (predicate prefix)"
```

---

### Task 7: Fix `_clone`/`_pull` naming — bare creation verbs (CROSS-FILE)

**Files:**
- Modify: `knowledge/fetch.py` (definition + internal call sites)
- Modify: `tests/test_fetch.py` (local imports by name)
- Modify: `tests/test_types.py` (qualname strings in parametrize)

**Problem:** `_clone` and `_pull` are bare verbs for creation/git operations. Per naming-conventions.md: "avoid bare verbs for creation (spawn, create, build) which need a target noun." These are imported by name in tests, so the rename spans 3 files.

**Change:** `_clone` → `_clone_source`, `_pull` → `_pull_source`

- [ ] **Step 1: Rename definitions and internal call sites in `knowledge/fetch.py`**

Definition `_clone` (line 115):
```python
# Before:
def _clone(source: Source, dest: Path, verbose: bool, git_timeout: int) -> bool:
    """Atomically clone a git repo into a temp dir, then rename."""
# After:
def _clone_source(source: Source, dest: Path, verbose: bool, git_timeout: int) -> bool:
    """Atomically clone a git repo into a temp dir, then rename."""
```

Call site in `_fetch_git_source` (line 111):
```python
# Before:
    return _clone(source, dest, verbose, git_timeout)
# After:
    return _clone_source(source, dest, verbose, git_timeout)
```

Definition `_pull` (line 173):
```python
# Before:
def _pull(source: Source, dest: Path, verbose: bool, git_timeout: int) -> bool:
    """Pull existing repo. Returns True if HEAD changed."""
# After:
def _pull_source(source: Source, dest: Path, verbose: bool, git_timeout: int) -> bool:
    """Pull existing repo. Returns True if HEAD changed."""
```

Call site in `_fetch_git_source` (line 112):
```python
# Before:
    return _pull(source, dest, verbose, git_timeout)
# After:
    return _pull_source(source, dest, verbose, git_timeout)
```

- [ ] **Step 2: Update local imports in `tests/test_fetch.py`**

Three places, each is a local `from knowledge.fetch import _clone` / `from knowledge.fetch import _pull`:

```python
# test_clone_invalid_url — update import:
from knowledge.fetch import _clone_source  # was: _clone

# test_pull_nonexistent_dir_returns_false — update import:
from knowledge.fetch import _pull_source  # was: _pull

# test_pull_local_repo_no_remote — update import:
from knowledge.fetch import _pull_source  # was: _pull
```

- [ ] **Step 3: Update qualname strings in `tests/test_types.py`**

```python
# Before:
("knowledge.fetch", "_clone", False),
("knowledge.fetch", "_pull", False),
# After:
("knowledge.fetch", "_clone_source", True),
("knowledge.fetch", "_pull_source", True),
```

(The `True` here is safe — both functions already have `-> bool` return annotations.)

- [ ] **Step 4: Verify**

```bash
uv run pytest tests/test_fetch.py tests/test_types.py -v
```
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add knowledge/fetch.py tests/test_fetch.py tests/test_types.py
git commit -m "fix: rename _clone/_pull to _clone_source/_pull_source (bare verb → verb_noun)"
```

---

### Task 8: Fix `_bm25_order` naming — verb_noun

**Files:**
- Modify: `knowledge/search.py`

**Problem:** `_bm25_order` is a noun phrase, not `verb_noun`. Per naming-conventions.md: "Functions — Use `verb_noun` imperative order."

**Change:** `_bm25_order` → `_get_bm25_order`

- [ ] **Step 1: Rename function definition**

```python
# Before (line 141):
def _bm25_order(tier: QueryTier) -> tuple[str, str]:
    """Return (select_expr, order_expr) with consistent column-weighted BM25."""

# After:
def _get_bm25_order(tier: QueryTier) -> tuple[str, str]:
    """Return (select_expr, order_expr) with consistent column-weighted BM25."""
```

- [ ] **Step 2: Update the call site in `cmd_search`**

```python
# Before (line 224):
        bm25_select, bm25_order = _bm25_order(tier)
# After:
        bm25_select, bm25_order = _get_bm25_order(tier)
```

The variable name `bm25_order` on the left side is fine — it's a local variable holding an expression string, not a function.

- [ ] **Step 3: Verify**

Run: `uv run pytest tests/test_search.py -v`
Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add knowledge/search.py
git commit -m "fix: rename _bm25_order to _get_bm25_order (noun → verb_noun)"
```

---

### Final Verification

After ALL tasks are complete:

- [ ] **Run ruff check — expect 0 errors**

Run: `uv run ruff check .`

- [ ] **Run pyright — expect 0 errors**

Run: `uv run pyright .`

- [ ] **Run ruff format check — expect no changes**

Run: `uv run ruff format --check .`

- [ ] **Run full test suite — expect 125/125 pass**

Run: `uv run pytest -v`
