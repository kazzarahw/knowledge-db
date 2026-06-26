# knowledge-db Agent Guide

**Offline semantic search for pentest documentation.** CLI that clones 70+ wikis,
chunks into sections, embeds via sentence-transformers, stores in SQLite+sqlite-vec.

## Quick start

```sh
uv run kdb fetch          # clone/pull all sources (runs git subprocesses)
uv run kdb index          # chunk → embed (sentence-transformers) → store
uv run kdb update         # fetch + index (one-shot)
uv run kdb search "wpa3 evil twin commands"   # semantic search
uv run kdb search --json --top-k 5 "bypass edr"  # JSON output
uv run kdb list-sources   # show all configured sources + clone status
uv run python -m knowledge search "query"  # same as kdb
```

Data directory resolution (priority): `--config PATH` > `$KNOWLEDGE_DB_DIR` > `$XDG_DATA_HOME/knowledge-db/` > `./data/` (when `pyproject.toml` is sibling and `./data/` exists).

## Tests

```sh
uv run test                # run ALL tests (pytest as console_scripts entry)
uv run pytest -v           # same, verbose
uv run pytest tests/test_chunk.py -v       # single file
uv run pytest -k "test_name" -v            # filter by name
uv run pytest -x --ff                      # stop on first failure, retry failed first
```

**Test quirks:**
- One `test_X.py` per `knowledge/X.py` module.
- Integration tests in `test_integration.py` (chunking, DB schema, index pipeline).
- All integration tests use `all-MiniLM-L6-v2` (dim=384, fast) — NOT the default `LiquidAI/LFM2.5-Embedding-350M` (dim=1024, 354M params, CUDA, slow).
- `conftest.py` has an auto-use fixture that **tracks every `sqlite3.connect()` call and asserts all connections are closed** after each test. Leak a connection = test failure.
- CLI tests (`test_cli.py`) are subprocess-based (`python -m knowledge`).
- Test classes for pytest organization are acceptable (per `docs/rules/python-paradigm.md`).
- Model loading tests (`test_embed.py`) use `unittest.mock.patch` to avoid 3-5s `SentenceTransformer` load time.

## Architecture

```
pyproject.toml  → kdb / knowledge-db entry: knowledge.cli:main
knowledge/
  cli.py        argparse, 5 subcommands (fetch/index/update/search/list-sources)
  sources.py    YAML manifest loader → Source dataclass
  fetch.py      git clone/pull with sparse-checkout + LFS detection
  chunk.py      heading-aware document chunker (ATX + setext, code-block fences, notebooks)
  embed.py      Embedder protocol + SentenceTransformerEmbedder (cached as function attribute)
  indexer.py    orchestrator: walk → chunk → embed (batched) → sqlite-vec insert
  db.py         SQLite WAL mode + sqlite-vec vec0 table (COSINE distance, PARTITION KEY on source)
  search.py     vec0 CTE with dimension validation
  config.py     typed Config dataclass, YAML loading, path resolution
```

## Config system

Two YAML files, usually side by side:

- **`sources.yaml`** — required, lists all doc sources (git repos, local dirs, notebooks). Each entry: name, url, type, branch, sparse paths, docs_dir, index_ext, category, title.
- **`config.yaml`** — optional, overrides defaults. Sections: `embed`, `fetch`, `index`, `search`. All fields have defaults.

Config search: `$config_dir/sources.yaml` → `cwd/sources.yaml` (stderr warning on fallback).

**Important:** Model change (e.g. different dim) invalidates index. Run `kdb index --force` to rebuild.

## Embedding model

| Property | Value |
|----------|-------|
| Default | `LiquidAI/LFM2.5-Embedding-350M` |
| Dimensions | 1024 |
| Params | 354M |
| Device | CUDA (auto-detect, falls back to CPU with warning) |
| `trust_remote_code` | `true` (required by many HF models) |
| Prompt | Model's own `prompts` dict from `config_sentence_transformers.json` — `embed_query()` uses `"query"` prompt if defined |

First model load takes 3-5s. Subsequent calls hit the function-attribute cache.

## Incremental indexing

- **Git sources:** compares git HEAD hash before/after `fetch`. Reindexes only if HEAD changed.
- **Local sources:** SHA-256 content signature (first 4KB of each file + file names + sizes). Reindexes on signature change.
- **`kdb index --force`** drops all index tables and rebuilds everything.
- Orphan cleanup: sources no longer in `sources.yaml` are auto-deleted from index on next `kdb index`.

## Code conventions

Three rule files under `docs/rules/` — agents MUST respect them:

1. **`naming-conventions.md`** — `verb_noun` imperative order, predicate prefixes (`is_`/`has_`/`can_`), retrieval verb trust levels (`get_`/`fetch_`/`find_`/`query_`/`list_`/`match_`).
2. **`python-paradigm.md`** — functions over classes, `@dataclass(frozen=True, slots=True)` by default, no inheritance (use Protocol/composition/pattern-matching), pure core + edge IO, PEP 695 generics, Google-style docstrings.
3. **`code-documentation.md`** — comments explain *why*, docstrings define contract, type hints enforce interface.

## Tech stack (Python 3.12+)

- **UV** for package management (`uv.lock` committed, `uv run` for commands)
- **pytest** for tests (CLI entry `uv run test`)
- **sentence-transformers** for embeddings
- **sqlite-vec** for vector search in SQLite (vec0 virtual table, COSINE distance)
- **PyYAML** for config
- **nbformat** for `.ipynb` notebook ingestion
- **argparse** for CLI (stdlib, no click/typer)

## Operational gotchas

- `kdb fetch` only processes git sources; `type: local` sources are skipped (their path must exist).
- `kdb fetch` uses `git clone --depth 1` with optional sparse-checkout for huge repos (e.g., Ghidra, Sliver).
- Git LFS is auto-detected via `git lfs track`. If the repo uses LFS but `git-lfs` is not installed, indexing files may produce placeholder text. Warning is printed.
- Indexing 70+ repos takes significant time and GPU memory. The full pipeline is meant as an offline batch job.
- The `update` command is `fetch` + `index` sequentially. For production use, run `fetch`, then `index` separately to verify sources before embedding.
- `search` validates embedding dimension against stored index metadata. Mismatch = error + `kdb index --force` required.
- The embedder cache is stored as a function attribute (`get_embedder._cached`) — intentional, avoids module-level global per `docs/rules/python-paradigm.md`.
- License: AGPL v3.
