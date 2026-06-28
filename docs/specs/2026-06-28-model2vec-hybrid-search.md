# Model2Vec Hybrid Search — FTS5 + Static Embeddings with RRF Fusion

> **Status:** Design spec — not implemented. Saved for future reference.
> **Context:** User asked about unifying search across 70+ wikis; research into MinishLab's
> `semble` (code search) and `semhash` (dedup) showed a clean mapping to knowledge-db.
>
> See also: https://github.com/MinishLab/semble, https://github.com/MinishLab/semhash,
> https://github.com/MinishLab/model2vec

## Problem

Knowledge-db indexes 70+ pentest wikis into ~100k sections. Search is pure FTS5 BM25 —
fast, zero-dependency, but no semantic understanding. Users want results that feel
like one coherent knowledge base, not a mashup of different wikis with different
titles/formatting.

The old system used sentence-transformers (LiquidAI/LFM2.5-Embedding-350M, 354M params,
CUDA, PyTorch 4.8GB) — removed because it was too heavy. This design proposes a
replacement that is ~1000× smaller and CPU-only.

## Solution Architecture

```
INDEX TIME
┌──────────────┐   ┌──────────────────┐   ┌──────────────────────┐
│ 70+ wikis     │──▶│ chunk_file()     │──▶│ Section[]            │
│ (git repos)   │   │ heading-aware    │   │ (source, title,      │
└──────────────┘   │ MD/RST/notebook  │   │  heading_path, body) │
                   └──────────────────┘   └───────┬───────────────┘
                                                  │
          ┌───────────────────────────────────────┼───────────────────────┐
          │                                       │                       │
          ▼                                       ▼                       ▼
┌────────────────────┐                  ┌──────────────────┐    ┌───────────────────┐
│ model2vec encode   │                  │ FTS5 insert      │    │ semhash dedup     │
│ potion-base-32M    │                  │ (porter + trigram)│    │ (optional, reuses │
│ 256-dim static     │                  │ column-weighted  │    │  embeddings)      │
│ ~10K sent/sec CPU  │                  │ BM25             │    │ threshold=0.88    │
└─────────┬──────────┘                  └──────────────────┘    │ → dedup_group IDs │
          │                                                     └───────────────────┘
          ▼
┌────────────────────┐
│ sqlite-vec vec0    │
│ dim=256            │
│ PARTITION KEY=source│
│ COSINE distance    │
└────────────────────┘

QUERY TIME
"wpa3 evil twin"
        │
        ▼
┌──────────────────────┐
│ 1. Embed query (μs)  │──▶ model2vec.encode([query])
└──────────┬───────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌─────────┐  ┌──────────┐
│ FTS5    │  │ vec0     │
│ BM25    │  │ cosine   │
│ top_k*5 │  │ top_k*5  │
└────┬────┘  └────┬─────┘
     │             │
     └──────┬──────┘
            ▼
┌──────────────────────┐
│ 2. RRF fusion (k=60) │
│                      │
│  α = f(query_tier):  │
│    CONCEPTUAL → 0.5  │
│    TOOL       → 0.2  │
│    EXACT      → 0.1  │
│    PATH       → 0.2  │
│                      │
│  combined[c] =       │
│    α * RRF_vec(c) +  │
│    (1-α) * RRF_ft(c) │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 3. Post-processing   │
│    • dedup_group     │
│      collapse        │
│      (max 1/group)   │
│    • code_block      │
│      boost (×1.5)    │
│      if query tool   │
│    • source metadata │
│      for display     │
└──────────┬───────────┘
           ▼
    Ranked results
```

## Model Selection

### Primary: `potion-base-32M`

| Property | Value |
|----------|-------|
| Parameters | 32.3M |
| Dimensions | 256 |
| MTEB Average | 52.13 |
| Retrieval (MTEB) | 32.67 |
| Disk size | ~120 MB (safetensors) |
| RAM at inference | ~150 MB |
| Speed | ~10K sentences/sec CPU single-core |
| Deps | `model2vec>=0.4.0` (~15 MB, pure numpy) |
| **Total new dep weight** | **~135 MB** (model + pip package) |

Compared to old system (LiquidAI 354M): 1000× smaller, no GPU, no torch, no transformers.

### Alternatives

| Model | Params | Dim | MTEB | Size | When to choose |
|-------|--------|-----|------|------|----------------|
| potion-base-8M | 7.56M | 256 | 51.08 | ~30 MB | Memory-constrained (RPi) |
| potion-base-32M | 32.3M | 256 | 52.13 | ~120 MB | Default — best general quality |
| potion-code-16M | ~16M | 256 | 37.05 CoIR | ~60 MB | Only if queries are pure code identifiers |
| all-MiniLM-L6-v2 | 22.7M | 384 | 55.93 | ~90 MB | ONNX baseline (needs onnxruntime ~300MB) |

**Rationale for potion-base-32M:** Pentest docs are prose-with-embedded-commands,
not pure source code. The 32M model's larger vocabulary captures technical terminology
(Kerberos, NTLM, Pass-the-Hash) better than code-specialized models, and its MTEB
retrieval score is solid. Inference speed is identical to 8M (same architecture,
more params only affects load time).

### Model2Vec vs ONNX Transformers

Model2Vec (static embeddings) is fundamentally different from transformer-based
ONNX models:

| Aspect | Model2Vec | ONNX transformer |
|--------|-----------|-----------------|
| Inference | Token lookup + average | 12-layer BERT forward pass |
| Speed | ~10K sent/sec CPU | ~50 sent/sec CPU |
| Deps | numpy + model2vec (15 MB) | onnxruntime + transformers (300 MB) |
| Quality | 92% of MiniLM-L6-v2 | Baseline |
| Context | No limit (effectively unbounded) | 512 tokens |
| Warm-up | Instant (<1ms load) | 3-5s model load |

The speed advantage means brute-force cosine search over ~100K embeddings is
feasible (~10ms) — no need for ANN indices at this scale.

## RRF Fusion

### Formula

```python
_K_RRF = 60

def _rrf_scores(
    chunk_ids: list[int], scores: list[float]
) -> dict[int, float]:
    """Convert (id, score) list to RRF score dict.
    Higher raw score → rank 1 → higher RRF score.
    """
    ranked = sorted(
        range(len(scores)), key=lambda i: -scores[i]
    )
    return {
        chunk_ids[i]: 1.0 / (_K_RRF + rank + 1)
        for rank, i in enumerate(ranked)
    }

# Fuse:
rrf_vec = _rrf_scores(vec_ids, vec_scores)
rrf_bm25 = _rrf_scores(bm25_ids, bm25_scores)

all_ids = set(rrf_vec) | set(rrf_bm25)
combined = {
    cid: alpha * rrf_vec.get(cid, 0.0)
         + (1 - alpha) * rrf_bm25.get(cid, 0.0)
    for cid in all_ids
}
ranked = sorted(combined.items(), key=lambda x: -x[1])[:top_k]
```

### Alpha by Query Tier

α = weight on semantic (vec0) results; (1-α) = weight on BM25 (FTS5) results.

Maps to existing `_classify_query()` in `knowledge/search.py`:

| QueryTier | α | Query example | Rationale |
|-----------|---|---------------|-----------|
| `CONCEPTUAL` | 0.5–0.6 | "bypass EDR unhooking" | Slightly favor semantic — user wants technique desc |
| `TOOL_COMMAND` | 0.2–0.3 | "responder -I eth0 -w" | Favor BM25 — exact flag docs |
| `EXACT` | 0.1–0.2 | "CVE-2025-31161" | Strongly FTS5 trigram |
| `PATH` | 0.2–0.3 | "/etc/nginx/nginx.conf" | Favor BM25 path matching |

K=60 (same as semble) is conservative — rank position decays slowly, giving
deeper results from each system a fair shot.

### Why RRF (not min-max normalization)

RRF uses only rank position, not raw scores. This matters because BM25 scores
vary wildly by query length and corpus statistics, while cosine similarities
are query-dependent. RRF makes α have consistent semantic meaning across all
queries — α=0.5 always means "equal weight to semantic and keyword ranks."

## Vector Storage

### Schema (sqlite-vec vec0)

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS section_vectors USING vec0(
    section_id INTEGER PRIMARY KEY,
    source TEXT PARTITION KEY,
    embedding FLOAT[256] DISTANCE_METRIC=COSINE
);
```

### Partition Key

`source` as PARTITION KEY enables efficient per-source filtering:
- `kdb search --source=hacktricks` → vec0 only searches hacktricks' vectors
- Without filter → searches all partitions (no performance cost — vec0 handles this)

### Migration

Embedding dim changes from old 1024 → 256. Full index rebuild required:
```sh
kdb index --force
```

## SemHash Deduplication (Optional Enhancement)

### Purpose

Many pentest wikis describe the same technique (e.g., "Kerberos silver ticket attack"
appears in HackTricks, ired.team, PayloadsAllTheThings). SemHash detects these
near-duplicates at index time so search results don't show 5 near-identical results.

### Algorithm

```
1. Encode all section bodies → embedding matrix (N × 256)
2. Build Vicinity ANN index (USearch default)
3. For each section, nearest-neighbor search against all others
4. If cosine similarity ≥ threshold (0.88) → mark as duplicate group
```

### Schema addition

```sql
ALTER TABLE sections ADD COLUMN dedup_group INTEGER;
CREATE INDEX idx_sections_dedup_group ON sections(dedup_group);
```

### Strategy: Moderate (flag + collapse)

- All sections remain in index (no data loss)
- Sections in same dedup_group flagged with same ID
- Search shows max 1 result per group by default
- `kdb search --show-all` reveals duplicates with source annotation
- CLI display: append `(+2 more sources)` to collapsed groups

### Threshold tuning

Start at 0.88. Use `semhash.DeduplicationResult.get_least_similar_from_duplicates()`
to find boundary cases and calibrate. Pentest content with high technical overlap
may need 0.85 for proper near-duplicate detection.

### Known risk: false positives

- Silver ticket vs Golden ticket: similar Kerberos prose but different commands
  → may group incorrectly if threshold is too loose
- Different tools for same task (mimikatz vs procdump for LSASS dump)
  → similar intent, different implementation → may be falsely grouped

Mitigation: threshold on the conservative side (0.88–0.90) and let the rare miss
be a miss rather than a false positive.

## Code-Block Detection (Optional Enhancement)

### Purpose

Sections containing code blocks (fenced commands) should rank higher when the
query looks like a tool command.

### Implementation

```python
# In chunk.py
import re

_CODE_FENCE_RE = re.compile(r"```\w*\n.*?\n```", re.DOTALL)

def _has_code_block(body: str) -> bool:
    return bool(_CODE_FENCE_RE.search(body))
```

Store as column:
```sql
ALTER TABLE sections ADD COLUMN has_code_block INTEGER DEFAULT 0;
```

At query time, if `_classify_query` returns `TOOL_COMMAND`, apply ×1.5 boost to
sections with `has_code_block=1`.

## Changes Required

### New file: `knowledge/embed.py`

Recreate from git history, adapted for model2vec:

```python
"""Model2Vec static embedding model loading."""

from __future__ import annotations

from model2vec import StaticModel


def get_embedder(
    model_name: str = "minishlab/potion-base-32M",
) -> StaticModel:
    """Return a cached StaticModel singleton.

    Uses function-attribute cache (not module-level global) per
    docs/rules/python-paradigm.md.
    """
    cached = getattr(get_embedder, "_cached", None)
    if cached is not None:
        return cached
    model = StaticModel.from_pretrained(model_name)
    get_embedder._cached = model
    return model
```

### Modified files

| File | Change |
|------|--------|
| `knowledge/db.py` | Add vec0 table to `ensure_schema()`; add `dedup_group` + `has_code_block` columns |
| `knowledge/indexer.py` | After chunking: embed sections → vec0 insert; optional semhash dedup |
| `knowledge/search.py` | Parallel FTS5 + vec0 query → RRF fusion → reranking; query-tier α mapping |
| `knowledge/chunk.py` | Add `_has_code_block()` extraction |
| `knowledge/config.py` | Add `embed.model` to Config (default "minishlab/potion-base-32M") |
| `pyproject.toml` | Add `model2vec>=0.4.0` dependency |

### Test files

| File | Tests |
|------|-------|
| `tests/test_embed.py` | Recreate: model loading, cache behavior, encode shape |
| `tests/test_search.py` | Extend: RRF fusion, α weighting, hybrid results ordering |
| `tests/test_indexer.py` | Extend: embedding storage in vec0 table |

## Dependency Footprint

```toml
# pyproject.toml addition
dependencies = [
    "model2vec>=0.4.0",
]
```

Total new disk: ~15 MB (pip package) + ~120 MB (model file downloaded to HF cache).
No GPU, no CUDA, no PyTorch, no transformers, no onnxruntime.

Compare to old system: sentence-transformers + torch + numpy + sqlite-vec ≈ 5 GB.

## Estimated Effort

| Phase | Task | Hours |
|-------|------|-------|
| P1 | Add dep, create `embed.py`, model loading + cache | 1 |
| P2 | vec0 table (dim=256) in `ensure_schema`, migration path | 2 |
| P3 | Embed sections at index time, vec0 insert | 4 |
| P4 | RRF fusion in `cmd_search`, α by query tier, `--hybrid` flag | 8 |
| P5 | SemHash dedup at index time (moderate flag-based) | 8 |
| P6 | Code-block detection + boost | 2 |
| P7 | Benchmark vs BM25-only vs pure semantic | 4 |
| **Total** | | **~29** |

## Backward Compatibility

- `kdb search` without flags → BM25-only (current behavior, unchanged)
- `kdb search --hybrid` → FTS5 + vec0 RRF fusion
- `kdb search --json` → `distance` field becomes RRF score when `--hybrid`
- `kdb index --force` → full rebuild (drops vec0, recreates with dim=256)
- `kdb index` (incremental) → rebuilds only changed sources, updates their vectors

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Embedding dim mismatch with old index | Search returns errors | `--force` rebuild; dim check in `cmd_search` |
| model2vec first-load latency (3-5s) | First search is slow | Function-attribute cache (already in codebase pattern) |
| SemHash threshold brittleness | False positives/negatives | Start at 0.88; `get_least_similar_from_duplicates()` for tuning |
| RRF query-time regression | 10-50ms added to search | Brute-force numpy on 100K × 256 is ~10ms; acceptable |
| JSON consumers expect BM25 distance | API breakage | Only changes with `--hybrid` flag; document in changelog |
