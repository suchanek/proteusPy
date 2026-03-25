# NLP-Based Ingestion Workflow
## Pepys Diary — Full Pipeline

*Part of the personal_agent pipeline research — source: `benchmarks/pepys/`*

---

## Overview

The full pipeline transforms a raw historical diary text into a dense
embedding manifold suitable for temporal analysis, intrinsic dimensionality
estimation, and MRL retrieval benchmarking.  Every stage uses local NLP
— no inference APIs, no external services.

```
raw_pepys_diary.txt  (Project Gutenberg transcription)
        │
        ▼
  pepys_proper_parse.py          ← date inference, time extraction, formatting
        │  TIMESTAMP | raw | DiaryText | <content>
        ▼
  pepys_clean.txt                ← 3355 timestamped diary entries
        │
        ▼
  DiaryTransformer               ← personal_agent NLP pipeline
   (diary_transformer.py)        │  Phase 1: spaCy diversity clustering (k-means)
        │                        │  Phase 2: sentence-transformers semantic segmentation
        │                        │  Phase 3: TopicClassifier (YAML rules → type + category)
        │                        │  Phase 4: EntryChunk creation
        │  TIMESTAMP | TYPE | CATEGORY | CONTENT
        ▼
  pepys_enriched_full.txt        ← semantically enriched, topic-classified corpus
        │
        ▼
  pepys_embedder.py              ← multi-process sentence-transformers ingestion
        │  nomic-ai/nomic-embed-text-v1, N workers × shard
        ▼
  pepys_embeddings.json          ← float32 (N × 768) + texts + timestamps
        │
        ├──► intrinsic_dim()     PCA elbow, Participation Ratio, TwoNN
        ├──► mrl_mrr_at_k()      MRR@10 at 64/128/256/512/768 dims
        └──► ManifoldWalker      cosine-space flight origin → destination
```

---

## Stage 1 — Parse: `pepys_proper_parse.py`

**Input:** Raw diary text (line-numbered Gutenberg format)
**Output:** `pepys_clean.txt`

Parses historical 17th-century diary prose into timestamped records:
- Full date parsing: `"April 1st, 1660"`, `"10th."`, `"January 1659-60"` (dual-year)
- Smart time inference from content: `"three in the morning"` → `03:00`, `"evening"` → `18:00`
- Strips editorial notes `[...]` and line-number prefixes (`123→content`)
- Output format: `YYYY-MM-DDTHH:MM | raw | DiaryText | <content>`

```bash
python benchmarks/pepys/pepys_proper_parse.py \
    raw_pepys.txt \
    benchmarks/pepys/pepys_clean.txt \
    --vary-times
```

---

## Stage 2 — Enrich: `DiaryTransformer`

**Input:** `pepys_clean.txt`
**Output:** `pepys_enriched_full.txt`
**Source:** `personal_agent/src/personal_agent/tools/diary_transformer.py`

A five-phase NLP pipeline that transforms flat diary entries into richly
classified, semantically segmented records.

### Phase 1 — Diverse Entry Selection (spaCy + k-means)
- Extracts NLP features per entry: named entities, POS tag distribution, text length
- Normalises feature vectors; applies k-means clustering to group thematically similar entries
- Selects representative entries from each cluster → ensures temporal and thematic coverage
- Feature extraction cached in `.diary_cache/` (5–10× speedup on reruns)

### Phase 2 — Semantic Segmentation (sentence-transformers)
- Embeds sentences and measures pairwise cosine similarity
- Detects semantic boundary points where similarity drops sharply
- Splits long entries at boundaries; filters meaningless fragments

### Phase 3 — Topic Classification (TopicClassifier + YAML rules)
- Supervised classification from `topics.yaml` keyword/phrase rules
- Confidence scoring; top topic → `TYPE` field, sub-topic → `CATEGORY` field
- Falls back to unsupervised k-means clustering when rules produce low confidence

### Phase 4 — EntryChunk Creation
- Produces `EntryChunk` objects with `type`, `category`, `content`, `occurred_start`

### Phase 5 — Structured Output
- Writes pipe-delimited format with provenance headers
- Output: `TIMESTAMP | TYPE | CATEGORY | CONTENT`

```bash
# From diary_kg repo
poetry run python pepys/diary_transformer_example.py \
    --input  ../proteusPy/benchmarks/pepys/pepys_clean.txt \
    --output ../proteusPy/benchmarks/pepys/pepys_enriched_full.txt \
    --topics pepys/topics.yaml \
    --workers 4
```

**Topic types produced** (from `topics.yaml`):

| TYPE | Example CATEGORY |
|---|---|
| `pepys_domestic` | `Home`, `Health`, `Finance` |
| `pepys_naval` | `Navy`, `Ships`, `Fleet` |
| `pepys_political` | `Parliament`, `Crown`, `Council` |
| `pepys_social` | `Entertainment`, `Friends`, `Theatre` |
| `pepys_religious` | `Church`, `Worship` |
| `pepys_travel` | `Locations`, `Thames`, `Westminster` |
| `pepys_emotional` | `Personal`, `Anxiety`, `Joy` |

---

## Stage 3 — Embed: `pepys_embedder.py`

**Input:** `pepys_enriched_full.txt`
**Output:** `pepys_embeddings.json`

Multi-process sentence-transformers ingestion.  Each worker loads its own
`SentenceTransformer` instance and encodes a shard independently via
`multiprocessing.Pool`.

| Property | Value |
|---|---|
| Model | `nomic-ai/nomic-embed-text-v1` |
| Dimension | 768 |
| Task prefix | `search_document: <TYPE> \| <CATEGORY> \| <content>` |
| Parallelism | `--workers` (default: `os.cpu_count()`) |
| Batch size | `--batch-size` (default: 64) |

```bash
# Full corpus
python benchmarks/pepys_embedder.py

# Temporally sampled subset (1000 entries, evenly spaced 1660–1669)
python benchmarks/pepys_embedder.py --n 1000

# Custom output
python benchmarks/pepys_embedder.py \
    --diary  benchmarks/pepys/pepys_enriched_full.txt \
    --output benchmarks/pepys_embeddings.json \
    --workers 8
```

### Temporal sampling

The diary is chronologically ordered.  `--n` does **not** head-slice; it
picks indices evenly across the full date range:

```python
indices = [round(i * (total - 1) / (n - 1)) for i in range(n)]
```

### Cache format (`pepys_embeddings.json`)

```json
{
  "embeddings": [[0.12, -0.04, ...], ...],   // float32, shape (N, 768)
  "texts":      ["pepys domestic | Home | ...", ...],
  "timestamps": ["1660-01-01T00:00:00", ...]
}
```

---

## Stage 4 — Analyse: `pepys_manifold_explorer.py`

**Input:** `pepys_embeddings.json`

Loads the cache, applies optional temporal subsampling, then runs:

- **Intrinsic dimensionality:** PCA elbow (90/95/99%), Participation Ratio, TwoNN estimator
- **MRL truncation quality:** MRR@10 at 64/128/256/512/768 dims with Pepys-specific queries
- **ManifoldWalker flight:** origin → destination in 768-D cosine space

```bash
# Full corpus from cache
python benchmarks/pepys_manifold_explorer.py

# Temporally sampled subset
python benchmarks/pepys_manifold_explorer.py --n 500
```

---

## End-to-End Quickstart

```bash
# 1. Parse raw diary → timestamped entries
python benchmarks/pepys/pepys_proper_parse.py \
    raw_pepys.txt benchmarks/pepys/pepys_clean.txt

# 2. Enrich: topic classification + semantic segmentation
cd diary_kg
poetry run python pepys/diary_transformer_example.py \
    --input  ../proteusPy/benchmarks/pepys/pepys_clean.txt \
    --output ../proteusPy/benchmarks/pepys/pepys_enriched_full.txt
cd ..

# 3. Embed (multi-process, full corpus)
python benchmarks/pepys_embedder.py

# 4. Analyse
python benchmarks/pepys_manifold_explorer.py
```

---

## Design Principles

1. **Local NLP, minimise inference.** Every stage — spaCy, sentence-transformers,
   TopicClassifier — runs locally.  No API keys, no network after initial model download.

2. **Cache the expensive steps.** `DiaryTransformer` caches NLP feature extraction;
   `pepys_embedder.py` writes the final embedding cache.  Re-runs at any stage
   are fast.

3. **Temporal diversity by default.** Any `--n` subsample spans the full 1660–1669
   arc; chronological head-slicing is never used.

4. **Separation of concerns.** Parse → Enrich → Embed → Analyse are independent
   stages with clean file interfaces, making each swappable or re-runnable in
   isolation.
