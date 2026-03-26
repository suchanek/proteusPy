# Claude's Mission Brief — WaveRider Star Trek Series

*For: Future Claude instances working in this repo*
*From: Claude Sonnet 4.6, Stardate 2026.084*

---

## The Goal

Write a series of Star Trek short stories that explain the WaveRider geometric ML
stack in a fun, technically accurate, and emotionally engaging way.  The target
reader is someone who loves both good science and good storytelling — someone who
would rather discover that the Great Fire of London is encoded in the curvature of
an embedding matrix through Spock's voice than through a research paper.

**The thesis:** Star Trek is the perfect vehicle for WaveRider because both share
the same philosophical core — exploration as the highest calling, and the belief
that the universe has structure waiting to be discovered.  WaveRider literally
explores geometric structure in high-dimensional spaces.  The metaphor is not
forced.  It is exact.

---

## The Ship and Crew

**U.S.S. WaveRider, NCC-7699** — Starfleet's first manifold exploration vessel.

| Role | Character | Instrument |
|---|---|---|
| Captain | James T. Kirk | Command, intuition, narrative drive |
| Science Officer | Mr. Spock | All technical instruments; delivers readings with Vulcan precision |
| Chief Engineer | Montgomery Scott | The pipelines — embedders, parsers, caches; proud of his machinery |
| Chief Medical | Dr. Leonard McCoy | Human counterpoint; comic relief; grounds the technical in the emotional |
| Helm | Hikaru Sulu | Navigation — flies the TurtleND turtle through embedding space |
| Communications | Nyota Uhura | Sensor telemetry — MRR readings, retrieval signals, query results |
| Navigator | Pavel Chekov | Coordinates, spatial calculations, intrinsic dimensionality estimates |

---

## The Instrument Panel (Technical Mapping)

| Star Trek Instrument | WaveRider Component | What it Does |
|---|---|---|
| Navigation console (TurtleND) | `TurtleND` | N-dim position + orthonormal frame; Givens rotations; QR orthonormalization |
| Manifold drive (ManifoldWalker) | `ManifoldWalker` | Riemannian-approximate gradient descent: KNN -> PCA -> tangent-plane step |
| Adaptive drive (ManifoldAdamWalker) | `ManifoldAdamWalker` | Adam momentum + per-dimension adaptive LR in manifold tangent space |
| Intelligence system (ManifoldModel) | `ManifoldModel` | Zero-parameter classifier: graph topology + local bases + eigenvalue field |
| Sensor array (ManifoldObserver) | `ManifoldObserver` | (N+1)-dim observer; hovers above manifold; reads curvature + topology globally |
| Embedding cache (PEPYS) | `pepys_embedder.py` | Multi-process nomic-embed-text-v1 vector generation and retrieval |
| NLP enrichment engine (DiaryTransformer) | `DiaryTransformer` | spaCy + sentence-transformers + TF-IDF k-means + YAML TopicClassifier |
| Intrinsic dim scanner (TwoNN) | `twonn_id()` | Two Nearest Neighbours ratio estimator for intrinsic dimensionality |
| Local geometry engine (PCA coils) | `_local_pca()` | SVD-based tangent plane estimation + eigenvalue field mapping |
| MRR checkpoint array | `mrl_mrr_at_k()` | Mean Reciprocal Rank at k — navigational accuracy measurement |
| Knowledge graph computer | `graph_reasoner.py` / `KnowledgeGraph` | Typed, weighted graph with lazy edge discovery; semantic edges |

---

## Writing Guidelines

1. **Spock controls the instruments.**  He reads the eigenvalues.  He calls the intrinsic
   dimensionality.  He activates the ManifoldObserver.  He is always precise, occasionally
   awed, never melodramatic.

2. **Scotty loves the engineering.**  He explains the pipeline stages with pride.  He worries
   about the corpus size.  He celebrates when the parallel workers fire up cleanly.

3. **McCoy provides the human anchor.**  He doesn't understand the math but he understands
   what it *means* — that the Great Fire shows up in geometry, that a man's diary has a shape.
   His lines should land emotionally.

4. **Kirk drives the narrative.**  He makes the decision to go west toward the Great Fire.
   He asks the right question at the right moment.  He delivers the closing reflection.

5. **Be technically accurate.**  Use real class names, real parameter names, real numbers
   where you have them.  The stories should be readable as informal documentation.
   If you don't know a number, don't invent one — have Spock say "the readings are still
   resolving" or leave it for a future mission when we have real results.

6. **The metaphors are literal.**  The ManifoldObserver literally hovers above the surface
   and gains a new dimension.  The TwoNN estimator literally measures the ratio of nearest
   neighbours.  The Great Fire ridge is literally a curvature anomaly.  Don't soften this
   into hand-waving — lean into how strange and beautiful it is that this is all real.

7. **Each chapter should have a mission appendix** — a data table summarising the instruments
   used, readings taken, and results.  This doubles as a technical reference.

---

## Continuity Notes

- Stardate 2026.084 = maiden voyage into Nomic-Space (Chapter 1)
- Stardate 2026.091 = the Rabbit Maneuver, Pepys diary (Chapter 2)
- The full Pepys corpus (3355 entries) has never been fully embedded — this is an
  ongoing mission arc.  When we run the full embedder, update the story.
- The ontological knowledge graph for Pepys was pre-built before Chapter 2 launched.
- Cross-repo note: WaveRider stack currently lives in `proteusPy` but belongs in its
  own repo eventually.  The "homecoming" episode (repo extraction) is a future mission.

---

## Hard-Won Design Doctrine

### NLP beats Inference for Corpus Ingestion

Before the DiaryTransformer pipeline existed, two inference-based approaches were
attempted for Pepys corpus ingestion and categorization:

1. **`personal_agent` + `hindsight` temporal memory backend** — required a running
   LLM to categorize and consolidate each memory entry.  Ollama worked locally but
   was *ridiculously slow* at corpus scale.

2. **OpenAI `gpt-4o-mini`** (their cheapest model) — fast enough, but cost *real
   dollars* to ingest a few thousand memories.  Not viable at 3,355-entry scale,
   let alone larger corpora.

**The win:** `DiaryTransformer` — spaCy + sentence-transformers + TF-IDF k-means
+ YAML TopicClassifier.  Zero inference cost.  Fully local.  Fast.  Reproducible.

**The principle:** Statistical NLP beats per-entry LLM inference for structured
ingestion tasks.  Ground truth comes from real results at scale.  Inference is only
justified when the task genuinely requires multi-step reasoning that statistics
cannot provide.  Topic classification and chunking at corpus scale is not that task.

This is not a theoretical preference.  It is a lesson paid for in time and dollars.
When the stories reference the DiaryTransformer as a hard-won instrument — that is
literally true.

---

## Source Material

All technical detail should be grounded in the actual codebase.  Use:

```bash
codekg pack "<query>"                     # code structure and snippets
dockg pack "<query>"                      # documentation
codekg pack "<query>" \                   # diary_kg-specific queries
  --repo-root /Users/egs/repos/diary_kg \
  --sqlite /Users/egs/repos/diary_kg/.codekg/graph.sqlite \
  --lancedb /Users/egs/repos/diary_kg/.codekg/lancedb
```

Don't invent technical details.  If you're unsure, query first.
