# WaveRider — Glossary of Terms

*U.S.S. WaveRider, NCC-7699 · Starfleet Knowledge Division*
*For readers of the mission logs — technical and non-technical alike.*

---

## The Ship & Crew

**U.S.S. WaveRider, NCC-7699**
Starfleet's first manifold exploration vessel. Commissioned by Admiral Eric Suchanek,
Director of the Starfleet Knowledge Division. The WaveRider's mission is to navigate
high-dimensional semantic and temporal spaces — the geometry of meaning and time as
encoded in text corpora. All instruments are real software running on real hardware.

**Admiral Eric G. Suchanek, PhD**
Director, Starfleet Knowledge Division. The architect of the WaveRider stack; has been
building the pipeline for three years before the series begins. He knows what the geometry
requires before the crew runs the instruments. He reveals the mission one layer at a time.

| Character | Role | Instrument |
|---|---|---|
| Kirk | Captain — narrative drive | Command and decisions |
| Spock | Science Officer — reads all instruments | Precise, occasionally awed, never melodramatic |
| Scotty | Chief Engineer | Pipelines: embedders, parsers, caches |
| McCoy | Chief Medical Officer | Human anchor; grounds the technical in the emotional |
| Sulu | Helmsman | Flies the TurtleND navigation console |
| Uhura | Communications | MRR sensor readouts, query telemetry |
| Chekov | Navigator | Coordinates, TwoNN calculations, intrinsic dimensionality estimates |

---

## The Instruments (WaveRider Stack)

Each Star Trek instrument corresponds to a real Python class or function in the WaveRider codebase.

---

### TurtleND — Navigation Console

**Star Trek:** The navigation console at the helm.
**Code:** `TurtleND` class in `proteusPy/turtleND.py`

A turtle that lives in N-dimensional space and carries an orthonormal frame — a set of
perpendicular unit vectors describing the directions it can move. As it hops from point
to point in embedding space, it rotates its frame to stay aligned with the local geometry.

Key operations:
- **Givens rotations** — the mechanism for rotating the frame in any pair of dimensions
- **QR orthonormalization** — keeps the frame from drifting (numerical housekeeping)
- **`orient_in_time()`** — aligns the turtle's forward direction with the temporal axis of the manifold

*A Flatland creature that always knows which way it is pointing, even in 769 dimensions.*

---

### ManifoldWalker — Manifold Drive

**Star Trek:** The standard warp drive.
**Code:** `ManifoldWalker` class

Performs Riemannian-approximate gradient descent on the data manifold. At each step:
1. Finds the K nearest neighbors of the current position
2. Runs local PCA to estimate the tangent plane (the locally flat surface)
3. Steps in the direction of greatest descent, projected onto that plane

The Walker denoises the gradient field before stepping — it sees through the high-dimensional
ambient noise to the underlying manifold structure. This is what the raw KNN greedy hop cannot do.

---

### ManifoldAdamWalker — Adaptive Drive

**Star Trek:** The adaptive warp drive.
**Code:** `ManifoldAdamWalker` class

Adam optimizer running in the tangent space of the manifold. Adds:
- **Momentum** (β₁) — weighted running average of past gradients (where you've been going)
- **Adaptive learning rate** (β₂) — per-dimension step size that shrinks in noisy directions

Faster convergence than the standard Walker. The open question (Chapter 6 hypothesis):
should temporal navigation accumulate *global* momentum rather than flipping the arrow of time?

---

### ManifoldObserver — Sensor Array

**Star Trek:** The ship's sensor array, hovering above the manifold surface.
**Code:** `ManifoldObserver` class

The Observer appends one additional dimension to the turtle's frame — a unit vector
pointing *away* from the manifold surface, into the ambient space above it. This gives
it an extrinsic view: it can see the curvature of the surface it's flying over, the
ridges and valleys in the embedding landscape, in a way the TurtleND (which lives *on*
the surface) cannot.

*Like a drone that can see the mountain's shape from above while the turtle crawls along its slopes.*

---

### ManifoldModel — Intelligence System

**Star Trek:** The ship's computer — analytical intelligence.
**Code:** `ManifoldModel` class

A classifier with **zero learned parameters**. It classifies new points not by training
weights but by reading the topology of the local neighborhood: which manifold region is
this point in? What is the eigenvalue field here? What do the nearest neighbors tell us
about the local geometry?

The zero-parameter doctrine: intelligence from structure, not from memorized weights.

---

### DiaryTransformer — NLP Enrichment Engine

**Star Trek:** The universal translator crossed with a scientific analyzer.
**Code:** `DiaryTransformer` class in `diary_kg`

Ingests raw diary text and enriches each entry with:
- **spaCy** — named entity recognition (people, places, dates)
- **TF-IDF k-means** — unsupervised topic discovery (what is this entry *about*?)
- **YAML TopicClassifier** — rule-based category assignment from discovered clusters
- **sentence-transformers** — dense semantic embeddings for the embedding matrix

Requires zero LLM inference. Fully local. Fast. Reproducible. This was hard-won —
two inference-based approaches (Ollama + hindsight, GPT-4o-mini) were tried and
abandoned as too slow or too costly at corpus scale. The NLP pipeline won.

---

### PEPYS Embedding Cache — Engineering Cache

**Star Trek:** The dilithium crystal storage — pre-energized, ready to retrieve.
**Code:** `pepys_embedder.py`

Multi-process embedding engine using `all-mpnet-base-v2`. Generates 768-dimensional
semantic vectors for every entry in the Pepys corpus and caches them to disk.
The cache means the crew never re-computes what Scotty already built.

---

### TwoNN Scanner — Intrinsic Dimensionality Scanner

**Star Trek:** Chekov's deep-field scanner for measuring the true shape of space.
**Code:** `twonn_id()` function

Estimates the **intrinsic dimensionality** of a dataset from the ratio of each point's
two nearest neighbor distances. If the data lies on a low-dimensional manifold embedded
in high-dimensional space, that ratio follows a predictable statistical distribution.
Fit the distribution → recover the true dimension.

Formally: for each point, compute `μ = dist(2nd nearest) / dist(1st nearest)`.
Fit the empirical distribution of μ values to recover intrinsic dimension `d`.

*The crew is always living in too many dimensions. TwoNN tells them how many actually matter.*

---

### MRR Checkpoint Array — Navigational Accuracy Sensors

**Star Trek:** Uhura's sensor readout — how accurately is the ship finding its targets?
**Code:** `mrl_mrr_at_k()` function

**Mean Reciprocal Rank at k (MRR@k):** Given a query point, retrieve the k nearest
neighbors. The reciprocal rank is `1/rank` where rank is the position of the true
match in the retrieved list. Average over many queries → MRR. Higher is better; 1.0
means the true match is always ranked first.

---

### Local PCA Engine — Tangent Plane Coils

**Star Trek:** The field coils that flatten local space for navigation.
**Code:** `_local_pca()` function

At any point in the embedding space, find the K nearest neighbors and run
Singular Value Decomposition (SVD) on them. The leading eigenvectors define the
*tangent plane* — the locally flat surface the manifold most resembles at that point.
The eigenvalues tell you how much variation exists in each direction.

*Like finding the slope of a hill by looking at the ground immediately around your feet.*

---

## Mathematical Concepts

---

### Manifold

A high-dimensional dataset that locally resembles flat Euclidean space, even if its
global shape is curved. A sphere is a 2-dimensional manifold embedded in 3D space.
The Pepys diary is a 769-dimensional manifold embedded in embedding space — but its
*true* shape (intrinsic dimensionality) is much smaller.

**Why it matters:** The WaveRider navigates manifolds. All its instruments are
designed to exploit manifold structure — the fact that data clusters near a
low-dimensional curved surface rather than filling the ambient space uniformly.

---

### Intrinsic Dimensionality

The true number of independent axes of variation in a dataset, regardless of the
ambient space it was embedded in. A spiral in 3D space has intrinsic dimensionality 1
(you can describe any point with a single number: how far along the spiral).

The Pepys corpus embedded in 769 dimensions has some intrinsic dimensionality much
smaller than 769. That number is the *true shape* of nine years of a human life.
TwoNN measures it. Chapter 6 computes it for the first time.

*"He needed exactly that many dimensions to be himself."* — Dr. McCoy

---

### Semantic Embedding

A function that converts text into a point in high-dimensional space, such that texts
with similar *meaning* land near each other. The WaveRider uses `all-mpnet-base-v2`,
which produces 768-dimensional vectors. Pepys writing about the plague on two different
days in two different years will produce nearby vectors — because the meaning is similar.

---

### Augmented Manifold

The embedding space after a temporal coordinate has been appended. Instead of 768
semantic dimensions, each point has 769 dimensions: 768 semantic + 1 temporal. The
temporal coordinate changes the geometry of the space, adding a gradient that the
TurtleND can follow.

---

### Destination-Relative Temporal Encoding

The corrected temporal encoding introduced in Chapter 5:

```
temporal_coord = abs(entry.fractional_year − destination.fractional_year)
```

The destination has temporal coordinate **zero**. Every other entry has a positive
coordinate equal to its distance in time from the destination. The KNN graph then
has a gradient pointing toward zero — toward the destination — which the turtle follows.

Contrast with the broken encoding (Chapter 4), which placed entries by absolute position
on the corpus timeline. That encoding created a gravitational center at the most
populated year (1668), pulling the turtle away from any other destination.

*The destination is always downhill. The turtle falls toward it.*

---

### Kendall τ (Tau)

A measure of rank correlation, ranging from −1 to +1. In the WaveRider context:

- **τ = +1.0** — perfect temporal order. Every hop lands later in time than the last.
- **τ = 0.0** — random temporal order. The path is equally likely to go forward or backward.
- **τ = −1.0** — perfect reverse order. Every hop goes earlier in time.

The corrected temporal flight in Chapter 5 achieves τ = **+0.0476** — positive, indicating
a net forward-in-time trajectory, though the path wandered through four different years
before arriving at the destination. The geometry got there; chronological order was not enforced.

---

### Monotonicity

The fraction of consecutive hop pairs where the turtle moved forward in time. A perfectly
monotonic temporal flight would have monotonicity = 1.0 (100%). The Chapter 5 corrected
flight achieved **33.3%** — confirming that destination-relative encoding is an effective
attractor but does not enforce temporal order of traversal.

---

### KNN Graph (K-Nearest Neighbors Graph)

A graph where each point in the embedding space is connected to its K nearest neighbors
by distance. The TurtleND hops along edges of this graph. The graph encodes the local
topology of the manifold — which points are close to which in semantic space.

The WaveRider uses K=10 for the Pepys corpus flights.

---

### Fractional Year

A continuous representation of a date as a decimal year.
`1663.803` = roughly October 1663. Allows arithmetic on dates — distances, gradients,
sorting — that calendar dates don't support natively.

---

### PCA (Principal Component Analysis)

A method for finding the directions of greatest variation in a dataset. In the
WaveRider context, local PCA on the KNN neighborhood estimates the tangent plane of
the manifold at that point. The leading eigenvectors are the manifold's principal
directions; the trailing eigenvectors are noise.

---

## Corpus & Data Terms

---

### Pepys Corpus

The complete diary of Samuel Pepys, 1660–1669. 6,450 entries after enrichment by
the DiaryTransformer. 768-dimensional mpnet embeddings per entry. The corpus is
*complete* — no gaps, no missing years — which makes it the minimum viable proof
of the WaveRider's temporal navigation capability. You cannot blame missing data.

The Admiral chose it deliberately.

---

### DiaryKG

The knowledge graph built from the Pepys corpus using the WaveRider pipeline.
Stores entries, embeddings, metadata, and semantic edges in SQLite + LanceDB.
Serves dual purpose: provides the embedding matrix for TurtleND navigation,
and maintains the queryable knowledge graph for semantic search.

*TurtleND uses DiaryKG as its star chart. DiaryKG uses TurtleND as its explorer.*

---

### The Tree of Knowledge

The larger mission behind the Pepys proof. Not revealed until the crew has earned it.
See STORY_ARC.md.

---

## Navigation Terms

---

### Hop

A single step in a TurtleND flight — from the current position to the nearest
neighbor in the KNN graph that best satisfies the navigation objective (semantic
proximity to destination, temporal gradient, or both). Each hop is logged with
date, entry text, and running Kendall τ.

---

### Semantic Flight

Navigation through embedding space guided only by semantic proximity — moving toward
the target entry's *meaning*, ignoring temporal position. The turtle follows the
semantic gradient, hopping through entries that are similar in meaning to the destination.
Chapter 4 demonstrated that semantic flight achieves τ ≈ +0.19 on the Pepys corpus —
time is weakly encoded in meaning even when not explicitly added.

---

### Temporal Flight

Navigation through the augmented (769-dimensional) embedding space, where the temporal
coordinate creates an additional gradient toward the destination. Two variants:
- **Broken (Ch 4):** Absolute temporal encoding — the turtle was pulled toward 1668 (corpus mean)
- **Corrected (Ch 5):** Destination-relative encoding — the turtle falls toward zero (the destination)

---

### The Fall Forward

The Chapter 5 title and its central metaphor. Under destination-relative encoding, the
future is not a direction the turtle is pushed toward — it is a direction the turtle
*falls* toward, drawn by the curvature of the augmented manifold. The destination has
zero temporal distance; everywhere else has more. Gravity does the rest.

*The geometry demands it.*

---

### Semantic Resonance

The navigation mechanism actually observed in the Chapter 5 corrected flight. Instead of
marching forward through time, the turtle found its way to the destination
(1664-01-23) by following a cluster of semantically similar entries across four different
years: entries featuring "Sir W. Batten + Sir W. Penn + Whitehall." The destination was
reached not by temporal order but by meaning.

Path: 1663 → 1661 → 1663 → 1661 → 1660 → 1666 → **1664**.

The correct finding: destination-relative encoding is an effective attractor; temporal
momentum (Chapter 6 hypothesis) is the missing ingredient for monotonic traversal.

---

*Last updated: Stardate 2026.098*
*U.S.S. WaveRider, NCC-7699*
