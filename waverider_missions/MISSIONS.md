# U.S.S. WaveRider — Mission Log

*Starfleet Geometric Intelligence Division*
*Classification: Scientific Exploration*

---

## Table of Contents

- [Completed Missions](#completed-missions)
- [Mission Queue](#mission-queue)
- [Proposed Future Missions](#proposed-future-missions)
- [Standing Orders](#standing-orders)

---

## Completed Missions

| # | Stardate | Title | File | Concept Covered |
|---|---|---|---|---|
| 1 | 2026.084 | **The Maiden Voyage** | [waverider_trek.md](waverider_trek.md) | TurtleND, ManifoldWalker, ManifoldAdamWalker, ManifoldModel (zero params), ManifoldObserver, PEPYS cache, intrinsic dimensionality |
| 2 | 2026.091 | **The Rabbit Maneuver** | [waverider_trek_ch2.md](waverider_trek_ch2.md) | DiaryTransformer pipeline, TF-IDF k-means category discovery, TopicClassifier, temporal manifold, Great Fire ridge, Plague valley, MRR checkpoints |
| 3 | 2026.095 | **The mpnet Manifold** | [waverider_trek_ch3.md](waverider_trek_ch3.md) | all-mpnet-base-v2 vs nomic geometry, full corpus 7282 chunks, TwoNN=13.54, MRL inversion (peak at 64D), dark flight panels, distillation vs dimensionality |

---

## Lab Notebook — Experiments & Findings

*Running record of sub-mission experiments that inform the narrative but are not full chapters.*

---

### Experiment T-1 — τ-Reversal Test
**Stardate:** 2026.086
**Instrument:** `pepys_temporal_flight.py` · `TurtleND.orient_in_time()`
**Corpus:** Pepys mpnet embeddings, 6450 entries, 768D → 769D augmented (α=1.0, k=10)

**Method:** Added `temporal_backward` flight mode (`forward=False`) alongside the existing
`temporal` forward mode.  Added `--negate-time` flag to flip the temporal axis sign.
Measured Kendall τ for both runs.

**Results:**

| Mode | Mono | Kendall τ | Net span |
|---|---|---|---|
| Semantic | 47.3% | +0.049 | +4.0 yr |
| Temporal → | 55.3% | **+0.457** | −0.2 yr |
| Temporal ← | 50.7% | −0.306 | +2.6 yr |
| Mixed 50/50 | 47.3% | −0.165 | +3.2 yr |

τ-reversal symmetry residual: 0.1507 (non-zero due to non-uniform entry density across years).

**Bug fixed:** `orient_in_time` was pointing toward `+e_t` (positive z-score = later dates).
Empirically, forward-in-time motion in the KNN graph requires the **negative** z-score direction.
Fixed in `TurtleND.orient_in_time`: `e_t[time_axis] = -1.0`.

**Open problem — Temporal Wandering:**
Even with the correct axis, temporal forward flight achieves only 55.3% monotonicity and
net span ≈ 0 yr over 151 hops.  Root cause: the temporal signal is **1 dimension in 769** —
roughly 3.6% of total step magnitude at α=1.0.  The 768D semantic field drowns it out.
The KNN graph was built on semantic proximity; its edges carry almost no temporal gradient.

**Resolution path:** ManifoldWalker addresses this correctly — it denoises the field via
local PCA on the KNN neighbourhood before stepping, isolating the true manifold directions
from ambient noise.  Temporal flight coherence should be revisited once the walk is
ManifoldWalker-driven rather than raw KNN greedy.

**Story hook:** Sulu can read the heading. The manifold is pulling him. But the warp field
can't feel a single wire in a 768-strand cable. The Walker is the instrument that will
let him hear it.

---

## Mission Queue

*Approved and ready to write — needs real benchmark data first where noted.*

- [ ] **Mission 4 — The Disulfide Labyrinth**
  The WaveRider dives into the 5-dimensional torsion-angle manifold of disulfide
  bonds — proteusPy's home territory.  175,277 bonds.  Five angles (χ1–χ5).
  Spock navigates the `disulfide_manifold_flight.py` benchmark.  McCoy discovers
  that protein geometry has the same intrinsic structure as human language.
  File to be created: `waverider_trek_ch4.md`

- [ ] **Mission 5 — The Adam Wars**
  A tactical episode: ManifoldWalker vs ManifoldAdamWalker, head to head on the
  Iris manifold (intrinsic dim 2.5, ambient dim 4).  Scotty argues for the
  standard drive; Spock advocates for the Adam variant.  Kirk lets the data decide.
  The reveal: Adam is faster but standard WaveRider reveals that 99% of the space
  is noise — a discovery Adam never makes because it doesn't look.
  File to be created: `waverider_trek_ch5.md`

- [ ] **Mission 6 — Flatland and the Observer** *(conceptual — no new data needed)*
  A deep-focus episode on the ManifoldObserver.  The crew re-reads Edwin Abbott's
  *Flatland* in the briefing room.  Spock explains why a TurtleND walking an
  N-dimensional manifold is a Flatland creature — and what it gains by appending
  one orthonormal dimension via QR.  The episode that explains the *philosophy*
  behind the Observer construction.
  File to be created: `waverider_trek_ch6.md`

---

## Proposed Future Missions

*Ideas — not yet approved.  Add stardate and move to queue when ready.*

- [ ] **The Zero-Parameter Doctrine** — ManifoldModel vs standard MLP on CIFAR-10.
  219x fewer parameters.  WaveRider wins.  Kirk delivers a speech about what
  "intelligence" really means.

- [ ] **The Digit Nebula** — MNIST and the 22-dimensional manifold of handwritten numbers.
  Chekov calculates: 784 ambient dimensions, 22 intrinsic.  97% is noise.

- [ ] **The Semantic Edge** — `graph_reasoner.py` and the `KnowledgeGraph` frontier.
  Spock introduces typed, weighted edges and lazy edge discovery.  The ship
  navigates not by geometry alone but by *meaning*.

- [ ] **The TwoNN Mystery** — A mystery episode.  The TwoNN scanner returns a
  negative intrinsic dimensionality.  Chaos on the bridge.  Scotty traces the
  bug to a sign error in the formula.  The fix: `len(mu) / sum(log(mu))`.
  (Based on the real bug fixed in pepys_manifold_explorer.py.)

- [ ] **The Homecoming** — The WaveRider stack is extracted from proteusPy and
  launched as its own starship — `waverider` repo, independent warp core, clean
  installability.  An episode about growing up and leaving home.

- [ ] **The Full Temporal Arc** — Once the full Pepys corpus is embedded *and*
  the ManifoldObserver has done a complete pass: a comprehensive cartographic
  episode.  The definitive map of nine years of one man's mind.

- [ ] **The Nomic Deep Field** — Return to Nomic-Space with a much larger corpus.
  What is the intrinsic dimensionality of all human knowledge?  Spock refuses
  to speculate.  Kirk asks him to anyway.

- [ ] **The Enrichment Engine** — A full episode dedicated to the DiaryTransformer:
  spaCy, k-means, sentence-transformers, YAML rules.  Scotty walks the crew
  through each phase.  McCoy is troubled by the idea that human experience
  can be clustered.  Spock is not.

---

## Standing Orders

1. Every mission must be technically grounded.  Query `codekg pack` and
   `dockg pack` before writing.  Use real class names, real parameters, real numbers.

2. Every completed mission gets a mission data appendix table.

3. Save as both `.md` and `.pdf` in this directory.

4. Mark missions complete in this file immediately upon finishing — include the
   filename in the table.

5. When real benchmark numbers become available (e.g. after a full embedder run),
   update in-progress missions to reflect actual data.
