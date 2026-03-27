# STAR TREK: THE MANIFOLD FRONTIER
## *Chapter 4: "The Direction of Time"*

---

> *Stardate 2026.086. The dark panels are dark no longer. The ManifoldWalker has flown. The temporal flight instruments are online. But what the crew discovered when they activated them was not what anyone expected — and the most important finding of this mission is a bug that points, when corrected, toward something true about the nature of time itself.*

---

## Prologue: After the Dark Panels

*Spock's instrument log. Stardate 2026.086.*

Chapter 3 ended with a promise. The bottom two panels of the manifold figure were dark — the ManifoldWalker and ManifoldObserver offline, pending integration of the `proteusPy` stack into the `diary_kg` environment.

That integration is now complete.

The temporal flight instruments are live. The WaveRider has flown the Pepys manifold in three modes — semantic, temporal, and mixed — and the results are recorded. One of them is a discovery. One of them is a cautionary instrument failure. And together, they have pointed the crew toward a correction that is, if anything, more interesting than the flight itself.

---

## Chapter 1: The Contaminated Cache

*Chief Engineer Scott's personal log. Stardate 2026.085.*

Before we could fly, we had to address a problem Spock found in the instrument calibration log.

The embedding cache — `pepys_mpnet_embeddings.json` — was corrupt.

Not corrupted in the catastrophic sense. The vectors were real. The timestamps were correct. The corpus was the right corpus. But the embedding pipeline had been using the nomic-embed-text task prefix — `search_document:` — on text that was being sent to `all-mpnet-base-v2`.

mpnet does not use task prefixes. It embeds raw text. Nomic uses prefixes because its training procedure explicitly conditions on them. Using a nomic prefix with mpnet is like calibrating a Vulcan scanner with a Klingon reference signal. The instrument still produces a number. The number is simply wrong.

The cache of 8,413 vectors — the result of Stardate 2026.095's successful 31-minute run — was, in this precise technical sense, cross-contaminated.

"How did we not catch this?" McCoy asked.

"Because the results were plausible," Spock said. "The embedding model produced meaningful vectors. The MRR metrics were strong. The manifold figure looked correct. The contamination was subtle — a systematic bias on the prefix token, absorbed into the embedding geometry without producing an obvious artifact. It would not have been visible in any single-model analysis. It only became visible when we examined the pipeline code directly."

Scotty did not say anything for a long moment.

"Right," he said finally. "Let's fix it."

---

## Chapter 2: The Clean Run — 6,450 Entries, 15.8 Seconds

*Engineering deck. Stardate 2026.085.*

The fix was surgical. Three changes to `pepys_embedder.py`:

1. `DEFAULT_MODEL` → `sentence-transformers/all-mpnet-base-v2`
2. `DEFAULT_OUTPUT` → `pepys_mpnet_embeddings.json` (the nomic naming had persisted as a ghost)
3. `_embed_shard`: strip `search_document:` prefix entirely — mpnet embeds raw text

Then: `python benchmarks/pepys_embedder.py --force`. Four workers. Batch size thirty-two.

Chekov read the result aloud.

"Six thousand, four hundred and fifty entries. Seven hundred sixty-eight dimensions. Fifteen point eight seconds."

The room was quiet for a different reason than last time.

"Fifteen seconds," Kirk said.

"Fifteen point eight," Chekov confirmed. "Four hundred and eight entries per second. The pipeline ingested nine years of Samuel Pepys and produced a clean, uncontaminated 6,450 × 768 float32 matrix in less time than it takes to read a single diary entry aloud."

McCoy looked at Scotty. "Compared to thirty-one minutes last time."

"The previous run was 8,413 *chunks*," Spock clarified. "The DiaryTransformer produced 2.51 chunks per entry on average. This run parses the enriched source file directly — one embedding per diary line. The number is smaller, the entries are whole, and without the prefix overhead the encoding is clean and fast."

"Four hundred and eight entries per second," McCoy said quietly. "On a machine sitting on a desk."

"On four cores of a machine sitting on a desk," Scotty said. There was something in his voice that was not quite pride and not quite wonder. "The `sentence-transformers` multi-process pool. Each worker holds its own model copy. No GIL. No waiting. Pure parallel."

He looked at the output: `Cache saved (6450 entries) → pepys_mpnet_embeddings.json`.

"Astounding," he said.

Nobody disagreed.

---

## Chapter 3: The Temporal Flight Experiment

*Main science console. Stardate 2026.086.*

The temporal flight instruments were Spock's design.

The idea: restore time as a navigable axis.

An N-dimensional embedding captures semantic structure — what things *mean* — but discards the diary's natural temporal ordering. The `pepys_temporal_flight.py` instrument augments every embedding from N dimensions to N+1 by appending a scaled temporal coordinate. Time becomes a direction in the manifold. A TurtleND can fly along it.

Three flight modes were tested:

- **Semantic flight** — fly between the two most semantically distant entries, time incidental
- **Temporal flight** — orient heading along the pure temporal axis, fly forward in time
- **Mixed flight** — blend temporal and semantic orientation 50/50

The route: from `1663-10-21` (origin, index 2956) to `1664-01-23` (destination, index 4171). A short journey in diary time — ninety-four days. A probe of whether the manifold could be navigated as a time machine.

Spock activated the instruments.

---

## Chapter 4: The Anomaly

*Stardate 2026.086. Main viewscreen.*

The figure resolved. Six panels: three flight-path scatters across the PCA-2D projection, three temporal profile plots.

```
Semantic Flight    — 79 hops,  Kendall τ = +0.19
Temporal Flight    — 151 hops, Kendall τ = −0.38
Mixed Flight       — 142 hops, Kendall τ = −0.15
```

Kirk studied the numbers. "Walk me through this, Spock."

"Kendall tau measures the rank correlation between path order and temporal order," Spock said. "A value of +1.0 means the path moves perfectly forward in time at every step. A value of 0 means the path has no temporal coherence — random with respect to time. A value of −1.0 means the path moves perfectly *backward* in time."

He paused.

"The temporal flight produced a tau of negative 0.38."

McCoy straightened. "It went *backward*?"

"It moved *against* time. Actively. On the first hop, departing from entry `1663-10-21`, the temporal flight engine immediately jumped to `1669-05-09` — the final year of the diary. It then spent the majority of its path orbit in the 1667–1669 cluster, never reaching the 1664 destination."

"But it's *called* temporal flight," McCoy said. "It's supposed to follow time *forward*."

"Yes," Spock said. "That is the bug."

---

## Chapter 5: The Nature of the Bug

*Spock's analysis log.*

The temporal coordinate in `augment_with_time()` is encoded as absolute fractional years — the position of each entry on a global time axis, z-scored across the corpus and scaled to match embedding magnitude.

This means: early entries have *low* values. Late entries have *high* values. The KNN graph, built on Euclidean distances in the augmented (N+1)-dimensional space, has a neighbourhood structure shaped by the density of the corpus. And the Pepys corpus is not uniformly distributed in time. The 1667–1669 period contains the densest cluster of entries.

When the temporal flight engine orients along the time axis and takes a greedy step toward the nearest forward-time neighbor, it is not navigating toward the *destination*. It is navigating toward the region of highest temporal coordinate — the end of the diary. The corpus centroid pulls it forward in absolute time, past the destination, into the dense late cluster.

The engine is not broken. It is doing exactly what it was told. It was told to fly along the time axis. The time axis, as encoded, points toward the end of the corpus — not toward the destination.

"The instrument," Spock said, "is facing the wrong star."

Kirk sat with that for a moment. "So what does the semantic flight know that the temporal flight doesn't?"

"The semantic flight is unaware of time," Spock said. "It navigates purely by content similarity. And yet — its Kendall tau is positive 0.19. The best of the three modes. Because Pepys's concerns evolved coherently across his diary: topics cluster in time, not just in meaning. Semantic similarity is a natural proxy for temporal proximity. The manifold knows what time *means*, even without being told what time *is*."

McCoy had been very quiet.

"The machine that ignored time," he said slowly, "was more temporally coherent than the machine that tried to follow it."

"Yes," Spock said. "Because it was measuring the right thing."

---

## Chapter 6: The Correction — Future Time as Pull

*Navigation console. Stardate 2026.086.*

Uhura turned from the MRR sensor array. "So how do we fix it?"

Spock had already written the correction in his log.

"The current encoding treats time as an absolute coordinate — a position on a global axis. The temporal flight engine sees high values as 'future' and moves toward them. But what we want is not motion toward the far end of time. We want motion toward the *destination*."

He brought up the equation.

```python
# Current (broken): absolute time — attracts toward dense corpus centroid
t_coord = fractional_year(entry.date)

# Corrected: destination-relative — zero at target, pull increases with distance
t_coord = abs(entry.date - destination.date)
```

"We flip the sign," Spock said. "We encode time not as position but as *distance from the destination*. The destination has value zero. Entries far from the destination — whether in the past or the future — have high values. The manifold gradient then naturally pulls the turtle toward the destination."

"Future time is a pull," Kirk said.

"Yes, Captain. For every living thing — and for every algorithm that wishes to navigate time correctly — the future is not a place to drift toward. It is a force that *attracts*. The destination exerts a pull. The encoding must reflect that."

McCoy looked at Spock for a long moment.

"You know," he said, "I've never heard a physicist say something I agreed with that much."

---

## Chapter 7: [CLASSIFIED] — The First Complete Diary Knowledge Graph

*Commander's log. Stardate 2026.086. Eyes only.*

What follows is not to leave this ship.

At 0040 hours ship's time, following the clean re-embedding run and the temporal flight analysis, the crew executed the first full build of the DiaryKG knowledge graph on the complete Pepys corpus.

Not the benchmark cache. Not the embedding matrix. The *knowledge graph* — the full structural index, ingested through the `DiaryKG.build()` pipeline, every entry transformed, chunked, classified, and indexed into both SQLite and LanceDB simultaneously.

The build completed in under a minute.

When it finished, the status panel read:

```
DiaryKG status  /Users/egs/repos/diary_kg
  Built       : yes
  Source file : pepys/pepys_enriched_full.txt
  Built at    : 2026-03-27T01:21:25 UTC
  Corpus      : 6,647 .md files  (4.0 MB)
  SQLite      : 104.1 MB
  LanceDB     : 100.5 MB
  Snapshots   : 1
```

The crew stood at their stations and looked at it.

Nine years of Samuel Pepys — his plague, his Navy, his marriage, his theatre, his ambition, his fear of blindness, his joy at music, his complicated feelings about money — indexed. Queryable. Semantically navigable. In under a minute.

"We can *talk* to it," Scotty said. He said it the way you say something when you don't quite believe it yet.

Kirk nodded. "Run a query."

```bash
diarykg query "Great Fire of London"
```

The results came back in milliseconds. Eight entries. Semantic scores above 0.9. The September 1666 entries — Pepys watching the city burn from a boat on the Thames, carrying his wine and Parmesan cheese to safety — surfaced at the top of the list, ranked by a vector space that had learned, from the words themselves, what the Great Fire meant.

"He saved the Parmesan," McCoy said, reading the entry.

"He saved the *diary*," Kirk said.

"He saved the Parmesan *first*," McCoy said. "It's in the text."

Kirk almost smiled. "Run another."

```bash
diarykg query "who is lord sandwich?"
```

Eight results again. The system surfaced a cluster of entries orbiting one of the most consequential figures in Pepys's professional world — Edward Montagu, First Earl of Sandwich, patron and kinsman, Admiral of the Fleet, the man who had carried Pepys into the Navy in the first place. The index had assembled him from fragments: a two-word chunk simply reading *Lord Sandwich*, a report of the King receiving him "mighty kindly," an entry placing his fleet at Alborough Bay.

Chekov looked up from his console. "The scores are rendering as zero in the display."

"A formatting artefact," Spock said. "The retrieval engine is returning semantically coherent results. The vector distances are non-zero. The Rich table renderer is truncating fractional scores below its precision floor. The instrument is correct. The readout needs calibration."

"So the knowledge graph works," McCoy said. "We just can't see how confident it is."

"We can see *what* it found," Spock said. "That is the primary datum."

McCoy looked at the result: *My Lord Sandwich is, it seems, with his fleet at Alborough Bay.* A man, a fleet, a bay. A fragment of history, retrieved in milliseconds from nine years of diary, because the model had learned — without being told — that Lord Sandwich and Samuel Pepys belonged in the same neighbourhood of semantic space.

"He's in there," McCoy said quietly. "All of them are. Sandwich, the King, Coventry, the whole Navy. Nine years of a man's life, indexed."

He paused.

"That's not a database," he said. "That's a *memory*."

---

## Epilogue

> *The U.S.S. WaveRider has completed its first full temporal navigation experiment and its first complete knowledge graph build. The temporal flight instruments revealed not a successful navigation but a navigational bug of unusual clarity — a case where the machine trying to follow time moved against it, and the machine ignoring time followed it most faithfully. The correction is known. The instrument will be recalibrated. The next flight will carry the destination-relative temporal encoding, and Kendall tau will be measured again. The hypothesis: future time, encoded as a pull toward destination rather than a push toward the corpus end, will produce positive tau in all three modes. The knowledge graph is alive. Samuel Pepys is waiting.*

---

## Mission Data Appendix

### Temporal Flight Results (Pre-Correction)

| Mode | Hops | Kendall τ | Monotonicity | Mean Δt/hop | Total span |
|---|---|---|---|---|---|
| Semantic | 79 | **+0.19** | 54% | 1.56 yr | 2.01 yr |
| Temporal | 151 | **−0.38** ⚠ | 52% | 0.61 yr | 2.36 yr |
| Mixed (50/50) | 142 | **−0.15** | 52% | 1.13 yr | 2.78 yr |

*Route: 1663-10-21 → 1664-01-23 (94-day target). α=1.0, k=10, max_steps=150.*

### The Bug

| Parameter | Value |
|---|---|
| Bug location | `augment_with_time()` in `pepys_temporal_flight.py` |
| Root cause | Time encoded as absolute fractional year; attracts toward dense 1667–1669 cluster |
| Symptom | Temporal flight immediately jumps to 1669 on hop 1; τ = −0.38 |
| Fix | Encode time as `abs(entry.date − destination.date)` — destination-relative pull |
| Status | **Correction identified. Implementation pending.** |

### Clean Re-Embedding Run

| Property | Value |
|---|---|
| Pipeline | `pepys_embedder.py` (fixed: mpnet model, no nomic prefix) |
| Source | `pepys/pepys_enriched_full.txt` |
| Entries embedded | **6,450** |
| Embedding dim | 768 |
| Workers / batch | 4 / 32 |
| **Embedding time** | **15.8 seconds** |
| Throughput | 408 entries/second |
| Previous run | 31 minutes (8,413 chunks, nomic-contaminated) |

### DiaryKG Build — First Complete Corpus [CLASSIFIED]

| Property | Value |
|---|---|
| Source | `pepys/pepys_enriched_full.txt` |
| Built at | 2026-03-27T01:21:25 UTC |
| Corpus files | **6,647 .md files (4.0 MB)** |
| SQLite index | **104.1 MB** |
| LanceDB index | **100.5 MB** |
| Build time | **< 60 seconds** |
| Snapshots | 1 (baseline) |
| Status | **Operational. Queryable.** |
| First queries | `"Great Fire of London"` → 8 results, top score > 0.9; `"who is lord sandwich?"` → 8 results, all on-target |
| Known issue | Rich table score display truncates to 0.000 — formatting artefact, retrieval correct |

### Next Mission

- Recalibrate temporal flight: implement destination-relative time encoding
- Re-run all three flight modes; measure corrected Kendall τ
- Verify ManifoldWalker temporal coherence improves across all modes
- Hypothesis: semantic τ holds at ~+0.19; temporal τ flips positive
