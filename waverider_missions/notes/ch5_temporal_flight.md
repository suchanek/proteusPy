# Mission Log — Chapter 5
## U.S.S. WaveRider, NCC-7699
### Stardate 2026.086 — Destination-Relative Temporal Encoding: First Confirmed Flight

---

> *"Captain, I have found the attractor."*
> — Science Officer Spock, Stardate 2026.086

---

## Situation Report

The WaveRider has spent the last three stardates navigating Pepys-Space — the
7,282-entry manifold of Samuel Pepys' diary (1660–1669), embedded by
`all-mpnet-base-v2` in 768 dimensions.

Previous missions mapped the manifold's intrinsic structure: TwoNN
dimensionality ≈ 14.26, participation ratio ≈ 22.81, MRR@5 at full 768D =
0.9615.  The manifold is real.  The topology is stable.

This chapter asks a harder question:

> *Can we navigate it in time?*

Not merely observe entries in sequence — but **fly** from a chosen departure
point to a chosen destination, guided by a gravitational field encoded directly
into the geometry.

---

## The Instrument: Destination-Relative Encoding

**Mr. Scott** (standing over the embedding console, hands on hips):
*"It's deceptively simple, Captain.  We take the 768-dimensional embedding
vector for every entry in the corpus and append one more number — the entry's
fractional distance in time from the destination.  The destination itself gets
zero.  Every other entry gets a positive value proportional to how far it is
from the target, whether that's a week ago or nine years hence."*

**Spock** (at the navigation console, precise):
*"Specifically, for destination at fractional year* `d` *and entry at
fractional year* `t`:*

```
temporal_coord = abs(t - d)
```

*The destination has temporal coordinate zero.  The manifold now has a
gravitational basin.  Every direction vector from any node toward the
destination has a negative temporal component — a pull toward zero.  The KNN
graph encodes this pull into its topology: entries near the destination in time
become neighbors even when they are semantically distant.*

*The critical parameter is* `α` *(alpha), which scales the temporal axis
relative to the 768 semantic axes.  At* `α = 1`*, the temporal axis contributes
as much as one semantic axis — one part in 768, negligible.  At*
`α ≈ √768 ≈ 27.7`*, the temporal axis matches the full aggregate semantic
contribution.  We found* `α = 150` *sufficient for coherent temporal flight."*

**McCoy** (arms folded):
*"So you're telling me you bolted a clock onto the side of the navigation
computer and called it gravity."*

**Spock**:
*"An imprecise but not inaccurate characterisation, Doctor."*

---

## The Experiment: Two Flights

### Flight 1 — The Short Hop (94 Days)

**Parameters:**

| Instrument | Setting |
|---|---|
| Origin | `[2884]` 1663-10-21 — *"To begin to keep myself as warm as I can."* |
| Destination | `[3109]` 1664-01-23 — *"Up, and with Sir W. Batten and Sir W. Penn to Whitehall"* |
| Temporal span | 94 days |
| Alpha | 150.0 |
| k-graph | 15 neighbours |
| Corpus | 7,282 entries × 768 dims |

**Navigational Readings (Chekov):**

| Metric | Value |
|---|---|
| Path length | 10 hops |
| Reached destination | **Yes** |
| First hop date | 1663-12-04 (44 days forward) |
| Final Kendall τ | **0.5556** |
| Monotonicity | 55.6% |

**Hop log:**

```
 0  1663-10-21  —      pepys weather | To begin to keep myself as warm as I can.
 1  1663-12-04  —      pepys domestic | Up pretty betimes, that is about 7 o'clock...
 2  1663-12-01  0.333  pepys social | Up and to the office...
 3  1663-12-16  0.667  pepys social | Up, and with my head and heart full of my business...
 4  1663-11-25  0.200  pepys court | He set me down in Fleet Street...
 5  1664-02-01  0.467  pepys court | Lay long in bed...
 6  1664-01-13  0.524  pepys court | Up betimes and walked to my Lord Bellasses's lodgings...
 7  1664-01-15  0.571  pepys court | Up, and after a little at my office...
 8  1664-01-12  0.500  pepys court | Up, and to Whitehall about getting a privy seal...
 9  1664-01-23  0.556  pepys court | Up, and with Sir W. Batten and Sir W. Penn to Whitehall
```

**Spock's analysis:**  The turtle departed October and landed in December on the
first hop — a 44-day forward jump guided by the gravitational basin.  The path
then circled through late 1663/early 1664 court and domestic entries,
converging on the destination through semantic similarity.  Final τ = 0.5556
represents genuine temporal ordering — the path is more forward than backward.

### Flight 2 — The Great Crossing (4.5 Years)

The decisive test.  Origin: the very first line of Pepys' diary, January 1st
1660.  Destination: a naval office entry from June 1664, 1,627 days and four
and a half years away.

**Parameters:**

| Instrument | Setting |
|---|---|
| Origin | `[0]` 1660-01-01 — *"Called up this morning by Mr. Moore..."* |
| Destination | `[3419]` 1664-06-15 — *"Up and by appointment with Captain Witham..."* |
| Temporal span | 1,627 days (4.46 years) |
| Alpha | 150.0 |
| k-graph | 15 neighbours |

**Navigational Readings:**

| Metric | Value |
|---|---|
| Path length | 114 hops |
| Reached destination | **Yes** |
| First hop date | 1660-02-27 (57 days forward) |
| Final Kendall τ | **0.4490** |
| Monotonicity | **63.7%** |
| Hops 0–24 τ | **1.0000** (perfect) |

**The opening run:**

The first 25 hops were extraordinary.  From January 1660, the turtle stepped
forward in near-perfect chronological order, hop by hop through the years:

```
 0  1660-01-01  —      pepys domestic | Called up this morning by Mr. Moore...
 1  1660-02-27  —      pepys domestic | At the office all the morning...
 2  1660-03-27  1.000  pepys domestic | Up early to see my workmen at work...
 3  1660-04-11  1.000  pepys domestic | A Gentleman came this morning from my Lord...
 4  1660-05-18  1.000  pepys social | After we had seen all, we light by chance...
 5  1660-07-26  1.000  pepys domestic | In the evening I met with T. Doling...
 6  1660-09-21  1.000  pepys domestic | I landed at the old Swan...
 7  1660-11-13  1.000  pepys domestic | Early going to my Lord's...
 8  1661-01-06  1.000  pepys domestic | This morning I sent my lute to the Paynter's...
 9  1661-03-06  1.000  pepys domestic | Up early, my mind full of business...
10  1661-06-04  1.000  pepys domestic | The Comptroller came this morning...
11  1661-09-14  1.000  pepys domestic | At the office all the morning...
12  1661-12-13  1.000  pepys domestic | At home all the morning, being by the cold weather...
13  1662-02-19  1.000  pepys domestic | Up and to my office...
14  1662-04-15  1.000  pepys domestic | At the office all the morning...
15  1662-07-09  1.000  pepys domestic | Dined at home, and so to the office again...
16  1662-09-14  1.000  pepys domestic | Sir George told me of a chest of drawers...
17  1662-11-20  1.000  pepys domestic | All the morning sitting at the office...
18  1663-01-28  1.000  pepys domestic | Up and to the office...
19  1663-03-31  1.000  pepys domestic | Up betimes, and to my office...
20  1663-05-20  1.000  pepys domestic | Up and to my office, and anon home...
21  1663-06-02  1.000  pepys domestic | So home, and seeing my wife had dined...
22  1663-08-11  1.000  pepys domestic | This morning, about two or three o'clock...
23  1663-10-26  1.000  pepys domestic | Thence Creed and I to one or two periwig shops...
24  1663-12-16  1.000  pepys social | Up, and with my head and heart full...
```

**Kendall τ = 1.0000 for 24 consecutive hops.**  The turtle traversed 3.5 years of
Pepys' life — 1660 to late 1663 — in perfect chronological order, moving through
the `pepys_domestic/Office` manifold strand that threads the early diary years.

After hop 24, the turtle entered the 1664 cluster around the destination and
began a tighter search — oscillating through nearby spring-summer 1664 entries,
converging to the June 15 naval appointment in 90 more hops.

---

## The Discovery

**Kirk** (leaning forward):
*"What exactly does this tell us, Spock?"*

**Spock**:
*"Two things, Captain.  First: the destination-relative encoding works.  The
temporal basin created by* `abs(t − d)` *is a genuine gravitational attractor in
the augmented space.  At sufficient* `α`*, the KNN graph topology enforces
temporal pull.*

*Second, and more significant: the manifold has a* **temporal spine**.  The
`pepys_domestic/Office` strand — entries recording Pepys' daily Navy Board
routine — forms a path through the manifold that is nearly monotone in time.
The turtle found it and followed it for 24 hops before the semantic topology
diversified.*

*This strand is not an artefact.  It reflects a genuine regularity in Pepys'
life: he woke, went to the office, dined at home, and wrote about it.  That
regularity created a thread in embedding space that our navigation system could
follow — forward in time, without backtracking, for 3.5 years."*

**McCoy**:
*"You're saying the man's routine was so predictable it left a groove in
the geometry."*

**Spock**:
*"Precisely, Doctor.  A geodesic of habit."*

---

## Technical Corrections Made During This Mission

The initial implementation used `abs()` encoding with `α = 1.0`, which was
insufficient — the temporal axis contributed only 1/768 of the total signal,
invisible against the semantic space.

A signed encoding `(t − d)` was tested to prevent overshooting.  It worked for
short hops but provided weaker gravitational pull than the symmetric `abs()`
basin for longer flights.

The decisive insight: **calibrating `α` to `√D ≈ 27.7` makes temporal equal
to the full semantic contribution**.  At `α = 150` (approximately `5√D`), the
temporal basin dominates local KNN topology while semantic content still guides
path selection.

**Calibration sweep (short hop, 1663-10-21 → 1664-01-23):**

| α | τ | Monotonicity | First hop |
|---|---|---|---|
| 1 | 0.143 | 42.9% | 1666-12-02 (overshoot) |
| 10 | −0.085 | 52.9% | 1664-01-02 |
| 28 | 0.361 | 58.6% | 1664-01-02 |
| 50 | 0.407 | 53.8% | 1664-01-02 |
| 150 | **0.556** | **55.6%** | 1663-12-04 |

---

## Mission Data Appendix

| Instrument | Reading |
|---|---|
| TurtleND navigation console | `ndim=769` (768D semantic + 1D temporal) |
| Corpus | `pepys_mpnet_embeddings.json` — 7,282 entries × 768D |
| Temporal encoding | `abs(fyear_i − fyear_dest)`, scaled by `α / √D` |
| Optimal α | 150 (empirically calibrated; ≈ 5√768) |
| KNN graph | k=15 neighbours, Euclidean metric in 769D |
| Short hop τ | 0.5556 (10 hops, 94-day span) |
| Long crossing τ | 0.4490 (114 hops, 1627-day span) |
| Opening run | τ=1.000 for first 24 hops (perfect monotone, 3.5 years) |
| Monotonicity | 63.7% (long crossing) |
| `pepys_domestic/Office` | Identified as the temporal spine of the 1660–1663 manifold |

---

## Status

Mission Chapter 5 complete.  Destination-relative temporal encoding validated
on the full Pepys mpnet corpus.

Next: Chapter 6 — multi-destination navigation.  Can the turtle fly a
waypoint route?

---

*Log sealed — Science Officer Spock, U.S.S. WaveRider NCC-7699*
*Stardate 2026.086*
