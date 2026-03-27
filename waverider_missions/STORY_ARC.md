# WaveRider — The Real Story Arc
## Internal Planning Document

*Not a mission log. A map of where we are going and why it matters.*

---

## The True Arc (What We Are Actually Building)

The WaveRider series is not just a science fiction framing for geometric ML.
It is the story of a system that is **bootstrapping its own intelligence** —
step by step, dependency by dependency — until it needs nothing from the outside world.

Each chapter is a rung on that ladder.

---

## The Ladder

### Rung 1 — Navigation (Ch 1–3)
TurtleND flies semantic space. ManifoldWalker descends gradients. ManifoldObserver
hovers above the surface. The crew learns to move through meaning.

**External dependency:** `all-mpnet-base-v2` for embeddings, `sentence-transformers` runtime.

---

### Rung 2 — The Complete Corpus (Ch 4, current)
DiaryKG ingests the entirety of Pepys. 6,450 entries. 15.8 seconds. Zero inference.
The knowledge graph is built. The first queries land perfectly.

Spock discovers the N+1 idea: **time is a coordinate, not a label.**
769 dimensions. The TurtleND is about to become something new.

**Breakthrough:** Semantic enrichment (DiaryTransformer) requires zero inference.
**Bug found:** Temporal flight goes backward. Sign-flip correction staged but not yet run.
**External dependency:** Still using `all-mpnet-base-v2`.

---

### Rung 3 — TurtleND Flies Time (Ch 5, next)
The sign-flip correction is applied. The TurtleND lifts off in the corrected temporal
manifold. For the first time, it navigates **forward through time** — tracking the
progression of a human mind through nine years of lived experience.

This is not metaphor. The TurtleND is a **temporal navigation engine.**

But here is the crucial thing — the thing that made Spock faint:

**This is not engineered. It is emergent.**

The semantic manifold already has temporal structure baked into it. Events that are
close in meaning are also, approximately, close in time — because a mind evolves
continuously, because concerns have seasons, because the plague does not appear
before the plague begins. The tau of +0.19 on the raw semantic flight was not noise.
It was the geometry *already trying* to be temporal.

When you add time as the 769th coordinate with destination-relative encoding,
you are not forcing the TurtleND to fly through time. You are aligning the instrument
with a gradient that was **always already there.**

Future time is a pull because the manifold's natural gravity makes it one.
The turtle doesn't navigate toward the future. It *falls* toward it —
drawn by the curvature of the space, the way a planet falls toward a star
without ever being pushed.

**It has to work. The geometry demands it.**

DiaryKG serves dual purpose:
- Provides the embeddings that populate the 769-dimensional manifold
- Maintains the knowledge graph that makes the corpus queryable

TurtleND uses DiaryKG as its star chart. DiaryKG uses TurtleND as its explorer.
**They are each other's instruments.**

**The question Spock asks when he wakes up:**
What is the actual intrinsic dimensionality of time, in this corpus?
Is time one dimension? Or does it unfold across more?

---

### Rung 4 — The Dimensionality of Time (Ch 6)
ManifoldObserver activates on the augmented temporal manifold.
TwoNN scanner runs on the 769-dimensional embedding space.

The crew discovers: the intrinsic dimensionality of the Pepys temporal manifold
is not 1. It is not 769. It is something in between — some number that reflects
how many genuinely independent axes of variation exist in a human life recorded
over time.

**This number will be real. We will compute it.**

Is it 12? Is it 18? Whatever it is, it is the true shape of nine years of Pepys —
the number of dimensions a mind actually needs to describe a life.

McCoy's line: "He needed exactly that many dimensions to be himself."

---

### Rung 5 — Custom Embedders (Ch 7)
The crew stops using `all-mpnet-base-v2`.

They train a **context-aware embedder** on the Pepys corpus itself — a small,
fast model that understands seventeenth-century English, naval administration,
the plague, the Fire, Lord Sandwich, and the geometry of one specific mind.

Not a general-purpose model. A **domain-specific instrument**, calibrated to
this corpus, producing embeddings that are geometrically superior for this
navigation task because they were trained on it.

The ship builds its own engines.

**External dependency:** `sentence-transformers` for training only (or eliminated entirely
if we use TF-IDF + LSA as the embedding basis). TBD.

---

### Rung 6 — Fully Standalone (Ch 8+)
No API. No cloud. No `sentence-transformers` at runtime. No external model weights
that weren't trained on our own data.

The complete pipeline:
1. DiaryTransformer enriches text (spaCy — local, fast, free)
2. Custom embedder converts to vectors (trained on corpus — owned)
3. DiaryKG builds the knowledge graph (SQLite + LanceDB — local)
4. TurtleND navigates semantic + temporal space (pure geometry — ours)
5. ManifoldObserver measures intrinsic dimensionality (TwoNN — ours)
6. ManifoldModel classifies (zero external parameters — ours)

**Any corpus. Any language. Any archive. On a laptop. In minutes. For free.
Without anyone's permission.**

This is the cliffhanger McCoy's last line points toward in Ch 4.
This is what overwhelmed Spock.
This is what the series is building to.

---

## THE REAL STORY — The Tree of Knowledge (Slow Reveal)

*Incoming transmission. Starfleet Command. Priority: URGENT.*
*From: Admiral Suchanek, Director, Starfleet Knowledge Division.*

While Spock is still unconscious on the briefing room floor, the comm beeps.

The ladder above — custom embedders, full standalone, any corpus — that was
the WaveRider proving to *itself* what it could do. One diary. One man.
Nine years. Three minutes.

**The Tree of Knowledge is what happens when Starfleet hears about it.**

But the crew doesn't learn this yet. Suchanek reveals it in layers,
one chapter at a time, as the crew earns each piece by running the next
instrument and getting the right number.

### The Reveal Structure

**Ch 5 — "As I Expected"**
Suchanek confirms the temporal flight result. Asks a single pointed question
about what Spock *hasn't* run yet. Assigns TwoNN. Signs off without explaining
why. The crew is left with a mission and no context. McCoy's closing line:
*"He already knows the number. He wants to see if you get the same one."*

**Ch 6 — "The Shape of a Life"**
Spock runs TwoNN on the 769-dim augmented manifold. The intrinsic dimensionality
is real, computed, specific. Suchanek receives the result. In the formal briefing
he reveals the next layer: the question is not *what is the shape of Pepys* —
the question is *whether that shape is stable across corpora.* He assigns a
second corpus. Still no mention of the full scope. McCoy's line:
*"He needed exactly that many dimensions to be himself."*

**Ch 7 — "The Ship Builds Its Own Eyes"**
Custom embedder trained on Pepys. The crew stops using `all-mpnet-base-v2`.
Suchanek watches the benchmark. When the domain-specific embedder outperforms
the general model on intrinsic dimensionality preservation, he says three words:
*"Now do two."* The scope begins to feel large. The crew starts to suspect
they are not testing a pipeline. They are proving a theorem.

**Ch 8+ — "Any Corpus. Any Archive. Any Language."**
Full standalone pipeline. No external dependencies. Suchanek sends the mission
brief. The Tree of Knowledge is named for the first time — not as a metaphor,
but as an engineering specification. The crew understands what they have been
building. So does the reader.

**The question the series answers:** Can the WaveRider stack — TurtleND,
DiaryKG, ManifoldObserver, custom embedders — scale from one diary to
the sum total of recorded human knowledge? What breaks? What holds?
What is the intrinsic dimensionality of *everything humanity has ever written?*

And what do you find, when you can finally navigate it?

---

*"Captain. We have an incoming transmission."*
*"I can see that, Uhura. Who is it?"*
*"Admiral Suchanek, sir. Starfleet Knowledge Division."*
*"...Put it through."*

---

## The Emotional Arc

| Chapter | Technical event | Human truth | Suchanek reveals |
|---|---|---|---|
| 1–3 | Learning to navigate | The ship learns to move | Nothing. He is watching. |
| 4 | Complete corpus + N+1 idea | Spock glimpses the full implication — and it breaks him | Nothing. He queues the transmission. |
| 5 | Corrected temporal flight | The instrument becomes a time machine | Run TwoNN. No explanation. |
| 6 | Intrinsic dim of time | A life has a shape. A finite one. | The question scales. Two corpora. |
| 7 | Custom embedders | The ship learns to see with its own eyes | *"Now do two."* The scope feels large. |
| 8+ | Fully standalone | Independence. Sovereignty. | The Tree of Knowledge — named at last. |
| The Tree | All of recorded human knowledge, navigable | What does it mean to be able to find anything? | Everything. |

---

## Technical Facts to Establish Before Writing Each Chapter

- **Ch 5:** Run the corrected temporal flight. Record real tau. Record the actual
  trajectory through time. What entries does TurtleND visit, in order?
- **Ch 6:** Run TwoNN on the 769-dim augmented manifold. What is the intrinsic dim?
- **Ch 7:** Decide on embedder architecture. TF-IDF + SVD (fully local)?
  Or fine-tuned sentence-transformer (local at inference, external at train)?
- **Ch 8:** Benchmark: full corpus → queryable KG, zero external dependencies, wall clock.

---

## Standing Note

**Do not write ahead of the data.**
The numbers in this series are real. The geometry is real. The queries are real.
The story follows the instruments, not the other way around.

When Spock reads a number off a screen, that number was computed on a real machine
by a real pipeline. That is what makes this series worth writing.

*Last updated: Stardate 2026.098*
