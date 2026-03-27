# STAR TREK: THE MANIFOLD FRONTIER
## *Interlude: "The Package"*

---

> *Stardate 2026.091. The same morning as the Rabbit Maneuver. Before the briefing. Before the corpus was woken up. A sealed datacore arrived in the overnight queue, and someone had to be the first to open it.*

---

## 0512 Hours — Engineering Subdeck, Deck 7

Montgomery Scott had been awake since 0430.

This was not unusual. The WaveRider's pipeline infrastructure did not sleep, and neither, in any meaningful sense, did her Chief Engineer — not when there was a major corpus operation scheduled for 0730 and the PEPYS embedder cache was behaving in ways that required, in his professional estimation, a firm hand and some very careful attention before the officers arrived and started asking questions.

He had the secondary console to himself. The overnight queue was open on the left panel, showing the routine: calibration telemetry, a batch of corrected coordinate indices from Chekov's navigator log, three system notices that could wait until Monday. He scrolled through them with the practiced indifference of a man who has read overnight queues for eleven years and knows what a real item looks like.

He stopped.

The last item in the queue was not routine.

It was a sealed datacore transfer, timestamped 0147, flagged priority GAMMA-3. The routing tag read: `EYES: SCOTT, M. — CHIEF ENGINEER, U.S.S. WAVERIDER, NCC-7699`. No ship-of-origin. No standard header. The sender field contained only: `SUCHANEK, A. — STARFLEET KNOWLEDGE DIVISION`.

Scotty looked at it for a moment.

Then he poured himself a fresh mug of coffee, pulled the stool closer, and opened it.

---

## The Contents

The datacore was small — 5.7 megabytes of raw text, pre-formatted, structured. No cover document. No mission briefing. No formal orders and no chain of command reference.

Just a corpus.

He scanned the file manifest. `pepys_complete_enriched_v1/`. Beneath it: a structured archive — raw diary text, pre-parsed, pre-enriched, already formatted for pipeline ingestion. Entry headers intact. Date metadata clean. Category annotations applied. The work of the DiaryTransformer, done already, before anyone on this ship had been asked to do it.

At the bottom of the manifest: a single plain text file. `note.txt`. Forty-three bytes.

He opened it.

> *Build this. Navigate it. See what you find.*
>
> *— AS*

Scotty read it twice. Then he set down his coffee.

He had served under four captains and worked alongside admirals of every disposition Starfleet had to offer — the methodical ones, the ambitious ones, the ones who sent you twelve-page briefing documents and the ones who sent you twelve words. He had learned, over the years, to read the weight of brevity. Short orders from senior officers were never simple. They were compressed.

*Build this. Navigate it. See what you find.*

There was something in those nine words — not authority, exactly, but certainty. As if the Admiral already knew what was in the corpus and already knew, roughly, what they would find, and was not telling them because the finding was the point. The discovery had to be theirs.

Scotty sat back.

He thought about the PEPYS cache warm-up scheduled for 0730. He thought about the enriched archive in front of him — 1.3 million words, already cleaned, already structured, the kind of pre-processing that took weeks of iteration to get right, handed to them in the middle of the night with no explanation.

Then he opened a new terminal and began to stage the pipeline.

---

## 0541 Hours

The DiaryKG build command was simple. It always was, in his experience — the elegant solutions were always deceptively simple at the surface, with all the complexity buried in what had been solved before you got there. Four workers. Batch size thirty-two. The LanceDB index and the SQLite graph store, initializing in parallel on the secondary compute cluster.

He set it running on his subdeck console, quietly, so the output logs went to his personal buffer and not the main engineering display. This was not, technically, required by the GAMMA-3 flag or by any regulation he was currently recalling. But something about the hour and the tone of the note felt like it called for a certain discretion.

*See what you find.* Not *report what you find.* Not *submit findings to Starfleet Knowledge Division at your earliest convenience.* The Admiral had not asked for a report. He had asked for a discovery.

Scotty let the build run and went back to the PEPYS embedder calibration, which still needed his attention before 0730.

---

## 0558 Hours — Captain's Briefing Room, Deck 3

Kirk arrived early.

He found McCoy already there, standing at the viewport with his coffee, watching the stars with the expression he wore when something was bothering him and he hadn't decided yet whether to say it.

"You're up early, Bones."

"Couldn't sleep." McCoy turned. "Something's in the queue this morning."

Kirk looked at him. "What kind of something?"

"The kind that comes from Starfleet at 0147 with a three-word cover note." McCoy picked up his mug. "Scotty's already running it. He came to tell me an hour ago — I think he wanted someone else to know."

"What is it?"

McCoy was quiet for a moment. He had the look he got when the medicine ran out and the intuition took over — when there was no instrument that covered what he was sensing.

"It's a corpus," he said. "One man's diary. Nine years. Apparently Suchanek put it together himself. Or had it put together. And sent it with no explanation."

Kirk looked at the stars.

"He wants us to find something," McCoy said.

"What?"

McCoy looked at him. Something moved behind his eyes — not uncertainty, but a kind of careful recognition, like a doctor who has just identified the correct diagnosis and isn't sure yet whether to say it aloud.

"That's the question, isn't it," he said.

Kirk turned from the viewport. On the main display, the PEPYS subsystem was warming up. The enriched corpus was loading. Two decks below, Scotty's secondary console was running a build that none of them had been officially asked to run, because an Admiral had sent nine words in the middle of the night and trusted that the right people would understand what they meant.

Kirk looked at the display for a long moment.

"Get Spock," he said.

---

## Mission Data Appendix — Interlude

| Parameter | Value |
|---|---|
| Stardate | 2026.091 (same morning — Chapter 2 prequel) |
| Item | Sealed datacore, priority GAMMA-3 |
| Origin | Admiral Suchanek, Starfleet Knowledge Division |
| Recipient | Chief Engineer Scott, U.S.S. WaveRider |
| Contents | `pepys_complete_enriched_v1/` — complete Samuel Pepys diary corpus, pre-enriched, pipeline-ready |
| Note | *"Build this. Navigate it. See what you find. — AS"* |
| Corpus | 1.3 million words, 1660–1669, 9 years continuous |
| Pre-processing | DiaryTransformer enrichment complete; category annotations applied |
| Build initiated by | Scott, M. — Stardate 2026.091, 0541 |
| Official orders | None |

---

*End Interlude.*

*The DiaryKG build runs quietly on Scotty's secondary console while the rest of Chapter 2 proceeds above — the Rabbit Maneuver, the first embedding run, the first navigation of the Pepys manifold. By the time Chapter 4 opens on Stardate 2026.097, the build has completed. The corpus is navigable. And the crew is still wondering what they were meant to find.*

---

*"He wants us to find something."*
*"What?"*
*"That's the question, isn't it."*
*— Dr. Leonard McCoy, Stardate 2026.091*
