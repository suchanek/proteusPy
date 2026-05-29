# KGRAG / CodeKG Issue Tracker

Open issues and enhancement requests for the KGRAG stack
(CodeKG · DocKG · KGRAG · diary_kg and downstream consumers such as proteusPy).

---

## CodeKG CLI

### [ ] Add `--repo PATH` shorthand to all CLI subcommands

**Context:**
The CodeKG MCP server already accepts `--repo /path/to/repo` and resolves both
`.codekg/graph.sqlite` and `.codekg/lancedb` from that root automatically.
The CLI subcommands (`query`, `pack`, `explain`, `analyze`, `centrality`,
`architecture`, `snapshot`, …) each expose `--sqlite` and `--lancedb` separately,
so querying a *different* repo requires spelling out both paths:

```bash
# current — verbose, error-prone
codekg query "diary entry structure" \
    --sqlite /Users/egs/repos/diary_kg/.codekg/graph.sqlite \
    --lancedb /Users/egs/repos/diary_kg/.codekg/lancedb

# desired — mirrors the MCP server interface
codekg query "diary entry structure" --repo /Users/egs/repos/diary_kg
```

**Discovered when:** exploring `diary_kg` from the proteusPy session by running
the project-local venv binary (`diary_kg/.venv/bin/codekg`) as a workaround.
The canonical `codekg` binary works fine against its own CWD but has no way to
target a foreign repo's `.codekg` with a single flag.

**Proposed change in `code_kg`:**
Add a `--repo PATH` option (global or per-subcommand) that sets
`--sqlite PATH/.codekg/graph.sqlite`, `--lancedb PATH/.codekg/lancedb`, and
`--repo-root PATH` in one shot. Should be backward-compatible (defaults to
CWD as today).

**Affected subcommands:** `query`, `pack`, `explain`, `analyze`, `centrality`,
`architecture`, `snapshot list/show/diff`, `viz`, `viz3d`.

---

## ManifoldModel / fly_toward

### [x] Cycle detection in `fly_toward()` — **FIXED 2026-03-24**

`fly_toward()` oscillated between two nodes (A→B→A→B…) indefinitely because
`patience` reset to 0 on each "improving" half of the cycle.  Fixed by adding a
`visited: set[str]` in `fly_toward` and passing it as `excluded=` to `fly_step`,
which now skips any neighbour already in the set.  When all neighbours are
exhausted the walker stops cleanly rather than looping.

### [ ] Long-range graph connectivity for cross-octant flight

The disulfide SSflight demo picks the two most-distant **octant** centroids
(715° apart in 5D torsional space) as origin/destination.  With k=20 the
graph is locally well-connected but greedy navigation cannot guarantee a
through-path across regions that far apart.  Options to investigate:

1. Increase k (try k=50 or k=100) for denser long-range edges.
2. Use waypoints: navigate through binary-class centroids as intermediate
   targets before homing in on the final octant centroid.
3. Add an A\* / BFS fallback on the KnowledgeGraph when greedy stalls —
   gives a guaranteed shortest-path upper bound to compare against the
   manifold walk.

---
