> **Analysis Report Metadata**  
> - **Generated:** 2026-03-23T03:51:45Z  
> - **Version:** code-kg 0.10.0  
> - **Commit:** 685e12dc (feat.manifold)  
> - **Platform:** macOS 26.3.1 | arm64 (arm) | Turing | Python 3.12.13  
> - **Graph:** 22575 nodes · 28479 edges (1689 meaningful)  
> - **Included directories:** proteusPy  
> - **Excluded directories:** none  
> - **Elapsed time:** 8s  

# proteusPy Analysis

**Generated:** 2026-03-23 03:51:45 UTC

---

## Executive Summary

This report provides a comprehensive architectural analysis of the **proteusPy** repository using CodeKG's knowledge graph. The analysis covers complexity hotspots, module coupling, key call chains, and code quality signals to guide refactoring and architecture decisions.

| Overall Quality | Grade | Score |
|----------------|-------|-------|
| [B] **Good** | **B** | 80 / 100 |

---

## Baseline Metrics

| Metric | Value |
|--------|-------|
| **Total Nodes** | 22575 |
| **Total Edges** | 28479 |
| **Modules** | 122 (of 122 total) |
| **Functions** | 519 |
| **Classes** | 57 |
| **Methods** | 991 |

### Edge Distribution

| Relationship Type | Count |
|-------------------|-------|
| CALLS | 6581 |
| CONTAINS | 1567 |
| IMPORTS | 1385 |
| ATTR_ACCESS | 8234 |
| INHERITS | 33 |

---

## Fan-In Ranking

Most-called functions are potential bottlenecks or core functionality. These functions are heavily depended upon across the codebase.

| # | Function | Module | Callers |
|---|----------|--------|---------|
| 1 | `Vector3D()` | proteusPy/vector3D.py | **82** |
| 2 | `DisulfideList()` | proteusPy/DisulfideBase.py | **62** |
| 3 | `DisulfideList()` | build/lib/proteusPy/DisulfideBase.py | **34** |
| 4 | `DisulfideList()` | build/lib/proteusPy/DisulfideList.py | **33** |
| 5 | `DisulfideList()` | build/lib/proteusPy/oDisulfideList.py | **30** |
| 6 | `display()` | build/lib/proteusPy/qt5viewer.py | **14** |
| 7 | `display()` | proteusPy/qt5viewer.py | **14** |
| 8 | `create_deviation_dataframe()` | build/lib/proteusPy/DisulfideList.py | **12** |
| 9 | `create_deviation_dataframe()` | build/lib/proteusPy/DisulfideBase.py | **11** |
| 10 | `create_deviation_dataframe()` | proteusPy/DisulfideBase.py | **11** |
| 11 | `torsion_array()` | build/lib/proteusPy/DisulfideBase.py | **10** |
| 12 | `torsion_array()` | proteusPy/DisulfideBase.py | **10** |
| 13 | `Vector3D()` | build/lib/proteusPy/vector3D.py | **8** |
| 14 | `build_model()` | build/lib/proteusPy/Disulfide.py | **8** |
| 15 | `build_model()` | build/lib/proteusPy/DisulfideBase.py | **8** |


**Insight:** Functions with high fan-in are either core APIs or bottlenecks. Review these for:
- Thread safety and performance
- Clear documentation and contracts
- Potential for breaking changes

---

## High Fan-Out Functions (Orchestrators)

Functions that call many others may indicate complex orchestration logic or poor separation of concerns.

No extreme high fan-out functions detected. Well-balanced architecture.

---

## Module Architecture

Top modules by dependency coupling and cohesion (showing up to 10 with activity).
Cohesion = incoming / (incoming + outgoing + 1); higher = more internally focused.

| Module | Functions | Classes | Incoming | Outgoing | Cohesion |
|--------|-----------|---------|----------|----------|----------|
| `build/lib/proteusPy/DisulfideBase.py` | 2 | 2 | 34 | 15 | 0.30 |
| `proteusPy/DisulfideBase.py` | 2 | 2 | 43 | 15 | 0.25 |
| `build/lib/proteusPy/Disulfide.py` | 6 | 1 | 31 | 17 | 0.35 |
| `build/lib/proteusPy/DisulfideList.py` | 5 | 1 | 24 | 9 | 0.26 |
| `tests/test_disulfidelist.py` | 50 | 0 | 0 | 7 | 0.88 |
| `build/lib/proteusPy/oDisulfideList.py` | 0 | 1 | 21 | 3 | 0.12 |
| `build/lib/proteusPy/DisulfideLoader.py` | 2 | 1 | 30 | 12 | 0.28 |
| `proteusPy/DisulfideLoader.py` | 2 | 1 | 33 | 12 | 0.26 |
| `build/lib/proteusPy/DisulfideVisualization.py` | 3 | 1 | 2 | 3 | 0.50 |
| `proteusPy/DisulfideVisualization.py` | 3 | 1 | 12 | 3 | 0.19 |

---

## Key Call Chains

Deepest call chains in the codebase.

No deep call chains detected.

---

## Public API Surface

Identified public APIs (module-level functions with high usage).

| Function | Module | Fan-In | Type |
|----------|--------|--------|------|
| `Vector3D()` | proteusPy/vector3D.py | 82 | class |
| `DisulfideList()` | proteusPy/DisulfideBase.py | 62 | class |
| `Disulfide()` | proteusPy/DisulfideBase.py | 43 | class |
| `DisulfideList()` | build/lib/proteusPy/DisulfideBase.py | 34 | class |
| `DisulfideList()` | build/lib/proteusPy/DisulfideList.py | 33 | class |
| `set_plotly_theme()` | proteusPy/utility.py | 32 | function |
| `Disulfide()` | build/lib/proteusPy/Disulfide.py | 31 | class |
| `Disulfide()` | build/lib/proteusPy/DisulfideBase.py | 31 | class |
| `DisulfideList()` | build/lib/proteusPy/oDisulfideList.py | 30 | class |
| `Turtle3D()` | proteusPy/turtle3D.py | 24 | class |
---

## Docstring Coverage

Docstring coverage directly determines semantic retrieval quality. Nodes without
docstrings embed only structured identifiers (`KIND/NAME/QUALNAME/MODULE`), where
keyword search is as effective as vector embeddings. The semantic model earns its
value only when a docstring is present.

| Kind | Documented | Total | Coverage |
|------|-----------|-------|----------|
| `function` | 443 | 519 | [OK] 85.4% |
| `method` | 900 | 991 | [OK] 90.8% |
| `class` | 52 | 57 | [OK] 91.2% |
| `module` | 86 | 122 | [WARN] 70.5% |
| **total** | **1481** | **1689** | **[OK] 87.7%** |

---

## Structural Importance Ranking (SIR)

Weighted PageRank aggregated by module — reveals architectural spine. Cross-module edges boosted 1.5×; private symbols penalized 0.85×. Node-level detail: `codekg centrality --top 25`

| Rank | Score | Members | Module |
|------|-------|---------|--------|
| 1 | 0.121750 | 133 | `proteusPy/DisulfideBase.py` |
| 2 | 0.107883 | 133 | `build/lib/proteusPy/DisulfideBase.py` |
| 3 | 0.066010 | 60 | `build/lib/proteusPy/DisulfideList.py` |
| 4 | 0.064956 | 27 | `proteusPy/vector3D.py` |
| 5 | 0.054207 | 44 | `build/lib/proteusPy/oDisulfideList.py` |
| 6 | 0.048139 | 85 | `build/lib/proteusPy/Disulfide.py` |
| 7 | 0.039477 | 33 | `proteusPy/utility.py` |
| 8 | 0.036831 | 27 | `build/lib/proteusPy/vector3D.py` |
| 9 | 0.025529 | 35 | `proteusPy/DisulfideVisualization.py` |
| 10 | 0.021764 | 38 | `proteusPy/DisulfideLoader.py` |
| 11 | 0.021409 | 38 | `build/lib/proteusPy/DisulfideLoader.py` |
| 12 | 0.017424 | 26 | `old/DBViewer.py` |
| 13 | 0.016762 | 31 | `proteusPy/turtle3D.py` |
| 14 | 0.015965 | 15 | `proteusPy/DisulfideStats.py` |
| 15 | 0.015418 | 20 | `proteusPy/DisulfideClassGenerator.py` |



---

## Code Quality Issues

- [WARN] 4 orphaned functions found (`objective_function`, `task`, `R90`, `objective_function`) -- consider archiving or documenting

---

## Architectural Strengths

- Well-structured with 15 core functions identified
- No god objects or god functions detected
- Good docstring coverage: 87.7% of functions/methods/classes/modules documented

---

## Recommendations

### Immediate Actions
1. **Remove or archive orphaned functions** — `objective_function`, `task`, `R90`, `objective_function` have zero callers and add maintenance burden

### Medium-term Refactoring
1. **Harden high fan-in functions** — `Vector3D`, `DisulfideList`, `DisulfideList` are widely depended upon; review for thread safety, clear contracts, and stable interfaces
2. **Reduce module coupling** — consider splitting tightly coupled modules or introducing interface boundaries

### Long-term Architecture
1. **Version and stabilize the public API** — document breaking-change policies for `Vector3D`, `DisulfideList`, `Disulfide`
2. **Enforce layer boundaries** — add linting or CI checks to prevent unexpected cross-module dependencies as the codebase grows
3. **Monitor hot paths** — instrument the high fan-in functions identified here to catch performance regressions early

---

## Inheritance Hierarchy

**33** INHERITS edges across **35** classes. Max depth: **1**.

| Class | Module | Depth | Parents | Children |
|-------|--------|-------|---------|----------|
| `DisulfideConstructionWarning` | build/lib/proteusPy/DisulfideExceptions.py | 1 | 1 | 0 |
| `DisulfideParseWarning` | build/lib/proteusPy/DisulfideExceptions.py | 1 | 1 | 0 |
| `DisulfideConstructionWarning` | proteusPy/DisulfideExceptions.py | 1 | 1 | 0 |
| `DisulfideParseWarning` | proteusPy/DisulfideExceptions.py | 1 | 1 | 0 |
| `DisulfideList` | build/lib/proteusPy/DisulfideBase.py | 0 | 1 | 0 |
| `DisulfideConstructionException` | build/lib/proteusPy/DisulfideExceptions.py | 0 | 1 | 0 |
| `DisulfideException` | build/lib/proteusPy/DisulfideExceptions.py | 0 | 1 | 0 |
| `DisulfideIOException` | build/lib/proteusPy/DisulfideExceptions.py | 0 | 1 | 0 |
| `DisulfideList` | build/lib/proteusPy/DisulfideList.py | 0 | 1 | 0 |
| `ProteusPyWarning` | build/lib/proteusPy/ProteusPyWarning.py | 0 | 1 | 4 |
| `AngleAnnotation` | build/lib/proteusPy/angle_annotation.py | 0 | 1 | 0 |
| `DisulfideList` | build/lib/proteusPy/oDisulfideList.py | 0 | 1 | 0 |
| `DisulfideViewer` | build/lib/proteusPy/qt5viewer.py | 0 | 1 | 0 |
| `ReloadableApp` | build/lib/proteusPy/rcsb_viewer.py | 0 | 1 | 0 |
| `TestDisulfide` | build/lib/tests/test_disulfide.py | 0 | 1 | 0 |
| `TestTurtle3D` | build/lib/tests/test_turtle3d.py | 0 | 1 | 0 |
| `ReloadableApp` | old/DBViewer.py | 0 | 1 | 0 |
| `ReloadableApp` | old/rcsb_viewer.py | 0 | 1 | 0 |
| `DisulfideList` | proteusPy/DisulfideBase.py | 0 | 1 | 0 |
| `DisulfideConstructionException` | proteusPy/DisulfideExceptions.py | 0 | 1 | 0 |


---

## Snapshot History

No snapshots found. Run `codekg snapshot save <version>` to capture one.


---

## Appendix: Orphaned Code

Functions with zero callers (potential dead code):

| Function | Module | Lines |
|----------|--------|-------|
| `task()` | old/DisulfideClass_Analysis_mp.py | 84 |
| `R90()` | build/lib/proteusPy/angle_annotation.py | 8 |
| `objective_function()` | programs/minimize_dse.py | 4 |
| `objective_function()` | programs/minimize_dse.py | 4 |
---

## CodeRank -- Global Structural Importance

Weighted PageRank over CALLS + IMPORTS + INHERITS edges (test paths excluded). Scores are normalized to sum to 1.0. This ranking seeds Phase 2 fan-in discovery and Phase 15 concern queries.

| Rank | Score | Kind | Name | Module |
|------|-------|------|------|--------|
| 1 | 0.000378 | method | `DisulfideList.torsion_array` | proteusPy/DisulfideBase.py |
| 2 | 0.000378 | method | `DisulfideList.torsion_array` | build/lib/proteusPy/DisulfideBase.py |
| 3 | 0.000281 | class | `Vector3D` | proteusPy/vector3D.py |
| 4 | 0.000281 | class | `Vector3D` | build/lib/proteusPy/vector3D.py |
| 5 | 0.000198 | method | `Disulfide._compute_rho` | proteusPy/DisulfideBase.py |
| 6 | 0.000198 | method | `Disulfide._compute_rho` | build/lib/proteusPy/DisulfideBase.py |
| 7 | 0.000192 | method | `Disulfide._compute_rho` | build/lib/proteusPy/Disulfide.py |
| 8 | 0.000183 | method | `DisulfideList.resolution` | proteusPy/DisulfideBase.py |
| 9 | 0.000183 | method | `DisulfideList.resolution` | build/lib/proteusPy/DisulfideBase.py |
| 10 | 0.000183 | class | `DisulfideList` | proteusPy/DisulfideBase.py |
| 11 | 0.000183 | class | `DisulfideList` | build/lib/proteusPy/DisulfideBase.py |
| 12 | 0.000183 | method | `DisulfideList.get_torsion_array` | build/lib/proteusPy/DisulfideList.py |
| 13 | 0.000172 | method | `DisulfideEnergy._torad` | proteusPy/DisulfideEnergy.py |
| 14 | 0.000172 | method | `DisulfideEnergy._torad` | build/lib/proteusPy/DisulfideEnergy.py |
| 15 | 0.000167 | method | `DisulfideList.torsion_array` | build/lib/proteusPy/DisulfideList.py |
| 16 | 0.000160 | method | `Disulfide.rho` | proteusPy/DisulfideBase.py |
| 17 | 0.000160 | method | `Disulfide.rho` | build/lib/proteusPy/DisulfideBase.py |
| 18 | 0.000160 | method | `DisulfideList.validate_ss` | proteusPy/DisulfideBase.py |
| 19 | 0.000160 | method | `DisulfideList.validate_ss` | build/lib/proteusPy/DisulfideBase.py |
| 20 | 0.000160 | method | `DisulfideList.validate_ss` | build/lib/proteusPy/DisulfideList.py |

---

## Concern-Based Hybrid Ranking

Top structurally-dominant nodes per architectural concern (0.60 × semantic + 0.25 × CodeRank + 0.15 × graph proximity).

### Configuration Loading Initialization Setup

| Rank | Score | Kind | Name | Module |
|------|-------|------|------|--------|
| 1 | 0.7519 | function | `do_stuff` | build/lib/proteusPy/DisulfideExtractor_mp.py |
| 2 | 0.749 | method | `Turtle3D.new` | build/lib/proteusPy/turtle3D.py |
| 3 | 0.7484 | function | `do_stuff` | proteusPy/DisulfideExtractor_mp.py |
| 4 | 0.7468 | method | `Turtle3D.new` | proteusPy/turtle3D.py |
| 5 | 0.7466 | function | `build/lib/proteusPy/DisulfideBase.py.Disulfide._compute_torsional_energy.torad` | build/lib/proteusPy/DisulfideBase.py |

### Data Persistence Storage Database

| Rank | Score | Kind | Name | Module |
|------|-------|------|------|--------|
| 1 | 0.7507 | function | `do_stuff` | proteusPy/DisulfideExtractor_mp.py |
| 2 | 0.7468 | function | `do_stuff` | build/lib/proteusPy/DisulfideExtractor_mp.py |
| 3 | 0.7429 | function | `check_file` | proteusPy/ssparser.py |
| 4 | 0.7421 | function | `check_files` | programs/DisulfideChecker.py |
| 5 | 0.7415 | function | `check_file` | programs/DisulfideChecker.py |

### Query Search Retrieval Semantic

| Rank | Score | Kind | Name | Module |
|------|-------|------|------|--------|
| 1 | 0.7507 | function | `do_stuff` | proteusPy/DisulfideExtractor_mp.py |
| 2 | 0.7486 | function | `do_stuff` | build/lib/proteusPy/DisulfideExtractor_mp.py |
| 3 | 0.7462 | function | `_parse_remark_465` | data/parse_pdb_header_egs.py |
| 4 | 0.7411 | method | `DisulfideList.extract_distances` | build/lib/proteusPy/DisulfideBase.py |
| 5 | 0.7406 | function | `load_disulfides_from_id` | proteusPy/DisulfideIO.py |

### Graph Traversal Node Edge

| Rank | Score | Kind | Name | Module |
|------|-------|------|------|--------|
| 1 | 0.7472 | function | `to_carbonyl` | proteusPy/Residue.py |
| 2 | 0.7467 | function | `to_alpha` | build/lib/proteusPy/Residue.py |
| 3 | 0.7459 | method | `Turtle3D.new` | proteusPy/turtle3D.py |
| 4 | 0.7455 | function | `to_carbonyl` | build/lib/proteusPy/Residue.py |
| 5 | 0.7434 | function | `to_alpha` | proteusPy/Residue.py |



---

*Report generated by CodeKG Thorough Analysis Tool — analysis completed in 8.3s*
