> **Analysis Report Metadata**  
> - **Generated:** 2026-04-10T20:17:29Z  
> - **Version:** pycode-kg 0.13.0  
> - **Commit:** 82849cf0 (master)  
> - **Platform:** macOS 26.4 | arm64 (arm) | Turing | Python 3.12.13  
> - **Graph:** 10363 nodes · 9884 edges (737 meaningful)  
> - **Included directories:** proteusPy  
> - **Excluded directories:** none  
> - **Elapsed time:** 3s  

# proteusPy Analysis

**Generated:** 2026-04-10 20:17:29 UTC

---

## Executive Summary

This report provides a comprehensive architectural analysis of the **proteusPy** repository using PyCodeKG's knowledge graph. The analysis covers complexity hotspots, module coupling, key call chains, and code quality signals to guide refactoring and architecture decisions.

| Overall Quality | Grade | Score |
|----------------|-------|-------|
| [A] **Excellent** | **A** | 90 / 100 |

---

## Baseline Metrics

| Metric | Value |
|--------|-------|
| **Total Nodes** | 10363 |
| **Total Edges** | 9884 |
| **Modules** | 38 (of 38 total) |
| **Functions** | 163 |
| **Classes** | 40 |
| **Methods** | 496 |

### Edge Distribution

| Relationship Type | Count |
|-------------------|-------|
| CALLS | 2879 |
| CONTAINS | 699 |
| IMPORTS | 494 |
| ATTR_ACCESS | 3560 |
| INHERITS | 19 |

---

## Fan-In Ranking

Most-called functions are potential bottlenecks or core functionality. These functions are heavily depended upon across the codebase.

| # | Function | Module | Callers |
|---|----------|--------|---------|
| 1 | `copy()` | proteusPy/DisulfideBase.py | **49** |
| 2 | `copy()` | proteusPy/vector3D.py | **49** |
| 3 | `Vector3D()` | proteusPy/vector3D.py | **35** |
| 4 | `DisulfideList()` | proteusPy/DisulfideBase.py | **26** |
| 5 | `torsion_array()` | proteusPy/DisulfideBase.py | **10** |
| 6 | `_render_ss()` | proteusPy/DisulfideVisualization.py | **8** |
| 7 | `_rotate()` | proteusPy/turtleND.py | **6** |
| 8 | `unit()` | proteusPy/turtle3D.py | **6** |
| 9 | `create_deviation_dataframe()` | proteusPy/DisulfideBase.py | **6** |
| 10 | `__init__()` | proteusPy/DisulfideBase.py | **6** |
| 11 | `__init__()` | proteusPy/turtle3D.py | **6** |
| 12 | `roll()` | proteusPy/turtle3D.py | **6** |
| 13 | `validate_ss()` | proteusPy/DisulfideBase.py | **5** |
| 14 | `repr_ss_conformation()` | proteusPy/DisulfideBase.py | **5** |
| 15 | `repr_ss_info()` | proteusPy/DisulfideBase.py | **5** |


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
| `proteusPy/DisulfideBase.py` | 2 | 2 | 9 | 11 | 0.52 |
| `proteusPy/graph_reasoner.py` | 3 | 12 | 8 | 1 | 0.10 |
| `proteusPy/DisulfideLoader.py` | 2 | 1 | 7 | 9 | 0.53 |
| `proteusPy/DisulfideVisualization.py` | 3 | 1 | 5 | 3 | 0.33 |
| `proteusPy/rcsb_viewer.py` | 28 | 1 | 0 | 5 | 0.83 |
| `proteusPy/utility.py` | 32 | 0 | 5 | 6 | 0.50 |
| `proteusPy/turtleND.py` | 0 | 1 | 4 | 0 | 0.00 |
| `proteusPy/turtle3D.py` | 0 | 1 | 3 | 1 | 0.20 |
| `proteusPy/manifold_model.py` | 0 | 2 | 2 | 2 | 0.40 |
| `proteusPy/qt5viewer.py` | 4 | 1 | 0 | 4 | 0.80 |

---

## Key Call Chains

Deepest call chains in the codebase.

No deep call chains detected.

---

## Public API Surface

Identified public APIs (module-level functions with high usage).

| Function | Module | Fan-In | Type |
|----------|--------|--------|------|
| `Vector3D()` | proteusPy/vector3D.py | 35 | class |
| `DisulfideList()` | proteusPy/DisulfideBase.py | 26 | class |
| `set_plotly_theme()` | proteusPy/utility.py | 14 | function |
| `set_pyvista_theme()` | proteusPy/utility.py | 9 | function |
| `Disulfide()` | proteusPy/DisulfideBase.py | 8 | class |
| `plot()` | proteusPy/rcsb_viewer.py | 7 | function |
| `load_disulfides_from_id()` | proteusPy/DisulfideIO.py | 4 | function |
| `Load_PDB_SS()` | proteusPy/DisulfideLoader.py | 4 | function |
| `get_theme()` | proteusPy/utility.py | 4 | function |
| `ReasoningPath()` | proteusPy/graph_reasoner.py | 4 | class |
---

## Docstring Coverage

Docstring coverage directly determines semantic retrieval quality. Nodes without
docstrings embed only structured identifiers (`KIND/NAME/QUALNAME/MODULE`), where
keyword search is as effective as vector embeddings. The semantic model earns its
value only when a docstring is present.

| Kind | Documented | Total | Coverage |
|------|-----------|-------|----------|
| `function` | 147 | 163 | [OK] 90.2% |
| `method` | 434 | 496 | [OK] 87.5% |
| `class` | 40 | 40 | [OK] 100.0% |
| `module` | 32 | 38 | [OK] 84.2% |
| **total** | **653** | **737** | **[OK] 88.6%** |

---

## Structural Importance Ranking (SIR)

Weighted PageRank aggregated by module — reveals architectural spine. Cross-module edges boosted 1.5×; private symbols penalized 0.85×. Node-level detail: `pycodekg centrality --top 25`

| Rank | Score | Members | Module |
|------|-------|---------|--------|
| 1 | 0.261487 | 133 | `proteusPy/DisulfideBase.py` |
| 2 | 0.121144 | 27 | `proteusPy/vector3D.py` |
| 3 | 0.077429 | 59 | `proteusPy/graph_reasoner.py` |
| 4 | 0.069031 | 33 | `proteusPy/utility.py` |
| 5 | 0.048023 | 38 | `proteusPy/DisulfideLoader.py` |
| 6 | 0.045600 | 35 | `proteusPy/DisulfideVisualization.py` |
| 7 | 0.035868 | 32 | `proteusPy/turtleND.py` |
| 8 | 0.029210 | 34 | `proteusPy/rcsb_viewer.py` |
| 9 | 0.028232 | 31 | `proteusPy/turtle3D.py` |
| 10 | 0.027124 | 29 | `proteusPy/manifold_model.py` |
| 11 | 0.026377 | 23 | `proteusPy/DisulfideClassManager.py` |
| 12 | 0.026162 | 19 | `proteusPy/DisulfideEnergy.py` |
| 13 | 0.022856 | 27 | `proteusPy/qt5viewer.py` |
| 14 | 0.022193 | 15 | `proteusPy/DisulfideStats.py` |
| 15 | 0.020789 | 20 | `proteusPy/DisulfideClassGenerator.py` |



---

## Code Quality Issues

- [WARN] 1 orphaned functions found (`handle_help_menu`) -- consider archiving or documenting

---

## Architectural Strengths

- Well-structured with 15 core functions identified
- No god objects or god functions detected
- Good docstring coverage: 88.6% of functions/methods/classes/modules documented

---

## Recommendations

### Immediate Actions
1. **Remove or archive orphaned functions** — `handle_help_menu` have zero callers and add maintenance burden

### Medium-term Refactoring
1. **Harden high fan-in functions** — `copy`, `copy`, `Vector3D` are widely depended upon; review for thread safety, clear contracts, and stable interfaces
2. **Reduce module coupling** — consider splitting tightly coupled modules or introducing interface boundaries

### Long-term Architecture
1. **Version and stabilize the public API** — document breaking-change policies for `Vector3D`, `DisulfideList`, `set_plotly_theme`
2. **Enforce layer boundaries** — add linting or CI checks to prevent unexpected cross-module dependencies as the codebase grows
3. **Monitor hot paths** — instrument the high fan-in functions identified here to catch performance regressions early

---

## Inheritance Hierarchy

**19** INHERITS edges across **21** classes. Max depth: **1**.

| Class | Module | Depth | Parents | Children |
|-------|--------|-------|---------|----------|
| `DisulfideConstructionWarning` | proteusPy/DisulfideExceptions.py | 1 | 1 | 0 |
| `DisulfideParseWarning` | proteusPy/DisulfideExceptions.py | 1 | 1 | 0 |
| `DirectedDiscoverer` | proteusPy/graph_reasoner.py | 1 | 1 | 0 |
| `ExplorationSteering` | proteusPy/graph_reasoner.py | 1 | 1 | 0 |
| `GradientSteering` | proteusPy/graph_reasoner.py | 1 | 1 | 0 |
| `KNNDiscoverer` | proteusPy/graph_reasoner.py | 1 | 1 | 0 |
| `RadiusDiscoverer` | proteusPy/graph_reasoner.py | 1 | 1 | 0 |
| `TargetSteering` | proteusPy/graph_reasoner.py | 1 | 1 | 0 |
| `ManifoldAdamWalker` | proteusPy/manifold_walker.py | 1 | 1 | 0 |
| `DisulfideList` | proteusPy/DisulfideBase.py | 0 | 1 | 0 |
| `DisulfideConstructionException` | proteusPy/DisulfideExceptions.py | 0 | 1 | 0 |
| `DisulfideException` | proteusPy/DisulfideExceptions.py | 0 | 1 | 0 |
| `DisulfideIOException` | proteusPy/DisulfideExceptions.py | 0 | 1 | 0 |
| `ProteusPyWarning` | proteusPy/ProteusPyWarning.py | 0 | 1 | 2 |
| `AngleAnnotation` | proteusPy/angle_annotation.py | 0 | 1 | 0 |
| `EdgeDiscoverer` | proteusPy/graph_reasoner.py | 0 | 1 | 3 |
| `SteeringStrategy` | proteusPy/graph_reasoner.py | 0 | 1 | 3 |
| `ManifoldWalker` | proteusPy/manifold_walker.py | 0 | 0 | 1 |
| `DisulfideViewer` | proteusPy/qt5viewer.py | 0 | 1 | 0 |
| `ReloadableApp` | proteusPy/rcsb_viewer.py | 0 | 1 | 0 |


---

## Snapshot History

Recent snapshots in reverse chronological order. Δ columns show change vs. the immediately preceding snapshot.

| # | Timestamp | Branch | Version | Nodes | Edges | Coverage | Δ Nodes | Δ Edges | Δ Coverage |
|---|-----------|--------|---------|-------|-------|----------|---------|---------|------------|
| 1 | 2026-04-10 20:12:25 | master | 0.13.0 | 10363 | 9884 | 88.6% | — | — | — |


---

## Appendix: Orphaned Code

Functions with zero callers (potential dead code):

| Function | Module | Lines |
|----------|--------|-------|
| `handle_help_menu()` | proteusPy/rcsb_viewer.py | 10 |
---

## CodeRank -- Global Structural Importance

Weighted PageRank over CALLS + IMPORTS + INHERITS edges (test paths excluded). Scores are normalized to sum to 1.0. This ranking seeds Phase 2 fan-in discovery and Phase 15 concern queries.

| Rank | Score | Kind | Name | Module |
|------|-------|------|------|--------|
| 1 | 0.000776 | method | `DisulfideList.torsion_array` | proteusPy/DisulfideBase.py |
| 2 | 0.000578 | class | `Vector3D` | proteusPy/vector3D.py |
| 3 | 0.000407 | method | `Disulfide._compute_rho` | proteusPy/DisulfideBase.py |
| 4 | 0.000376 | method | `DisulfideList.resolution` | proteusPy/DisulfideBase.py |
| 5 | 0.000376 | class | `DisulfideList` | proteusPy/DisulfideBase.py |
| 6 | 0.000354 | method | `DisulfideEnergy._torad` | proteusPy/DisulfideEnergy.py |
| 7 | 0.000329 | method | `Disulfide.rho` | proteusPy/DisulfideBase.py |
| 8 | 0.000328 | method | `DisulfideList.validate_ss` | proteusPy/DisulfideBase.py |
| 9 | 0.000321 | method | `TurtleND._rotate` | proteusPy/turtleND.py |
| 10 | 0.000287 | method | `DisulfideEnergy.calculate_dse_components` | proteusPy/DisulfideEnergy.py |
| 11 | 0.000282 | method | `Disulfide._internal_coords` | proteusPy/DisulfideBase.py |
| 12 | 0.000268 | method | `Turtle3D.unit` | proteusPy/turtle3D.py |
| 13 | 0.000264 | method | `DisulfideViewer.display` | proteusPy/qt5viewer.py |
| 14 | 0.000251 | method | `DisulfideEnergy.set_dihedrals` | proteusPy/DisulfideEnergy.py |
| 15 | 0.000246 | method | `AngleAnnotation.get_theta` | proteusPy/angle_annotation.py |
| 16 | 0.000246 | method | `DisulfideList.create_deviation_dataframe` | proteusPy/DisulfideBase.py |
| 17 | 0.000246 | method | `Disulfide.get_chains` | proteusPy/DisulfideBase.py |
| 18 | 0.000237 | method | `Disulfide.repr_ss_conformation` | proteusPy/DisulfideBase.py |
| 19 | 0.000237 | method | `Disulfide.repr_ss_info` | proteusPy/DisulfideBase.py |
| 20 | 0.000235 | method | `Disulfide.bond_angle_ideality` | proteusPy/DisulfideBase.py |

---

## Concern-Based Hybrid Ranking

Top structurally-dominant nodes per architectural concern (0.60 × semantic + 0.25 × CodeRank + 0.15 × graph proximity).

### Configuration Loading Initialization Setup

| Rank | Score | Kind | Name | Module |
|------|-------|------|------|--------|
| 1 | 0.75 | method | `DisulfideLoader.__post_init__` | proteusPy/DisulfideLoader.py |
| 2 | 0.7388 | method | `ReloadableApp.__init__` | proteusPy/rcsb_viewer.py |
| 3 | 0.7246 | method | `DisulfideClassManager.build_classes` | proteusPy/DisulfideClassManager.py |
| 4 | 0.7162 | method | `DisulfideClassGenerator._initialize_data` | proteusPy/DisulfideClassGenerator.py |
| 5 | 0.7134 | method | `DisulfideLoader.__setitem__` | proteusPy/DisulfideLoader.py |

### Data Persistence Storage Database

| Rank | Score | Kind | Name | Module |
|------|-------|------|------|--------|
| 1 | 0.7507 | method | `DisulfideLoader.describe` | proteusPy/DisulfideLoader.py |
| 2 | 0.7359 | function | `Load_PDB_SS` | proteusPy/DisulfideLoader.py |
| 3 | 0.7329 | function | `load_data` | proteusPy/rcsb_viewer.py |
| 4 | 0.7305 | function | `do_stuff` | proteusPy/DisulfideExtractor_mp.py |
| 5 | 0.7278 | method | `DisulfideClassGenerator.prepare_energy_data` | proteusPy/DisulfideClassGenerator.py |

### Query Search Retrieval Semantic

| Rank | Score | Kind | Name | Module |
|------|-------|------|------|--------|
| 1 | 0.7444 | method | `ManifoldModel._predict_single` | proteusPy/manifold_model.py |
| 2 | 0.7422 | method | `KnowledgeGraph.__contains__` | proteusPy/graph_reasoner.py |
| 3 | 0.739 | method | `ManifoldObserver.locate` | proteusPy/manifold_observer.py |
| 4 | 0.7314 | method | `KNNDiscoverer.discover` | proteusPy/graph_reasoner.py |
| 5 | 0.7284 | method | `ManifoldModel.predict` | proteusPy/manifold_model.py |

### Graph Traversal Node Edge

| Rank | Score | Kind | Name | Module |
|------|-------|------|------|--------|
| 1 | 0.7542 | method | `GraphReasoner.step` | proteusPy/graph_reasoner.py |
| 2 | 0.7443 | method | `EdgeDiscoverer.discover` | proteusPy/graph_reasoner.py |
| 3 | 0.7166 | method | `KnowledgeGraph.discover_neighbors` | proteusPy/graph_reasoner.py |
| 4 | 0.6997 | function | `graph_from_disulfides` | proteusPy/graph_reasoner.py |
| 5 | 0.6507 | class | `EdgeDiscoverer` | proteusPy/graph_reasoner.py |



---

*Report generated by PyCodeKG Thorough Analysis Tool — analysis completed in 4.0s*
