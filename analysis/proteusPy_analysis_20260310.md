> **Analysis Report Metadata**  
> - **Generated:** 2026-03-10T21:41:37Z  
> - **Version:** code-kg 0.7.1  
> - **Commit:** 6939c640 (v0.99.36.dev0)  

# proteusPy Analysis

**Generated:** 2026-03-10 21:41:37 UTC

---

## 📊 Executive Summary

This report provides a comprehensive architectural analysis of the **proteusPy** repository using CodeKG's knowledge graph. The analysis covers complexity hotspots, module coupling, critical call chains, and code quality signals to guide refactoring and architecture decisions.

| Overall Quality | Grade | Score |
|----------------|-------|-------|
| 🟢 **Good** | **B** | 75 / 100 |

---

## 📈 Baseline Metrics

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

## 🔥 Complexity Hotspots (High Fan-In)

Most-called functions are potential bottlenecks or core functionality. These functions are heavily depended upon across the codebase.

| # | Function | Module | Callers | Risk Level |
|---|----------|--------|---------|-----------|
| 1 | `__init__()` | proteusPy/DisulfideClassManager.py | **13** | 🟡 MEDIUM |
| 2 | `__init__()` | build/lib/proteusPy/DisulfideClass_Constructor.py | **13** | 🟡 MEDIUM |
| 3 | `__init__()` | build/lib/proteusPy/DisulfideClassManager.py | **13** | 🟡 MEDIUM |
| 4 | `octant_classes_vs_cutoff()` | old/DisulfideClass_Analysis_mp.py | **1** | 🟢 LOW |
| 5 | `analyze_classes()` | old/DisulfideClass_Analysis_mp.py | **1** | 🟢 LOW |
| 6 | `main()` | programs/example_class_generator.py | **0** | 🟢 LOW |
| 7 | `main()` | proteusPy/display_class_disulfides.py | **0** | 🟢 LOW |
| 8 | `test_pprint_methods()` | tests/test_disulfidelist.py | **0** | 🟢 LOW |
| 9 | `wrapper()` | programs/DisulfideChecker.py | **0** | 🟢 LOW |
| 10 | `main()` | programs/DisulfideClass_Analysis.py | **0** | 🟢 LOW |
| 11 | `main()` | old/DisulfideClass_Analysis_mp.py | **0** | 🟢 LOW |
| 12 | `main()` | programs/class_analysis_mp.py | **0** | 🟢 LOW |
| 13 | `main()` | build/lib/proteusPy/display_class_disulfides.py | **0** | 🟢 LOW |
| 14 | `main()` | programs/dse_calculator.py | **0** | 🟢 LOW |
| 15 | `timer()` | programs/DisulfideChecker.py | **0** | 🟢 LOW |


**Insight:** Functions with high fan-in are either core APIs or bottlenecks. Review these for:
- Thread safety and performance
- Clear documentation and contracts
- Potential for breaking changes

---

## 🔗 High Fan-Out Functions (Orchestrators)

Functions that call many others may indicate complex orchestration logic or poor separation of concerns.

✓ No extreme high fan-out functions detected. Well-balanced architecture.

---

## 📦 Module Architecture

Top modules by dependency coupling and cohesion (showing up to 10 with activity).

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

## 🔗 Critical Call Chains

Deepest call chains in the codebase. These represent critical execution paths.

**Chain 1** (depth: 2)

```
analyze_classes_multiprocess → octant_classes_vs_cutoff
```

**Chain 2** (depth: 4)

```
main → analyze_classes → analyze_classes_multiprocess → octant_classes_vs_cutoff
```

**Chain 3** (depth: 4)

```
main → generate_for_class → _generate_disulfides_for_class → Disulfide
```

**Chain 4** (depth: 2)

```
main → display_class_disulfides
```

**Chain 5** (depth: 3)

```
test_pprint_methods → pprint → repr_ss_ca_dist
```

---

## 🔓 Public API Surface

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

## 📝 Docstring Coverage

Docstring coverage directly determines semantic retrieval quality. Nodes without
docstrings embed only structured identifiers (`KIND/NAME/QUALNAME/MODULE`), where
keyword search is as effective as vector embeddings. The semantic model earns its
value only when a docstring is present.

| Kind | Documented | Total | Coverage |
|------|-----------|-------|----------|
| `function` | 443 | 519 | 🟢 85.4% |
| `method` | 900 | 991 | 🟢 90.8% |
| `class` | 52 | 57 | 🟢 91.2% |
| `module` | 86 | 122 | 🟡 70.5% |
| **total** | **1481** | **1689** | **🟢 87.7%** |



---

## ⚠️  Code Quality Issues

- ⚠️  6 orphaned functions found (`handle_help_menu`, `handle_help_menu`, `handle_help_menu`, `init_worker`, `main`, `wrapper`) — consider archiving or documenting

---

## ✅ Architectural Strengths

- ✓ Well-structured with 15 core functions identified
- ✓ No god objects or god functions detected
- ✓ Good docstring coverage: 87.7% of functions/methods/classes/modules documented

---

## 💡 Recommendations

### Immediate Actions
1. **Remove or archive orphaned functions** — `handle_help_menu`, `handle_help_menu`, `handle_help_menu`, `init_worker`, `main` (and 1 more) have zero callers and add maintenance burden

### Medium-term Refactoring
1. **Harden high fan-in functions** — `__init__`, `__init__`, `__init__` are widely depended upon; review for thread safety, clear contracts, and stable interfaces
2. **Reduce module coupling** — consider splitting tightly coupled modules or introducing interface boundaries
3. **Add tests for critical call chains** — the identified call chains represent high-risk execution paths that benefit most from regression coverage

### Long-term Architecture
1. **Version and stabilize the public API** — document breaking-change policies for `Vector3D`, `DisulfideList`, `Disulfide`
2. **Enforce layer boundaries** — add linting or CI checks to prevent unexpected cross-module dependencies as the codebase grows
3. **Monitor hot paths** — instrument the high fan-in functions identified here to catch performance regressions early

---

## 🧬 Inheritance Hierarchy

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

## 📋 Appendix: Orphaned Code

Functions with zero callers (potential dead code):

| Function | Module | Lines |
|----------|--------|-------|
| `main()` | old/DisulfideClass_Analysis_mp.py | 46 |
| `handle_help_menu()` | viewer/rcsb_viewer.py | 10 |
| `handle_help_menu()` | build/lib/proteusPy/rcsb_viewer.py | 10 |
| `handle_help_menu()` | proteusPy/rcsb_viewer.py | 10 |
| `wrapper()` | programs/DisulfideChecker.py | 5 |
| `init_worker()` | old/DisulfideClass_Analysis_mp.py | 3 |


---

*Report generated by CodeKG Thorough Analysis Tool — analysis completed in 10.3s*
