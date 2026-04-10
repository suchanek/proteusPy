# Release Notes — v0.99.62

> Released: 2026-04-10


### Added

- **`docs/suchanek_disulfide_chapter_2026.pdf`** — completed Springer book chapter on disulfide bond geometry and the proteusPy analysis methodology. The chapter is now finished and the final PDF is included in the repository.
- **ManifoldModel baseline in MNIST benchmark** (`mnist_manifold_architecture.py`) — `ManifoldModel` (zero-parameter, pure-geometry classifier) is now included as a named architecture entry alongside the neural models. Uses `run_trial_sklearn()` path; reports intrinsic dimensionality and noise-suppression percentage. `ManifoldModel` is skipped in the parameter-efficiency table (0 params) but participates in the winner comparison with the label "Uses ZERO learned parameters — pure manifold geometry".
- **`.claude/settings.json`** — Claude Code project settings file.
- **`.pre-commit-config.yaml`** — pre-commit configuration adapted from `pycode_kg`: standard file hygiene (`pre-commit-hooks` v5.0.0), `poetry check --lock`, and `ruff` + `ruff-format`. Resolves the missing-config error that blocked commits when the codekg pre-commit hook invoked `pre-commit run`.

### Fixed

- **Apple BLAS float32 overflow in local PCA** (`digits_manifold_architecture.py`, `digits_manifold_knn.py`, `mnist_manifold_architecture.py`) — covariance matrix computation now casts centered neighbors to `float64` and uses `np.einsum("ij,ik->jk", …)` instead of `centered.T @ centered`. Prevents silent numerical overflow on Apple Silicon where Apple BLAS accumulates float32 products into a float32 accumulator, producing `inf` / `NaN` eigenvalues.
- **`np.nan_to_num` after `StandardScaler`** (`digits_manifold_architecture.py`, `mnist_manifold_architecture.py`) — constant-valued pixels (std = 0) produce `NaN` after division by zero in `StandardScaler.fit_transform()`; clamped to zero before casting to `float32`.
- **Numerically stable loss + gradient clipping in MNIST neural models** (`mnist_manifold_architecture.py`) — all Keras models now use `from_logits=True` (fused log-softmax, avoids log(≈0) on Metal GPU) and `clipnorm=1.0` via a shared `_compile()` helper. Previously separate `model.compile()` calls with `"sparse_categorical_crossentropy"` (implying pre-normalised softmax) could overflow on Apple Metal float32.
- **PCA float64 promotion for cross-validation folds** (`digits_manifold_architecture.py`) — `pca_fold.fit_transform` and `pca_fold.transform` now receive `X.astype(np.float64)` to prevent the same Apple BLAS overflow inside scikit-learn's PCA matmul.

### Changed

- **`tensorflow` upgraded to 2.18.0; `tensorflow-metal` upgraded to 1.2.0** (`pyproject.toml`) — `tensorflow` moves from 2.16.2; `tensorflow-metal` moves from 1.1.0 (previously optional extra) into the `ml` dependency group as a required peer of TensorFlow 2.18 on Apple Silicon.
- **`ftree-kg` switched to local develop path** (`pyproject.toml`) — dependency now resolves from `../ftreekg` with `develop = true` instead of the upstream git remote, enabling in-place edits during active development.
- **Removed `metal` extras entry** (`pyproject.toml`) — `tensorflow-metal` is now a direct `ml` group dependency; the separate `[extras.metal]` group and its entry in `[extras.all]` have been removed.

### Removed

- **`turtlend_package/`** — entire in-tree package (source, tests, and docs) removed. The `TurtleND`, `Turtle3D`, `ManifoldWalker`, and `ManifoldModel` implementations have been extracted to [flux-frontiers/WaveRider](https://github.com/flux-frontiers/WaveRider); the in-tree copies are no longer needed.
- **`waverider_missions/`** — mission logs, story arc, glossary, chapter drafts (ch1–ch5, interlude), and supporting notes removed. Content migrated to [flux-frontiers/WaveRider](https://github.com/flux-frontiers/WaveRider).
- **`docs/waverider/article/`** — LaTeX source, compiled PDF, auxiliary files, and figures for the WaveRider arXiv manuscript removed (now maintained in the WaveRider repo).
- **`docs/manifold_observer/manifold_observer.md`**, **`docs/manifold_walker_spec/`** — specification documents removed (superseded by the extracted package documentation).

### Added

- **`benchmarks/canonical_tests/report_generator.py`** — automated PDF + Markdown report generator for canonical benchmark runs. Reads a JSON results file, aggregates per-architecture statistics (mean/std accuracy, loss, wall time, parameter count, efficiency), embeds the result figure, and emits provenance-rich reports covering manifold discovery tables, per-class intrinsic dimensionality, architecture comparison, and key findings. Supports MNIST, CIFAR-10, CIFAR-100, Digits, and Iris via `--all` batch mode.
- **Canonical benchmark reports** — generated PDF and Markdown reports for all four datasets: MNIST (`mnist_report.md/.pdf`), CIFAR-10 (`cifar10_report.md/.pdf`), CIFAR-100 (`cifar100_report.md/.pdf`), and Digits (`digits_report.md/.pdf`), each including experimental setup, manifold discovery statistics, per-class intrinsic dimensionality, architecture comparison table, and key findings.

### Fixed

- **`digits_manifold_architecture.py` / `mnist_manifold_architecture.py`** — removed dead `d = max(intrinsic_dim, n_classes)` assignments in `build_pca_intrinsic_dim_model()` (and `build_manifold_observer_model()` in MNIST); the computed value was never used downstream, so the lines were pure dead code.
- **`digits_manifold_architecture.py`** — removed unused `sklearn_names` variable from `plot_results()`.

### Changed

- **WaveRider references** updated throughout README, docs, and benchmarks to point to the new [flux-frontiers/WaveRider](https://github.com/flux-frontiers/WaveRider) repository.

- **WaveRider arXiv manuscript** (`docs/waverider/article/waverider_arxiv.tex`) — relocated from `docs/waverider/` to `docs/waverider/article/`; updated Introduction to justify intrinsic-dimensionality claims with the actual discovery mechanism: local PCA cumulative-variance threshold (τ) applied per data point, with forward references to Algorithm 1 (`\label{alg:walker}`) and §3.2 (`\label{sec:manifoldwalker}`); corrected the noise-fraction figures per dataset (71–83% for UCI Digits, 98.8–99.3% for CIFAR-10; previously misstated as ">99%" for both). Added "Intrinsic dimensionality estimation" paragraph in Related Work covering TwoNN (Facco et al. 2017), participation ratio, and PCA-elbow methods. Added bibliography entries for Facco et al. 2017 (TwoNN estimator) and Pope et al. 2021 (intrinsic dimension of images), cited for corroboration.

- **`pepys_manifold_explorer.py` — TwoNN fix**: corrected sign error in `twonn_id()` — formula now returns `len(mu) / sum(log(mu))` (positive), fixing negative intrinsic-dimensionality estimates.
- **`pepys_manifold_explorer.py` — NaN-safe MRR plot**: `make_figure()` filters `NaN` values from `mrrs` before calling `max()`, preventing crash when some MRL checkpoints yield no valid retrievals.
- **`pepys_manifold_explorer.py` — use `DEFAULT_MODEL` for query embedding**: `main()` now passes `DEFAULT_MODEL` instead of `args.model` to `embed_local()`, ensuring query and corpus embeddings always use the same model.
- **Removed stale Pepys benchmark artefacts**: deleted `benchmarks/pepys_small_embeddings.json`, `benchmarks/pepys_small_results.json`, and `benchmarks/pepys_small_results.png` (superseded by the full-corpus run); updated `benchmarks/pepys_manifold_results.json` and `.png` with results from the corrected run.
- **Removed `benchmarks/pepys/tests/`**: deleted the entire test sub-package (`__init__.py`, `conftest.py`, and nine test modules) that targeted the now-removed `DiaryKG`/`DiaryTransformer` in-explorer code paths; no replacement needed as that logic lives in `pepys_embedder.py`.

### Added

- **`pepys_embedder.py`** — standalone multi-process ingestion pipeline: parses a pipe-delimited diary file, applies temporally diverse subsampling, and embeds via `sentence-transformers` (`nomic-ai/nomic-embed-text-v1`) using `multiprocessing.Pool` (one worker per CPU core, each loading the model independently). Produces a JSON cache compatible with `pepys_manifold_explorer.py`. Replaces the in-explorer `--init` flow.
- **`nlp_ingestion_workflow.md`** — full pipeline documentation covering all four stages: raw-text parse (`pepys_proper_parse.py`), NLP enrichment (`DiaryTransformer` — spaCy diversity clustering + sentence-transformers segmentation + YAML TopicClassifier), multi-process embedding (`pepys_embedder.py`), and manifold analysis (`pepys_manifold_explorer.py`).
- **Claude Code memory** — `feedback_nlp_principles.md` records the project's NLP-first principle: prefer `sentence-transformers` / HuggingFace locally, minimise inference API calls.

### Changed

- **`pepys_manifold_explorer.py` — cache-only reader**: removed all embedding/ingestion code (`--init`, `--diary`, `--model`, `--max-chars` args, `embed_local()` call, partial-checkpoint logic). Explorer now requires a pre-built cache from `pepys_embedder.py`; prints the exact command if no cache is found.
- **`pepys_manifold_explorer.py` — temporal sampling on cache load**: `--n` now applies `temporally_sample()` against the full cached corpus, guaranteeing the subset spans the entire 1660–1669 arc rather than head-slicing.
- **Renamed** `benchmarks/pepys/COMPLETE_TECHNICAL_ARTICLE.md` → `personal_agent_pipeline_article.md` and `COMPLETE_TECHNICAL_ARTICLE_internal.md` → `personal_agent_pipeline_article_internal.md` to reflect their source (personal_agent pipeline research).

- **Pepys benchmark enriched embeddings** (`pepys_embeddings.json`) — precomputed nomic-embed-text-4k embeddings for the full Pepys diary corpus, cached for reproducible benchmark runs.
- **Pepys benchmark results** (`pepys_manifold_results.json`) — manifold flight metrics (MRL retrieval, ManifoldWalker path, observer heights/curvatures) from the enriched-corpus run.
- **5 new Pepys query topics** — Church/religion, Travel/locations, Money/finance, Social gathering, and Emotion/personal feelings added to `PEPYS_QUERIES` for broader semantic coverage of the diary corpus.
- **`_NumpyEncoder`** in `pepys_manifold_explorer.py` — custom `json.JSONEncoder` subclass that serialises numpy scalar and array types, preventing `TypeError` when saving results that include numpy integers/floats.

### Changed

- **`pepys_manifold_explorer.py` — enriched embedding format**: `parse_diary()` now prepends `entry_type | category |` to each diary entry before embedding, so topic-level signal from `pepys_enriched_full.txt` is preserved in the embedding space. Comment lines (`#`) and blank lines are now silently skipped.
- **`ManifoldModel` constructor** — `n_neighbors` parameter renamed to `k_graph` in all benchmark call-sites (`nomic_manifold_explorer.py`, `pepys_manifold_explorer.py`).
- **`ManifoldModel.set_position()`** → **`ManifoldModel.fly_to()`** — updated `pepys_manifold_explorer.py` to use the renamed positioning method.
- **`ManifoldObserver` constructor** — simplified from `ManifoldObserver(mm._graph, mm._geometries)` to `ManifoldObserver(mm)` in `pepys_manifold_explorer.py`.
- **Observer path API** — `flight_obs["mean_height"]` / `flight_obs["mean_curvature"]` replaced by averaging over raw `flight_obs["heights"]` / `flight_obs["curvatures"]` lists, matching the updated `observe_path()` return schema.
- **Code formatting** (`nomic_manifold_explorer.py`, `pepys_manifold_explorer.py`) — Black-style line-length fixes throughout; no logic changes.

- **`ManifoldObserver.observe_path()`** — "pen-down view from above": given a walker's traced node path, the observer measures per-hop height (reconstruction error from the local tangent plane) and curvature (principal angle between consecutive tangent subspaces), returning a structured dict that summarises the full trajectory from one dimension above.
- **`disulfide_manifold_flight.py`** benchmark — canonical end-to-end demonstration of manifold flight through the 5D disulfide torsional space (χ₁–χ₅) using the full proteusPy database (175 277 bonds). Selects the two most-distant **octant-class** centroids as origin and destination (the finest-grained, most distant landmarks in the hierarchy), flies the graph with `ManifoldWalker`, records per-hop class membership at all four hierarchy levels (binary/quadrant/sextant/octant), and feeds the path to `ManifoldObserver.observe_path()`.
- **`disulfide_flight_visualizer.py`** — post-run visualisation: trajectory plots, curvature/height profiles, class-boundary crossing charts.
- **`disulfide_flight_results.json` / `.png` / `_report.md`** — artefacts from the first full-database flight run (k=20, τ=0.9, w=0.8, 200-step budget, 992 s elapsed).
- **`mlp_expand_first.py`** benchmark — "expand-first" MLP architecture comparison baseline against `ManifoldModel` on structured biological data.
- **`nomic_manifold_explorer.py`** benchmark — intrinsic dimensionality analysis of nomic-embed-text-v1 embeddings (PCA elbow, Participation Ratio, TwoNN estimator) at all MRL checkpoints (64–768 D) with MRR@10 retrieval quality comparison.
- **WaveRider arXiv manuscript** (`docs/waverider/waverider_arxiv.tex` / `.pdf` / `_draft.md`) — full paper describing the four-layer WaveRider geometric ML stack: TurtleND → ManifoldWalker → ManifoldModel/ManifoldObserver.

### Changed

- **`ManifoldModel.fly_toward()`** — added `patience` parameter (default 5). With sparse graphs a greedy step can move sideways; `patience` lets the walker continue past brief non-improving detours rather than terminating at the first stall.
- **`pyproject.toml`** — extended pytest `--ignore` list to cover `turtlend_package/` so its tests do not run in the main test suite.
- **`.gitignore`** — added `proteusPy/data/PDB_SS_ALL_LOADER.pkl` to avoid committing the large binary loader cache.

### Fixed

- Added missing BSD/Flux-Frontiers header blocks to `manifold_observer.py`, `manifold_model.py`, `manifold_walker.py`, `turtleND.py`, and `disulfide_tree.py`.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
