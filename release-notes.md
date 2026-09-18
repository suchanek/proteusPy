# Release Notes — v0.100.3

> Released: 2026-09-18

The disulfide database and prebuilt loaders move off git entirely and onto
GitHub Release assets, with real checksums now published. CI and release
automation land for the first time. Packaging metadata converts to PEP 621,
matching the rest of the fleet, and a tornado transitive dependency is bumped
to close three Dependabot alerts.

## Why this release exists

v0.100.2 was documentation-only; nothing under `proteusPy/` had changed since
v0.100.1. This release is the opposite: it's the first to ship the data-asset
distribution work, the first built and published through the new CI/release
pipeline, and the first with PEP 621 packaging metadata.

## What changed

### Data distribution is now live

The large `.pkl` files — the disulfide database (`PDB_all_ss.pkl`, 457 MB) and
the prebuilt loaders (`PDB_SS_ALL_LOADER.pkl`, 482 MB; `PDB_SS_SUBSET_LOADER.pkl`,
14 MB) — are GitHub Release assets on a dedicated `data-v1.0` tag rather than
git-lfs objects. git-lfs bandwidth used to be exhausted by a single clone of
this repo; release-asset downloads are unmetered and need no credentials.

`proteusPy/data_fetch.py` is new: `fetch_data_file()` streams a download to a
`.part` file, verifies it against a sha256 recorded in `DATA_RELEASE_SHA256`,
and only then moves it into place, so an interrupted transfer can't leave a
truncated pickle where the loader expects a whole one. `Load_PDB_SS()` fetches
the prebuilt loader directly now instead of rebuilding it — 14 MB for
`subset=True` against the 457 MB master list and the minutes a rebuild used to
cost. Google Drive is a fallback if the release asset can't be fetched, not
the primary source; `gdown` is imported lazily, only on that path.

The `data-v1.0` release is published with all three assets and their real
sha256 checksums in `proteusPy/ProteusGlobals.py` — this release is what
completes that: the release existed but `DATA_RELEASE_SHA256` still held empty
placeholders until now. Verified by fetching all three into a clean directory
and confirming each downloads and checksum-verifies. The four small
consensus/metrics files (`SS_consensus_class_32.pkl`, `SS_consensus_class_oct.pkl`,
`binary_class_metrics.pkl`, `octant_class_metrics.pkl`) are restored as
ordinary git blobs, since their git-lfs objects are gone from the server for
good (`410 Object does not exist`) and what remained in the tree was 130-byte
pointer text under a `.pkl` name — worse than absent, since the loader would
try to unpickle a pointer instead of fetching the real file.

### CI and release automation

`.github/workflows/ci.yml` and `release.yml` are new, adapted from the doc_kg
templates. CI runs on every push and pull request to `master`: `ruff format
--check` plus `ruff check`, the pytest suite, and a wheel smoke test that
builds the package, installs it into an empty venv, and imports it, so a wheel
missing a data file or a dependency fails here rather than on a user's
machine. Release fires on `v*` tags: builds once, creates the GitHub Release
with this file as the body, and publishes the same artifacts to PyPI. The old
`pytest.yml` workflow and a `publish-to-pypi.yml` that lived outside
`workflows/` and so never ran are both removed.

CI's first run caught a real bug: `programs/DisulfideCluster.py` and
`programs/DisulfidePruner.py` imported `proteusPy.proteusGlobals`; the module
is `ProteusGlobals`. macOS's case-insensitive filesystem had hidden the
mismatch — ruff resolved it as a first-party import locally — but on the
Linux CI runner it doesn't resolve, and both scripts would have raised
`ModuleNotFoundError` for anyone running them.

### Packaging converts to PEP 621

`pyproject.toml` metadata that only Poetry read — name, version, description,
dependencies, extras, scripts, urls, classifiers — now lives under `[project]`
and `[project.optional-dependencies]`, matching every other repo in the fleet.
`license = "BSD"` becomes the SPDX identifier `"BSD-3-Clause"` with
`license-files = ["LICENSE"]`, replacing the ambiguous
`License :: OSI Approved :: BSD License` / `License :: Other/Proprietary
License` classifier pair PyPI has been showing with an unambiguous
`License-Expression` in the wheel metadata. A built wheel and sdist were
diffed against a build from before this change: only the license fields, two
added Python-version classifiers, and the extra-marker form differ — no
dependency changed. `[tool.poetry.group.kg]` floors are raised to current
PyPI: `doc-kg` 0.22.0 → 0.26.0, `pycode-kg` 0.23.1 → 0.27.0, `ftree-kg` 0.14.0
→ 0.16.0.

### Security

`poetry.lock`: tornado 6.5.7 → 6.5.9, closing three Dependabot alerts
(GHSA-wwv5-g3v4-889x, GHSA-8423-8fgw-73vq, GHSA-mpf4-983q-p7j4 — cookie
attribute injection, multipart memory amplification, and an event-loop stall
from urlencoded body parsing). tornado is not a direct dependency; it arrives
transitively through bokeh and the jupyter stack, both optional groups, so
`pip install proteusPy` is unaffected.

### Repository tooling

`.pre-commit-config.yaml` now pins `ruff-pre-commit` at v0.16.0, the version
Poetry resolves, instead of v0.9.10 — the two had drifted far enough apart
that the old hook enforced a rule newer ruff has retired, so a tree that
passed `poetry run ruff check` could still fail at commit time. 41 files were
reformatted at the 100-column width `pyproject.toml` already declared; every
one was AST-compared against the prior version, and only
`programs/compare_class_disulfides.py` has a behavioral-looking diff, where
pyupgrade replaced `typing.Dict` with `dict`. `.gitignore` now excludes
`**/.agentkg/`, matching the rest of the fleet.

### The repository clones normally again

`.gitattributes` still routed notebooks through Git LFS after the data moved
off it, and GitHub no longer serves this repository's LFS objects, so a plain
`git clone` failed at checkout. Git LFS is now gone from the repository: the
19 files it tracked are ordinary blobs, two very large notebooks had their
outputs cleared, and `data/PDB_SS_classes_master2.csv` holds its data rather
than LFS pointer text. Checking out a commit from before this release still
needs `GIT_LFS_SKIP_SMUDGE=1`.

The pre-commit hook now runs the quality checks first, and rebuilds the KG
indices and saves snapshots only when `PROTEUSPY_SNAPSHOT=1` is set. The 27
snapshots the old per-commit hook wrote are removed.

## Upgrading

```console
$ pip install --upgrade proteusPy
```

No API changes. The one behavioral difference: `Load_PDB_SS()` now downloads
the prebuilt loader from the `data-v1.0` release on first use instead of
rebuilding it locally, which is faster and no longer requires the 457 MB
master list to be present up front.
