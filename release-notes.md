# Release Notes — v0.100.1

> Released: 2026-08-21

This release clears every open Dependabot advisory, fixes proteusPy's citation
trail, and adds a standalone backbone-dihedral loader for WaveRider.

## What changed

**Security.** All outstanding advisories across every dependency group are
resolved. Two exact version pins — `notebook` and `panel` — were transitively
holding vulnerable `jupyterlab` and `bokeh` releases in place; both are now
relaxed to ranges that resolve past the affected versions. Pillow, Requests,
gdown, and python-multipart are bumped past their respective CVEs, and a
stale `old/pyproject.toml` left over from a January 2025 setuptools manifest
was deleted — it pinned an ancient Pillow and was the sole source of ten
Pillow advisories, since Dependabot scans every file named `pyproject.toml`
regardless of whether anything references it.

**Citation trail fixed.** The Zenodo DOI badge on the README was broken in
two independent ways: the badge image itself 502'd for every reader (Zenodo
rate-limits GitHub's image proxy), and the link it carried pointed at a
two-year-old version DOI instead of the stable concept DOI. Both are fixed —
the badge is now a shields.io static badge linking the concept DOI
(`10.5281/zenodo.11148440`) — and a `CITATION.cff` is added so GitHub's "Cite
this repository" button works, with the JOSS paper as the preferred
citation.

**Backbone dihedral extraction.** A new `BackboneLoader` module extracts
backbone φ/ψ/ω dihedral angles from PDB files in parallel, producing a flat
list of `BackboneResidue` records with secondary-structure annotation. Its
attribute interface is consumed directly by WaveRider's manifold-fitting
pipeline.

**Packaging.** The `doc-kg`, `pycode-kg`, and `ftree-kg` knowledge-graph
packages move from a PyPI `[extras]` entry — which PyPI forbids for
private, GitHub-hosted dependencies — to an optional Poetry dependency
group (`poetry install --with kg`). A plain `pip install proteusPy` is
unaffected.

## Upgrading

No action needed for a plain `pip install proteusPy` upgrade. Anyone
installing the `kg` extra should switch to `poetry install --with kg`
instead, since the extra no longer exists.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
