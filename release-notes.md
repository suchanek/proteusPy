# Release Notes -- v0.100.7

> Released: 2026-10-22

A maintenance release. The disulfide renderer is ready for PyVista 0.50, and
the RCSB viewer's Docker image is rebuilt on a slim Python base in place of
Miniconda.

## What changed

**PyVista 0.50 compatibility.** `DisulfideVisualization` passed `center` and
`direction` to `pv.Cylinder` by position. PyVista 0.49 warns about that, and
0.50 will raise `TypeError`, which would break bond rendering in every style
that draws cylinders. Both are now passed by keyword. Every rendering style
runs on PyVista 0.49.1 with no deprecation warnings.

**A smaller viewer image.** The `rcsb_viewer` image now starts from
`python:3.12-slim` and installs proteusPy and Panel with pip, with no conda.
Torch, which proteusPy needs through `pydssp`, comes from PyTorch's CPU-only
build; the default amd64 build adds several GB of CUDA libraries the viewer
never uses. The image is built for `linux/amd64` and `linux/arm64`. The
makefile targets keep the build cache, so rebuilds reuse unchanged layers.

**The viewer reports the real version.** The viewer's title bar and About
dialog showed a hard-coded `0.99.1`. They now show proteusPy's version.

**Docker documentation.** The README's Docker section builds with
`make docker`, corrects the `docker run` command, and states the memory need:
the loaded database peaks at about 7 GB, so Docker needs at least 8 GB.

## Upgrading

Run `pip install -U proteusPy`. No code changes are needed. To run the viewer
image, pull `egsuchanek/rcsb_viewer:latest` and give Docker at least 8 GB of
memory (Docker Desktop: **Settings > Resources > Memory**).

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
