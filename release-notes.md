# Release Notes -- v0.101.1

> Released: 2026-10-26

A security release. It changes no code; it raises the floors of two
dependencies and relocks five more, patching every open Dependabot alert and
one vulnerability a full `pip-audit` found that Dependabot had not flagged.

## What changed

**Pillow 13.** Pillow carried 41 alerts, 5 of them critical. The published
0.101.0 declares `pillow>=12.3.0`, so a fresh install could resolve to a
vulnerable Pillow; 0.101.1 declares `pillow>=13.0.0`. proteusPy's own Pillow
calls (`Image.open`, `Resampling`, `ImageFont`) work unchanged, and the test
suite passes on it.

**setuptools and the lock.** The setuptools floor is now 84.1.0. The lock
moves certifi to 2026.10.23 (one critical alert), filelock to 4.2.4, and the
development tools virtualenv to 21.14.13, nbconvert to 7.17.2 and
jupyter-server to 2.21.1 (PYSEC-2026-4054).

**Checked on a full install.** With every dependency group and extra
installed, `pip-audit` finds no known vulnerabilities in the 246 installed
packages, nor in the locked versions of the 25 Linux, CUDA and Windows
packages a macOS install leaves out. `pip check` finds no broken
requirements.

## Upgrading

Run `pip install -U proteusPy`. No code changes are needed.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
