# Release Notes -- v0.100.4

> Released: 2026-10-05

A maintenance release. It fixes a dependency that a clean install was
missing, corrects the torsional energy equation as written in the
documentation, and moves the turtle classes to the standalone `turtlend`
package. The public API is unchanged and every computed value is the same.

## What changed

**Clean installs can load the full database.** `pyarrow` was used but never
declared, so `Load_PDB_SS(subset=False)` failed on a fresh install with
`ModuleNotFoundError`. It is now a main dependency.

**The energy equation in the docs is corrected.** The `Disulfide` docstring
and the `DisulfideAnalysis` notebook gave the `chi5` term a coefficient of 1.0.
The code has always used 2.0 for both `chi1` and `chi5`, so no energy changes;
only the written equation was wrong. The API documentation is regenerated to
match.

**`Turtle3D`, `TurtleND` and `Vector3D` now come from `turtlend`.** proteusPy
and WaveRider each carried a copy of these classes; the `turtlend` package is
now the single source. The old import paths still work, and built structures
are identical to nine decimal places. This also brings in three `Turtle3D`
fixes made in `turtlend`.

**Repository housekeeping.** The `kg` Poetry group and the FileTreeKG index are
gone, the `ruff` and `pytest` floors match the rest of the fleet, the README
gains a version badge and an APA citation, and stale "Last revision" date lines
are removed from source headers.

## Upgrading

Run `pip install -U proteusPy`. The upgrade pulls in `turtlend` and `pyarrow`.
No code changes are needed.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
