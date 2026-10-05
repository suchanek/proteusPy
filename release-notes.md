# Release Notes -- v0.100.5

> Released: 2026-10-05

A dependency-only release. proteusPy now requires `turtlend` 0.1.1, the
current release of the package that provides `Turtle3D`, `TurtleND` and
`Vector3D`. 0.100.4 shipped still allowing and locking 0.1.0.

## What changed

**`turtlend` floor raised to 0.1.1.** Nothing in proteusPy's own code
changes, and the public API is the same. The full test suite passes against
`turtlend` 0.1.1.

## Upgrading

Run `pip install -U proteusPy`. It pulls in `turtlend` 0.1.1 if you have an
older copy. No code changes are needed.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
