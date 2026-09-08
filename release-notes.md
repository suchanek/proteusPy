# Release Notes — v0.100.2

> Released: 2026-09-08

A documentation release. The Springer book chapter built on proteusPy is now
published, and this release exists to say so where it matters most: on PyPI.

## Why this release exists

`pyproject.toml` sets `readme = "README.md"`, so the README is rendered as the
project's long description on PyPI — and PyPI does not allow the metadata of an
already-published release to be edited. The 0.100.1 page therefore still
announced the chapter as "going to print" and "in press," and linked the book
DOI, which resolves to a landing page that never names the chapter. Uploading a
new version is the only way to correct that.

**No library code changed.** Nothing under `proteusPy/` differs from v0.100.1,
and no runtime dependency moved, so upgrading changes nothing about how the
package behaves.

## What changed

**The chapter is published.** *Structural Analysis of Disulfide Bonds in the
RCSB Protein Data Bank Using proteusPy* is out as chapter 2, pp. 15–32, of
*Functional Disulphide Bonds: Methods and Protocols*, 2nd edition, edited by
Philip J. Hogg (Methods in Molecular Biology, vol. 3016; Springer US / Humana
Press, New York, NY, 2026). Print ISBN 978-1-0716-5157-5, eBook ISBN
978-1-0716-5158-2.

Every citation in the repository now points at the **chapter** DOI,
`10.1007/978-1-0716-5158-2_2`. The previously used `10.1007/978-1-0716-5158-2`
is the book DOI; it survives only as an explicit book-level SpringerLink
pointer. `docs/suchanek_disulfide_chapter_2026.pdf` is relabelled from a
pre-publication PDF to the author's accepted manuscript, now that a version of
record exists.

**Machine-readable citation.** `CITATION.cff` gains the chapter under
`references:`, so GitHub's "Cite this repository" panel and downstream citation
tooling surface it alongside the software. `preferred-citation` is deliberately
unchanged and still names the JOSS paper, which remains the correct citation
for the package itself. CFF 1.2.0 has no book-chapter reference type, so the
entry uses `type: generic` with the containing work expressed structurally:
`collection-title` for the series, `volume`/`volume-title` for the book, and
`section` for the chapter. The file validates against the CFF 1.2.0 schema.

**Tooling fix.** `scripts/rebuild-kg.sh` called `pycodekg build-lancedb`, a
subcommand retired along with LanceDB, so the script failed outright on its
second step. It now calls `build-index`, the sqlite-vec equivalent, which takes
the same `--repo`/`--wipe` flags. This is repository tooling and is not part of
the distributed package.

**Repository maintenance.** The fleet knowledge-graph packages were relocked to
current PyPI versions — `doc-kg` 0.22.0, `pycode-kg` 0.23.1, `ftree-kg` 0.14.0,
with `kgmodule-utils` 0.18.0 as a transitive resolution. These live in the
optional `kg` Poetry group, which is dev-only and never ships in the wheel, so
`pip install proteusPy` is unaffected. `.pre-commit-config.yaml` gained a
top-level `exclude: '^old/'`, and `.gitignore` dropped stale `lancedb/` rules
that named a path nothing produces since the sqlite-vec migration.

## Upgrading

```console
$ pip install --upgrade proteusPy
```

Nothing to do. There are no API changes, no behavioural changes, and no
migration steps.
