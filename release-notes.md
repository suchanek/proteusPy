# Release Notes -- v0.101.0

> Released: 2026-10-26

A feature release. proteusPy can now export its disulfide database for the
web, as the data behind The Knowledge Press: The Disulfides, a 3-D orchard in
which every structural class is a tree. The export holds the published set of
disulfides, exactly the bonds the Springer chapter describes.

## What changed

**A web export of the database.** The new console script
`proteusPy.export_web --database subset|full --out DIR` writes the class
hierarchy as JSON: the 32 binary classes, their quadrant and octant classes,
the class families as groves, a sample of each octant class's real bonds as
bond cards with their coordinates, and the rendering tables proteusPy draws
disulfides with. The tiers are binary, quadrant and octant, computed with
`classify_angles` at bases 2, 4 and 8, which nest exactly; the sextant tier
does not nest under the quadrant tier and is left out. Every class carries the
consensus torsions, energy and Ca distance of all its bonds, and binary and
octant classes carry their consensus model disulfides. The full export loads
in about 5 s and writes about 52 MB.

**The published set, exactly.** `Load_PDB_SS` returns its prebuilt loader
unfiltered, 175,277 bonds on the full database. The export keeps only the bonds
within `CA_CUTOFF` (6.71 A) and `SG_CUTOFF` (2.12 A): 158,965 bonds in 35,367
structures, the chapter's numbers to the bond, and 4,575 of the subset's 5,000.
`--ca-cutoff` and `--sg-cutoff` change the cutoffs, and -1 keeps every bond.
The constants are the full database's 95th percentiles of the Ca and Sg
distances, rounded; `DisulfideLoader(percentile=95)` uses them unrounded and
keeps 158,938 bonds, the count in the analysis notebooks. `ProteusGlobals`
now documents the difference.

**Two programs run again.** `programs/class_analysis_mp.py` and
`programs/test_hexbin.py` still passed cutoff arguments `Load_PDB_SS` stopped
taking in v0.99.34, so both failed at once. The class analysis now builds its
loader with `CA_CUTOFF` and `SG_CUTOFF`: the published set, with 8,534 octant
classes, in about 24 s from the master list.

## Upgrading

Run `pip install -U proteusPy`. No code changes are needed. To export the
database for the web, run `proteusPy.export_web --database full --out DIR`;
the first run downloads the prebuilt loader if it is not already local.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
