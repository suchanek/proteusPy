#!/usr/bin/env python3
# backbone_loader.py
# Copyright (c) 2026 Eric G. Suchanek, PhD, Flux-Frontiers
# https://github.com/Flux-Frontiers
# License: BSD
# Last revised: 2026-05-08 -egs-

"""
backbone_loader.py

Parallel loader that extracts backbone φ/ψ/ω dihedral angles from a directory
of PDB files and returns a flat list of ``BackboneResidue`` objects.

The resulting objects satisfy the interface expected by
``waverider.backbone_angles.BackboneAngleList.from_proteuspy()``:
each object exposes ``.phi``, ``.psi``, ``.omega``, ``.residue_name``,
``.chain_id``, ``.seq_pos``, ``.pdb_id``, and ``.secondary_structure``.

Terminal residues carry ``math.nan`` for undefined angles; WaveRider's
``.valid()`` filter removes them before manifold fitting.

Parsing a large PDB directory is slow (minutes for thousands of structures).
Use :meth:`BackboneLoader.save` to serialise the result to a zstd-compressed
Parquet file and :meth:`BackboneLoader.load_cache` to reload it instantly on
subsequent runs.  The cache preserves all fields verbatim — no information is
lost relative to a fresh parse.

Typical workflow
----------------
**First run** — parse the directory and write the cache::

    from proteusPy.backbone_loader import BackboneLoader

    loader = BackboneLoader(pdb_dir="/data/pdb/good", workers=8)
    residues = loader.load()                           # all pdb*.ent files
    BackboneLoader.save(residues, "/data/pdb/good_backbone.parquet")

**Subsequent runs** — skip parsing entirely::

    residues = BackboneLoader.load_cache("/data/pdb/good_backbone.parquet")

**Selective loads**::

    residues = loader.load(pdb_ids=["1abc", "2xyz"])   # specific structures
    residues = loader.load(chain_ids=["A"])             # chain filter
    residues = loader.load(max_files=100)               # cap file count (timing runs)

Performance notes
-----------------
- Workers are capped at 12 regardless of ``cpu_count()`` — beyond that,
  process-spawn overhead outweighs parallelism gains on typical NFS-backed
  PDB mirrors.
- Files are split into ``min(workers, n_files)`` chunks so each worker
  processes a contiguous batch rather than a single file, reducing pool
  dispatch overhead.
- A 200-file sample (~85 k residues) parses in ~15 s on a 12-core machine.
  The full ``good/`` mirror (~4 k structures) takes roughly 5–10 min;
  reloading the Parquet cache takes under 1 s.

Author: Eric G. Suchanek, PhD
Affiliation: Flux-Frontiers
License: BSD
Last revised: 2026-05-08 -egs-
"""

# pylint: disable=C0301

import logging
import math
import multiprocessing
from dataclasses import dataclass
from os import cpu_count
from pathlib import Path

import pandas as pd

from proteusPy.logger_config import create_logger
from proteusPy.vector3D import Vector3D, calc_dihedral

_logger = create_logger(__name__)


# ── BackboneResidue dataclass ─────────────────────────────────────────────────


@dataclass
class BackboneResidue:
    """Single-residue backbone geometry record.

    Intentionally parallel to ``Disulfide`` — same attribute-access pattern
    that WaveRider already uses for disulfide torsions.

    Terminal residues carry ``math.nan`` for undefined angles:

    - ``phi``:  ``math.nan`` for the N-terminal residue of each chain
    - ``psi``:  ``math.nan`` for the C-terminal residue of each chain
    - ``omega``: ``math.nan`` for the first residue of each chain

    :param phi: φ dihedral (degrees): C(i-1)–N(i)–Cα(i)–C(i)
    :param psi: ψ dihedral (degrees): N(i)–Cα(i)–C(i)–N(i+1)
    :param omega: ω dihedral (degrees): Cα(i-1)–C(i-1)–N(i)–Cα(i)
    :param residue_name: 3-letter residue code (e.g. 'ALA')
    :param chain_id: PDB chain identifier
    :param seq_pos: Residue sequence number from the ATOM record
    :param pdb_id: 4-character PDB ID (lowercase, e.g. '1abc')
    :param secondary_structure: H/E/C/U — from HELIX/SHEET records when
        available, else 'U' (unknown)
    """

    phi: float
    psi: float
    omega: float
    residue_name: str
    chain_id: str
    seq_pos: int
    pdb_id: str
    secondary_structure: str = "U"


# ── module-level helpers (must be pickleable for multiprocessing) ─────────────


def _build_ss_map(lines: list) -> dict:
    """Parse HELIX and SHEET records; return {(chain, seq_num): 'H'|'E'}.

    All HELIX classes (α, π, 3₁₀) map to 'H'.  SHEET strands map to 'E'.
    Residues absent from the map default to 'U' in the caller.
    """
    ss_map: dict = {}
    for line in lines:
        if line.startswith("HELIX"):
            try:
                chain = line[19].strip() or line[31].strip()
                start = int(line[21:25])
                end = int(line[33:37])
                for seq in range(start, end + 1):
                    ss_map[(chain, seq)] = "H"
            except (ValueError, IndexError):
                pass
        elif line.startswith("SHEET"):
            try:
                chain = line[21].strip() or line[32].strip()
                start = int(line[22:26])
                end = int(line[33:37])
                for seq in range(start, end + 1):
                    ss_map[(chain, seq)] = "E"
            except (ValueError, IndexError):
                pass
    return ss_map


def _parse_pdb_backbone(args: tuple) -> list:
    """Parse one PDB file; return a list of BackboneResidue. Module-level for pickle.

    :param args: ``(path, chain_filter)`` where *chain_filter* is
        ``list[str] | None``.  ``None`` includes all chains.
    :returns: List of BackboneResidue objects for every resolved residue that
        has complete backbone atoms (N, Cα, C).
    """
    path, chain_filter = args

    stem = Path(path).stem  # e.g. "pdb5rsa"
    pdb_id = stem[3:] if stem.startswith("pdb") else stem

    # atoms[(chain_id, seq_num)] = {"N": Vector3D, "CA": Vector3D, "C": Vector3D}
    atoms: dict = {}
    resnames: dict = {}  # (chain_id, seq_num) -> 3-letter residue name

    try:
        with open(path) as fh:
            lines = fh.readlines()
    except OSError:
        return []

    ss_map = _build_ss_map(lines)

    for line in lines:
        record = line[:6].strip()
        if record not in ("ATOM", "HETATM"):
            continue

        atom_name = line[12:16].strip()
        if atom_name not in ("N", "CA", "C"):
            continue

        # skip alternate conformations — first occurrence wins
        alt_loc = line[16]
        if alt_loc not in (" ", "A"):
            continue

        chain_id = line[21]
        if chain_filter and chain_id not in chain_filter:
            continue

        try:
            seq_num = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue

        res_name = line[17:20].strip()
        key = (chain_id, seq_num)

        if key not in atoms:
            atoms[key] = {}
            resnames[key] = res_name

        # first occurrence wins (handles alternate conformations)
        if atom_name not in atoms[key]:
            atoms[key][atom_name] = Vector3D(x, y, z)

    if not atoms:
        return []

    # group residues by chain, sorted by sequence number
    chains: dict = {}
    for chain_id, seq_num in sorted(atoms.keys(), key=lambda k: (k[0], k[1])):
        chains.setdefault(chain_id, []).append(seq_num)

    result: list = []

    for chain_id, seq_nums in chains.items():
        n_res = len(seq_nums)

        for i, seq_num in enumerate(seq_nums):
            key = (chain_id, seq_num)
            res_atoms = atoms[key]

            # skip residues missing any backbone heavy atom
            if not all(a in res_atoms for a in ("N", "CA", "C")):
                continue

            n_i = res_atoms["N"]
            ca_i = res_atoms["CA"]
            c_i = res_atoms["C"]

            # φ: C(i-1) – N(i) – Cα(i) – C(i)
            if i > 0:
                prev_atoms = atoms[(chain_id, seq_nums[i - 1])]
                phi = (
                    calc_dihedral(prev_atoms["C"], n_i, ca_i, c_i)
                    if "C" in prev_atoms
                    else math.nan
                )
            else:
                phi = math.nan

            # ψ: N(i) – Cα(i) – C(i) – N(i+1)
            if i < n_res - 1:
                next_atoms = atoms[(chain_id, seq_nums[i + 1])]
                psi = (
                    calc_dihedral(n_i, ca_i, c_i, next_atoms["N"])
                    if "N" in next_atoms
                    else math.nan
                )
            else:
                psi = math.nan

            # ω: Cα(i-1) – C(i-1) – N(i) – Cα(i)
            if i > 0:
                prev_atoms = atoms[(chain_id, seq_nums[i - 1])]
                omega = (
                    calc_dihedral(prev_atoms["CA"], prev_atoms["C"], n_i, ca_i)
                    if "CA" in prev_atoms and "C" in prev_atoms
                    else math.nan
                )
            else:
                omega = math.nan

            result.append(
                BackboneResidue(
                    phi=phi,
                    psi=psi,
                    omega=omega,
                    residue_name=resnames[key],
                    chain_id=chain_id,
                    seq_pos=seq_num,
                    pdb_id=pdb_id,
                    secondary_structure=ss_map.get(key, "U"),
                )
            )

    return result


def _parse_pdb_backbone_chunk(args: tuple) -> list:
    """Parse a list of PDB files; return a flat list of BackboneResidue.

    :param args: ``(paths, chain_filter)`` where *paths* is a list of file
        path strings and *chain_filter* is ``list[str] | None``.
    :returns: Flat list of BackboneResidue from all files in the chunk.
    """
    paths, chain_filter = args
    result: list = []
    for path in paths:
        result.extend(_parse_pdb_backbone((path, chain_filter)))
    return result


def _dssp_annotate_one(path: str) -> "tuple[str, dict]":
    """Run pydssp on one PDB file.

    :returns: ``(pdb_id, {(chain_id, seq_pos): ss_code})``.
    """
    try:
        import pydssp
    except ImportError:
        return ("", {})

    _C3_MAP = {"H": "H", "E": "E", "-": "C"}

    stem = Path(path).stem
    pdb_id = stem[3:] if stem.startswith("pdb") else stem

    try:
        with open(path) as fh:
            pdbtext = fh.read()
    except OSError:
        return (pdb_id, {})

    keys: list = []
    seen: set = set()
    for line in pdbtext.splitlines():
        if not line.startswith("ATOM"):
            continue
        resid = line[21:26]
        if resid in seen:
            continue
        seen.add(resid)
        chain_id = line[21]
        try:
            seq_pos = int(line[22:26])
        except ValueError:
            continue
        keys.append((chain_id, seq_pos))

    if not keys:
        return (pdb_id, {})

    try:
        coord = pydssp.pdbio.read_pdbtext_no_checking(pdbtext)
        if coord.ndim != 3 or coord.shape[1] != 4 or len(coord) != len(keys):
            return (pdb_id, {})
        labels = pydssp.assign(coord, out_type="c3")
    except Exception:
        return (pdb_id, {})

    return (pdb_id, {key: _C3_MAP.get(str(label), "C") for key, label in zip(keys, labels)})


def _dssp_annotate_chunk(args: tuple) -> dict:
    """Run pydssp on a list of PDB files; return merged {(pdb_id, chain, seq): ss}.

    :param args: ``(paths,)`` — list of path strings.
    :returns: Dict mapping ``(pdb_id, chain_id, seq_pos)`` to ss_code.
    """
    (paths,) = args
    merged: dict = {}
    for path in paths:
        pdb_id, per_file = _dssp_annotate_one(path)
        for (chain_id, seq_pos), ss in per_file.items():
            merged[(pdb_id, chain_id, seq_pos)] = ss
    return merged


def _worker_log_init() -> None:
    """Disable all logging in pool workers to avoid console noise."""
    logging.disable(logging.CRITICAL)


# ── BackboneLoader ────────────────────────────────────────────────────────────


class BackboneLoader:
    """Parallel loader for backbone dihedral angles from a PDB file directory.

    Follows the same multiprocessing pattern as
    ``DisulfideExtractor_mp.extract_disulfides_chunk``: a pool of workers
    parse individual ``.ent`` files and results are flattened into a single
    list.

    :param pdb_dir: Directory containing PDB files in ``pdb{id}.ent`` format.
    :param workers: Number of parallel worker processes.
        Defaults to ``os.cpu_count()``.

    Example::

        loader = BackboneLoader("/data/pdb/good", workers=8)
        residues = loader.load()
        # residues: list[BackboneResidue]
    """

    def __init__(self, pdb_dir: "str | Path", workers: int = None):
        self.pdb_dir = Path(pdb_dir)
        self.workers = min(workers if workers is not None else (cpu_count() or 4), 12)

    def load(
        self,
        pdb_ids: "list[str] | None" = None,
        chain_ids: "list[str] | None" = None,
        max_files: "int | None" = None,
    ) -> list:
        """Return a flat list of BackboneResidue from all requested PDB files.

        Terminal residues (undefined φ, ψ, or ω) are included with
        ``math.nan`` — WaveRider's ``.valid()`` filter removes them before
        manifold fitting.

        :param pdb_ids: 4-character PDB IDs to load.  ``None`` loads every
            ``pdb*.ent`` file found in *pdb_dir*.
        :param chain_ids: Chain identifiers to include.  ``None`` includes all
            chains.
        :param max_files: Cap the number of PDB files processed (useful for
            quick test runs).
        :returns: Flat list of BackboneResidue, one entry per resolved residue
            with complete backbone atoms.
        """
        if pdb_ids is not None:
            paths = [self.pdb_dir / f"pdb{pid.lower()}.ent" for pid in pdb_ids]
            paths = [p for p in paths if p.exists()]
        else:
            paths = sorted(self.pdb_dir.glob("pdb*.ent"))

        if not paths:
            _logger.warning("BackboneLoader: no PDB files found in %s", self.pdb_dir)
            return []

        if max_files is not None:
            paths = paths[:max_files]

        nworkers = min(self.workers, len(paths))
        chunk_size = max(1, len(paths) // nworkers)
        chunks = [paths[i : i + chunk_size] for i in range(0, len(paths), chunk_size)]
        chunk_args = [([str(p) for p in chunk], chain_ids) for chunk in chunks]

        print(
            f"  BackboneLoader: {len(paths):,} files → {len(chunks)} chunks / {nworkers} workers … ",
            end="",
            flush=True,
        )
        with multiprocessing.Pool(nworkers, initializer=_worker_log_init) as pool:
            results = pool.map(_parse_pdb_backbone_chunk, chunk_args)
        print("done.")

        flat: list = []
        for chunk in results:
            flat.extend(chunk)

        _logger.info(
            "BackboneLoader: loaded %d residues from %d files",
            len(flat),
            len(paths),
        )
        return flat

    # ------------------------------------------------------------------
    # Cache (Parquet)
    # ------------------------------------------------------------------

    @staticmethod
    def save(residues: list, path: "str | Path") -> None:
        """Serialise a list of BackboneResidue objects to a Parquet file.

        The file can be reloaded with :meth:`load_cache`, bypassing PDB
        parsing entirely on subsequent runs.

        :param residues: List of BackboneResidue objects (from :meth:`load`).
        :param path: Destination ``.parquet`` file path.
        """
        path = Path(path)
        df = pd.DataFrame(
            {
                "phi": [r.phi for r in residues],
                "psi": [r.psi for r in residues],
                "omega": [r.omega for r in residues],
                "residue_name": [r.residue_name for r in residues],
                "chain_id": [r.chain_id for r in residues],
                "seq_pos": [r.seq_pos for r in residues],
                "pdb_id": [r.pdb_id for r in residues],
                "secondary_structure": [r.secondary_structure for r in residues],
            }
        )
        df["seq_pos"] = df["seq_pos"].astype("int32")
        for col in ("residue_name", "chain_id", "secondary_structure"):
            df[col] = df[col].astype("category")

        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, compression="zstd")
        print(f"  BackboneLoader: saved {len(residues):,} residues → {path}")

    def build_cache(
        self,
        path: "str | Path",
        dssp: bool = False,
        batch_files: int = 2000,
        max_files: "int | None" = None,
    ) -> None:
        """Build a Parquet cache from the PDB directory in memory-bounded batches.

        Processes *batch_files* PDB files at a time — load, optionally DSSP-
        annotate, write to Parquet — then frees that batch before loading the
        next.  Peak memory is bounded to one batch rather than the full corpus.

        :param path: Destination ``.parquet`` file path.
        :param dssp: Run pydssp SS annotation on each batch before writing.
        :param batch_files: Number of PDB files per batch (default 2000).
        :param max_files: Cap total files processed (useful for test runs).
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        all_paths = sorted(self.pdb_dir.glob("pdb*.ent"))
        if max_files is not None:
            all_paths = all_paths[:max_files]

        n_total = len(all_paths)
        n_batches = max(1, (n_total + batch_files - 1) // batch_files)
        print(
            f"  build_cache: {n_total:,} files / {batch_files} per batch = {n_batches} batches → {path.name}"
        )

        writer = None
        total_residues = 0

        try:
            for b, start in enumerate(range(0, n_total, batch_files)):
                batch_paths = all_paths[start : start + batch_files]
                print(
                    f"  batch {b + 1}/{n_batches}  ({len(batch_paths)} files) … ",
                    end="",
                    flush=True,
                )

                # Load
                nworkers = min(self.workers, len(batch_paths))
                chunk_size = max(1, len(batch_paths) // nworkers)
                chunks = [
                    batch_paths[i : i + chunk_size] for i in range(0, len(batch_paths), chunk_size)
                ]
                chunk_args = [([str(p) for p in chunk], None) for chunk in chunks]
                with multiprocessing.Pool(nworkers, initializer=_worker_log_init) as pool:
                    results = pool.map(_parse_pdb_backbone_chunk, chunk_args)
                residues = [
                    r
                    for chunk in results
                    for r in chunk
                    if not (__import__("math").isnan(r.phi) or __import__("math").isnan(r.psi))
                ]

                # DSSP
                if dssp and residues:
                    dssp_paths = [str(p) for p in batch_paths]
                    dssp_chunk_size = max(1, len(dssp_paths) // nworkers)
                    dssp_chunks = [
                        dssp_paths[i : i + dssp_chunk_size]
                        for i in range(0, len(dssp_paths), dssp_chunk_size)
                    ]
                    dssp_args = [(chunk,) for chunk in dssp_chunks]
                    with multiprocessing.Pool(nworkers, initializer=_worker_log_init) as pool:
                        dssp_results = pool.map(_dssp_annotate_chunk, dssp_args)
                    ss_map: dict = {}
                    for d in dssp_results:
                        ss_map.update(d)
                    for r in residues:
                        ss = ss_map.get((r.pdb_id, r.chain_id, r.seq_pos))
                        if ss is not None:
                            r.secondary_structure = ss

                # Write batch to Parquet
                df = pd.DataFrame(
                    {
                        "phi": [r.phi for r in residues],
                        "psi": [r.psi for r in residues],
                        "omega": [r.omega for r in residues],
                        "residue_name": [r.residue_name for r in residues],
                        "chain_id": [r.chain_id for r in residues],
                        "seq_pos": [r.seq_pos for r in residues],
                        "pdb_id": [r.pdb_id for r in residues],
                        "secondary_structure": [r.secondary_structure for r in residues],
                    }
                )
                df["seq_pos"] = df["seq_pos"].astype("int32")
                table = pa.Table.from_pandas(df, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema, compression="zstd")
                writer.write_table(table)

                total_residues += len(residues)
                print(f"{len(residues):,} residues  (total {total_residues:,})")

        finally:
            if writer is not None:
                writer.close()

        print(f"  build_cache: done — {total_residues:,} residues → {path}")

    @staticmethod
    def load_cache(path: "str | Path") -> list:
        """Load BackboneResidue objects from a Parquet cache file.

        :param path: Path to a ``.parquet`` file written by :meth:`save`.
        :returns: Flat list of BackboneResidue, ready for
            ``BackboneAngleList.from_proteuspy()``.
        """
        path = Path(path)
        df = pd.read_parquet(path)
        residues = [
            BackboneResidue(
                phi=row.phi,
                psi=row.psi,
                omega=row.omega,
                residue_name=row.residue_name,
                chain_id=row.chain_id,
                seq_pos=int(row.seq_pos),
                pdb_id=row.pdb_id,
                secondary_structure=row.secondary_structure,
            )
            for row in df.itertuples(index=False)
        ]
        print(f"  BackboneLoader: loaded {len(residues):,} residues ← {path}")
        return residues

    def annotate_dssp(self, residues: list) -> list:
        """Re-annotate secondary structure using pydssp.

        Replaces the HELIX/SHEET-record SS labels set during :meth:`load` with
        coordinate-derived DSSP labels.  This eliminates the large ``U``
        (unknown) class caused by PDB files that omit HELIX/SHEET records.

        DSSP c3 labels map to the proteusPy single-letter scheme as:
        ``H`` (helix), ``E`` (strand), ``-`` → ``C`` (coil).

        Requires ``pydssp`` (``pip install pydssp``).  Files that fail to parse
        are left with their original SS labels.

        :param residues: List of BackboneResidue objects (from :meth:`load` or
            :meth:`load_cache`).
        :returns: The same list with ``secondary_structure`` fields updated
            in-place.

        Example::

            loader = BackboneLoader(pdb_dir="/data/pdb/good")
            residues = loader.load()
            residues = loader.annotate_dssp(residues)
            BackboneLoader.save(residues, "/data/pdb/good_dssp.parquet")
        """
        try:
            import pydssp  # noqa: F401
        except ImportError:
            raise ImportError(
                "pydssp is required for annotate_dssp.  Install with: pip install pydssp"
            )

        # Group residues by pdb_id so each file is parsed only once.
        from collections import defaultdict

        by_pdb: dict = defaultdict(list)
        for i, r in enumerate(residues):
            by_pdb[r.pdb_id].append(i)

        # Resolve file paths (same stem convention as load).
        paths = []
        for pdb_id in by_pdb:
            candidate = self.pdb_dir / f"pdb{pdb_id.lower()}.ent"
            if candidate.exists():
                paths.append(str(candidate))

        if not paths:
            _logger.warning("annotate_dssp: no matching .ent files found in %s", self.pdb_dir)
            return residues

        nworkers = min(self.workers, len(paths))
        chunk_size = max(1, len(paths) // nworkers)
        chunks = [paths[i : i + chunk_size] for i in range(0, len(paths), chunk_size)]
        chunk_args = [(chunk,) for chunk in chunks]

        print(
            f"  BackboneLoader.annotate_dssp: {len(paths):,} files "
            f"→ {len(chunks)} chunks / {nworkers} workers … ",
            end="",
            flush=True,
        )
        with multiprocessing.Pool(nworkers, initializer=_worker_log_init) as pool:
            results = pool.map(_dssp_annotate_chunk, chunk_args)
        print("done.")

        # Merge chunk dicts — already keyed by (pdb_id, chain_id, seq_pos).
        ss_map: dict = {}
        for chunk_result in results:
            ss_map.update(chunk_result)

        n_updated = 0
        for r in residues:
            ss = ss_map.get((r.pdb_id, r.chain_id, r.seq_pos))
            if ss is not None:
                r.secondary_structure = ss
                n_updated += 1

        print(f"  BackboneLoader.annotate_dssp: updated {n_updated:,} / {len(residues):,} residues")
        return residues
