"""
Export the disulfide database for The Knowledge Press: The Disulfides.

Writes the data the web orchard reads: ``catalog.json`` (groves, lines and
classes), one class card per line and class under ``classes/``, one bond
card per sampled member under ``bonds/``, and ``atoms.json`` (the atom
order, colors, radii and bond tables of ``DisulfideVisualization._render_ss``).

Only the bonds within ``CA_CUTOFF`` and ``SG_CUTOFF`` are exported, the set
the Springer chapter describes (158,965 bonds on the full database).
``Load_PDB_SS`` returns the prebuilt loader unfiltered (175,277 bonds), so
the export filters it here. The constants are the 95th percentiles of the
Ca and Sg distances, rounded; ``DisulfideLoader(percentile=95)`` uses them
unrounded and keeps 158,938 bonds (see ``ProteusGlobals``).

The class hierarchy is binary > quadrant > octant > members. The tiers are
computed here with :func:`proteusPy.disulfide_tree.classify_angles` at bases
2, 4 and 8, which partition each other exactly, and not through
``DisulfideTree``'s parent maps, whose sextant tier does not nest. Every class
id carries its level suffix (``b``, ``q``, ``o``).

Part of the program proteusPy, https://github.com/suchanek/proteusPy,
a Python package for the manipulation and analysis of macromolecules.

Author: Eric G. Suchanek, PhD
License: BSD
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from proteusPy._version import __version__
from proteusPy.atoms import ATOM_COLORS, ATOM_RADII_CPK, BOND_COLOR, BOND_RADIUS, BS_SCALE
from proteusPy.disulfide_tree import LEVEL_SUFFIXES, _circular_mean_deg, classify_angles
from proteusPy.DisulfideBase import Disulfide
from proteusPy.DisulfideLoader import Load_PDB_SS
from proteusPy.ProteusGlobals import (
    CA_CUTOFF,
    DATA_DIR,
    OCTANT_CLASS_METRICS_FILE,
    SG_CUTOFF,
    SS_CONSENSUS_BIN_FILE,
    SS_CONSENSUS_OCT_FILE,
)

#: The export format; bumped on a breaking change.
FORMAT = 1
CHI = ["chi1", "chi2", "chi3", "chi4", "chi5"]
CLASS_DEFINITIONS_FILE = "PDB_SS_class_definitions.csv"
RCSB_URL = "https://www.rcsb.org/structure/{}"

#: proteusPy's 16-atom order (``_render_ss``): each cysteine's N, CA, C, O,
#: CB, SG, then the proximal C before and N after, and the distal C before
#: and N after.
ATOM_NAMES = (
    "N",
    "CA",
    "C",
    "O",
    "CB",
    "SG",
    "N",
    "CA",
    "C",
    "O",
    "CB",
    "SG",
    "C",
    "N",
    "C",
    "N",
)
#: The keys into ``ATOM_COLORS`` and ``ATOM_RADII_CPK`` that ``_render_ss``
#: uses for the same atoms.
ATOM_ELEMENTS = ("N", "C", "C", "O", "C", "SG", "N", "C", "C", "O", "C", "SG", "C", "N", "C", "N")

#: ``_render_ss``'s full bond table; a bond with missing atoms draws rows 0-10.
BONDS_FULL = (
    (0, 1),
    (1, 2),
    (2, 3),
    (1, 4),
    (4, 5),
    (6, 7),
    (7, 8),
    (8, 9),
    (7, 10),
    (10, 11),
    (5, 11),
    (12, 0),
    (2, 13),
    (14, 6),
    (8, 15),
)
#: ``_render_ss``'s table for a modelled disulfide: no carbonyl O, which a
#: model built from torsions alone does not define.
BONDS_BACKBONE = (
    (0, 1),
    (1, 2),
    (1, 4),
    (4, 5),
    (6, 7),
    (7, 8),
    (7, 10),
    (10, 11),
    (5, 11),
)
MODEL_ATOMS = (0, 1, 2, 4, 5, 6, 7, 8, 10, 11)


def _r(x: float, nd: int = 3) -> float:
    """Round a float for JSON.

    :param x: The value.
    :param nd: Decimal places.
    :return: The rounded float.
    """
    return round(float(x), nd)


def class_ids(chis: np.ndarray) -> tuple[str, str, str]:
    """Return the binary, quadrant and octant class ids of one bond, with suffixes.

    :param chis: The five torsions in degrees.
    :return: ``(binary, quadrant, octant)``, for example ``("22222b", "4444q", ...)``.
    """
    return (
        classify_angles(chis, 2) + LEVEL_SUFFIXES["binary"],
        classify_angles(chis, 4) + LEVEL_SUFFIXES["quadrant"],
        classify_angles(chis, 8) + LEVEL_SUFFIXES["octant"],
    )


def circular_stats(angles: np.ndarray) -> tuple[list[float], list[float]]:
    """Circular mean and circular standard deviation of each torsion.

    :param angles: Shape ``(n, 5)``, degrees.
    :return: ``(means, stds)``, five each, in degrees.
    """
    mean = _circular_mean_deg(angles if len(angles) > 1 else angles.reshape(-1))
    rad = np.deg2rad(angles.reshape(-1, angles.shape[-1]))
    r = np.hypot(np.sin(rad).mean(axis=0), np.cos(rad).mean(axis=0))
    std = np.rad2deg(np.sqrt(-2.0 * np.log(np.clip(r, 1e-12, 1.0))))
    return [_r(a, 2) for a in mean], [_r(s, 2) for s in std]


def _centered(coords: np.ndarray, atoms: tuple[int, ...]) -> list[dict]:
    """Atoms as ``{name, element, x, y, z}``, centered on their own centroid.

    :param coords: The bond's internal coordinates, ``(16, 3)``.
    :param atoms: Indices into proteusPy's atom order to keep.
    :return: The atom records, in the order of ``atoms``.
    """
    xyz = coords[list(atoms)]
    xyz = xyz - xyz.mean(axis=0)
    return [
        {
            "name": ATOM_NAMES[i],
            "element": ATOM_ELEMENTS[i],
            "x": _r(p[0]),
            "y": _r(p[1]),
            "z": _r(p[2]),
        }
        for i, p in zip(atoms, xyz)
    ]


def molecule(ss: Disulfide) -> dict:
    """Ball-and-stick geometry for one disulfide, as ``_render_ss`` draws it.

    A model (built from torsions) has 10 atoms and the 9-row backbone table;
    a real bond with missing atoms 12 atoms and 11 bonds; a complete real
    bond 16 atoms and 15 bonds. Coordinates are proteusPy's internal frame
    (CA of the proximal cysteine at the origin), so every molecule faces the
    same way, centered on its atoms' centroid, in angstroms.

    :param ss: The disulfide.
    :return: ``{"atoms": [...], "bonds": [[i, j], ...]}``, bonds indexing ``atoms``.
    """
    coords = np.asarray(ss.internal_coords, dtype=float)
    if ss.modelled:
        keep = MODEL_ATOMS
        table = BONDS_BACKBONE
    elif ss.missing_atoms:
        keep = tuple(range(12))
        table = BONDS_FULL[:11]
    else:
        keep = tuple(range(16))
        table = BONDS_FULL
    where = {a: n for n, a in enumerate(keep)}
    return {
        "atoms": _centered(coords, keep),
        "bonds": [[where[a], where[b]] for a, b in table],
    }


def model_from_torsions(name: str, chis: list[float]) -> dict:
    """Build a model disulfide from five torsions and return its geometry.

    :param name: The model's name.
    :param chis: chi1..chi5 in degrees.
    :return: As :func:`molecule`.
    """
    return molecule(Disulfide(name, torsions=list(chis)))


def atoms_table() -> dict:
    """The rendering tables the web shares with ``_render_ss``.

    :return: The ``atoms.json`` document.
    """
    return {
        "order": list(ATOM_NAMES),
        "elements": list(ATOM_ELEMENTS),
        "colors": dict(ATOM_COLORS),
        "radii": dict(ATOM_RADII_CPK),
        "bs_scale": BS_SCALE,
        "bond_radius": BOND_RADIUS,
        "bond_color": BOND_COLOR,
        "bonds_full": [list(b) for b in BONDS_FULL],
        "bonds_model": [list(b) for b in BONDS_BACKBONE],
    }


def _rcsb(key: str) -> str:
    """The RCSB entry page for a disulfide key such as ``2q7q_75D_140D``.

    :param key: The disulfide's name.
    :return: The URL.
    """
    return RCSB_URL.format(key.split("_")[0].upper())


def _grove_of(name: str, class_id: str) -> tuple[str, str]:
    """The grove (class family) of a binary class.

    :param name: ``SS_Classname`` from the class definitions, such as ``+/-RHHook``.
    :param class_id: The five binary digits.
    :return: ``(grove id, grove name)``: ``("rhhook", "RHHook")`` for a named
        class, ``("b-00002", "00002b")`` for an unnamed one.
    """
    stem = name.lstrip("+-/")
    if not stem or stem.isdigit():
        return f"b-{class_id}", f"{class_id}b"
    return stem.lower(), stem


def _summary(rows: pd.DataFrame, total: int) -> dict:
    """Count, percentage, consensus and spreads of one class's members.

    :param rows: The members' rows of the torsion table.
    :param total: Bonds in the database.
    :return: The shared class fields.
    """
    chis = rows[CHI].to_numpy(dtype=float)
    mean, std = circular_stats(chis)
    return {
        "count": len(rows),
        "percentage": _r(100.0 * len(rows) / total, 4),
        "consensus": {"chi": mean, "std": std},
        "energy": {"mean": _r(rows.energy.mean()), "std": _r(rows.energy.std(ddof=0))},
        "ca_distance": {
            "mean": _r(rows.ca_distance.mean()),
            "std": _r(rows.ca_distance.std(ddof=0)),
        },
    }


def _bond_card(ss: Disulfide, ids: tuple[str, str, str]) -> dict:
    """The bond card for one real disulfide.

    :param ss: The disulfide.
    :param ids: Its binary, quadrant and octant class ids.
    :return: The ``bonds/<key>.json`` document.
    """
    return {
        "key": ss.name,
        "pdb_id": ss.pdb_id,
        "proximal": ss.proximal,
        "distal": ss.distal,
        "proximal_chain": ss.proximal_chain,
        "distal_chain": ss.distal_chain,
        "resolution": _r(ss.resolution, 2),
        "torsions": [_r(a, 2) for a in ss.torsion_array],
        "energy": _r(ss.energy),
        "ca_distance": _r(ss.ca_distance),
        "sg_distance": _r(ss.sg_distance),
        "classes": {"binary": ids[0], "quadrant": ids[1], "octant": ids[2]},
        **molecule(ss),
        "rcsb": _rcsb(ss.name),
    }


def build(
    loader,
    database: str,
    min_members: int | None = None,
    members_per_class: int = 12,
    home: str | None = None,
    ca_cutoff: float = CA_CUTOFF,
    sg_cutoff: float = SG_CUTOFF,
) -> tuple[dict, dict[str, dict], dict[str, dict]]:
    """Build the catalog, class cards and bond cards in memory.

    Only the bonds within the distance cutoffs are exported, by the rule of
    ``DisulfideList.filter_by_distance``: 0 < distance < cutoff. The defaults
    are proteusPy's ``CA_CUTOFF`` and ``SG_CUTOFF``, which give the published
    set on the full database (158,965 bonds in 35,367 structures).

    :param loader: A ``DisulfideLoader`` (``Load_PDB_SS``), unfiltered as it is distributed.
    :param database: ``"subset"`` or ``"full"``, recorded in the catalog.
    :param min_members: Octant classes with fewer members fold into an
        ``<quadrant id>-other`` tip; default 2 on the subset, 20 on the full database.
    :param members_per_class: Members sampled per octant class, lowest energy first.
    :param home: The home disulfide's key; default the lowest-energy bond.
    :param ca_cutoff: Largest Cα-Cα distance kept, Å; -1 keeps every bond.
    :param sg_cutoff: Largest Sγ-Sγ distance kept, Å; -1 keeps every bond.
    :return: ``(catalog, class cards by id, bond cards by key)``.
    """
    if min_members is None:
        min_members = 2 if database == "subset" else 20
    # Row labels stay the bonds' positions in loader.SSList, for the bond cards.
    df = loader.TorsionDF.reset_index(drop=True)
    unfiltered = len(df)
    if ca_cutoff > 0:
        df = df[(df.ca_distance > 0) & (df.ca_distance < ca_cutoff)]
    if sg_cutoff > 0:
        df = df[(df.sg_distance > 0) & (df.sg_distance < sg_cutoff)]
    total = len(df)
    chis = df[CHI].to_numpy(dtype=float)
    tiers = [class_ids(row) for row in chis]
    df = df.assign(
        binary=[t[0] for t in tiers],
        quadrant=[t[1] for t in tiers],
        octant=[t[2] for t in tiers],
    )

    defs = pd.read_csv(Path(DATA_DIR) / CLASS_DEFINITIONS_FILE, dtype=str)
    with open(Path(DATA_DIR) / SS_CONSENSUS_BIN_FILE, "rb") as fh:
        bin_models = {ss.name: ss for ss in pickle.load(fh)}
    with open(Path(DATA_DIR) / SS_CONSENSUS_OCT_FILE, "rb") as fh:
        oct_models = {ss.name: ss for ss in pickle.load(fh)}
    oct_metrics = pd.read_pickle(Path(DATA_DIR) / OCTANT_CLASS_METRICS_FILE)
    oct_means = {
        f"{row['class']}o": [float(row[f"{c}_mean"]) for c in CHI]
        for _, row in oct_metrics.iterrows()
    }

    home_key = home or str(df.ss_id[df.energy.idxmin()])
    if not (df.ss_id == home_key).any():
        raise ValueError(f"home disulfide {home_key} is not among the bonds within the cutoffs")
    lines: list[dict] = []
    classes: list[dict] = []
    cards: dict[str, dict] = {}
    sampled: dict[str, tuple[int, tuple[str, str, str]]] = {}
    groves: dict[str, dict] = {}

    for _, d in defs.iterrows():
        bid = f"{d.class_id}b"
        grove_id, grove_name = _grove_of(d.SS_Classname, d.class_id)
        function = "" if d.FXN == "UNK" else d.FXN
        name = d.SS_Classname if not d.SS_Classname.isdigit() else bid
        rows = df[df.binary == bid]
        line = {
            "id": bid,
            "name": name,
            "grove": grove_id,
            "function": function,
            **(_summary(rows, total) if len(rows) else {"count": 0, "percentage": 0.0}),
            "model": molecule(bin_models[bid]),
        }
        lines.append(line)
        grove = groves.setdefault(
            grove_id, {"id": grove_id, "name": grove_name, "functions": [], "lines": []}
        )
        grove["lines"].append(bid)
        if function and function not in grove["functions"]:
            grove["functions"].append(function)

        line_children = []
        for qid, qrows in sorted(rows.groupby("quadrant"), key=lambda g: (-len(g[1]), g[0])):
            line_children.append(qid)
            quad = {"id": qid, "level": "quadrant", "parent": bid, "line": bid}
            quad.update(_summary(qrows, total))
            q_children: list[str] = []
            folded: list[dict] = []
            octs = sorted(qrows.groupby("octant"), key=lambda g: (-len(g[1]), g[0]))
            for oid, orows in octs:
                if len(orows) < min_members:
                    folded.append({"id": oid, "count": len(orows)})
                    continue
                q_children.append(oid)
                take = orows.sort_values(["energy", "ss_id"]).head(members_per_class)
                members = [{"key": str(r.ss_id), "energy": _r(r.energy)} for r in take.itertuples()]
                for idx, r in zip(take.index, take.itertuples()):
                    sampled[str(r.ss_id)] = (idx, (bid, qid, oid))
                octant = {"id": oid, "level": "octant", "parent": qid, "line": bid}
                octant.update(_summary(orows, total))
                if oid in oct_models:
                    octant["model"] = molecule(oct_models[oid])
                    octant["model_source"] = "consensus"
                elif oid in oct_means:
                    octant["model"] = model_from_torsions(oid, oct_means[oid])
                    octant["model_source"] = "metrics"
                else:
                    octant["model"] = model_from_torsions(oid, octant["consensus"]["chi"])
                    octant["model_source"] = "members"
                octant["members"] = members
                classes.append(octant)
            if folded:
                other_id = f"{qid}-other"
                q_children.append(other_id)
                classes.append(
                    {
                        "id": other_id,
                        "level": "other",
                        "parent": qid,
                        "line": bid,
                        "count": sum(f["count"] for f in folded),
                        "percentage": _r(100.0 * sum(f["count"] for f in folded) / total, 4),
                        "folded": folded,
                    }
                )
            quad["children"] = q_children
            classes.append(quad)
        line["children"] = line_children

    for line in lines:
        cards[line["id"]] = {"level": "binary", **line}
    for c in classes:
        card = dict(c)
        if c["level"] == "octant":
            card["members"] = [
                {**m, "pdb_id": m["key"].split("_")[0], "rcsb": _rcsb(m["key"])}
                for m in c["members"]
            ]
        cards[c["id"]] = card
    # The catalog lists quadrants before their children; the web derives
    # children from each class's parent.
    ordered = [c for c in classes if c["level"] == "quadrant"]
    ordered += [c for c in classes if c["level"] != "quadrant"]
    ordered = [{k: v for k, v in c.items() if k != "children"} for c in ordered]

    if home_key not in sampled:
        idx = int(df.index[df.ss_id == home_key][0])
        row = df.loc[idx]
        sampled[home_key] = (idx, (row.binary, row.quadrant, row.octant))
    bonds: dict[str, dict] = {}
    for key, (idx, ids) in sampled.items():
        ss = loader.SSList[idx]
        if ss.name != key:
            raise ValueError(f"torsion table row {idx} is {key}, the list has {ss.name}")
        bonds[key] = _bond_card(ss, ids)

    line_count = {ln["id"]: ln["count"] for ln in lines}
    grove_list = []
    for g in groves.values():
        g["lines"].sort(key=lambda bid: (-line_count[bid], bid))
        count = sum(line_count[b] for b in g["lines"])
        grove_list.append(
            {
                "id": g["id"],
                "name": g["name"],
                "function": ", ".join(g["functions"]),
                "lines": g["lines"],
                "count": count,
                "percentage": _r(100.0 * count / total, 4),
                "named": not g["id"].startswith("b-"),
            }
        )
    # Named families first, each tier by occupancy.
    grove_list.sort(key=lambda g: (not g["named"], -g["count"], g["id"]))

    catalog = {
        "meta": {
            "format": FORMAT,
            "generator": f"proteusPy {__version__}",
            "generated": datetime.now(UTC).isoformat(timespec="seconds"),
            "database": {
                "name": database,
                "bonds": total,
                "pdb_entries": int(df.source.nunique()),
                "unfiltered_bonds": unfiltered,
                "filter": {
                    "ca_cutoff": ca_cutoff if ca_cutoff > 0 else None,
                    "sg_cutoff": sg_cutoff if sg_cutoff > 0 else None,
                },
            },
            "min_members": min_members,
            "members_per_class": members_per_class,
            "counts": {
                "groves": len(grove_list),
                "lines": len(lines),
                "quadrant_classes": sum(c["level"] == "quadrant" for c in classes),
                "octant_classes": sum(c["level"] == "octant" for c in classes),
                "other_tips": sum(c["level"] == "other" for c in classes),
                "members_sampled": sum(len(c.get("members", [])) for c in classes),
                "members_total": total,
            },
        },
        "groves": grove_list,
        "lines": [{k: v for k, v in ln.items() if k != "children"} for ln in lines],
        "classes": ordered,
        "home": home_key,
    }
    return catalog, cards, bonds


def _write_json(path: Path, data) -> None:
    """Write compact JSON.

    :param path: Destination file.
    :param data: The document.
    """
    path.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")


def write(out: Path, catalog: dict, cards: dict, bonds: dict, force: bool = False) -> None:
    """Write an export into ``out``, replacing it whole.

    :param out: The forest directory, such as ``web/public/forests/subset``.
    :param catalog: From :func:`build`.
    :param cards: Class cards by id.
    :param bonds: Bond cards by key.
    :param force: Replace an existing ``out``.
    :raises FileExistsError: If ``out`` exists and ``force`` is not set.
    """
    out = Path(out)
    if out.exists() and not force:
        raise FileExistsError(f"{out} exists; pass --force to replace it")
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "classes").mkdir(parents=True)
    (tmp / "bonds").mkdir()
    _write_json(tmp / "catalog.json", catalog)
    _write_json(tmp / "atoms.json", atoms_table())
    for cid, card in cards.items():
        _write_json(tmp / "classes" / f"{cid}.json", card)
    for key, card in bonds.items():
        _write_json(tmp / "bonds" / f"{key}.json", card)
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)


def main(argv: list[str] | None = None) -> int:
    """Command line: ``proteusPy.export_web --database subset --out DIR``.

    :param argv: Arguments; default ``sys.argv[1:]``.
    :return: The exit status.
    """
    parser = argparse.ArgumentParser(
        description="Export the disulfide database for The Knowledge Press: The Disulfides.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--database", choices=["subset", "full"], default="subset")
    parser.add_argument("--out", type=Path, required=True, help="Forest directory to write.")
    parser.add_argument(
        "--min-members",
        type=int,
        default=None,
        help="Fold smaller octant classes into an 'other' tip (default 2 subset, 20 full).",
    )
    parser.add_argument("--members-per-class", type=int, default=12)
    parser.add_argument("--home", default=None, help="Home disulfide key (default lowest energy).")
    parser.add_argument(
        "--ca-cutoff",
        type=float,
        default=CA_CUTOFF,
        help="Keep bonds with Ca-Ca distance below this, A; -1 keeps every bond.",
    )
    parser.add_argument(
        "--sg-cutoff",
        type=float,
        default=SG_CUTOFF,
        help="Keep bonds with Sg-Sg distance below this, A; -1 keeps every bond.",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing export.")
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    loader = Load_PDB_SS(subset=args.database == "subset", verbose=False)
    t1 = time.perf_counter()
    catalog, cards, bonds = build(
        loader,
        args.database,
        min_members=args.min_members,
        members_per_class=args.members_per_class,
        home=args.home,
        ca_cutoff=args.ca_cutoff,
        sg_cutoff=args.sg_cutoff,
    )
    write(args.out, catalog, cards, bonds, force=args.force)
    t2 = time.perf_counter()
    counts = catalog["meta"]["counts"]
    print(
        f"{args.out}: {catalog['meta']['database']['bonds']} bonds "
        f"(of {catalog['meta']['database']['unfiltered_bonds']} before the cutoffs), "
        f"{counts['groves']} groves, {counts['lines']} lines, "
        f"{counts['quadrant_classes']} quadrant and {counts['octant_classes']} octant classes, "
        f"{counts['members_sampled']} members sampled; home {catalog['home']}; "
        f"load {t1 - t0:.1f} s, export {t2 - t1:.1f} s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
