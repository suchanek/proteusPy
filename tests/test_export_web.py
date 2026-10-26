"""
Tests for proteusPy.export_web, the data export for The Knowledge Press:
The Disulfides, on the 5,000-bond subset (4,575 within the cutoffs).

Author: Eric G. Suchanek, PhD
"""

import json
import re

import pandas as pd
import pytest

from proteusPy import Load_PDB_SS
from proteusPy.export_web import build, main, write
from proteusPy.ProteusGlobals import CA_CUTOFF, DATA_DIR, SG_CUTOFF

CLASS_ID = re.compile(r"^(\d{5}[bqo]|[1-4]{5}q-other)$")


@pytest.fixture(scope="module")
def loader():
    return Load_PDB_SS(subset=True, verbose=False)


@pytest.fixture(scope="module")
def export(loader):
    return build(loader, "subset")


def _quadrant_digit(octant_digit: int) -> int:
    """The quadrant digit whose 90-degree segment holds an octant segment."""
    return 4 - (8 - octant_digit) // 2


def _binary_digit(octant_digit: int) -> int:
    """The binary digit (2 positive, 0 negative) of an octant segment."""
    return 2 if octant_digit > 4 else 0


def _parent_by_digit(oid: str) -> tuple[str, str]:
    digits = [int(c) for c in oid[:5]]
    q = "".join(str(_quadrant_digit(d)) for d in digits) + "q"
    b = "".join(str(_binary_digit(d)) for d in digits) + "b"
    return q, b


def test_tiers_match_the_loader_strings(loader, export):
    """The computed binary and octant ids agree with the torsion table's own."""
    _, _, bonds = export
    df = loader.TorsionDF.set_index("ss_id")
    for key, card in bonds.items():
        assert card["classes"]["binary"] == df.loc[key, "binary_class_string"] + "b"
        assert card["classes"]["octant"] == df.loc[key, "octant_class_string"] + "o"


def test_partition_invariant(export):
    """Octant > quadrant > binary nest by digit, and counts sum down every limb."""
    catalog, _, bonds = export
    total = catalog["meta"]["database"]["bonds"]
    lines = {ln["id"]: ln for ln in catalog["lines"]}
    classes = {c["id"]: c for c in catalog["classes"]}
    assert len(classes) == len(catalog["classes"]), "a class appears twice"

    for card in bonds.values():
        q, b = _parent_by_digit(card["classes"]["octant"])
        assert card["classes"]["quadrant"] == q
        assert card["classes"]["binary"] == b

    children: dict[str, list[str]] = {}
    for c in catalog["classes"]:
        children.setdefault(c["parent"], []).append(c["id"])
        if c["level"] == "quadrant":
            assert c["parent"] in lines
            assert c["parent"] == c["line"]
            # A quadrant digit is a binary digit refined: 1, 2 negative, 3, 4 positive.
            assert c["parent"][:5] == "".join("2" if d in "34" else "0" for d in c["id"][:5])
        else:
            parent = classes[c["parent"]]
            assert parent["level"] == "quadrant"
            assert c["line"] == parent["line"]
            if c["level"] == "octant":
                assert _parent_by_digit(c["id"]) == (c["parent"], c["line"])
            for f in c.get("folded", []):
                assert _parent_by_digit(f["id"]) == (c["parent"], c["line"])

    assert sum(ln["count"] for ln in lines.values()) == total
    for lid, ln in lines.items():
        assert sum(classes[k]["count"] for k in children.get(lid, [])) == ln["count"]
    for c in catalog["classes"]:
        if c["level"] == "quadrant":
            assert sum(classes[k]["count"] for k in children[c["id"]]) == c["count"]
        if c["level"] == "other":
            assert sum(f["count"] for f in c["folded"]) == c["count"]
            assert all(f["count"] < catalog["meta"]["min_members"] for f in c["folded"])


def test_every_id_carries_its_suffix(export):
    catalog, cards, bonds = export
    for ln in catalog["lines"]:
        assert re.fullmatch(r"[02]{5}b", ln["id"])
    for c in catalog["classes"]:
        assert CLASS_ID.match(c["id"]), c["id"]
        assert CLASS_ID.match(c["parent"]), c["parent"]
        for f in c.get("folded", []):
            assert re.fullmatch(r"[1-8]{5}o", f["id"])
    for cid in cards:
        assert CLASS_ID.match(cid), cid
    for card in bonds.values():
        assert re.fullmatch(r"[02]{5}b", card["classes"]["binary"])
        assert re.fullmatch(r"[1-4]{5}q", card["classes"]["quadrant"])
        assert re.fullmatch(r"[1-8]{5}o", card["classes"]["octant"])


def test_every_sampled_member_has_a_bond_card(export):
    catalog, cards, bonds = export
    sampled = [m["key"] for c in catalog["classes"] for m in c.get("members", [])]
    assert len(sampled) == catalog["meta"]["counts"]["members_sampled"]
    assert len(set(sampled)) == len(sampled)
    for key in sampled:
        assert key in bonds
    for c in catalog["classes"]:
        energies = [m["energy"] for m in c.get("members", [])]
        assert energies == sorted(energies)
        assert len(energies) <= catalog["meta"]["members_per_class"]
    for c in catalog["classes"]:
        assert c["id"] in cards
    for ln in catalog["lines"]:
        assert ln["id"] in cards


def test_models_and_bonds_draw_as_render_ss(export):
    """A model has 10 atoms and the 9-row backbone table; a subset bond 12 and 11."""
    catalog, _, bonds = export
    models = [ln["model"] for ln in catalog["lines"]]
    models += [c["model"] for c in catalog["classes"] if c["level"] == "octant"]
    for m in models:
        assert len(m["atoms"]) == 10
        assert len(m["bonds"]) == 9
        assert "O" not in {a["name"] for a in m["atoms"]}
    for card in bonds.values():
        assert len(card["atoms"]) == 12
        assert len(card["bonds"]) == 11
    for m in [*models, *(b for b in bonds.values())]:
        for axis in "xyz":
            assert abs(sum(a[axis] for a in m["atoms"]) / len(m["atoms"])) < 0.01


def test_lines_cover_the_binary_classes(export):
    catalog, _, _ = export
    defs = pd.read_csv(f"{DATA_DIR}/PDB_SS_class_definitions.csv", dtype=str)
    assert sorted(ln["id"] for ln in catalog["lines"]) == sorted(f"{c}b" for c in defs.class_id)
    assert len(catalog["lines"]) == 32


def test_groves_follow_the_class_names(export):
    """Six named families hold the 20 named classes; each unnamed class is its own grove."""
    catalog, _, _ = export
    defs = pd.read_csv(f"{DATA_DIR}/PDB_SS_class_definitions.csv", dtype=str)
    named = defs[~defs.SS_Classname.str.isdigit()]
    assert len(named) == 20
    groves = {g["id"]: g for g in catalog["groves"]}
    assert len(groves) == 18
    families = ["lhspiral", "rhspiral", "lhhook", "rhhook", "lhstaple", "rhstaple"]
    assert sorted(g for g in groves if not g.startswith("b-")) == sorted(families)
    lines = {ln["id"]: ln for ln in catalog["lines"]}
    for _, d in named.iterrows():
        line = lines[f"{d.class_id}b"]
        assert line["name"] == d.SS_Classname
        assert line["grove"] == d.SS_Classname.lstrip("+-/").lower()
    assert sum(len(g["lines"]) for g in groves.values() if g["named"]) == 20
    assert groves["rhstaple"]["function"] == "Allosteric"
    assert groves["rhhook"]["function"] == "Catalytic"
    assert lines["00200b"]["function"] == "Allosteric"
    assert lines["20220b"]["function"] == "Catalytic"
    # Named families first, then by occupancy.
    order = catalog["groves"]
    assert [g["named"] for g in order] == sorted((g["named"] for g in order), reverse=True)
    for a, b in zip(order, order[1:]):
        if a["named"] == b["named"]:
            assert a["count"] >= b["count"]
    assert order[0]["id"] == "lhspiral"


def _kept(df: pd.DataFrame) -> pd.DataFrame:
    """The bonds within proteusPy's cutoffs, by filter_by_distance's rule."""
    return df[
        (df.ca_distance > 0)
        & (df.ca_distance < CA_CUTOFF)
        & (df.sg_distance > 0)
        & (df.sg_distance < SG_CUTOFF)
    ]


def test_home_is_the_lowest_energy_bond(loader, export):
    catalog, _, bonds = export
    df = _kept(loader.TorsionDF)
    assert catalog["home"] == df.ss_id[df.energy.idxmin()]
    assert catalog["home"] in bonds


def test_only_bonds_within_the_cutoffs(loader, export):
    """The export holds exactly the bonds within CA_CUTOFF and SG_CUTOFF, and says so."""
    catalog, _, bonds = export
    db = catalog["meta"]["database"]
    kept = _kept(loader.TorsionDF)
    assert db["bonds"] == len(kept) < db["unfiltered_bonds"] == len(loader.TorsionDF)
    assert db["pdb_entries"] == kept.source.nunique()
    assert db["filter"] == {"ca_cutoff": CA_CUTOFF, "sg_cutoff": SG_CUTOFF}
    assert sum(ln["count"] for ln in catalog["lines"]) == db["bonds"]
    for card in bonds.values():
        assert 0 < card["ca_distance"] < CA_CUTOFF + 0.001
        assert 0 < card["sg_distance"] < SG_CUTOFF + 0.001


def test_cutoffs_can_be_turned_off(loader):
    catalog, _, _ = build(loader, "subset", ca_cutoff=-1, sg_cutoff=-1)
    db = catalog["meta"]["database"]
    assert db["bonds"] == db["unfiltered_bonds"] == len(loader.TorsionDF)
    assert db["filter"] == {"ca_cutoff": None, "sg_cutoff": None}


def test_a_home_outside_the_cutoffs_is_refused(loader):
    df = loader.TorsionDF
    outside = df.ss_id[df.sg_distance >= SG_CUTOFF].iloc[0]
    with pytest.raises(ValueError, match="within the cutoffs"):
        build(loader, "subset", home=outside)


def test_write_and_cli(tmp_path, export):
    catalog, cards, bonds = export
    out = tmp_path / "subset"
    write(out, catalog, cards, bonds)
    with pytest.raises(FileExistsError):
        write(out, catalog, cards, bonds)
    assert (
        main(["--database", "subset", "--out", str(out), "--force", "--members-per-class", "2"])
        == 0
    )
    written = json.loads((out / "catalog.json").read_text())
    assert written["meta"]["members_per_class"] == 2
    atoms = json.loads((out / "atoms.json").read_text())
    assert len(atoms["order"]) == 16 and len(atoms["bonds_full"]) == 15
    assert len(atoms["bonds_model"]) == 9
    for c in written["classes"]:
        for m in c.get("members", []):
            assert (out / "bonds" / f"{m['key']}.json").exists()
        assert (out / "classes" / f"{c['id']}.json").exists()
    assert not (tmp_path / "subset.tmp").exists()
