"""
DisulfideTree: Hierarchical classification tree for KGRAG integration.

Builds a multi-level tree of disulfide structural classes:
  binary (2^5=32) → quadrant (4^5=1024) → sextant (6^5=7776) → octant (8^5=32768) → members

Each level refines the torsion-angle classification.  Branch lengths are
proportional to occupancy (member count).  The tree is represented as a
:class:`~proteusPy.graph_reasoner.KnowledgeGraph` so the
:class:`~proteusPy.graph_reasoner.GraphReasoner` can traverse it.

KGRAG integration: each node produces a structured snippet suitable for
retrieval-augmented generation over protein structural knowledge.

Part of the program proteusPy, https://github.com/suchanek/proteusPy,
a Python package for the manipulation and analysis of macromolecules.

Author: Eric G. Suchanek, PhD
"""

from __future__ import annotations

__pdoc__ = {"__all__": True}

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from proteusPy.graph_reasoner import KnowledgeGraph, SemanticEdge

# ---------------------------------------------------------------------------
# Hierarchy constants
# ---------------------------------------------------------------------------

#: Classification levels in refinement order.
LEVELS = ("binary", "quadrant", "sextant", "octant")

#: Base (number of bins per dihedral) at each level.
LEVEL_BASES = {"binary": 2, "quadrant": 4, "sextant": 6, "octant": 8}

#: Suffix character for class IDs at each level.
LEVEL_SUFFIXES = {"binary": "b", "quadrant": "q", "sextant": "s", "octant": "o"}

#: Number of dihedral angles per disulfide.
N_CHI = 5


# ---------------------------------------------------------------------------
# Digit-level parent mapping
# ---------------------------------------------------------------------------


def _segment_center(digit: int, base: int) -> float:
    """Return the center angle (degrees) of a segment digit for a given base.

    Segments are numbered ``base`` down to ``1``, where digit ``base``
    corresponds to ``[0, 360/base)`` and digit ``1`` to
    ``[360 - 360/base, 360)``.
    """
    segment_size = 360.0 / base
    # digit = base - floor(angle / segment_size)  ⇒  angle_start = (base - digit) * segment_size
    angle_start = (base - digit) * segment_size
    return angle_start + segment_size / 2.0


def _classify_angle(angle_deg: float, base: int) -> int:
    """Return the segment digit for *angle_deg* at the given base.

    For base 2, returns 0 (negative, [180, 360)) or 2 (positive, [0, 180))
    to match proteusPy's ``get_binary_quadrant`` convention.
    For other bases, returns digits 1..base via the standard formula.
    """
    angle_deg = angle_deg % 360.0
    if base == 2:
        return 2 if angle_deg < 180.0 else 0
    segment_size = 360.0 / base
    return base - int(angle_deg // segment_size)


def _digit_parent_map(child_base: int, parent_base: int) -> dict[int, int]:
    """Build a mapping from child digits to parent digits via center angles.

    For each child segment, compute its center angle and classify it at
    the parent base.
    """
    mapping = {}
    for digit in range(1, child_base + 1):
        center = _segment_center(digit, child_base)
        mapping[digit] = _classify_angle(center, parent_base)
    return mapping


# Pre-compute the three parent maps for the hierarchy.
_QUADRANT_TO_BINARY = _digit_parent_map(4, 2)
_SEXTANT_TO_QUADRANT = _digit_parent_map(6, 4)
_OCTANT_TO_SEXTANT = _digit_parent_map(8, 6)

_PARENT_MAPS = {
    ("quadrant", "binary"): _QUADRANT_TO_BINARY,
    ("sextant", "quadrant"): _SEXTANT_TO_QUADRANT,
    ("octant", "sextant"): _OCTANT_TO_SEXTANT,
}


def parent_class_id(child_id: str, child_level: str, parent_level: str) -> str:
    """Map a child class ID string to its parent class ID string.

    Parameters
    ----------
    child_id : str
        5-character class ID (digits only, no suffix).
    child_level : str
        One of ``"quadrant"``, ``"sextant"``, ``"octant"``.
    parent_level : str
        The next coarser level.

    Returns
    -------
    str
        5-character parent class ID.
    """
    dmap = _PARENT_MAPS[(child_level, parent_level)]
    child_base = LEVEL_BASES[child_level]

    parent_digits = []
    for ch in child_id[:N_CHI]:
        # Handle character encoding (base > 9 uses hex-like chars)
        d = int(ch, 16) if ch.isalpha() else int(ch)
        parent_digits.append(str(dmap[d]))
    return "".join(parent_digits)


def classify_angles(angles: list[float] | np.ndarray, base: int) -> str:
    """Classify a set of 5 dihedral angles at the given base.

    Parameters
    ----------
    angles : array-like
        Five dihedral angles in degrees.
    base : int
        Classification base (2, 4, 6, or 8).

    Returns
    -------
    str
        5-character class ID.
    """
    return "".join(str(_classify_angle(float(a), base)) for a in angles)


# ---------------------------------------------------------------------------
# Tree node data
# ---------------------------------------------------------------------------


@dataclass
class TreeNodeData:
    """Payload stored at each tree node.

    Attributes
    ----------
    class_id : str
        5-character class ID (e.g. ``"00000"``).
    level : str
        One of ``"root"``, ``"binary"``, ``"quadrant"``, ``"sextant"``,
        ``"octant"``, ``"member"``.
    occupancy : int
        Number of disulfide members in this class (and all children).
    occupancy_pct : float
        Percentage of total database.
    class_name : str
        Human-readable name (e.g. ``"-LHSpiral"``), if known.
    functional_annotation : str
        Functional annotation (e.g. ``"Allosteric"``), if known.
    consensus_torsions : list[float] or None
        Consensus dihedral angles for this class.
    """

    class_id: str
    level: str
    occupancy: int = 0
    occupancy_pct: float = 0.0
    class_name: str = ""
    functional_annotation: str = ""
    consensus_torsions: Optional[list[float]] = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# KGRAG snippet generation
# ---------------------------------------------------------------------------


def snippet_for_node(node_data: TreeNodeData) -> dict:
    """Generate a KGRAG-compatible snippet for a tree node.

    Parameters
    ----------
    node_data : TreeNodeData
        The tree node payload.

    Returns
    -------
    dict
        Structured snippet with entity_type, identifiers, properties,
        and a natural-language description.
    """
    level = node_data.level
    cid = node_data.class_id
    suffix = LEVEL_SUFFIXES.get(level, "")

    desc_parts = [f"{level.capitalize()} class {cid}{suffix}"]
    if node_data.class_name:
        desc_parts[0] += f" ({node_data.class_name})"
    desc_parts.append(
        f"{node_data.occupancy} members ({node_data.occupancy_pct:.2f}%)"
    )
    if node_data.functional_annotation:
        desc_parts.append(f"Function: {node_data.functional_annotation}")
    if node_data.consensus_torsions:
        angles_str = ", ".join(f"{a:.1f}" for a in node_data.consensus_torsions)
        desc_parts.append(f"Consensus torsions: [{angles_str}]")

    return {
        "entity_type": f"disulfide_{level}_class",
        "class_id": f"{cid}{suffix}",
        "level": level,
        "occupancy": node_data.occupancy,
        "occupancy_pct": node_data.occupancy_pct,
        "class_name": node_data.class_name or None,
        "functional_annotation": node_data.functional_annotation or None,
        "consensus_torsions": node_data.consensus_torsions,
        "description": ". ".join(desc_parts) + ".",
    }


def snippet_for_disulfide(ss) -> dict:
    """Generate a KGRAG-compatible snippet for an individual disulfide.

    Parameters
    ----------
    ss : Disulfide
        A proteusPy Disulfide object.

    Returns
    -------
    dict
        Structured snippet with identifiers, torsion angles, energy,
        class memberships, and a natural-language description.
    """
    torsions = list(ss.torsion_array)
    angles_str = ", ".join(f"{a:.1f}" for a in torsions)

    classes = {}
    for level, base in LEVEL_BASES.items():
        classes[level] = classify_angles(torsions, base)

    return {
        "entity_type": "disulfide",
        "node_id": f"{ss.pdb_id}_{ss.proximal}_{ss.distal}",
        "pdb_id": ss.pdb_id,
        "proximal": ss.proximal,
        "distal": ss.distal,
        "torsions": torsions,
        "energy": float(ss.torsion_energy),
        "ca_distance": float(ss.ca_distance),
        "classes": classes,
        "description": (
            f"Disulfide bond in {ss.pdb_id} between "
            f"Cys{ss.proximal}-Cys{ss.distal}: "
            f"torsion angles [{angles_str}] deg, "
            f"energy {ss.torsion_energy:.2f} kcal/mol, "
            f"binary class {classes['binary']}."
        ),
    }


# ---------------------------------------------------------------------------
# DisulfideTree builder
# ---------------------------------------------------------------------------


class DisulfideTree:
    """Hierarchical classification tree for disulfide bonds.

    Builds a four-level tree (binary → quadrant → sextant → octant) with
    individual disulfides as leaves.  Branch weights reflect occupancy.
    The tree is stored as a :class:`KnowledgeGraph` for integration with
    the :class:`GraphReasoner` and KGRAG.

    Parameters
    ----------
    sslist : DisulfideList
        Source disulfide bonds.
    class_names : dict, optional
        Mapping from binary class ID to human-readable name
        (e.g. ``{"00000": "-LHSpiral"}``).
    functional_annotations : dict, optional
        Mapping from binary class ID to function
        (e.g. ``{"00200": "Allosteric"}``).

    Examples
    --------
    >>> tree = DisulfideTree(my_sslist)
    >>> graph = tree.graph
    >>> snippet = tree.snippet("00000b")
    """

    def __init__(
        self,
        sslist=None,
        class_names: Optional[dict[str, str]] = None,
        functional_annotations: Optional[dict[str, str]] = None,
    ):
        self._class_names = class_names or {}
        self._functional_annotations = functional_annotations or {}
        self._node_data: dict[str, TreeNodeData] = {}
        self._total = 0

        # The graph uses 5D embeddings (consensus torsion angles or class centers)
        self._graph = KnowledgeGraph(ndim=N_CHI, name="disulfide_tree")

        if sslist is not None:
            self.build(sslist)

    @property
    def graph(self) -> KnowledgeGraph:
        """The underlying knowledge graph."""
        return self._graph

    def build(self, sslist) -> None:
        """Build the full tree from a DisulfideList.

        Parameters
        ----------
        sslist : DisulfideList
            Source disulfides.
        """
        self._total = len(sslist)
        if self._total == 0:
            return

        # Step 1: classify every disulfide at all 4 levels
        members: dict[str, dict[str, list]] = {
            level: {} for level in LEVELS
        }
        member_angles: dict[str, list[np.ndarray]] = {
            level: {} for level in LEVELS
        }

        for ss in sslist:
            torsions = np.array(ss.torsion_array, dtype="d")
            for level, base in LEVEL_BASES.items():
                cid = classify_angles(torsions, base)
                members[level].setdefault(cid, []).append(ss)
                member_angles[level].setdefault(cid, []).append(torsions)

        # Step 2: create nodes at each level
        for level in LEVELS:
            suffix = LEVEL_SUFFIXES[level]
            for cid, ss_list in members[level].items():
                node_key = f"{cid}{suffix}"
                occupancy = len(ss_list)
                occupancy_pct = 100.0 * occupancy / self._total

                # Compute centroid embedding (mean of member torsion angles)
                angle_stack = np.array(member_angles[level][cid])
                # Circular mean for periodic angles
                centroid = _circular_mean_deg(angle_stack)

                data = TreeNodeData(
                    class_id=cid,
                    level=level,
                    occupancy=occupancy,
                    occupancy_pct=occupancy_pct,
                    class_name=self._class_names.get(cid, ""),
                    functional_annotation=self._functional_annotations.get(cid, ""),
                    consensus_torsions=centroid.tolist(),
                )
                self._node_data[node_key] = data
                self._graph.add_node(node_key, centroid, node=data)

        # Step 3: add root node
        all_angles = np.array(
            [np.array(ss.torsion_array, dtype="d") for ss in sslist]
        )
        root_centroid = _circular_mean_deg(all_angles)
        root_data = TreeNodeData(
            class_id="root",
            level="root",
            occupancy=self._total,
            occupancy_pct=100.0,
        )
        self._node_data["root"] = root_data
        self._graph.add_node("root", root_centroid, node=root_data)

        # Step 4: add hierarchical edges
        # root → binary
        for cid in members["binary"]:
            node_key = f"{cid}b"
            weight = self._node_data[node_key].occupancy_pct / 100.0
            self._graph.add_edge(
                SemanticEdge("root", node_key, weight, "hierarchy")
            )

        # binary → quadrant → sextant → octant
        level_pairs = [
            ("quadrant", "binary"),
            ("sextant", "quadrant"),
            ("octant", "sextant"),
        ]
        for child_level, parent_level in level_pairs:
            child_suffix = LEVEL_SUFFIXES[child_level]
            parent_suffix = LEVEL_SUFFIXES[parent_level]
            for cid in members[child_level]:
                child_key = f"{cid}{child_suffix}"
                pid = parent_class_id(cid, child_level, parent_level)
                parent_key = f"{pid}{parent_suffix}"
                if parent_key in self._graph:
                    weight = self._node_data[child_key].occupancy_pct / 100.0
                    self._graph.add_edge(
                        SemanticEdge(parent_key, child_key, weight, "hierarchy")
                    )

        # Step 5: add leaf edges (octant → individual disulfides)
        for cid, ss_list in members["octant"].items():
            oct_key = f"{cid}o"
            for ss in ss_list:
                member_key = f"{ss.pdb_id}_{ss.proximal}_{ss.distal}"
                if member_key not in self._graph:
                    emb = np.array(ss.torsion_array, dtype="d")
                    self._graph.add_node(member_key, emb, node=ss)
                self._graph.add_edge(
                    SemanticEdge(oct_key, member_key, 1.0, "membership")
                )

    def snippet(self, node_key: str) -> dict:
        """Return a KGRAG snippet for the given node.

        Parameters
        ----------
        node_key : str
            A node key like ``"00000b"`` (class) or ``"1egs_24_84"`` (member).

        Returns
        -------
        dict
            Structured snippet for KGRAG consumption.
        """
        if node_key in self._node_data:
            return snippet_for_node(self._node_data[node_key])

        # Must be a leaf disulfide
        node = self._graph.get_node(node_key)
        if node is not None and hasattr(node, "torsion_array"):
            return snippet_for_disulfide(node)

        return {"entity_type": "unknown", "node_id": node_key, "description": ""}

    def children(self, node_key: str) -> list[str]:
        """Return the child node keys for a given node.

        Parameters
        ----------
        node_key : str
            Parent node key.

        Returns
        -------
        list[str]
            Child node keys, sorted by occupancy (descending).
        """
        edges = self._graph._edges.get(node_key, [])
        child_keys = [e.target_id for e in edges if e.edge_type == "hierarchy"]

        # Sort by occupancy (descending)
        def _occ(k):
            data = self._node_data.get(k)
            return data.occupancy if data else 0

        child_keys.sort(key=_occ, reverse=True)
        return child_keys

    def members(self, node_key: str) -> list[str]:
        """Return leaf member keys for a given node.

        Parameters
        ----------
        node_key : str
            Class node key (e.g. ``"00000o"``).

        Returns
        -------
        list[str]
            Member disulfide node keys.
        """
        edges = self._graph._edges.get(node_key, [])
        return [e.target_id for e in edges if e.edge_type == "membership"]

    def node_data(self, node_key: str) -> Optional[TreeNodeData]:
        """Return the :class:`TreeNodeData` for *node_key*."""
        return self._node_data.get(node_key)

    @property
    def levels(self) -> dict[str, int]:
        """Number of populated classes at each level."""
        counts = {level: 0 for level in LEVELS}
        for data in self._node_data.values():
            if data.level in counts:
                counts[data.level] += 1
        return counts

    def __repr__(self) -> str:
        lvl = self.levels
        return (
            f"DisulfideTree(total={self._total}, "
            f"binary={lvl['binary']}, quadrant={lvl['quadrant']}, "
            f"sextant={lvl['sextant']}, octant={lvl['octant']})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _circular_mean_deg(angles: np.ndarray) -> np.ndarray:
    """Compute the circular mean of angle arrays (in degrees).

    Parameters
    ----------
    angles : np.ndarray
        Shape ``(n_samples, n_angles)`` or ``(n_angles,)``.

    Returns
    -------
    np.ndarray
        Circular mean angle(s) in degrees, shape ``(n_angles,)``.
    """
    if angles.ndim == 1:
        return angles.copy()
    rad = np.deg2rad(angles)
    mean_sin = np.mean(np.sin(rad), axis=0)
    mean_cos = np.mean(np.cos(rad), axis=0)
    return np.rad2deg(np.arctan2(mean_sin, mean_cos))


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "LEVELS",
    "LEVEL_BASES",
    "LEVEL_SUFFIXES",
    "classify_angles",
    "parent_class_id",
    "TreeNodeData",
    "DisulfideTree",
    "snippet_for_node",
    "snippet_for_disulfide",
]


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# End of file
