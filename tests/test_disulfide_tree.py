"""Tests for DisulfideTree — verifies hierarchical classification tree
construction, parent mapping, snippet generation, and KGRAG integration."""

import importlib
import os
import sys
import types
import unittest

import numpy as np

# ---------------------------------------------------------------------------
# Direct-load modules without triggering proteusPy/__init__.py
# ---------------------------------------------------------------------------
_pkg_dir = os.path.join(os.path.dirname(__file__), "..", "proteusPy")

_stub = types.ModuleType("proteusPy")
_stub.__path__ = [os.path.abspath(_pkg_dir)]
sys.modules.setdefault("proteusPy", _stub)

# Load turtleND
_tnd_spec = importlib.util.spec_from_file_location(
    "proteusPy.turtleND", os.path.join(_pkg_dir, "turtleND.py")
)
_tnd_mod = importlib.util.module_from_spec(_tnd_spec)
sys.modules["proteusPy.turtleND"] = _tnd_mod
_tnd_spec.loader.exec_module(_tnd_mod)

# Load graph_reasoner
_gr_spec = importlib.util.spec_from_file_location(
    "proteusPy.graph_reasoner", os.path.join(_pkg_dir, "graph_reasoner.py")
)
_gr_mod = importlib.util.module_from_spec(_gr_spec)
sys.modules["proteusPy.graph_reasoner"] = _gr_mod
_gr_spec.loader.exec_module(_gr_mod)

# Load disulfide_tree
_dt_spec = importlib.util.spec_from_file_location(
    "proteusPy.disulfide_tree", os.path.join(_pkg_dir, "disulfide_tree.py")
)
_dt_mod = importlib.util.module_from_spec(_dt_spec)
sys.modules["proteusPy.disulfide_tree"] = _dt_mod
_dt_spec.loader.exec_module(_dt_mod)

# Import symbols
classify_angles = _dt_mod.classify_angles
parent_class_id = _dt_mod.parent_class_id
DisulfideTree = _dt_mod.DisulfideTree
TreeNodeData = _dt_mod.TreeNodeData
snippet_for_node = _dt_mod.snippet_for_node
KnowledgeGraph = _gr_mod.KnowledgeGraph
SemanticEdge = _gr_mod.SemanticEdge

_QUADRANT_TO_BINARY = _dt_mod._QUADRANT_TO_BINARY
_SEXTANT_TO_QUADRANT = _dt_mod._SEXTANT_TO_QUADRANT
_OCTANT_TO_SEXTANT = _dt_mod._OCTANT_TO_SEXTANT


# ---------------------------------------------------------------------------
# Mock Disulfide for testing (avoids heavy DisulfideBase import)
# ---------------------------------------------------------------------------


class MockDisulfide:
    """Lightweight stand-in for a Disulfide object."""

    def __init__(self, pdb_id, proximal, distal, torsions, energy=5.0, ca_dist=5.5):
        self.pdb_id = pdb_id
        self.proximal = proximal
        self.distal = distal
        self.torsion_array = np.array(torsions, dtype="d")
        self.torsion_energy = energy
        self.ca_distance = ca_dist

    def __repr__(self):
        return f"MockSS({self.pdb_id}_{self.proximal}_{self.distal})"


def _make_mock_sslist(n=20, seed=42):
    """Generate a list of mock disulfides with varied torsion angles."""
    rng = np.random.RandomState(seed)
    sslist = []
    for i in range(n):
        angles = rng.uniform(-180, 180, size=5)
        ss = MockDisulfide(
            pdb_id=f"pdb{i:03d}",
            proximal=10 + i,
            distal=50 + i,
            torsions=angles,
            energy=rng.uniform(2, 20),
        )
        sslist.append(ss)
    return sslist


# ---------------------------------------------------------------------------
# Tests: digit-level parent mapping
# ---------------------------------------------------------------------------


class TestDigitParentMapping(unittest.TestCase):
    def test_quadrant_to_binary_coverage(self):
        """Every quadrant digit (1-4) maps to a valid binary digit (0 or 2)."""
        for digit, parent in _QUADRANT_TO_BINARY.items():
            self.assertIn(parent, [0, 2], f"digit {digit} mapped to {parent}")

    def test_quadrant_to_binary_partition(self):
        """Two quadrant digits map to each binary half."""
        neg = [d for d, p in _QUADRANT_TO_BINARY.items() if p == 0]
        pos = [d for d, p in _QUADRANT_TO_BINARY.items() if p == 2]
        self.assertEqual(len(neg), 2, f"negative: {neg}")
        self.assertEqual(len(pos), 2, f"positive: {pos}")

    def test_sextant_to_quadrant_coverage(self):
        """Every sextant digit (1-6) maps to a valid quadrant digit (1-4)."""
        for digit, parent in _SEXTANT_TO_QUADRANT.items():
            self.assertIn(parent, [1, 2, 3, 4])

    def test_octant_to_sextant_coverage(self):
        """Every octant digit (1-8) maps to a valid sextant digit (1-6)."""
        for digit, parent in _OCTANT_TO_SEXTANT.items():
            self.assertIn(parent, [1, 2, 3, 4, 5, 6])

    def test_octant_to_sextant_total(self):
        """All 8 octant digits are mapped."""
        self.assertEqual(len(_OCTANT_TO_SEXTANT), 8)


# ---------------------------------------------------------------------------
# Tests: classify_angles
# ---------------------------------------------------------------------------


class TestClassifyAngles(unittest.TestCase):
    def test_binary_positive(self):
        """All positive angles → binary digit 2."""
        result = classify_angles([45, 90, 135, 10, 170], base=2)
        self.assertEqual(result, "22222")

    def test_binary_negative(self):
        """All negative angles → binary digit 0."""
        result = classify_angles([-45, -90, -135, -10, -170], base=2)
        # Negative angles: -45 % 360 = 315 → [180, 360) → segment 0
        self.assertEqual(result, "00000")

    def test_binary_mixed(self):
        result = classify_angles([45, -90, 90, -135, 170], base=2)
        self.assertEqual(result, "20202")

    def test_quadrant_all_bins(self):
        """Each quadrant bin is reachable."""
        # base 4, segment_size = 90°
        # digit 4: [0°, 90°), digit 3: [90°, 180°), digit 2: [180°, 270°), digit 1: [270°, 360°)
        result = classify_angles([45, 135, 225, 315, 45], base=4)
        self.assertEqual(result, "43214")

    def test_octant_length(self):
        result = classify_angles([0, 45, 90, 135, 180], base=8)
        self.assertEqual(len(result), 5)


# ---------------------------------------------------------------------------
# Tests: parent_class_id
# ---------------------------------------------------------------------------


class TestParentClassId(unittest.TestCase):
    def test_quadrant_to_binary(self):
        """Quadrant class maps to correct binary parent."""
        # All digits in positive half → binary 2
        result = parent_class_id("44444", "quadrant", "binary")
        self.assertEqual(result, "22222")

        # All digits in negative half → binary 0
        result = parent_class_id("11111", "quadrant", "binary")
        self.assertEqual(result, "00000")

    def test_mixed_quadrant_to_binary(self):
        result = parent_class_id("41234", "quadrant", "binary")
        # digit 4→2, 1→0, 2→0, 3→2, 4→2
        self.assertEqual(result, "20022")

    def test_all_negative_quadrant_to_binary(self):
        """Quadrant digits 1,2 are negative half → binary 0."""
        result = parent_class_id("11111", "quadrant", "binary")
        self.assertEqual(result, "00000")

    def test_octant_to_sextant(self):
        """Octant class maps to a valid sextant class."""
        result = parent_class_id("88888", "octant", "sextant")
        # Each result digit should be 1-6
        for ch in result:
            self.assertIn(int(ch), [1, 2, 3, 4, 5, 6])


# ---------------------------------------------------------------------------
# Tests: DisulfideTree
# ---------------------------------------------------------------------------


class TestDisulfideTree(unittest.TestCase):
    def setUp(self):
        self.sslist = _make_mock_sslist(50, seed=123)
        self.tree = DisulfideTree(self.sslist)

    def test_tree_builds(self):
        self.assertGreater(len(self.tree.graph), 50)  # nodes > just members

    def test_root_exists(self):
        self.assertIn("root", self.tree.graph)

    def test_binary_classes_created(self):
        levels = self.tree.levels
        self.assertGreater(levels["binary"], 0)
        self.assertLessEqual(levels["binary"], 32)

    def test_quadrant_classes_created(self):
        levels = self.tree.levels
        self.assertGreater(levels["quadrant"], 0)

    def test_sextant_classes_created(self):
        levels = self.tree.levels
        self.assertGreater(levels["sextant"], 0)

    def test_octant_classes_created(self):
        levels = self.tree.levels
        self.assertGreater(levels["octant"], 0)

    def test_hierarchy_edges_exist(self):
        """Root should have children."""
        children = self.tree.children("root")
        self.assertGreater(len(children), 0)
        # Children should be binary classes
        for c in children:
            self.assertTrue(c.endswith("b"))

    def test_binary_has_quadrant_children(self):
        """Each binary class should have quadrant children."""
        binary_nodes = [
            k for k, v in self.tree._node_data.items() if v.level == "binary"
        ]
        has_children = False
        for bk in binary_nodes:
            kids = self.tree.children(bk)
            if kids:
                has_children = True
                for k in kids:
                    self.assertTrue(k.endswith("q"))
        self.assertTrue(has_children)

    def test_octant_has_members(self):
        """At least some octant classes should have member disulfides."""
        oct_nodes = [
            k for k, v in self.tree._node_data.items() if v.level == "octant"
        ]
        has_members = False
        for ok in oct_nodes:
            mems = self.tree.members(ok)
            if mems:
                has_members = True
                break
        self.assertTrue(has_members)

    def test_occupancy_sums(self):
        """Binary-level occupancies should sum to total."""
        binary_occ = sum(
            v.occupancy
            for v in self.tree._node_data.values()
            if v.level == "binary"
        )
        self.assertEqual(binary_occ, len(self.sslist))

    def test_children_sorted_by_occupancy(self):
        """Children should be returned in descending occupancy order."""
        children = self.tree.children("root")
        if len(children) >= 2:
            occs = [self.tree.node_data(c).occupancy for c in children]
            self.assertEqual(occs, sorted(occs, reverse=True))


# ---------------------------------------------------------------------------
# Tests: snippets
# ---------------------------------------------------------------------------


class TestSnippets(unittest.TestCase):
    def setUp(self):
        self.sslist = _make_mock_sslist(20, seed=99)
        self.tree = DisulfideTree(
            self.sslist,
            class_names={"22222": "+RHSpiral"},
            functional_annotations={"00200": "Allosteric"},
        )

    def test_class_snippet_structure(self):
        # Find a binary node
        binary_key = None
        for k, v in self.tree._node_data.items():
            if v.level == "binary":
                binary_key = k
                break
        self.assertIsNotNone(binary_key)

        snip = self.tree.snippet(binary_key)
        self.assertEqual(snip["entity_type"], "disulfide_binary_class")
        self.assertIn("occupancy", snip)
        self.assertIn("description", snip)

    def test_snippet_for_node_direct(self):
        data = TreeNodeData(
            class_id="00000",
            level="binary",
            occupancy=5000,
            occupancy_pct=25.0,
            class_name="-LHSpiral",
            functional_annotation="Allosteric",
        )
        snip = snippet_for_node(data)
        self.assertIn("-LHSpiral", snip["description"])
        self.assertIn("Allosteric", snip["description"])
        self.assertEqual(snip["occupancy"], 5000)

    def test_unknown_node_snippet(self):
        snip = self.tree.snippet("nonexistent_node")
        self.assertEqual(snip["entity_type"], "unknown")


# ---------------------------------------------------------------------------
# Tests: repr
# ---------------------------------------------------------------------------


class TestRepr(unittest.TestCase):
    def test_repr(self):
        sslist = _make_mock_sslist(10)
        tree = DisulfideTree(sslist)
        s = repr(tree)
        self.assertIn("DisulfideTree", s)
        self.assertIn("binary=", s)
        self.assertIn("octant=", s)


if __name__ == "__main__":
    unittest.main()
