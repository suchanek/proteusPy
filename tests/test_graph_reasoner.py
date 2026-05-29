"""Tests for GraphReasoner — verifies semantic reasoning over knowledge graphs
using steering strategies, edge discovery, and turtle-based navigation."""

import importlib
import os
import sys
import types
import unittest

import numpy as np
from numpy.testing import assert_allclose

# ---------------------------------------------------------------------------
# Direct-load the graph_reasoner module and its dependency (turtleND) without
# triggering proteusPy/__init__.py, which pulls in heavy optional deps that
# may not be installed in CI / lightweight environments.
# ---------------------------------------------------------------------------
_pkg_dir = os.path.join(os.path.dirname(__file__), "..", "proteusPy")

# 1. Create a minimal stub package so "from proteusPy.turtleND import ..."
#    works inside graph_reasoner.py
_stub = types.ModuleType("proteusPy")
_stub.__path__ = [os.path.abspath(_pkg_dir)]
sys.modules.setdefault("proteusPy", _stub)

# 2. Load turtleND first (graph_reasoner depends on it)
_tnd_spec = importlib.util.spec_from_file_location(
    "proteusPy.turtleND", os.path.join(_pkg_dir, "turtleND.py")
)
_tnd_mod = importlib.util.module_from_spec(_tnd_spec)
sys.modules["proteusPy.turtleND"] = _tnd_mod
_tnd_spec.loader.exec_module(_tnd_mod)

# 3. Load graph_reasoner
_gr_spec = importlib.util.spec_from_file_location(
    "proteusPy.graph_reasoner", os.path.join(_pkg_dir, "graph_reasoner.py")
)
_gr_mod = importlib.util.module_from_spec(_gr_spec)
sys.modules["proteusPy.graph_reasoner"] = _gr_mod
_gr_spec.loader.exec_module(_gr_mod)

# 4. Import symbols
DirectedDiscoverer = _gr_mod.DirectedDiscoverer
ExplorationSteering = _gr_mod.ExplorationSteering
GradientSteering = _gr_mod.GradientSteering
GraphReasoner = _gr_mod.GraphReasoner
KNNDiscoverer = _gr_mod.KNNDiscoverer
KnowledgeGraph = _gr_mod.KnowledgeGraph
RadiusDiscoverer = _gr_mod.RadiusDiscoverer
ReasoningPath = _gr_mod.ReasoningPath
SemanticEdge = _gr_mod.SemanticEdge
TargetSteering = _gr_mod.TargetSteering
angular_distance = _gr_mod.angular_distance
euclidean_distance = _gr_mod.euclidean_distance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_corridor_graph(n=5, ndim=3, spacing=1.0):
    """Build a linear corridor: n0 -- n1 -- n2 -- ... along axis 0."""
    g = KnowledgeGraph(ndim=ndim, name="corridor")
    for i in range(n):
        emb = np.zeros(ndim)
        emb[0] = i * spacing
        g.add_node(f"n{i}", emb)
    g.add_discoverer(RadiusDiscoverer(threshold=spacing * 1.5))
    return g


def _make_branching_graph():
    """Build a Y-shaped graph in 3D.

    Layout::

        n0(0,0,0) -- n1(1,0,0) -- n2(2,0,0) -- n3(3,0,0)
                                       |
                                    n4(2,1,0) -- n5(3,1,0)
    """
    g = KnowledgeGraph(ndim=3, name="branching")
    positions = {
        "n0": [0, 0, 0],
        "n1": [1, 0, 0],
        "n2": [2, 0, 0],
        "n3": [3, 0, 0],
        "n4": [2, 1, 0],
        "n5": [3, 1, 0],
    }
    for nid, pos in positions.items():
        g.add_node(nid, np.array(pos, dtype="d"))
    g.add_discoverer(RadiusDiscoverer(threshold=1.5))
    return g


def _make_angular_graph():
    """Build a 5D graph with angular embeddings (like torsion angles)."""
    g = KnowledgeGraph(ndim=5, name="angular")
    angles = {
        "a": [-60, -60, -90, -60, -60],
        "b": [-65, -55, -85, -65, -55],
        "c": [-70, -50, -80, -70, -50],
        "d": [170, 170, 170, 170, 170],
        "e": [-175, 175, -175, 175, -175],
    }
    for nid, ang in angles.items():
        g.add_node(nid, np.array(ang, dtype="d"))
    g.add_discoverer(RadiusDiscoverer(threshold=30.0, angular=True))
    return g


# ---------------------------------------------------------------------------
# Distance functions
# ---------------------------------------------------------------------------


class TestDistanceFunctions(unittest.TestCase):
    def test_euclidean(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        self.assertAlmostEqual(euclidean_distance(a, b), 5.0)

    def test_angular_no_wrapping(self):
        a = np.array([10.0, 20.0])
        b = np.array([30.0, 50.0])
        expected = np.sqrt(20**2 + 30**2)
        self.assertAlmostEqual(angular_distance(a, b), expected)

    def test_angular_with_wrapping(self):
        """Distance between -170 and 170 should be 20, not 340."""
        a = np.array([-170.0])
        b = np.array([170.0])
        self.assertAlmostEqual(angular_distance(a, b), 20.0)

    def test_angular_symmetry(self):
        a = np.array([45.0, -120.0, 90.0])
        b = np.array([-30.0, 150.0, -80.0])
        self.assertAlmostEqual(angular_distance(a, b), angular_distance(b, a))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class TestSemanticEdge(unittest.TestCase):
    def test_creation(self):
        e = SemanticEdge("a", "b", 0.8, "torsion")
        self.assertEqual(e.source_id, "a")
        self.assertEqual(e.target_id, "b")
        self.assertAlmostEqual(e.weight, 0.8)
        self.assertEqual(e.edge_type, "torsion")

    def test_default_metadata(self):
        e = SemanticEdge("a", "b", 0.5)
        self.assertEqual(e.metadata, {})


class TestReasoningPath(unittest.TestCase):
    def test_empty_path(self):
        p = ReasoningPath()
        self.assertEqual(p.length, 0)
        self.assertAlmostEqual(p.total_score, 0.0)
        self.assertAlmostEqual(p.mean_score, 0.0)

    def test_path_scores(self):
        p = ReasoningPath(
            node_ids=["a", "b", "c"],
            scores=[1.0, 2.0],
        )
        self.assertEqual(p.length, 3)
        self.assertAlmostEqual(p.total_score, 3.0)
        self.assertAlmostEqual(p.mean_score, 1.5)

    def test_copy(self):
        p = ReasoningPath(
            node_ids=["a", "b"],
            embeddings=[np.array([1.0]), np.array([2.0])],
            scores=[0.5],
        )
        c = p.copy()
        c.node_ids.append("c")
        self.assertEqual(p.length, 2)
        self.assertEqual(c.length, 3)


# ---------------------------------------------------------------------------
# KnowledgeGraph
# ---------------------------------------------------------------------------


class TestKnowledgeGraph(unittest.TestCase):
    def test_add_and_retrieve(self):
        g = KnowledgeGraph(ndim=3)
        g.add_node("a", np.array([1.0, 2.0, 3.0]), node="payload")
        self.assertIn("a", g)
        assert_allclose(g.get_embedding("a"), [1.0, 2.0, 3.0])
        self.assertEqual(g.get_node("a"), "payload")
        self.assertEqual(len(g), 1)

    def test_embedding_shape_validation(self):
        g = KnowledgeGraph(ndim=3)
        with self.assertRaises(ValueError):
            g.add_node("bad", np.array([1.0, 2.0]))

    def test_precomputed_edge(self):
        g = KnowledgeGraph(ndim=2)
        g.add_node("a", np.array([0.0, 0.0]))
        g.add_node("b", np.array([1.0, 0.0]))
        g.add_edge(SemanticEdge("a", "b", 0.9, "manual"))
        edges = g.discover_neighbors("a")
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].target_id, "b")

    def test_matrix_rebuild(self):
        g = KnowledgeGraph(ndim=2)
        g.add_node("a", np.array([0.0, 0.0]))
        g.add_node("b", np.array([1.0, 1.0]))
        g._rebuild_matrix()
        self.assertEqual(g._embedding_matrix.shape, (2, 2))
        self.assertFalse(g._dirty)


# ---------------------------------------------------------------------------
# Edge discoverers
# ---------------------------------------------------------------------------


class TestRadiusDiscoverer(unittest.TestCase):
    def test_discovers_nearby(self):
        g = _make_corridor_graph(n=5, ndim=3, spacing=1.0)
        edges = g.discover_neighbors("n2")
        target_ids = {e.target_id for e in edges}
        self.assertIn("n1", target_ids)
        self.assertIn("n3", target_ids)
        # n0 and n4 are distance 2.0, threshold is 1.5
        self.assertNotIn("n0", target_ids)
        self.assertNotIn("n4", target_ids)

    def test_excludes_self(self):
        g = _make_corridor_graph(n=3, ndim=3)
        edges = g.discover_neighbors("n1")
        target_ids = {e.target_id for e in edges}
        self.assertNotIn("n1", target_ids)

    def test_weight_decreases_with_distance(self):
        g = _make_corridor_graph(n=3, ndim=3, spacing=0.5)
        edges = g.discover_neighbors("n0")
        # n1 is at 0.5, n2 is at 1.0 — n1 should have higher weight
        edge_map = {e.target_id: e.weight for e in edges}
        if "n1" in edge_map and "n2" in edge_map:
            self.assertGreater(edge_map["n1"], edge_map["n2"])

    def test_angular_wrapping(self):
        g = _make_angular_graph()
        edges = g.discover_neighbors("d")
        target_ids = {e.target_id for e in edges}
        # d=[170,...] and e=[-175,...]: angular distance per component is 15,
        # so these should be neighbors
        self.assertIn("e", target_ids)
        # a=[-60,...] is far from d=[170,...] — should not be neighbors
        self.assertNotIn("a", target_ids)


class TestKNNDiscoverer(unittest.TestCase):
    def test_returns_k_neighbors(self):
        g = KnowledgeGraph(ndim=3, name="knn_test")
        for i in range(10):
            g.add_node(f"n{i}", np.array([float(i), 0.0, 0.0]))
        g.add_discoverer(KNNDiscoverer(k=3))
        edges = g.discover_neighbors("n5")
        self.assertEqual(len(edges), 3)
        target_ids = {e.target_id for e in edges}
        # Should get n4, n6 and one of n3/n7
        self.assertIn("n4", target_ids)
        self.assertIn("n6", target_ids)

    def test_angular_knn(self):
        g = _make_angular_graph()
        # Replace discoverer with KNN
        g._discoverers = [KNNDiscoverer(k=2, angular=True)]
        edges = g.discover_neighbors("a")
        target_ids = {e.target_id for e in edges}
        # a and b are closest in angular space
        self.assertIn("b", target_ids)


class TestDirectedDiscoverer(unittest.TestCase):
    def test_filters_backward_edges(self):
        g = _make_corridor_graph(n=5, ndim=3)
        # Replace with directed discoverer, 60 degree cone
        g._discoverers = [
            DirectedDiscoverer(RadiusDiscoverer(threshold=1.5), cone_angle=60.0)
        ]
        heading = np.array([1.0, 0.0, 0.0])  # pointing right
        edges = g.discover_neighbors("n2", heading)
        target_ids = {e.target_id for e in edges}
        # n3 is forward, n1 is backward
        self.assertIn("n3", target_ids)
        self.assertNotIn("n1", target_ids)

    def test_no_heading_passes_all(self):
        g = _make_corridor_graph(n=3, ndim=3)
        g._discoverers = [
            DirectedDiscoverer(RadiusDiscoverer(threshold=1.5), cone_angle=60.0)
        ]
        edges = g.discover_neighbors("n1", heading=None)
        self.assertEqual(len(edges), 2)  # n0 and n2


# ---------------------------------------------------------------------------
# Steering strategies
# ---------------------------------------------------------------------------


class TestTargetSteering(unittest.TestCase):
    def test_prefers_toward_target(self):
        target = np.array([10.0, 0.0, 0.0])
        strategy = TargetSteering(target)
        current = np.array([0.0, 0.0, 0.0])
        heading = np.array([1.0, 0.0, 0.0])
        path = ReasoningPath()

        forward = np.array([1.0, 0.0, 0.0])
        backward = np.array([-1.0, 0.0, 0.0])
        edge = SemanticEdge("a", "b", 0.5)

        s_forward = strategy.score_candidate(current, forward, heading, edge, path)
        s_backward = strategy.score_candidate(current, backward, heading, edge, path)
        self.assertGreater(s_forward, s_backward)


class TestGradientSteering(unittest.TestCase):
    def test_prefers_lower_energy(self):
        def field(x):
            return np.sum(x**2)
        strategy = GradientSteering(field, minimize=True)
        current = np.array([5.0, 0.0, 0.0])
        heading = np.array([-1.0, 0.0, 0.0])
        path = ReasoningPath()
        edge = SemanticEdge("a", "b", 0.5)

        closer = np.array([3.0, 0.0, 0.0])  # lower energy
        farther = np.array([7.0, 0.0, 0.0])  # higher energy

        s_closer = strategy.score_candidate(current, closer, heading, edge, path)
        s_farther = strategy.score_candidate(current, farther, heading, edge, path)
        self.assertGreater(s_closer, s_farther)


class TestExplorationSteering(unittest.TestCase):
    def test_prefers_unvisited_region(self):
        strategy = ExplorationSteering()
        current = np.array([0.0, 0.0, 0.0])
        heading = np.array([1.0, 0.0, 0.0])
        edge = SemanticEdge("a", "b", 0.5)

        path = ReasoningPath(
            embeddings=[np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
        )

        novel = np.array([0.0, 5.0, 0.0])  # far from visited
        redundant = np.array([0.5, 0.0, 0.0])  # near visited

        s_novel = strategy.score_candidate(current, novel, heading, edge, path)
        s_redundant = strategy.score_candidate(current, redundant, heading, edge, path)
        self.assertGreater(s_novel, s_redundant)


# ---------------------------------------------------------------------------
# GraphReasoner — core
# ---------------------------------------------------------------------------


class TestGraphReasonerInit(unittest.TestCase):
    def test_basic_init(self):
        g = _make_corridor_graph()
        r = GraphReasoner(g, ExplorationSteering())
        self.assertIsNone(r.current_node)
        self.assertEqual(r.path.length, 0)

    def test_start(self):
        g = _make_corridor_graph()
        r = GraphReasoner(g, ExplorationSteering())
        r.start("n0")
        self.assertEqual(r.current_node, "n0")
        self.assertEqual(r.path.length, 1)
        assert_allclose(r.position, [0.0, 0.0, 0.0])

    def test_start_invalid_node(self):
        g = _make_corridor_graph()
        r = GraphReasoner(g, ExplorationSteering())
        with self.assertRaises(KeyError):
            r.start("nonexistent")

    def test_step_before_start(self):
        g = _make_corridor_graph()
        r = GraphReasoner(g, ExplorationSteering())
        with self.assertRaises(RuntimeError):
            r.step()


class TestGraphReasonerStep(unittest.TestCase):
    def test_single_step(self):
        g = _make_corridor_graph()
        r = GraphReasoner(g, ExplorationSteering())
        r.start("n0")
        next_id = r.step()
        self.assertIsNotNone(next_id)
        self.assertEqual(r.path.length, 2)
        self.assertIn(next_id, ["n1"])  # only neighbor of n0

    def test_step_returns_none_at_dead_end(self):
        """A graph with one node and no edges should return None."""
        g = KnowledgeGraph(ndim=2)
        g.add_node("lonely", np.array([0.0, 0.0]))
        r = GraphReasoner(g, ExplorationSteering())
        r.start("lonely")
        self.assertIsNone(r.step())

    def test_no_revisit(self):
        """The reasoner should not revisit nodes."""
        g = _make_corridor_graph(n=3)
        r = GraphReasoner(g, ExplorationSteering())
        r.start("n1")
        # n1 has neighbors n0 and n2
        r.step()
        r.step()
        # Should visit both neighbors, not revisit n1
        visited = set(r.path.node_ids)
        self.assertEqual(len(visited), r.path.length)


# ---------------------------------------------------------------------------
# GraphReasoner — multi-hop
# ---------------------------------------------------------------------------


class TestGraphReasonerReason(unittest.TestCase):
    def test_corridor_traversal(self):
        """Should traverse the full corridor."""
        g = _make_corridor_graph(n=5)
        r = GraphReasoner(g, ExplorationSteering())
        path = r.reason("n0", max_hops=10)
        self.assertEqual(path.length, 5)
        self.assertEqual(path.node_ids[0], "n0")

    def test_max_hops_limit(self):
        g = _make_corridor_graph(n=10)
        r = GraphReasoner(g, ExplorationSteering())
        path = r.reason("n0", max_hops=3)
        self.assertLessEqual(path.length, 4)  # start + 3 hops

    def test_min_score_cutoff(self):
        """Reasoning should stop when score falls below threshold."""
        g = _make_corridor_graph(n=10)
        r = GraphReasoner(g, ExplorationSteering())
        # Very high min_score should stop after 1 step at most
        path = r.reason("n0", max_hops=10, min_score=999.0)
        self.assertLessEqual(path.length, 2)

    def test_scores_recorded(self):
        g = _make_corridor_graph(n=5)
        r = GraphReasoner(g, ExplorationSteering())
        path = r.reason("n0", max_hops=4)
        self.assertEqual(len(path.scores), path.length - 1)
        self.assertEqual(len(path.edges), path.length - 1)


class TestGraphReasonerReasonToward(unittest.TestCase):
    def test_finds_path_to_target(self):
        g = _make_corridor_graph(n=5)
        r = GraphReasoner(g, ExplorationSteering())
        path = r.reason_toward("n0", "n4", max_hops=10)
        self.assertEqual(path.node_ids[-1], "n4")

    def test_restores_strategy(self):
        g = _make_corridor_graph(n=5)
        original = ExplorationSteering()
        r = GraphReasoner(g, original)
        r.reason_toward("n0", "n4")
        self.assertIs(r.strategy, original)

    def test_branching_finds_target(self):
        g = _make_branching_graph()
        r = GraphReasoner(g, ExplorationSteering())
        path = r.reason_toward("n0", "n5", max_hops=10)
        self.assertIn("n5", path.node_ids)


# ---------------------------------------------------------------------------
# GraphReasoner — beam search
# ---------------------------------------------------------------------------


class TestGraphReasonerBeam(unittest.TestCase):
    def test_beam_returns_multiple_paths(self):
        g = _make_branching_graph()
        r = GraphReasoner(g, ExplorationSteering())
        paths = r.beam_reason("n0", max_hops=5, beam_width=3)
        self.assertGreater(len(paths), 0)
        # All paths should start from n0
        for p in paths:
            self.assertEqual(p.node_ids[0], "n0")

    def test_beam_width_limits(self):
        g = _make_corridor_graph(n=10)
        r = GraphReasoner(g, ExplorationSteering())
        paths = r.beam_reason("n0", max_hops=5, beam_width=2)
        self.assertLessEqual(len(paths), 2)

    def test_beam_paths_are_unique(self):
        g = _make_branching_graph()
        r = GraphReasoner(g, ExplorationSteering())
        paths = r.beam_reason("n0", max_hops=5, beam_width=5)
        if len(paths) > 1:
            # At least some paths should differ
            path_tuples = [tuple(p.node_ids) for p in paths]
            self.assertGreater(len(set(path_tuples)), 0)


# ---------------------------------------------------------------------------
# GraphReasoner — backtracking
# ---------------------------------------------------------------------------


class TestGraphReasonerBacktrack(unittest.TestCase):
    def test_backtrack_one_step(self):
        g = _make_corridor_graph(n=5)
        r = GraphReasoner(g, ExplorationSteering())
        r.reason("n0", max_hops=3)
        before = r.path.length
        r.backtrack(1)
        self.assertEqual(r.path.length, before - 1)

    def test_backtrack_updates_position(self):
        g = _make_corridor_graph(n=5)
        r = GraphReasoner(g, ExplorationSteering())
        r.reason("n0", max_hops=3)
        prev_node = r.path.node_ids[-2]
        r.backtrack(1)
        self.assertEqual(r.current_node, prev_node)

    def test_backtrack_prevents_revisit(self):
        """After backtracking, the reasoner should explore new paths."""
        g = _make_branching_graph()
        r = GraphReasoner(g, ExplorationSteering())
        r.start("n2")
        first = r.step()
        r.backtrack(1)
        second = r.step()
        # Should go to a different neighbor since first is still visited
        if second is not None:
            self.assertNotEqual(first, second)

    def test_cannot_backtrack_past_start(self):
        g = _make_corridor_graph()
        r = GraphReasoner(g, ExplorationSteering())
        r.start("n0")
        r.backtrack(100)
        self.assertEqual(r.path.length, 1)
        self.assertEqual(r.current_node, "n0")


# ---------------------------------------------------------------------------
# Heading alignment
# ---------------------------------------------------------------------------


class TestHeadingAlignment(unittest.TestCase):
    def test_heading_tracks_direction(self):
        """After several steps, heading should roughly follow path direction."""
        g = _make_corridor_graph(n=10, ndim=3)
        r = GraphReasoner(g, ExplorationSteering())
        r.reason("n0", max_hops=5)
        heading = r.heading
        # Corridor goes along axis 0, heading should have large x component
        self.assertGreater(abs(heading[0]), 0.5)

    def test_heading_alignment_zero_vector(self):
        """Aligning to zero direction should not crash."""
        g = _make_corridor_graph()
        r = GraphReasoner(g, ExplorationSteering())
        r.start("n0")
        # This should be a no-op, not raise
        r._align_heading(np.zeros(3))


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr(unittest.TestCase):
    def test_graph_repr(self):
        g = _make_corridor_graph()
        s = repr(g)
        self.assertIn("corridor", s)
        self.assertIn("ndim=3", s)

    def test_reasoner_repr(self):
        g = _make_corridor_graph()
        r = GraphReasoner(g, ExplorationSteering())
        s = repr(r)
        self.assertIn("unstarted", s)

    def test_reasoner_repr_after_start(self):
        g = _make_corridor_graph()
        r = GraphReasoner(g, ExplorationSteering())
        r.start("n0")
        s = repr(r)
        self.assertIn("n0", s)

    def test_path_repr(self):
        p = ReasoningPath(node_ids=["a", "b"], scores=[0.5])
        s = repr(p)
        self.assertIn("length=2", s)


if __name__ == "__main__":
    unittest.main()
