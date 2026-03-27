#!/usr/bin/env python3
"""
Disulfide Knowledge Graph Demo
==============================

Demonstrates the full WaveRider KG stack applied to disulfide bonds:

1. Load disulfides and build the hierarchical classification tree
2. Print the tree structure (text visualizer)
3. Render a PNG schematic of the tree
4. Query the KG — walk from root down through the hierarchy
5. Build a flat torsion-angle KG and reason between disulfides
6. Generate KGRAG snippets for retrieval-augmented generation

Usage
-----
    python programs/disulfide_kg_demo.py

Author: Eric G. Suchanek, PhD
"""

from __future__ import annotations

import time


def main():
    from proteusPy import (
        DisulfideTree,
        ExplorationSteering,
        GraphReasoner,
        TargetSteering,
        graph_from_disulfides,
    )
    from proteusPy.DisulfideLoader import Load_PDB_SS
    from proteusPy.tree_visualizer import png_tree, text_tree

    # ------------------------------------------------------------------
    # 1. Load disulfide data
    # ------------------------------------------------------------------
    print("=" * 70)
    print("DISULFIDE KNOWLEDGE GRAPH DEMO")
    print("=" * 70)

    print("\n[1] Loading disulfide database (subset)...")
    t0 = time.time()
    PDB_SS = Load_PDB_SS(verbose=False, subset=True)
    sslist = PDB_SS.SSList
    print(f"    Loaded {len(sslist):,} disulfides in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 2. Build the hierarchical DisulfideTree
    # ------------------------------------------------------------------
    print("\n[2] Building DisulfideTree (binary → quadrant → sextant → octant)...")
    t0 = time.time()
    tree = DisulfideTree(sslist)
    print(f"    {tree}")
    print(f"    Built in {time.time() - t0:.2f}s")
    print(f"    Graph nodes: {len(tree.graph):,}")

    levels = tree.levels
    print(f"    Levels: {levels}")

    # ------------------------------------------------------------------
    # 3. Text tree visualization
    # ------------------------------------------------------------------
    print("\n[3] Text tree (top 2 levels, top 5 children per node):")
    print("-" * 70)
    txt = text_tree(tree, max_depth=2, max_children=5)
    print(txt)
    print("-" * 70)

    print("\n    Full depth view (top 3 children per node):")
    print("-" * 70)
    txt_full = text_tree(tree, max_depth=4, max_children=3)
    print(txt_full)
    print("-" * 70)

    # ------------------------------------------------------------------
    # 4. PNG tree rendering
    # ------------------------------------------------------------------
    print("\n[4] Rendering PNG tree schematic...")
    outfile = png_tree(
        tree,
        filename="disulfide_tree.png",
        max_depth=2,
        max_children=6,
        title="Disulfide Classification Hierarchy",
        dpi=150,
    )
    print(f"    Saved: {outfile}")

    # ------------------------------------------------------------------
    # 5. Walk the hierarchy using GraphReasoner
    # ------------------------------------------------------------------
    print("\n[5] Walking the hierarchy with GraphReasoner...")

    # Find the most populated binary class
    binary_children = tree.children("root")
    if binary_children:
        top_binary = binary_children[0]
        top_data = tree.node_data(top_binary)
        print(f"    Most populated binary class: {top_binary}")
        print(
            f"      occupancy={top_data.occupancy:,} "
            f"({top_data.occupancy_pct:.1f}%)"
        )
        if top_data.consensus_torsions:
            angles = ", ".join(f"{a:.1f}" for a in top_data.consensus_torsions)
            print(f"      consensus torsions: [{angles}]")

        # Drill down: binary → quadrant → sextant → octant
        print(f"\n    Drilling down from {top_binary}:")
        node = top_binary
        for depth in range(3):
            children = tree.children(node)
            if not children:
                break
            top_child = children[0]
            child_data = tree.node_data(top_child)
            indent = "      " + "  " * depth
            print(
                f"{indent}→ {top_child}  "
                f"n={child_data.occupancy:,} ({child_data.occupancy_pct:.1f}%)"
            )
            node = top_child

        # Show members of the deepest octant class
        members = tree.members(node)
        if members:
            print(f"\n    Leaf members of {node} ({len(members)} disulfides):")
            for m in members[:5]:
                ss_node = tree.graph.get_node(m)
                if hasattr(ss_node, "torsion_array"):
                    angles = ", ".join(f"{a:.1f}" for a in ss_node.torsion_array)
                    energy = (
                        f"{ss_node.torsion_energy:.2f}"
                        if hasattr(ss_node, "torsion_energy")
                        else "?"
                    )
                    print(f"      {m}: [{angles}] E={energy} kcal/mol")
                else:
                    print(f"      {m}")
            if len(members) > 5:
                print(f"      ... +{len(members) - 5} more")

    # ------------------------------------------------------------------
    # 6. KGRAG snippet generation
    # ------------------------------------------------------------------
    print("\n[6] KGRAG snippet example:")
    if binary_children:
        snippet = tree.snippet(binary_children[0])
        print(f"    Entity type: {snippet['entity_type']}")
        print(f"    Class ID:    {snippet['class_id']}")
        print(f"    Description: {snippet['description']}")

    # ------------------------------------------------------------------
    # 7. Flat torsion-angle KG with GraphReasoner
    # ------------------------------------------------------------------
    print("\n[7] Building flat torsion-angle KG and reasoning...")

    # Use a small sample for the flat graph demo
    sample_size = min(200, len(sslist))
    sample = sslist[:sample_size]

    t0 = time.time()
    kg = graph_from_disulfides(sample, edge_threshold=15.0)
    print(f"    Built KG with {len(kg)} nodes in {time.time() - t0:.2f}s")

    # Pick two disulfides and find a path between them
    node_ids = kg.node_ids
    start_id = node_ids[0]
    target_id = node_ids[-1]

    print(f"\n    Reasoning from {start_id} toward {target_id}...")
    reasoner = GraphReasoner(kg, TargetSteering(kg.get_embedding(target_id)))
    path = reasoner.reason_toward(start_id, target_id, max_hops=10)

    print(f"    Path length: {path.length} hops")
    if path.length > 0:
        print(f"    Mean score:  {path.mean_score:.3f}")
        print(f"    Path: {' → '.join(path.node_ids[:8])}")
        if path.length > 8:
            print(f"          ... ({path.length - 8} more hops)")

    # Exploration-mode reasoning
    print(f"\n    Exploration from {start_id} (max 8 hops)...")
    explorer = GraphReasoner(kg, ExplorationSteering())
    explore_path = explorer.reason(start_id, max_hops=8)

    print(f"    Explored {explore_path.length} nodes")
    if explore_path.scores:
        print(f"    Score range: [{min(explore_path.scores):.3f}, {max(explore_path.scores):.3f}]")
    print(f"    Path: {' → '.join(explore_path.node_ids[:6])}")
    if explore_path.length > 6:
        print(f"          ... ({explore_path.length - 6} more)")

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Disulfides loaded:     {len(sslist):,}")
    print(f"  Tree nodes (total):    {len(tree.graph):,}")
    print(f"  Hierarchy levels:      {levels}")
    print(f"  Flat KG nodes:         {len(kg)}")
    print(f"  Targeted path length:  {path.length}")
    print(f"  Explored path length:  {explore_path.length}")
    print(f"  Output PNG:            {outfile}")
    print("=" * 70)


if __name__ == "__main__":
    main()
