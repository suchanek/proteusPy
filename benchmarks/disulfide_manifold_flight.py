#!/usr/bin/env python3
# disulfide_manifold_flight.py
# Copyright (c) 2026 Eric G. Suchanek, PhD, Flux-Frontiers
# https://github.com/Flux-Frontiers
# License: BSD
# Last revised: 2026-03-23 -egs-
"""
disulfide_manifold_flight.py
-----------------------------
Canonical demonstration of manifold flight through disulfide torsional space.

Dataset:  proteusPy disulfide database — 36 000+ experimentally determined
          disulfide bonds, each described by 5 backbone-independent dihedral
          angles: χ1, χ2, χ3, χ4, χ5  (degrees).

Class hierarchy (primary branches of the DisulfideTree):
  binary   (base 2, 2^5 =    32 classes): sign of each χ angle (0=neg, 2=pos)
  quadrant (base 4, 4^5 =  1024 classes): 90° quadrant of each χ
  sextant  (base 6, 6^5 =  7776 classes): 60° sextant of each χ
  octant   (base 8, 8^5 = 32768 classes): 45° octant of each χ

Each level refines the previous.  A disulfide's full position in the tree is:
  binary_id → quadrant_id → sextant_id → octant_id → individual bond

This script:
  1. Loads the database and extracts the 5-angle feature matrix.
  2. Computes all four class IDs for every disulfide (binary/quadrant/sextant/octant).
  3. Fits ManifoldModel on a manageable subset (~2000 points).
  4. Identifies binary-class centroids (the coarsest, most meaningful landmarks)
     and picks the two most-distant ones as origin and destination.
  5. Flies origin → destination along the graph, recording per-hop:
       - position in 5D torsional space
       - class at each level of the hierarchy
       - local intrinsic dimensionality
       - class boundary crossings at all four levels
  6. Runs ManifoldObserver.observe_path() to show the path from above:
       - observer height (reconstruction error) at each hop
       - curvature at each hop
  7. Prints rich tables and saves results as JSON.

Usage:
  python benchmarks/disulfide_manifold_flight.py [--n N] [--k K] [--subset]
"""

import argparse
import json
import sys
import time

import numpy as np
from rich.console import Console
from rich.progress import track
from rich.table import Table

console = Console()

# Hierarchy constants (mirrors disulfide_tree.py)
LEVELS = ("binary", "quadrant", "sextant", "octant")
LEVEL_BASES = {"binary": 2, "quadrant": 4, "sextant": 6, "octant": 8}


def parse_args():
    p = argparse.ArgumentParser(description="Manifold flight through disulfide torsional space")
    p.add_argument("--n", type=int, default=0,
                   help="Number of disulfides to sample (0 = full database)")
    p.add_argument("--k", type=int, default=20, help="KNN for graph construction")
    p.add_argument("--subset", action="store_true", help="Use subset loader (faster)")
    p.add_argument("--tau", type=float, default=0.90, help="PCA variance threshold")
    p.add_argument("--w", type=float, default=0.8, help="Manifold weight in blending")
    p.add_argument("--max-steps", type=int, default=500,
                   help="Maximum graph hops for fly_toward")
    p.add_argument("--patience", type=int, default=15,
                   help="Non-improving hops before stopping")
    return p.parse_args()


def all_class_ids(angles):
    """Return {level: class_id_str} for all four levels of the hierarchy."""
    from proteusPy.disulfide_tree import classify_angles
    return {level: classify_angles(angles, LEVEL_BASES[level]) for level in LEVELS}


def load_disulfides(subset: bool, n_sample: int, rng: np.random.Generator):
    """Load database, extract angles and full class hierarchy for each bond.

    :param subset: If True, load the smaller subset database.
    :param n_sample: Number to sample; 0 means use the full database.
    :param rng: NumPy random generator for reproducible sampling.
    """
    console.print("  Loading proteusPy disulfide database …")
    import proteusPy as pp

    loader = pp.Load_PDB_SS(verbose=False, subset=subset)
    all_ss = loader.SSList
    console.print(f"  Total disulfides available: {len(all_ss):,}")

    if n_sample == 0 or n_sample >= len(all_ss):
        sample = list(all_ss)
        console.print("  Using [bold]full database[/bold]")
    else:
        indices = rng.choice(len(all_ss), size=n_sample, replace=False)
        sample = [all_ss[int(i)] for i in indices]
        console.print(f"  Sampled {n_sample:,} from {len(all_ss):,}")

    angles_list = []
    classes_list = []   # list of dicts: {level: class_id_str}
    ids = []

    for ss in track(sample, description="  Extracting angles + classes"):
        try:
            chi = list(ss.torsion_array)  # [chi1, chi2, chi3, chi4, chi5]
            if len(chi) != 5 or any(np.isnan(c) for c in chi):
                continue
            angles_list.append(chi)
            classes_list.append(all_class_ids(chi))
            ids.append(ss.name)
        except Exception:
            pass

    X = np.array(angles_list, dtype=np.float32)
    console.print(f"  Valid disulfides: {len(X):,}  |  5D torsional space")
    console.print(
        f"  Hierarchy: binary (2^5={2**5}) → quadrant (4^5={4**5}) → "
        f"sextant (6^5={6**5}) → octant (8^5={8**5})"
    )
    return X, classes_list, ids


def binary_label_array(classes_list):
    """Return integer array of binary class indices (0..31) for ML use."""
    unique = sorted(set(c["binary"] for c in classes_list))
    lut = {v: i for i, v in enumerate(unique)}
    return np.array([lut[c["binary"]] for c in classes_list], dtype=int), unique


def _write_report(result, path_data, level_crossings, obs_summary, png_path, md_path):
    """Write a professional Markdown flight report.

    :param result: Full result dict from the flight run.
    :param path_data: Per-hop list of dicts (node, chi, class labels, etc.).
    :param level_crossings: Dict of boundary crossing counts per level.
    :param obs_summary: Observer summary dict (height, curvature, etc.).
    :param png_path: Path to the saved figure PNG (or None if unavailable).
    :param md_path: Output path for the Markdown report.
    """
    cfg = result["config"]
    n_ss = result["n_disulfides"]
    hcp = result["hierarchy_classes_present"]
    ts = result.get("run_timestamp", "unknown")
    elapsed = result.get("elapsed_seconds", 0)
    origin = result["origin_octant"]
    dest = result["dest_octant"]
    dist = result["centroid_distance_deg"]
    n_hops = result["path_length"]
    arrived = result["arrived"]
    mean_d = result["mean_path_local_dim"]

    lines = [
        "# Disulfide Manifold Flight Report",
        "",
        "> **WaveRider · ManifoldWalker demo** — navigating the 5D disulfide torsional manifold",
        "> via the proteusPy structural database.",
        "",
        "---",
        "",
        "## Provenance",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Run timestamp (UTC) | `{ts}` |",
        f"| Elapsed time | {elapsed:.1f} s |",
        "| proteusPy database | full (subset=False) |",
        f"| Disulfides loaded | {n_ss:,} |",
        "| Script | `benchmarks/disulfide_manifold_flight.py` |",
        "| Report generated by | WaveRider / ManifoldObserver stack |",
        "",
        "---",
        "",
        "## Run Parameters",
        "",
        "| Parameter | Value | Description |",
        "|-----------|-------|-------------|",
        f"| `--n` | {cfg['n'] or 'all'} | Disulfides sampled (0 = full DB) |",
        f"| `--k` | {cfg['k']} | KNN for manifold graph construction |",
        f"| `--tau` | {cfg['tau']} | PCA variance threshold (intrinsic dim) |",
        f"| `--w` | {cfg['w']} | Manifold weight in blending function |",
        f"| `--patience` | {cfg.get('patience', 15)} | Non-improving hops before stopping |",
        f"| `--max-steps` | {cfg.get('max_steps', 500)} | Maximum graph hops |",
        "",
        "---",
        "",
        "## Class Hierarchy",
        "",
        "Disulfide bonds are classified by the sign/sector of each of the five",
        "backbone-independent dihedral angles (χ₁–χ₅) at four levels of refinement:",
        "",
        "| Level | Base | Classes (theory) | Classes present |",
        "|-------|------|-----------------|-----------------|",
        f"| binary   | 2 | 2⁵ = 32     | {hcp['binary']} |",
        f"| quadrant | 4 | 4⁵ = 1,024  | {hcp['quadrant']} |",
        f"| sextant  | 6 | 6⁵ = 7,776  | {hcp['sextant']} |",
        f"| octant   | 8 | 8⁵ = 32,768 | {hcp['octant']} |",
        "",
        "Each level *refines* the previous. Octant is the finest grain (45° sectors).",
        "A flight between maximally distant octant centroids must cross all coarser",
        "boundaries along the way.",
        "",
        "---",
        "",
        "## Flight Summary",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Origin octant class | `{origin}` |",
        f"| Destination octant class | `{dest}` |",
        f"| Centroid distance | {dist:.1f}° (5D Euclidean, torsional) |",
        f"| Path length | {n_hops} hops |",
        f"| Arrived at destination | {'✓ Yes' if arrived else '✗ No (stopped early)'} |",
        f"| Mean local intrinsic dim | {mean_d:.1f} |",
        "",
        "### Boundary Crossings",
        "",
        "| Level | Crossings | Meaning |",
        "|-------|-----------|---------|",
        f"| binary   | {level_crossings['binary']} | Sign-level transitions (coarsest) |",
        f"| quadrant | {level_crossings['quadrant']} | 90° sector transitions |",
        f"| sextant  | {level_crossings['sextant']} | 60° sector transitions |",
        f"| octant   | {level_crossings['octant']} | 45° sector transitions (finest) |",
        "",
    ]

    if obs_summary:
        lines += [
            "### Observer (N+1 Dimensional View)",
            "",
            "The ManifoldObserver lifts each hop into (N+1)-space. The extra coordinate",
            "is the reconstruction error — how far the node sits above its local tangent plane.",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Mean height h | {obs_summary.get('mean_height', '—')} |",
            f"| Mean curvature κ | {obs_summary.get('mean_curvature', '—')} |",
            f"| High-curvature hops | {obs_summary.get('high_curvature_hops', [])} |",
            "",
        ]

    # Flight path table (all hops)
    lines += [
        "---",
        "",
        "## Flight Path",
        "",
        "| Hop | Node | d | binary | quadrant | sextant | octant | χ₁ | χ₂ | χ₃ | χ₄ | χ₅ |",
        "|-----|------|---|--------|----------|---------|--------|----|----|----|----|-----|",
    ]
    for hop, pd in enumerate(path_data):
        chi = pd.get("chi", [])
        chi_str = " | ".join(f"{c:.1f}" for c in chi) if chi else "— | — | — | — | —"
        lines.append(
            f"| {hop} | {pd['node']} | {pd['local_dim']} "
            f"| {pd['binary']} | {pd['quadrant']} | {pd['sextant']} | {pd['octant']} "
            f"| {chi_str} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Figure",
        "",
    ]
    if png_path:
        fig_rel = png_path.replace("benchmarks/", "")
        lines += [
            f"![Disulfide Manifold Flight]({fig_rel})",
            "",
            "**Figure.** Four-panel flight report. *(Top-left)* PCA projection of all",
            f"{n_ss:,} disulfides coloured by binary class; flight path overlaid in white.",
            "*(Top-right)* All five χ angles along the path. *(Bottom-left)* Observer",
            "height h (reconstruction error) per hop — zero means on the manifold.",
            "*(Bottom-right)* Local curvature κ; vertical dashed lines mark class",
            "boundary crossings at the octant level.",
        ]
    else:
        lines.append("*Figure not available — run the visualizer separately.*")

    # Travel Journal
    landmarks = result.get("landmarks", [])
    obs = result.get("observer", {})
    if landmarks:
        lines += [
            "---",
            "",
            "## Explorer's Travel Journal",
            "",
            f"The explorer departed from octant class **{result['origin_octant']}** and",
            f"navigated toward octant class **{result['dest_octant']}**, {result['centroid_distance_deg']:.1f}° away",
            f"in 5-dimensional torsional space. The journey covered **{result['path_length']} hops**",
            "across the disulfide manifold, crossing class boundaries at all four levels",
            "of the hierarchy.",
            "",
            "### Waypoints",
            "",
            "| Hop | Node | Tag | κ | h | octant | binary |",
            "|-----|------|-----|---|---|--------|--------|",
        ]
        for lm in landmarks:
            tag_str = ", ".join(lm["tags"])
            lines.append(
                f"| {lm['hop']} | {lm['node']} | **{tag_str}** "
                f"| {lm['curvature']:.4f} | {lm['height']:.4f} "
                f"| `{lm['octant']}` | `{lm['binary']}` |"
            )
        peak_hop = obs.get("peak_hop")
        valley_hop = obs.get("valley_hop")
        lines += [
            "",
            "### Terrain Notes",
            "",
        ]
        if peak_hop is not None and peak_hop < len(path_data):
            pk = path_data[peak_hop]
            lines.append(
                f"- **Highest peak** (max curvature) at hop {peak_hop}: "
                f"octant `{pk['octant']}`, binary `{pk['binary']}`, "
                f"κ = {pk.get('curvature', '?')}"
            )
        if valley_hop is not None and valley_hop < len(path_data):
            vl = path_data[valley_hop]
            lines.append(
                f"- **Deepest valley** (min curvature) at hop {valley_hop}: "
                f"octant `{vl['octant']}`, binary `{vl['binary']}`, "
                f"κ = {vl.get('curvature', '?')}"
            )
        lines.append(
            f"- Mean curvature κ = {obs.get('mean_curvature', '?')} ± "
            f"{obs.get('std_curvature', '?')}"
        )
        lines.append(
            f"- Mean observer height h = {obs.get('mean_height', '?')} ± "
            f"{obs.get('std_height', '?')}"
        )
        lines += [""]

    lines += [
        "",
        "---",
        "",
        "## Notes",
        "",
        "- **Blending function:** `dist = 0.8·manifold_dist + 0.2·euclidean_dist`",
        "  weights tangent-plane geometry while preventing degenerate edges in sparse",
        "  neighbourhoods.",
        "- **Edge weights:** `1 / (1 + dist_blend)` — maps to (0, 1], scale-invariant.",
        "- **Normal existence:** the observer's extra basis vector is constructed by",
        "  padding the N-dim tangent frame and applying QR decomposition — orthogonality",
        "  is guaranteed by construction (Normal Existence Proposition, WaveRider §3.3).",
        "",
        "---",
        "*Generated by proteusPy · WaveRider stack · "
        "© 2026 Eric G. Suchanek, PhD · Flux-Frontiers · BSD License*",
    ]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    rng = np.random.default_rng(42)

    console.rule("[bold blue]Disulfide Manifold Flight — Binary→Quadrant→Sextant→Octant")
    console.print(f"\n  Config: n={args.n}  k={args.k}  τ={args.tau}  w={args.w}\n")

    t0 = time.perf_counter()

    # -----------------------------------------------------------------------
    # 1. Load data with full class hierarchy
    # -----------------------------------------------------------------------
    X, classes_list, _ = load_disulfides(
        subset=args.subset, n_sample=args.n, rng=rng
    )

    # Binary class as integer label for ManifoldModel
    y, _ = binary_label_array(classes_list)

    # Show class distribution at each level
    for level in LEVELS:
        n_classes = len(set(c[level] for c in classes_list))
        console.print(f"  {level:10s}: {n_classes:5d} distinct classes present")

    # -----------------------------------------------------------------------
    # 2. Fit ManifoldModel
    # -----------------------------------------------------------------------
    console.print("\n[bold]Step 2:[/bold] Fitting ManifoldModel …")
    from proteusPy.manifold_model import ManifoldModel

    mm = ManifoldModel(
        k_graph=args.k,
        k_pca=min(args.k * 5, len(X) - 1),
        variance_threshold=args.tau,
        manifold_weight=args.w,
    )
    mm.fit(X, y)
    console.print(f"  Fit complete in {time.perf_counter() - t0:.1f}s")

    local_dims = [
        mm._geometries[f"n{i}"].intrinsic_dim
        for i in range(len(X))
        if f"n{i}" in mm._geometries
    ]
    console.print(
        f"  Local intrinsic dim: mean={np.mean(local_dims):.1f}  "
        f"median={np.median(local_dims):.0f}  "
        f"range=[{min(local_dims)}, {max(local_dims)}]"
    )

    # -----------------------------------------------------------------------
    # 3. Identify octant-class centroids (finest level = maximum reach)
    #
    # Why octant → octant traverses the WHOLE manifold:
    #   - Octant is the finest grain: 45° sectors, 8^5 = 32768 possible classes.
    #   - Two maximally distant octant centroids sit at OPPOSITE ENDS of 5D space.
    #   - Any path between them MUST cross binary, quadrant, and sextant boundaries
    #     along the way — coarser boundaries are nested inside finer ones, so you
    #     can't get from one extreme octant to another without traversing the whole
    #     hierarchy from the inside out.
    #   - Going binary→binary only guarantees one sign-level crossing (too coarse).
    # -----------------------------------------------------------------------
    console.print("\n[bold]Step 3:[/bold] Octant-class centroids (finest level → maximum traversal) …")

    octant_ids_present = sorted(set(c["octant"] for c in classes_list))
    centroids = {}
    counts = {}
    for oid in octant_ids_present:
        mask = np.array([c["octant"] == oid for c in classes_list])
        counts[oid] = int(mask.sum())
        if counts[oid] >= 3:
            centroids[oid] = X[mask].mean(axis=0)

    present = sorted(centroids.keys())
    # Find the pair of octant centroids furthest apart in 5D torsional space
    best_dist = -1.0
    origin_bid, dest_bid = present[0], present[-1]
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            a, b = present[i], present[j]
            d = float(np.linalg.norm(centroids[a] - centroids[b]))
            if d > best_dist:
                best_dist = d
                origin_bid, dest_bid = a, b

    # Show the top 12 most-populated octant classes
    top12 = sorted(present, key=lambda k: -counts[k])[:12]
    cen_table = Table(title="Octant Class Centroids (top 12 by population)", show_header=True)
    cen_table.add_column("octant ID")
    cen_table.add_column("Count", justify="right")
    cen_table.add_column("χ1", justify="right")
    cen_table.add_column("χ2", justify="right")
    cen_table.add_column("χ3", justify="right")
    cen_table.add_column("χ4", justify="right")
    cen_table.add_column("χ5", justify="right")
    for oid in top12:
        c = centroids[oid]
        marker = " ←ORIGIN" if oid == origin_bid else (" ←DEST" if oid == dest_bid else "")
        cen_table.add_row(
            oid + marker, str(counts[oid]),
            f"{c[0]:.1f}", f"{c[1]:.1f}", f"{c[2]:.1f}", f"{c[3]:.1f}", f"{c[4]:.1f}",
        )
    console.print(cen_table)
    console.print(
        f"\n  Flight: octant {origin_bid} → octant {dest_bid}  "
        f"(centroid dist = {best_dist:.1f}°)"
    )
    origin_binary = next(c["binary"] for c in classes_list if c["octant"] == origin_bid)
    dest_binary = next(c["binary"] for c in classes_list if c["octant"] == dest_bid)
    console.print(
        f"  [dim]Origin binary: {origin_binary}  Dest binary: {dest_binary}[/dim]"
    )

    # -----------------------------------------------------------------------
    # 4. Fly the manifold
    # -----------------------------------------------------------------------
    console.print("\n[bold]Step 4:[/bold] Flying along the manifold graph …")
    mm.fly_to_nearest(centroids[origin_bid])
    path = mm.fly_toward(centroids[dest_bid],
                         max_steps=args.max_steps,
                         patience=args.patience)

    if not path:
        console.print("[red]  No path found — try increasing --n or --k[/red]")
        sys.exit(1)
    console.print(f"  Path length: {len(path)} hops")

    # Build per-hop class record at all four levels
    path_data = []
    for node_id in path:
        idx = int(node_id[1:])
        cls = classes_list[idx] if idx < len(classes_list) else {}
        geom = mm._geometries.get(f"n{idx}")
        local_d = geom.intrinsic_dim if geom else 0
        chi = X[idx].tolist() if idx < len(X) else []
        path_data.append({
            "node": node_id,
            "idx": idx,
            "local_dim": local_d,
            "chi": chi,
            "binary": cls.get("binary", "?"),
            "quadrant": cls.get("quadrant", "?"),
            "sextant": cls.get("sextant", "?"),
            "octant": cls.get("octant", "?"),
        })

    # Flight table
    flight_table = Table(title="Manifold Flight Path (all four class levels)", show_header=True)
    flight_table.add_column("Hop", justify="right")
    flight_table.add_column("Node")
    flight_table.add_column("d", justify="right")
    flight_table.add_column("binary")
    flight_table.add_column("quadrant")
    flight_table.add_column("sextant")
    flight_table.add_column("octant")
    flight_table.add_column("χ1", justify="right")
    flight_table.add_column("χ2", justify="right")
    flight_table.add_column("χ3", justify="right")

    for hop, pd in enumerate(path_data):
        chi = pd["chi"]
        flight_table.add_row(
            str(hop), pd["node"], str(pd["local_dim"]),
            pd["binary"], pd["quadrant"], pd["sextant"], pd["octant"],
            *(f"{c:.1f}" for c in chi[:3]) if chi else ["—", "—", "—"],
        )
    console.print(flight_table)

    # Boundary crossings at each level
    console.print("\n  [bold]Class boundary crossings:[/bold]")
    for level in LEVELS:
        crossings = []
        for i in range(1, len(path_data)):
            if path_data[i][level] != path_data[i - 1][level]:
                crossings.append(
                    f"  hop {i:2d}: {path_data[i-1][level]} → {path_data[i][level]}"
                )
        label = f"    {level:10s} ({len(crossings)} crossings)"
        if crossings:
            console.print(f"[cyan]{label}[/cyan]")
            for line in crossings:
                console.print(f"[dim]{line}[/dim]")
        else:
            console.print(f"[dim]{label}  (no boundary crossing)[/dim]")

    # -----------------------------------------------------------------------
    # 5. Observer view
    # -----------------------------------------------------------------------
    landmarks = []   # populated inside try block; always defined for result dict
    console.print("\n[bold]Step 5:[/bold] ManifoldObserver.observe_path() — view from above …")
    try:
        from proteusPy.manifold_observer import ManifoldObserver

        obs = ManifoldObserver(mm)
        path_view = obs.observe_path(path)

        obs_table = Table(title="Observer View — Height & Curvature Along Path", show_header=True)
        obs_table.add_column("Hop", justify="right")
        obs_table.add_column("binary")
        obs_table.add_column("quadrant")
        obs_table.add_column("Height h", justify="right")
        obs_table.add_column("Curvature κ", justify="right")
        obs_table.add_column("Local d", justify="right")

        for hop, node_id in enumerate(path_view["path"]):
            pd = path_data[hop]
            h = path_view["heights"][hop]
            k = path_view["curvatures"][hop]
            d = path_view["intrinsic_dims"][hop]
            obs_table.add_row(
                str(hop), pd["binary"], pd["quadrant"],
                f"{h:.4f}", f"{k:.4f}", str(d),
            )
        console.print(obs_table)

        # Highlight high-curvature hops (potential class boundaries)
        mean_k = float(np.mean(path_view["curvatures"]))
        std_k = float(np.std(path_view["curvatures"]))
        high_curv = [
            (i, path_view["curvatures"][i])
            for i in range(len(path))
            if path_view["curvatures"][i] > mean_k + std_k
        ]
        if high_curv:
            console.print(
                f"\n  High-curvature hops (κ > μ+σ = {mean_k:.3f}+{std_k:.3f}): "
                + ", ".join(f"hop {i} (κ={k:.3f})" for i, k in high_curv)
            )

        # Write per-hop observer data back into path_data for JSON serialisation
        for hop_i, node_id in enumerate(path_view["path"]):
            path_data[hop_i]["height"] = round(float(path_view["heights"][hop_i]), 6)
            path_data[hop_i]["curvature"] = round(float(path_view["curvatures"][hop_i]), 6)

        # Detect peaks (κ > μ+σ) and valleys (κ < μ-σ)
        kappas = path_view["curvatures"]
        mean_k = float(np.mean(kappas))
        std_k  = float(np.std(kappas))
        heights = path_view["heights"]
        mean_h = float(np.mean(heights))
        std_h  = float(np.std(heights))

        high_curv = [
            (i, float(kappas[i]))
            for i in range(len(path))
            if kappas[i] > mean_k + std_k
        ]
        low_curv = [
            (i, float(kappas[i]))
            for i in range(len(path))
            if kappas[i] < max(0, mean_k - std_k)
        ]

        # Build landmarks list (ordered by hop)
        landmarks = []
        for hop_i in range(len(path)):
            tags = []
            if hop_i == 0:
                tags.append("ORIGIN")
            if hop_i == len(path) - 1:
                tags.append("DESTINATION")
            if kappas[hop_i] > mean_k + std_k:
                tags.append("PEAK")
            if kappas[hop_i] < max(0, mean_k - std_k):
                tags.append("VALLEY")
            for level in LEVELS:
                if hop_i > 0 and path_data[hop_i][level] != path_data[hop_i - 1][level]:
                    tags.append(f"CROSS_{level.upper()}")
            if tags:
                landmarks.append({
                    "hop": hop_i,
                    "node": path_data[hop_i]["node"],
                    "tags": tags,
                    "curvature": round(float(kappas[hop_i]), 6),
                    "height": round(float(heights[hop_i]), 6),
                    "octant": path_data[hop_i]["octant"],
                    "binary": path_data[hop_i]["binary"],
                    "chi": path_data[hop_i]["chi"],
                })

        if high_curv:
            console.print(
                f"\n  Peaks (κ > μ+σ = {mean_k:.3f}+{std_k:.3f}): "
                + ", ".join(f"hop {i} (κ={k:.3f})" for i, k in high_curv)
            )

        obs_summary = {
            "mean_height": round(mean_h, 4),
            "std_height": round(std_h, 4),
            "mean_curvature": round(mean_k, 4),
            "std_curvature": round(std_k, 4),
            "boundary_crossings_observer": path_view["boundary_crossings"],
            "high_curvature_hops": [i for i, _ in high_curv],
            "low_curvature_hops": [i for i, _ in low_curv],
            "peak_hop": int(np.argmax(kappas)) if len(kappas) else None,
            "valley_hop": int(np.argmin(kappas)) if len(kappas) else None,
        }
    except Exception as exc:
        console.print(f"  [yellow]Observer skipped: {exc}[/yellow]")
        obs_summary = {}
        landmarks = []

    # -----------------------------------------------------------------------
    # 6. Summary
    # -----------------------------------------------------------------------
    end_cls = path_data[-1]["octant"] if path_data else "?"
    arrived = end_cls == dest_bid
    level_crossings = {}
    for level in LEVELS:
        n_cross = sum(
            1 for i in range(1, len(path_data))
            if path_data[i][level] != path_data[i - 1][level]
        )
        level_crossings[level] = n_cross

    console.rule("[bold]Summary[/bold]")
    console.print(f"""
  Origin octant class   : {origin_bid}
  Dest   octant class   : {dest_bid}
  Centroid distance     : {best_dist:.1f}°
  Path length           : {len(path)} hops
  Arrived at dest       : {"YES ✓" if arrived else f"NO (ended at octant {end_cls})"}
  Mean local intrinsic d: {np.mean([d['local_dim'] for d in path_data]):.1f}
  Total time            : {time.perf_counter() - t0:.1f}s

  Boundary crossings per level:
    binary   : {level_crossings['binary']}
    quadrant : {level_crossings['quadrant']}
    sextant  : {level_crossings['sextant']}
    octant   : {level_crossings['octant']}
""")

    elapsed = time.perf_counter() - t0
    run_ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    result = {
        "config": vars(args),
        "run_timestamp": run_ts,
        "elapsed_seconds": round(elapsed, 1),
        "n_disulfides": len(X),
        "hierarchy_classes_present": {
            level: len(set(c[level] for c in classes_list))
            for level in LEVELS
        },
        "octant_classes_with_centroids": len(present),
        "origin_octant": origin_bid,
        "dest_octant": dest_bid,
        "centroid_distance_deg": round(best_dist, 2),
        "path_length": len(path),
        "arrived": bool(arrived),
        "mean_path_local_dim": round(
            float(np.mean([d["local_dim"] for d in path_data])), 2
        ),
        "level_crossings": level_crossings,
        "observer": obs_summary,
        "landmarks": landmarks,
        "path": path_data,
    }
    out_path = "benchmarks/disulfide_flight_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    console.print(f"[dim]Results saved → {out_path}[/dim]")

    # -----------------------------------------------------------------------
    # 7. Auto-generate figure
    # -----------------------------------------------------------------------
    import importlib.util as _ilu
    import os as _os
    _spec = _ilu.spec_from_file_location(
        "disulfide_flight_visualizer",
        _os.path.join(_os.path.dirname(__file__), "disulfide_flight_visualizer.py"),
    )
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    make_figure = _mod.make_figure

    png_path = out_path.replace(".json", ".png")
    try:
        make_figure(out_path, png_path)
        console.print(f"[green]Figure saved  → {png_path}[/green]")
    except Exception as exc:
        console.print(f"[yellow]Figure generation skipped: {exc}[/yellow]")
        png_path = None

    # -----------------------------------------------------------------------
    # 8. Markdown report
    # -----------------------------------------------------------------------
    md_path = out_path.replace(".json", "_report.md")
    _write_report(result, path_data, level_crossings, obs_summary, png_path, md_path)
    console.print(f"[green]Report saved  → {md_path}[/green]")


if __name__ == "__main__":
    main()
