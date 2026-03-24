#!/usr/bin/env python3
# disulfide_flight_visualizer.py
# Copyright (c) 2026 Eric G. Suchanek, PhD, Flux-Frontiers
# https://github.com/Flux-Frontiers
# License: BSD
# Last revised: 2026-03-23 -egs-
"""
disulfide_flight_visualizer.py
-------------------------------
Load a disulfide_flight_results.json and produce a professional 4-panel PNG
figure summarising the manifold flight.

Panels
------
1 (top-left)    PCA(2D) scatter of path nodes coloured by binary class;
                flight path overlaid as a white→red gradient line with
                origin (★) and destination (✕) marked.
2 (top-right)   All five χ angles (χ₁–χ₅) plotted along the hop axis;
                vertical dashed lines mark octant boundary crossings.
3 (bottom-left) Observer height h (reconstruction error) per hop — zero
                means the node sits exactly on the local tangent plane.
4 (bottom-right) Local curvature κ per hop; vertical dashed lines mark
                octant boundary crossings; high-curvature hops highlighted.

Usage
-----
  # standalone
  python benchmarks/disulfide_flight_visualizer.py \\
      [--input benchmarks/disulfide_flight_results.json] \\
      [--output benchmarks/disulfide_flight_results.png]

  # programmatic
  from disulfide_flight_visualizer import make_figure
  make_figure("benchmarks/disulfide_flight_results.json",
              "benchmarks/disulfide_flight_results.png")
"""

from __future__ import annotations

import argparse
import json

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA

mpl.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
    "font.size":        9,
})

BINARY_CMAP = plt.get_cmap("tab20", 32)
CHI_COLORS  = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657"]
CHI_LABELS  = ["χ₁", "χ₂", "χ₃", "χ₄", "χ₅"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _octant_crossings(path_data: list[dict]) -> list[int]:
    """Return hop indices where the octant class changes."""
    return [
        i for i in range(1, len(path_data))
        if path_data[i]["octant"] != path_data[i - 1]["octant"]
    ]


def _gradient_path(ax, xy: np.ndarray, cmap: str = "autumn_r",
                   lw: float = 2.5, zorder: int = 5):
    """Draw a line with a colour gradient along the path."""
    points = xy.reshape(-1, 1, 2)
    segs   = np.concatenate([points[:-1], points[1:]], axis=1)
    t      = np.linspace(0, 1, len(segs))
    lc     = LineCollection(segs.tolist(), array=t, cmap=cmap, linewidth=lw,
                            zorder=zorder, capstyle="round")
    ax.add_collection(lc)
    return lc


# ---------------------------------------------------------------------------
# panel builders
# ---------------------------------------------------------------------------

def _panel_observer_map(ax, result: dict) -> None:
    """Panel 1 — Observer's Map: top-down terrain view of the path.

    The observer sits one dimension above the manifold and sees the
    walker's route as a relief map.  Curvature κ is rendered as terrain
    colour (blue=valley, red=peak); the path is drawn over the terrain;
    peaks (▲) and valleys (▼) are marked; class boundary crossings are
    vertical tick marks.
    """
    path_data = result["path"]
    landmarks = result.get("landmarks", [])
    if not path_data:
        ax.text(0.5, 0.5, "No path data", ha="center", va="center",
                transform=ax.transAxes)
        return

    path_chi = np.array(
        [p["chi"] for p in path_data if p.get("chi") and len(p["chi"]) == 5],
        dtype=np.float32,
    )
    if path_chi.shape[0] < 2:
        ax.text(0.5, 0.5, "Insufficient data for map", ha="center",
                va="center", transform=ax.transAxes)
        return

    pca = PCA(n_components=2)
    xy  = pca.fit_transform(path_chi)
    var = pca.explained_variance_ratio_

    # Curvature as terrain colour
    kappas = np.array([p.get("curvature", np.nan) for p in path_data],
                      dtype=float)
    valid  = ~np.isnan(kappas)
    if valid.any():
        kmin, kmax = kappas[valid].min(), kappas[valid].max()
        norm = mcolors.Normalize(vmin=kmin, vmax=kmax)
        cmap_terrain = plt.get_cmap("RdYlBu_r")
        sc = ax.scatter(xy[:, 0], xy[:, 1],
                        c=kappas, cmap=cmap_terrain, norm=norm,
                        s=45, zorder=4, edgecolors="#0d1117", linewidths=0.3)
        cb = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label("κ (curvature)", fontsize=7)
        cb.ax.tick_params(labelsize=6)
    else:
        ax.scatter(xy[:, 0], xy[:, 1], color="#58a6ff", s=45, zorder=4)

    # Flight path gradient line
    if len(xy) >= 2:
        _gradient_path(ax, xy, cmap="Wistia", lw=2.0)

    # Landmark markers
    peak_hops   = {lm["hop"] for lm in landmarks if "PEAK" in lm["tags"]}
    valley_hops = {lm["hop"] for lm in landmarks if "VALLEY" in lm["tags"]}
    cross_hops  = {lm["hop"] for lm in landmarks
                   if any(t.startswith("CROSS_") for t in lm["tags"])}

    valid_idx = [i for i, p in enumerate(path_data)
                 if p.get("chi") and len(p["chi"]) == 5]
    idx_map = {orig: new for new, orig in enumerate(valid_idx)}

    for h in peak_hops:
        if h in idx_map:
            i = idx_map[h]
            ax.scatter(*xy[i], s=160, marker="^", color="#f78166",
                       zorder=7, label="Peak ▲" if h == min(peak_hops) else "")
            ax.annotate(f"↑{h}", xy=xy[i], fontsize=6, color="#f78166",
                        xytext=(2, 4), textcoords="offset points")

    for h in valley_hops:
        if h in idx_map:
            i = idx_map[h]
            ax.scatter(*xy[i], s=160, marker="v", color="#58a6ff",
                       zorder=7, label="Valley ▼" if h == min(valley_hops) else "")
            ax.annotate(f"↓{h}", xy=xy[i], fontsize=6, color="#58a6ff",
                        xytext=(2, -8), textcoords="offset points")

    for h in cross_hops:
        if h in idx_map:
            i = idx_map[h]
            ax.scatter(*xy[i], s=60, marker="|", color="#ffa657",
                       zorder=6, linewidths=1.5)

    # Origin / destination
    if 0 in idx_map:
        ax.scatter(*xy[idx_map[0]], s=220, marker="*", color="#3fb950",
                   zorder=8, label="Origin ★")
    last = len(path_data) - 1
    if last in idx_map:
        ax.scatter(*xy[idx_map[last]], s=180, marker="X", color="#f78166",
                   zorder=8, label="Dest ✕")

    ax.set_title(
        f"Observer's Map — terrain = κ  |  PC1+PC2 = {var[0]+var[1]:.0%} variance",
        fontsize=9, pad=6,
    )
    ax.set_xlabel(f"PC1 ({var[0]:.0%})")
    ax.set_ylabel(f"PC2 ({var[1]:.0%})")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right",
              fontsize=7, facecolor="#21262d", edgecolor="#30363d")
    ax.grid(True, ls="--", alpha=0.3)


def _panel_chi(ax, result: dict) -> None:
    """Panel 2 — all χ angles along the path with octant boundary markers."""
    path_data = result["path"]
    hops      = list(range(len(path_data)))
    crossings = _octant_crossings(path_data)

    for ci, (label, color) in enumerate(zip(CHI_LABELS, CHI_COLORS)):
        vals = [
            p["chi"][ci] if p.get("chi") and len(p["chi"]) > ci else np.nan
            for p in path_data
        ]
        ax.plot(hops, vals, color=color, lw=1.8, label=label,
                marker="o", markersize=3.5)

    for cx in crossings:
        ax.axvline(cx, color="#ffa657", lw=0.8, ls="--", alpha=0.7)

    ax.set_title("Dihedral angles along path  (dashed = octant boundary)",
                 fontsize=9, pad=6)
    ax.set_xlabel("Hop")
    ax.set_ylabel("Angle (°)")
    ax.set_ylim(-185, 185)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(60))
    ax.legend(loc="upper right", ncol=5, fontsize=7,
              facecolor="#21262d", edgecolor="#30363d")
    ax.grid(True, ls="--", alpha=0.4)


def _panel_height(ax, result: dict) -> None:
    """Panel 3 — observer height (reconstruction error) per hop."""
    obs       = result.get("observer", {})
    path_data = result["path"]
    n         = len(path_data)

    heights = [p.get("height", np.nan) for p in path_data]
    all_nan = all(np.isnan(h) if isinstance(h, float) else False
                  for h in heights)

    if all_nan:
        mean_h = obs.get("mean_height")
        if mean_h is not None:
            ax.axhline(mean_h, color="#58a6ff", lw=1.5, ls="--",
                       label=f"mean h = {mean_h:.4f}")
        ax.text(0.5, 0.5,
                "Per-hop heights not in JSON\n(mean shown as dashed line)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color="#8b949e")
    else:
        hops = list(range(n))
        ax.fill_between(hops, heights, alpha=0.25, color="#58a6ff")
        ax.plot(hops, heights, color="#58a6ff", lw=1.8, marker="o",
                markersize=3.5)
        ax.axhline(0, color="#3fb950", lw=0.8, ls=":",
                   label="h = 0 (on manifold)")

    ax.set_title("Observer height h (tangent-plane reconstruction error)",
                 fontsize=9, pad=6)
    ax.set_xlabel("Hop")
    ax.set_ylabel("h")
    ax.legend(loc="upper right", fontsize=7,
              facecolor="#21262d", edgecolor="#30363d")
    ax.grid(True, ls="--", alpha=0.4)


def _panel_curvature(ax, result: dict) -> None:
    """Panel 4 — local curvature κ per hop with crossing markers."""
    obs       = result.get("observer", {})
    path_data = result["path"]
    n         = len(path_data)
    crossings = _octant_crossings(path_data)
    high_hops = obs.get("high_curvature_hops", [])

    kappas  = [p.get("curvature", np.nan) for p in path_data]
    all_nan = all(np.isnan(k) if isinstance(k, float) else False
                  for k in kappas)

    if all_nan:
        mean_k = obs.get("mean_curvature")
        if mean_k is not None:
            ax.axhline(mean_k, color="#d2a8ff", lw=1.5, ls="--",
                       label=f"mean κ = {mean_k:.4f}")
        ax.text(0.5, 0.5,
                "Per-hop curvature not in JSON\n(mean shown as dashed line)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color="#8b949e")
    else:
        hops = list(range(n))
        ax.fill_between(hops, kappas, alpha=0.20, color="#d2a8ff")
        ax.plot(hops, kappas, color="#d2a8ff", lw=1.8, marker="o",
                markersize=3.5)
        for hh in high_hops:
            if hh < n:
                ax.scatter(hh, kappas[hh], color="#f78166", s=70, zorder=5,
                           label="high κ" if hh == high_hops[0] else "")

    for cx in crossings:
        ax.axvline(cx, color="#ffa657", lw=0.8, ls="--", alpha=0.7)

    ax.set_title("Local curvature κ  (dashed = octant boundary)",
                 fontsize=9, pad=6)
    ax.set_xlabel("Hop")
    ax.set_ylabel("κ (mean principal angle, rad)")
    ax.legend(loc="upper right", fontsize=7,
              facecolor="#21262d", edgecolor="#30363d")
    ax.grid(True, ls="--", alpha=0.4)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def make_figure(json_path: str, png_path: str, dpi: int = 150) -> None:
    """Load flight results and write a 4-panel PNG figure.

    :param json_path: Path to ``disulfide_flight_results.json``.
    :param png_path:  Output PNG path.
    :param dpi:       Figure resolution (default 150).
    """
    with open(json_path, encoding="utf-8") as f:
        result = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Disulfide Manifold Flight Report\n"
        f"origin {result['origin_octant']} → dest {result['dest_octant']}  |  "
        f"{result['n_disulfides']:,} disulfides  |  "
        f"{result['path_length']} hops  |  "
        f"dist {result['centroid_distance_deg']:.1f}°  |  "
        f"k={result['config']['k']}  τ={result['config']['tau']}  "
        f"w={result['config']['w']}\n"
        f"{result.get('run_timestamp', '')}",
        fontsize=10, color="#c9d1d9", y=0.995,
    )
    fig.subplots_adjust(hspace=0.38, wspace=0.30, top=0.87)

    _panel_observer_map(axes[0, 0], result)
    _panel_chi(axes[0, 1], result)
    _panel_height(axes[1, 0], result)
    _panel_curvature(axes[1, 1], result)

    fig.text(
        0.5, 0.004,
        "proteusPy · WaveRider · ManifoldObserver  |  "
        "© 2026 Eric G. Suchanek, PhD · Flux-Frontiers · BSD",
        ha="center", fontsize=7, color="#484f58",
    )

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Figure written → {png_path}")


def main():
    """Standalone CLI entry point."""
    p = argparse.ArgumentParser(
        description="Visualise disulfide manifold flight results"
    )
    p.add_argument("--input",  default="benchmarks/disulfide_flight_results.json",
                   help="Path to flight results JSON")
    p.add_argument("--output", default="benchmarks/disulfide_flight_results.png",
                   help="Output PNG path")
    p.add_argument("--dpi",    type=int, default=150, help="Figure DPI")
    args = p.parse_args()
    make_figure(args.input, args.output, dpi=args.dpi)


if __name__ == "__main__":
    main()
