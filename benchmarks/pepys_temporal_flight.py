#!/usr/bin/env python3
# pepys_temporal_flight.py
# Copyright (c) 2026 Eric G. Suchanek, PhD, Flux-Frontiers
# https://github.com/Flux-Frontiers
# License: Elastic 2.0
# Last revised: 2026-03-26 19:17:39 -egs-
"""
pepys_temporal_flight.py
-------------------------
Temporally-grounded manifold flight through the Samuel Pepys diary.

The Idea
--------
An N-dimensional embedding captures *semantic* structure but discards
the diary's natural *temporal* axis.  This experiment restores time as
a first-class navigable dimension:

1. Load N-dim Pepys embeddings (from the manifold-explorer cache).
2. Parse ISO timestamps → fractional-year scalars.
3. Normalise the temporal scalar to match embedding scale.
4. Augment every embedding from N-dim → (N+1)-dim by appending the
   temporal coordinate.  The result is a *temporally-grounded* space
   where time is literally a direction the turtle can fly.
5. Build a ManifoldModel on the (N+1)-dim space.
6. Demonstrate three flight modes:
   a. **Semantic flight** — fly between the two most-distant entries
      (heading follows the embedding gradient, time is incidental).
   b. **Temporal flight** — orient_in_time(), then fly forward in
      pure chronological order through the manifold.
   c. **Mixed flight** — partial orient toward time, partial toward
      a semantic target — the turtle spirals through time *and* meaning.
7. Compare: how much temporal coherence does each mode exhibit?
8. Visualise the results.

New TurtleND Primitives Used
-----------------------------
- ``expand_dim()`` — grow the turtle's basis set by one axis
- ``orient_in_time()`` — rotate heading to face the temporal axis
- ``orient_toward(direction)`` — rotate heading toward any direction

Usage
-----
  # requires a cached embedding file from pepys_manifold_explorer.py
  python benchmarks/pepys_temporal_flight.py

  # specify cache and output paths
  python benchmarks/pepys_temporal_flight.py \\
      --cache benchmarks/pepys_small_embeddings.json \\
      --alpha 0.3

Requirements
------------
  numpy, scikit-learn, matplotlib, rich  (all in proteusPy env).
  A pre-computed embedding cache JSON from pepys_manifold_explorer.py.
"""

from __future__ import annotations

import argparse

# Direct import of turtleND to avoid pulling in heavy optional deps (pyvista etc.)
# via proteusPy.__init__.  We load the module file directly.
import importlib.util as _ilu  # noqa: E402
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

_tnd_path = str(
    Path(__file__).resolve().parent.parent.parent / "proteusPy" / "proteusPy" / "turtleND.py"
)
_spec = _ilu.spec_from_file_location("turtleND", _tnd_path)
_tnd_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_tnd_mod)
TurtleND = _tnd_mod.TurtleND

console = Console()

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CACHE = str(Path(__file__).parent / "pepys_mpnet_embeddings.json")
DEFAULT_OUT_JSON = str(Path(__file__).parent / "pepys_temporal_flight_results.json")
DEFAULT_OUT_PNG = str(Path(__file__).parent / "pepys_temporal_flight_results.png")


# ---------------------------------------------------------------------------
# Temporal coordinate helpers
# ---------------------------------------------------------------------------
def timestamps_to_fractional_years(timestamps: list[datetime]) -> np.ndarray:
    """Convert datetimes to fractional years (e.g. 1660.5 = July 1660).

    :param timestamps: List of datetime objects.
    :return: 1-D float64 array of fractional years.
    """
    fyears = np.array(
        [
            ts.year + (ts.timetuple().tm_yday - 1) / 365.25 + ts.hour / (365.25 * 24)
            for ts in timestamps
        ],
        dtype=np.float64,
    )
    return fyears


def augment_with_time(
    embeddings: np.ndarray,
    time_values: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Append a scaled temporal coordinate as the (N+1)-th dimension.

    The temporal values are z-scored then multiplied by *alpha* times the
    mean L2 norm of the embedding vectors.  This makes ``alpha=1`` mean
    "time contributes as much as one typical embedding axis."

    :param embeddings: (N, D) float array — original embeddings.
    :param time_values: (N,) float array — temporal scalars (e.g. fractional years).
    :param alpha: Temporal weight.  0 = ignore time, 1 = equal weight, >1 = time-dominant.
    :return: (N, D+1) float array — temporally-grounded embeddings.
    """
    # z-score normalise time
    mu = time_values.mean()
    sigma = time_values.std()
    if sigma < 1e-12:
        t_norm = np.zeros_like(time_values)
    else:
        t_norm = (time_values - mu) / sigma

    # scale to match embedding magnitude
    emb_norms = np.linalg.norm(embeddings, axis=1)
    mean_norm = emb_norms.mean()
    t_scaled = t_norm * alpha * (mean_norm / math.sqrt(embeddings.shape[1]))

    return np.column_stack([embeddings, t_scaled])


# ---------------------------------------------------------------------------
# Embedding cache loader (same format as pepys_manifold_explorer)
# ---------------------------------------------------------------------------
def load_cache(path: str) -> tuple[np.ndarray, list[str], list[datetime]]:
    """Load embeddings + metadata from the JSON cache.

    :param path: Path to cache JSON.
    :return: (embeddings, texts, timestamps).
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    embeddings = np.array(data["embeddings"], dtype=np.float32)
    texts = data["texts"]
    timestamps = [datetime.fromisoformat(ts) for ts in data["timestamps"]]
    return embeddings, texts, timestamps


# ---------------------------------------------------------------------------
# Simple KNN-graph flight (no full ManifoldModel dep — keep it lightweight)
# ---------------------------------------------------------------------------
class TemporalFlyer:
    """Lightweight graph-flight engine over temporally-augmented embeddings.

    Builds a KNN graph, positions a TurtleND in (D+1)-space, and
    supports semantic, temporal, and mixed flight modes.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        timestamps: list[datetime],
        texts: list[str],
        k: int = 10,
        alpha: float = 1.0,
        negate_time: bool = False,
    ):
        from sklearn.neighbors import NearestNeighbors

        self.texts = texts
        self.timestamps = timestamps
        self.alpha = alpha
        self.negate_time = negate_time

        # Fractional years & augmented embeddings
        self.fyears = timestamps_to_fractional_years(timestamps)
        fyears_for_aug = -self.fyears if negate_time else self.fyears
        self.E_orig = embeddings.copy()
        self.E_aug = augment_with_time(embeddings, fyears_for_aug, alpha=alpha)
        self.N, self.D = self.E_aug.shape  # D = original_dim + 1
        self.time_axis = self.D - 1

        # L2-normalise the augmented embeddings for cosine-like flight
        norms = np.linalg.norm(self.E_aug, axis=1, keepdims=True)
        self.E_norm = self.E_aug / np.clip(norms, 1e-8, None)

        # Build KNN graph
        self.k = min(k, self.N - 1)
        nbrs = NearestNeighbors(n_neighbors=self.k + 1, metric="euclidean")
        nbrs.fit(self.E_aug)
        self.distances, self.indices = nbrs.kneighbors(self.E_aug)

        # Turtle in (D)-dimensional augmented space
        self.turtle = TurtleND(ndim=self.D, name="PepysTemporal")
        self._current = None
        self._path = []

    def fly_to(self, idx: int) -> None:
        """Position turtle at entry *idx*."""
        self.turtle.position = self.E_aug[idx].astype(np.float64)
        self._current = idx
        self._path = [idx]

    def _neighbors(self, idx: int) -> list[int]:
        """Return KNN neighbor indices for entry *idx* (excluding self)."""
        return [int(j) for j in self.indices[idx] if j != idx]

    def fly_step(
        self,
        direction: np.ndarray | None = None,
        excluded: set | None = None,
    ) -> int | None:
        """Take one graph step toward *direction* (or turtle heading).

        :param direction: Preferred direction in augmented space.
        :param excluded: Indices to skip.
        :return: Index moved to, or None if stuck.
        """
        if self._current is None:
            raise RuntimeError("Call fly_to() first")

        if direction is not None:
            heading = np.asarray(direction, dtype=np.float64)
            n = np.linalg.norm(heading)
            heading = heading / n if n > 1e-10 else self.turtle.heading
        else:
            heading = self.turtle.heading

        current_emb = self.E_aug[self._current]
        best_score = -np.inf
        best_idx = None

        for j in self._neighbors(self._current):
            if excluded and j in excluded:
                continue
            step = self.E_aug[j] - current_emb
            norm = np.linalg.norm(step)
            if norm < 1e-10:
                continue
            alignment = float(np.dot(step / norm, heading))
            if alignment > best_score:
                best_score = alignment
                best_idx = j

        if best_idx is None:
            return None

        self.turtle.position = self.E_aug[best_idx].astype(np.float64)
        self._current = best_idx
        self._path.append(best_idx)
        return best_idx

    # ------- Flight modes -------

    def semantic_flight(self, origin: int, dest: int, max_steps: int = 100) -> list[int]:
        """Fly from *origin* toward *dest* using pure semantic heading."""
        self.fly_to(origin)
        target = self.E_aug[dest].astype(np.float64)
        visited: set[int] = {origin}

        for _ in range(max_steps):
            direction = target - self.E_aug[self._current]
            nxt = self.fly_step(direction=direction, excluded=visited)
            if nxt is None:
                break
            visited.add(nxt)
            if nxt == dest:
                break
        return list(self._path)

    def temporal_flight(self, origin: int, max_steps: int = 100, forward: bool = True) -> list[int]:
        """Fly from *origin* along the pure temporal axis.

        :param forward: True = advance in time, False = go backward.
        """
        self.fly_to(origin)
        self.turtle.orient_in_time(self.time_axis)
        if not forward:
            self.turtle.rotate(180, 0, self.time_axis)

        visited: set[int] = {origin}
        for _ in range(max_steps):
            nxt = self.fly_step(excluded=visited)
            if nxt is None:
                break
            visited.add(nxt)
        return list(self._path)

    def mixed_flight(
        self,
        origin: int,
        dest: int,
        time_blend: float = 0.5,
        max_steps: int = 100,
    ) -> list[int]:
        """Fly with a blend of semantic and temporal heading.

        :param time_blend: 0 = pure semantic, 1 = pure temporal, 0.5 = equal mix.
        """
        self.fly_to(origin)
        target = self.E_aug[dest].astype(np.float64)

        # Temporal direction unit vector
        e_t = np.zeros(self.D, dtype=np.float64)
        sign = 1.0 if self.fyears[dest] >= self.fyears[origin] else -1.0
        e_t[self.time_axis] = sign

        visited: set[int] = {origin}
        for _ in range(max_steps):
            semantic_dir = target - self.E_aug[self._current]
            sn = np.linalg.norm(semantic_dir)
            if sn > 1e-10:
                semantic_dir /= sn
            blended = (1 - time_blend) * semantic_dir + time_blend * e_t
            nxt = self.fly_step(direction=blended, excluded=visited)
            if nxt is None:
                break
            visited.add(nxt)
            if nxt == dest:
                break
        return list(self._path)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def temporal_coherence(path: list[int], fyears: np.ndarray) -> dict:
    """Measure how temporally ordered a flight path is.

    Returns metrics:
    - monotonicity: fraction of consecutive hops that move forward in time
    - mean_dt: average time step (years) per hop
    - total_span: temporal span from first to last entry on path
    - kendall_tau: Kendall rank correlation between path order and time
    """
    if len(path) < 2:
        return {
            "monotonicity": 1.0,
            "mean_dt": 0.0,
            "total_span": 0.0,
            "kendall_tau": 1.0,
        }

    times = fyears[path]
    dts = np.diff(times)

    monotonicity = float(np.mean(dts > 0))
    mean_dt = float(np.mean(np.abs(dts)))
    total_span = float(times[-1] - times[0])

    # Kendall tau (manual — avoid scipy dependency)
    n = len(times)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if times[j] > times[i]:
                concordant += 1
            elif times[j] < times[i]:
                discordant += 1
    denom = concordant + discordant
    tau = (concordant - discordant) / denom if denom > 0 else 0.0

    return {
        "monotonicity": round(monotonicity, 4),
        "mean_dt_years": round(mean_dt, 4),
        "total_span_years": round(total_span, 2),
        "kendall_tau": round(tau, 4),
    }


def find_distant_pair(E: np.ndarray, n_sample: int = 500) -> tuple[int, int]:
    """Find the two entries most distant in cosine space (sampled)."""
    rng = np.random.default_rng(42)
    n = min(len(E), n_sample)
    idx = rng.choice(len(E), size=n, replace=False)
    E_s = E[idx]
    norms = np.linalg.norm(E_s, axis=1, keepdims=True)
    E_s_n = E_s / np.clip(norms, 1e-8, None)
    sims = E_s_n @ E_s_n.T
    np.fill_diagonal(sims, 1.0)
    i_loc, j_loc = np.unravel_index(np.argmin(sims), sims.shape)
    return int(idx[i_loc]), int(idx[j_loc])


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def make_figure(
    flyer: TemporalFlyer,
    paths: dict[str, list[int]],
    coherence: dict[str, dict],
    out_path: str,
    dpi: int = 150,
) -> None:
    """8-panel figure: 4 flight-path scatters + 4 temporal profiles."""

    dark = {
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "text.color": "#c9d1d9",
        "grid.color": "#21262d",
        "grid.linewidth": 0.6,
    }
    mpl.rcParams.update(dark)

    from sklearn.decomposition import PCA

    plot_modes = ["semantic", "temporal", "temporal_backward", "mixed"]
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle(
        "Pepys Diary · Temporal Flight Experiment",
        fontsize=15,
        color="#c9d1d9",
        y=0.98,
    )

    # PCA-2D of augmented space for scatter plots
    pca = PCA(n_components=2)
    coords = pca.fit_transform(flyer.E_aug)

    colors = {
        "semantic": "#58a6ff",
        "temporal": "#3fb950",
        "temporal_backward": "#f0a500",
        "mixed": "#f78166",
    }
    mode_labels = {
        "semantic": "Semantic Flight",
        "temporal": "Temporal Flight (→)",
        "temporal_backward": "Temporal Backward (←)",
        "mixed": "Mixed Flight (50/50)",
    }

    for col, mode in enumerate(plot_modes):
        path = paths[mode]
        coh = coherence[mode]
        color = colors[mode]

        # --- Top row: PCA scatter with flight path ---
        ax = axes[0, col]
        # Background points coloured by year
        years = np.array([ts.year for ts in flyer.timestamps])
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=years,
            cmap="plasma",
            s=3,
            alpha=0.25,
            linewidths=0,
        )

        # Flight path
        path_coords = coords[path]
        ax.plot(
            path_coords[:, 0],
            path_coords[:, 1],
            color=color,
            lw=1.2,
            alpha=0.8,
            zorder=3,
        )
        ax.scatter(
            path_coords[0, 0],
            path_coords[0, 1],
            c="white",
            s=60,
            marker="o",
            zorder=5,
            edgecolors=color,
            linewidths=2,
        )
        ax.scatter(
            path_coords[-1, 0],
            path_coords[-1, 1],
            c="white",
            s=60,
            marker="*",
            zorder=5,
            edgecolors=color,
            linewidths=2,
        )

        ax.set_title(
            f"{mode_labels[mode]}\n({len(path)} hops, τ={coh['kendall_tau']:.2f})",
            color="#c9d1d9",
            fontsize=10,
        )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=8)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- Bottom row: temporal profile along path ---
        ax2 = axes[1, col]
        path_times = flyer.fyears[path]
        hops = np.arange(len(path))
        ax2.plot(hops, path_times, color=color, lw=1.5, alpha=0.9)
        ax2.fill_between(hops, path_times, alpha=0.15, color=color)
        ax2.set_xlabel("Hop", fontsize=9)
        ax2.set_ylabel("Year (fractional)", fontsize=9)
        ax2.set_title(
            f"Time along path · mono={coh['monotonicity']:.0%}  "
            f"span={coh['total_span_years']:.1f}yr",
            color="#c9d1d9",
            fontsize=10,
        )
        ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]Figure saved → {out_path}[/green]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pepys diary temporal flight experiment")
    p.add_argument(
        "--cache",
        default=DEFAULT_CACHE,
        help="Path to embedding cache JSON from pepys_manifold_explorer.py",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Temporal axis weight (0=ignore time, 1=equal, >1=time-dominant)",
    )
    p.add_argument(
        "--k",
        type=int,
        default=10,
        help="KNN graph neighbors (default: 10)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=150,
        help="Max hops per flight (default: 150)",
    )
    p.add_argument(
        "--time-blend",
        type=float,
        default=0.5,
        help="Temporal blend for mixed flight (0=semantic, 1=temporal)",
    )
    p.add_argument(
        "--out-json",
        default=DEFAULT_OUT_JSON,
        help="Path for results JSON",
    )
    p.add_argument(
        "--out-png",
        default=DEFAULT_OUT_PNG,
        help="Path for figure PNG",
    )
    p.add_argument(
        "--negate-time",
        action="store_true",
        default=False,
        help="Negate the temporal coordinate before augmenting (reverses the time axis sign)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    console.rule("[bold blue]Pepys Diary · Temporal Flight Experiment")

    # ------------------------------------------------------------------
    # Step 1: Load cached embeddings
    # ------------------------------------------------------------------
    cache_path = Path(args.cache)
    if not cache_path.exists():
        console.print(
            f"[red]Embedding cache not found: {cache_path}\n"
            "Run pepys_manifold_explorer.py first to generate embeddings.[/red]"
        )
        sys.exit(1)

    console.print(f"\n[bold]Step 1:[/bold] Loading embeddings from {cache_path} …")
    E, texts, timestamps = load_cache(str(cache_path))
    N, orig_dim = E.shape
    console.print(
        f"  {N} entries × {orig_dim} dims, {min(timestamps).date()} → {max(timestamps).date()}"
    )

    # L2-normalise
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    E = E / np.clip(norms, 1e-8, None)

    # ------------------------------------------------------------------
    # Step 2: Build temporally-augmented space
    # ------------------------------------------------------------------
    console.print(
        f"\n[bold]Step 2:[/bold] Augmenting with temporal axis (α={args.alpha}"
        f"{', time axis negated' if args.negate_time else ''}) …"
    )
    fyears = timestamps_to_fractional_years(timestamps)
    fyears_for_aug = -fyears if args.negate_time else fyears
    E_aug = augment_with_time(E, fyears_for_aug, alpha=args.alpha)
    console.print(
        f"  Augmented: {E_aug.shape[0]} × {E_aug.shape[1]} dims "
        f"(+1 temporal axis at index {E_aug.shape[1] - 1})"
    )

    # Show temporal axis stats
    t_col = E_aug[:, -1]
    console.print(
        f"  Temporal axis: min={t_col.min():.3f}, max={t_col.max():.3f}, std={t_col.std():.3f}"
    )

    # ------------------------------------------------------------------
    # Step 3: Build flyer and find distant pair
    # ------------------------------------------------------------------
    console.print(f"\n[bold]Step 3:[/bold] Building KNN graph (k={args.k}) …")
    t0 = time.time()
    flyer = TemporalFlyer(
        E,
        timestamps,
        texts,
        k=args.k,
        alpha=args.alpha,
        negate_time=args.negate_time,
    )
    elapsed = time.time() - t0
    console.print(f"  Built in {elapsed:.1f}s")

    i_orig, i_dest = find_distant_pair(E)
    console.print(f'  Origin : [{i_orig}] {timestamps[i_orig].date()} — "{texts[i_orig][:60]}…"')
    console.print(f'  Dest   : [{i_dest}] {timestamps[i_dest].date()} — "{texts[i_dest][:60]}…"')

    # ------------------------------------------------------------------
    # Step 4: Three flight modes
    # ------------------------------------------------------------------
    console.print(f"\n[bold]Step 4:[/bold] Flying three modes (max {args.max_steps} hops) …")

    paths = {}
    coherence = {}

    # 4a: Semantic flight
    console.print("  [cyan]Semantic flight …[/cyan]")
    paths["semantic"] = flyer.semantic_flight(i_orig, i_dest, max_steps=args.max_steps)
    coherence["semantic"] = temporal_coherence(paths["semantic"], flyer.fyears)

    # 4b: Temporal flight (forward from origin)
    console.print("  [green]Temporal flight …[/green]")
    paths["temporal"] = flyer.temporal_flight(i_orig, max_steps=args.max_steps, forward=True)
    coherence["temporal"] = temporal_coherence(paths["temporal"], flyer.fyears)

    # 4c: Mixed flight
    console.print("  [red]Mixed flight …[/red]")
    paths["mixed"] = flyer.mixed_flight(
        i_orig, i_dest, time_blend=args.time_blend, max_steps=args.max_steps
    )
    coherence["mixed"] = temporal_coherence(paths["mixed"], flyer.fyears)

    # 4d: Temporal backward — reversed tau test
    console.print("  [yellow]Temporal backward (τ-reversal test) …[/yellow]")
    paths["temporal_backward"] = flyer.temporal_flight(
        i_orig, max_steps=args.max_steps, forward=False
    )
    coherence["temporal_backward"] = temporal_coherence(paths["temporal_backward"], flyer.fyears)

    # ------------------------------------------------------------------
    # Step 5: Results table
    # ------------------------------------------------------------------
    all_modes = ["semantic", "temporal", "temporal_backward", "mixed"]
    console.print()
    table = Table(title="Temporal Coherence by Flight Mode", show_header=True)
    table.add_column("Mode", style="cyan")
    table.add_column("Hops", justify="right")
    table.add_column("Monotonicity", justify="right", style="green")
    table.add_column("Kendall τ", justify="right", style="green")
    table.add_column("Mean Δt (yr)", justify="right")
    table.add_column("Span (yr)", justify="right")

    for mode in all_modes:
        c = coherence[mode]
        table.add_row(
            mode.replace("_", " ").capitalize(),
            str(len(paths[mode])),
            f"{c['monotonicity']:.1%}",
            f"{c['kendall_tau']:.3f}",
            f"{c['mean_dt_years']:.3f}",
            f"{c['total_span_years']:.1f}",
        )
    console.print(table)

    # τ-reversal symmetry check
    tau_fwd = coherence["temporal"]["kendall_tau"]
    tau_bwd = coherence["temporal_backward"]["kendall_tau"]
    tau_sum = tau_fwd + tau_bwd
    console.print(
        f"\n[bold]τ-reversal symmetry:[/bold]  "
        f"τ(forward)={tau_fwd:+.3f}  τ(backward)={tau_bwd:+.3f}  "
        f"sum={tau_sum:+.3f}  (ideal: 0.000)"
    )

    # ------------------------------------------------------------------
    # Step 6: TurtleND primitive demo
    # ------------------------------------------------------------------
    console.print("\n[bold]Step 5:[/bold] TurtleND temporal primitive demo …")
    t = TurtleND(ndim=orig_dim, name="DemoTurtle")
    console.print(f"  Created {orig_dim}D turtle")

    time_idx = t.expand_dim()
    console.print(f"  expand_dim() → ndim={t.ndim}, time_axis={time_idx}")

    t.position = flyer.E_aug[i_orig].astype(np.float64)
    ang = t.orient_in_time(time_idx)
    console.print(f"  orient_in_time() rotated heading {ang:.1f}° toward axis {time_idx}")
    console.print(f"  heading[time_axis] = {t.heading[time_idx]:.4f}")

    # Move forward in time
    t.recording = True
    t.move(0.1)
    console.print(
        f"  move(0.1) → temporal coord went from "
        f"{flyer.E_aug[i_orig, -1]:.3f} to {t.position[time_idx]:.3f}"
    )

    # ------------------------------------------------------------------
    # Step 7: Save results
    # ------------------------------------------------------------------
    tau_symmetry = round(
        coherence["temporal"]["kendall_tau"] + coherence["temporal_backward"]["kendall_tau"],
        4,
    )
    results = {
        "corpus_size": N,
        "original_dim": orig_dim,
        "augmented_dim": int(E_aug.shape[1]),
        "alpha": args.alpha,
        "negate_time": args.negate_time,
        "time_blend": args.time_blend,
        "k_graph": args.k,
        "max_steps": args.max_steps,
        "origin_idx": i_orig,
        "dest_idx": i_dest,
        "origin_date": timestamps[i_orig].isoformat(),
        "dest_date": timestamps[i_dest].isoformat(),
        "tau_reversal_symmetry": tau_symmetry,
        "flights": {},
    }
    for mode in ["semantic", "temporal", "temporal_backward", "mixed"]:
        results["flights"][mode] = {
            "path_length": len(paths[mode]),
            "coherence": coherence[mode],
            "path_dates": [timestamps[i].isoformat() for i in paths[mode]],
        }

    class _NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, cls=_NpEncoder)
    console.print(f"\n[dim]Results saved → {args.out_json}[/dim]")

    # ------------------------------------------------------------------
    # Step 8: Figure
    # ------------------------------------------------------------------
    console.print("\n[bold]Generating figure …[/bold]")
    try:
        make_figure(flyer, paths, coherence, args.out_png)  # now 8-panel (4 modes)
    except Exception as exc:
        console.print(f"[yellow]Figure generation failed: {exc}[/yellow]")
        import traceback

        traceback.print_exc()

    console.rule("[bold green]Done")


if __name__ == "__main__":
    main()
