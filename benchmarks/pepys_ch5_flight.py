#!/usr/bin/env python3
# pepys_ch5_flight.py
# Copyright (c) 2026 Eric G. Suchanek, PhD, Flux-Frontiers
# https://github.com/Flux-Frontiers
# License: Elastic 2.0
# Last revised: 2026-03-27 -egs-
"""
pepys_ch5_flight.py
--------------------
The Chapter 5 experiment: destination-relative temporal encoding.

This is the *corrected* temporal flight described in WaveRider Chapter 5 —
not the z-scored absolute encoding from T-1, but the destination-relative
formulation:

    temporal_coord = abs(entry.fractional_year - destination.fractional_year)

Under this encoding, the destination has temporal coordinate 0.  Every other
entry has a positive coordinate equal to its distance in time from the
destination.  The KNN graph pulls the turtle toward zero — toward the
destination — as a gravitational effect of the geometry.

Specific flight: Pepys diary, 1663-10-21 → 1664-01-23.
Uses the full 6450-entry mpnet corpus from diary_kg.

Outputs every hop with date, text, and running Kendall tau.
Saves results JSON and prints mission data appendix.

Confirmed Results (2026-03-27)
------------------------------
Flight: 1663-10-21 → 1664-01-23  (94-day span, Pepys mpnet, 6450 entries)
Path length : 7 hops  |  Reached destination: YES
Final Kendall τ : 0.0476  |  Monotonicity: 33.3 %

The manifold did NOT navigate by falling forward through time.
It navigated by *semantic resonance*.  The dominant pattern in hops 3–6
is the phrase cluster "Sir W. Batten + Sir W. Penn + Whitehall", a
combination that appears in the destination entry (1664-01-23) and recurs
across three different years (1661, 1660, 1666) before landing.

Hop log (actual corpus indices and dates):
  Hop 0  1663-10-21  [idx 2645]  "weather | General | To begin to keep myself …"
  Hop 1  1661-07-22  [idx 1052]  "Up by three, and going by four on my way to London …"
  Hop 2  1663-01-22  [idx 2106]  "Up, and it being a brave morning, with a gally to Woolwich …"
  Hop 3  1661-06-01  [idx  971]  "…Sir W. Batten and my Lady, who are gone this morning …"
  Hop 4  1660-10-09  [idx  537]  "…Sir W. Batten with Colonel Birch … Sir W. Penn and I …"
  Hop 5  1666-02-03  [idx 4269]  "…Sir W. Batten and [Sir] W. Penn to Whitehall …"
  Hop 6  1664-01-23  [idx 2835]  "…Sir W. Batten and Sir W. Penn to Whitehall …"  ← DEST

Running Kendall τ by hop:
  Hops 0-1: —       (need ≥3 points)
  Hop 2:    -0.3333
  Hop 3:    -0.6667
  Hop 4:    -0.8000
  Hop 5:    -0.2000
  Hop 6:     0.0476  ← final

Interpretation
--------------
The clock went everywhere: 1663 → 1661 → 1663 → 1661 → 1660 → 1666 → 1664.
τ near zero is the honest score for a path that arrived at the right
destination via the wrong temporal route.  The geometric gravity of the
destination-relative encoding pulled the turtle to the correct *place*
in semantic space; chronological order was not preserved.

This is the correct finding for Chapter 5: destination-relative encoding
is an effective attractor, but it does not enforce monotonic traversal.
Temporal momentum (Chapter 6 hypothesis) is the missing ingredient.

Usage
-----
  python benchmarks/pepys_ch5_flight.py

  # or with custom corpus path
  python benchmarks/pepys_ch5_flight.py \\
      --corpus /path/to/pepys_mpnet_embeddings.json \\
      --origin-date 1663-10-21 \\
      --dest-date 1664-01-23
"""

from __future__ import annotations

import argparse

# Direct import of turtleND
import importlib.util as _ilu
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

_tnd_path = str(Path(__file__).resolve().parent.parent / "proteusPy" / "turtleND.py")
_spec = _ilu.spec_from_file_location("turtleND", _tnd_path)
_tnd_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_tnd_mod)
TurtleND = _tnd_mod.TurtleND

console = Console()

DEFAULT_CORPUS = str(
    Path(__file__).resolve().parent.parent.parent
    / "diary_kg"
    / "benchmarks"
    / "pepys_mpnet_embeddings.json"
)
DEFAULT_OUT = str(Path(__file__).parent / "pepys_ch5_flight_results.json")
DEFAULT_OUT_PNG = str(Path(__file__).parent / "pepys_ch5_flight_results.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fractional_year(dt: datetime) -> float:
    return dt.year + (dt.timetuple().tm_yday - 1) / 365.25 + dt.hour / (365.25 * 24)


def load_corpus(path: str) -> tuple[np.ndarray, list[str], list[datetime]]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    embeddings = np.array(data["embeddings"], dtype=np.float32)
    texts = data["texts"]
    timestamps = [datetime.fromisoformat(ts) for ts in data["timestamps"]]
    return embeddings, texts, timestamps


def augment_dest_relative(
    embeddings: np.ndarray,
    fyears: np.ndarray,
    dest_fyear: float,
    alpha: float = 1.0,
) -> np.ndarray:
    """Append destination-relative temporal coordinate as (D+1)-th dimension.

    temporal_coord = abs(fyear_i - fyear_dest)

    Normalised so that the mean L2 norm contribution of the temporal axis
    equals alpha times that of a typical semantic axis.

    :param embeddings: (N, D) float array.
    :param fyears: (N,) fractional years.
    :param dest_fyear: fractional year of the destination entry.
    :param alpha: temporal weight relative to a single semantic axis.
    :return: (N, D+1) augmented array.
    """
    t_raw = np.abs(fyears - dest_fyear)  # destination has coord 0

    # Scale to match embedding axis magnitude
    emb_norms = np.linalg.norm(embeddings, axis=1)
    mean_norm = emb_norms.mean()
    scale = alpha * (mean_norm / math.sqrt(embeddings.shape[1]))

    # Normalise t_raw to unit range then apply scale
    t_max = t_raw.max()
    t_scaled = (t_raw / t_max) * scale if t_max > 1e-12 else t_raw * scale

    return np.column_stack([embeddings, t_scaled])


def kendall_tau(times: np.ndarray) -> float:
    """Manual Kendall tau — rank correlation between path order and time."""
    n = len(times)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if times[j] > times[i]:
                concordant += 1
            elif times[j] < times[i]:
                discordant += 1
    denom = concordant + discordant
    return (concordant - discordant) / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def make_figure(
    E_aug: np.ndarray,
    fyears: np.ndarray,
    timestamps: list,
    hops: list[dict],
    out_path: str,
    dpi: int = 150,
) -> None:
    """2-panel figure: PCA scatter with flight path + temporal profile.

    :param E_aug: Augmented embedding array (N, D+1).
    :param fyears: Fractional-year array (N,).
    :param timestamps: List of datetime objects aligned with fyears.
    :param hops: Hop records from run_flight().
    :param out_path: Output PNG path.
    :param dpi: Figure resolution.
    """
    from sklearn.decomposition import PCA

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

    path = [h["idx"] for h in hops]
    color = "#58a6ff"

    # PCA-2D of augmented space
    pca = PCA(n_components=2)
    coords = pca.fit_transform(E_aug)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f"WaveRider Ch5 · Destination-Relative Temporal Flight\n"
        f"{hops[0]['date']} → {hops[-1]['date']}  "
        f"({len(hops)} hops)",
        fontsize=13,
        color="#c9d1d9",
        y=0.98,
    )

    # --- Left panel: PCA scatter + flight path ---
    years = np.array([ts.year for ts in timestamps])
    sc = ax1.scatter(
        coords[:, 0],
        coords[:, 1],
        c=years,
        cmap="plasma",
        s=3,
        alpha=0.25,
        linewidths=0,
    )
    plt.colorbar(sc, ax=ax1, label="Year", fraction=0.03, pad=0.04)

    path_coords = coords[path]
    ax1.plot(
        path_coords[:, 0],
        path_coords[:, 1],
        color=color,
        lw=1.5,
        alpha=0.9,
        zorder=3,
    )
    # Annotate each hop
    for i, (x, y) in enumerate(path_coords):
        ax1.annotate(
            str(i),
            (x, y),
            fontsize=7,
            color="#c9d1d9",
            ha="center",
            va="bottom",
            zorder=6,
        )
    ax1.scatter(
        path_coords[0, 0], path_coords[0, 1],
        c="white", s=80, marker="o", zorder=5, edgecolors=color, linewidths=2,
    )
    ax1.scatter(
        path_coords[-1, 0], path_coords[-1, 1],
        c="white", s=80, marker="*", zorder=5, edgecolors=color, linewidths=2,
    )
    tau_final = hops[-1]["running_tau"]
    tau_str = f"{tau_final:.4f}" if tau_final is not None else "—"
    ax1.set_title(
        f"PCA-2D of augmented space · τ={tau_str}",
        color="#c9d1d9",
        fontsize=10,
    )
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=8)
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Right panel: temporal profile + running tau ---
    path_fyears = fyears[path]
    hop_nums = np.arange(len(path))

    ax2.plot(hop_nums, path_fyears, color=color, lw=1.5, alpha=0.9, label="Year")
    ax2.fill_between(hop_nums, path_fyears, alpha=0.15, color=color)

    # Mark each hop with its date label
    for h in hops:
        ax2.annotate(
            h["date"],
            (h["hop"], fyears[h["idx"]]),
            fontsize=6,
            color="#8b949e",
            rotation=30,
            ha="left",
            va="bottom",
        )

    # Running tau on secondary axis
    tau_hops = [h["hop"] for h in hops if h["running_tau"] is not None]
    tau_vals = [h["running_tau"] for h in hops if h["running_tau"] is not None]
    if tau_hops:
        ax2b = ax2.twinx()
        ax2b.plot(tau_hops, tau_vals, color="#f78166", lw=1.2, linestyle="--", alpha=0.8, label="τ")
        ax2b.axhline(0, color="#f78166", lw=0.5, alpha=0.4)
        ax2b.set_ylabel("Running Kendall τ", fontsize=8, color="#f78166")
        ax2b.tick_params(axis="y", colors="#f78166")
        ax2b.set_ylim(-1.1, 1.1)

    ax2.set_xlabel("Hop", fontsize=9)
    ax2.set_ylabel("Fractional Year", fontsize=9)
    ax2.set_title("Time along path · running Kendall τ", color="#c9d1d9", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]Figure saved → {out_path}[/green]")


# ---------------------------------------------------------------------------
# Flight
# ---------------------------------------------------------------------------


def run_flight(
    E_aug: np.ndarray,
    fyears: np.ndarray,
    texts: list[str],
    timestamps: list[datetime],
    origin_idx: int,
    dest_idx: int,
    k: int = 10,
    max_steps: int = 150,
) -> list[dict]:
    """Run destination-directed greedy KNN flight.

    At each step, choose the neighbour whose direction in augmented space
    most aligns with the vector pointing toward the destination.

    :return: List of hop records (index, date, text snippet, running tau).
    """
    from sklearn.neighbors import NearestNeighbors

    N = len(E_aug)
    k = min(k, N - 1)

    console.print(f"  Building KNN graph (k={k}, N={N}) …")
    t0 = time.time()
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(E_aug)
    _, indices = nbrs.kneighbors(E_aug)
    console.print(f"  KNN built in {time.time() - t0:.1f}s")

    turtle = TurtleND(ndim=E_aug.shape[1], name="PepysCh5")
    turtle.position = E_aug[origin_idx].astype(np.float64)

    current = origin_idx
    path = [origin_idx]
    visited = {origin_idx}

    dest_emb = E_aug[dest_idx].astype(np.float64)

    for step in range(max_steps):
        neighbors = [
            int(j) for j in indices[current] if j != current and j not in visited
        ]
        if not neighbors:
            break

        direction = dest_emb - E_aug[current]
        dn = np.linalg.norm(direction)
        if dn < 1e-10:
            break
        direction /= dn

        best_score = -np.inf
        best_idx = None
        for j in neighbors:
            step_vec = E_aug[j] - E_aug[current]
            norm = np.linalg.norm(step_vec)
            if norm < 1e-10:
                continue
            score = float(np.dot(step_vec / norm, direction))
            if score > best_score:
                best_score = score
                best_idx = j

        if best_idx is None:
            break

        turtle.position = E_aug[best_idx].astype(np.float64)
        current = best_idx
        path.append(best_idx)
        visited.add(best_idx)

        if best_idx == dest_idx:
            break

    # Build hop records
    path_times = fyears[path]
    hops = []
    for hop_n, idx in enumerate(path):
        running_tau = kendall_tau(path_times[: hop_n + 1]) if hop_n > 1 else None
        hops.append(
            {
                "hop": hop_n,
                "idx": int(idx),
                "date": timestamps[idx].strftime("%Y-%m-%d"),
                "text": texts[idx][:120],
                "fyear": round(float(fyears[idx]), 4),
                "running_tau": (
                    round(running_tau, 4) if running_tau is not None else None
                ),
            }
        )

    return hops


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WaveRider Chapter 5 temporal flight")
    p.add_argument(
        "--corpus", default=DEFAULT_CORPUS, help="Path to pepys_mpnet_embeddings.json"
    )
    p.add_argument(
        "--origin-date", default="1663-10-21", help="Target origin date (YYYY-MM-DD)"
    )
    p.add_argument(
        "--origin-entry",
        type=int,
        default=0,
        help="Which entry on origin date to use (0-indexed)",
    )
    p.add_argument(
        "--dest-date", default="1664-01-23", help="Target destination date (YYYY-MM-DD)"
    )
    p.add_argument(
        "--dest-entry",
        type=int,
        default=0,
        help="Which entry on dest date to use (0-indexed)",
    )
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--out-png", default=DEFAULT_OUT_PNG, help="Path for results PNG figure")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    console.rule(
        "[bold blue]WaveRider Chapter 5 — Destination-Relative Temporal Flight"
    )

    # ------------------------------------------------------------------
    # Load corpus
    # ------------------------------------------------------------------
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        console.print(f"[red]Corpus not found: {corpus_path}[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Loading corpus:[/bold] {corpus_path.name} …")
    E, texts, timestamps = load_corpus(str(corpus_path))
    N, D = E.shape
    console.print(
        f"  {N} entries × {D} dims  "
        f"({min(timestamps).date()} → {max(timestamps).date()})"
    )

    # L2-normalise
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    E = E / np.clip(norms, 1e-8, None)

    fyears = np.array([fractional_year(t) for t in timestamps])

    # ------------------------------------------------------------------
    # Find origin and destination indices
    # ------------------------------------------------------------------
    def entries_on_date(date_str: str) -> list[int]:
        return [
            i for i, t in enumerate(timestamps) if t.strftime("%Y-%m-%d") == date_str
        ]

    origin_candidates = entries_on_date(args.origin_date)
    dest_candidates = entries_on_date(args.dest_date)

    if not origin_candidates:
        console.print(f"[red]No entries found for origin date {args.origin_date}[/red]")
        sys.exit(1)
    if not dest_candidates:
        console.print(f"[red]No entries found for dest date {args.dest_date}[/red]")
        sys.exit(1)

    origin_idx = origin_candidates[min(args.origin_entry, len(origin_candidates) - 1)]
    dest_idx = dest_candidates[min(args.dest_entry, len(dest_candidates) - 1)]

    console.print(
        f"\n[bold]Origin:[/bold]  [{origin_idx}] {timestamps[origin_idx].date()}"
    )
    console.print(f"  {texts[origin_idx][:100]}")
    console.print(f"\n[bold]Dest:  [/bold] [{dest_idx}] {timestamps[dest_idx].date()}")
    console.print(f"  {texts[dest_idx][:100]}")

    dest_fyear = fyears[dest_idx]
    span_days = int((fyears[dest_idx] - fyears[origin_idx]) * 365.25)
    console.print(f"\n  Temporal span: {span_days} days")

    # ------------------------------------------------------------------
    # Build destination-relative augmented space
    # ------------------------------------------------------------------
    console.print(
        f"\n[bold]Augmenting:[/bold] destination-relative encoding (α={args.alpha}) …"
    )
    E_aug = augment_dest_relative(E, fyears, dest_fyear, alpha=args.alpha)
    console.print(f"  {D}D → {E_aug.shape[1]}D  (temporal axis appended)")
    t_col = E_aug[:, -1]
    console.print(
        f"  Temporal axis: origin={t_col[origin_idx]:.4f}, "
        f"dest={t_col[dest_idx]:.4f}, "
        f"max={t_col.max():.4f}"
    )

    # ------------------------------------------------------------------
    # Fly
    # ------------------------------------------------------------------
    console.print(
        f"\n[bold]Flying:[/bold] {args.origin_date} → {args.dest_date} "
        f"(max {args.max_steps} hops, k={args.k}) …"
    )
    hops = run_flight(
        E_aug,
        fyears,
        texts,
        timestamps,
        origin_idx,
        dest_idx,
        k=args.k,
        max_steps=args.max_steps,
    )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    path_times = fyears[[h["idx"] for h in hops]]
    final_tau = kendall_tau(path_times)
    monotonicity = (
        float(np.mean(np.diff(path_times) > 0)) if len(path_times) > 1 else 1.0
    )
    reached = hops[-1]["idx"] == dest_idx

    console.print()
    table = Table(
        title=f"Hop Log — {len(hops)} hops  τ={final_tau:.4f}  reached={reached}",
        show_header=True,
    )
    table.add_column("Hop", justify="right", style="dim")
    table.add_column("Date", style="cyan")
    table.add_column("τ (running)", justify="right", style="green")
    table.add_column("Entry (truncated)")

    for h in hops:
        tau_str = f"{h['running_tau']:.3f}" if h["running_tau"] is not None else "—"
        table.add_row(str(h["hop"]), h["date"], tau_str, h["text"][:80])
    console.print(table)

    # ------------------------------------------------------------------
    # Mission Data Appendix
    # ------------------------------------------------------------------
    console.rule("[bold]Mission Data Appendix")
    appendix = Table(show_header=False, box=None)
    appendix.add_column("Parameter", style="bold cyan")
    appendix.add_column("Value")
    rows = [
        ("Corpus", f"Pepys mpnet, {N} entries, {D}D"),
        ("Encoding", "Destination-relative: abs(fyear_i − fyear_dest)"),
        (
            "Origin",
            f"[{origin_idx}] {timestamps[origin_idx].date()} — {texts[origin_idx][:70]}",
        ),
        (
            "Destination",
            f"[{dest_idx}] {timestamps[dest_idx].date()} — {texts[dest_idx][:70]}",
        ),
        ("Temporal span", f"{span_days} days"),
        ("α", str(args.alpha)),
        ("k", str(args.k)),
        ("Path length", str(len(hops))),
        ("Reached destination", str(reached)),
        ("Final Kendall τ", f"{final_tau:.4f}"),
        ("Monotonicity", f"{monotonicity:.1%}"),
        ("First hop date", hops[1]["date"] if len(hops) > 1 else "—"),
        ("Last hop date", hops[-1]["date"]),
    ]
    for k_, v in rows:
        appendix.add_row(k_, v)
    console.print(appendix)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    results = {
        "corpus": str(corpus_path),
        "N": N,
        "D": D,
        "alpha": args.alpha,
        "k": args.k,
        "encoding": "destination_relative",
        "origin_idx": origin_idx,
        "dest_idx": dest_idx,
        "origin_date": timestamps[origin_idx].isoformat(),
        "dest_date": timestamps[dest_idx].isoformat(),
        "span_days": span_days,
        "path_length": len(hops),
        "reached_destination": reached,
        "final_kendall_tau": round(final_tau, 4),
        "monotonicity": round(monotonicity, 4),
        "hops": hops,
    }

    class _NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return super().default(obj)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, cls=_NpEncoder)
    console.print(f"\n[dim]Results saved → {args.out}[/dim]")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    console.print("\n[bold]Generating figure …[/bold]")
    try:
        make_figure(E_aug, fyears, timestamps, hops, args.out_png)
    except Exception as exc:
        console.print(f"[yellow]Figure generation failed: {exc}[/yellow]")
        import traceback
        traceback.print_exc()

    console.rule("[bold green]Done")


if __name__ == "__main__":
    main()
