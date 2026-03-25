#!/usr/bin/env python3
# pepys_manifold_explorer.py
# Copyright (c) 2026 Eric G. Suchanek, PhD, Flux-Frontiers
# https://github.com/Flux-Frontiers
# License: BSD
# Last revised: 2026-03-24 -egs-
"""
pepys_manifold_explorer.py
--------------------------
Explore the nomic-embed-text manifold using the Samuel Pepys diary as corpus.

The diary (1660–1669) provides 3355 temporally grounded entries — each dated
to the day — giving a ground-truth "time axis" against which we can measure
how much temporal structure the embedding manifold retains.

Pipeline
--------
1. Parse pepys_clean.txt (pipe-delimited: TIMESTAMP | TYPE | CATEGORY | CONTENT).
2. Subsample if requested; embed via sentence-transformers (nomic-ai/nomic-embed-text-v1).
   Embeddings are cached to JSON so re-runs are instant.
3. Intrinsic dimensionality:
     - PCA explained-variance elbow (90 / 95 / 99 %)
     - Participation Ratio: PR = (Σλ)² / Σλ²
     - TwoNN estimator (Facco et al. 2017)
4. MRL truncation quality: MRR@10 at 64 / 128 / 256 / 512 / 768 D using
   Pepys-specific queries tied to known historical events in the diary.
5. ManifoldWalker flight: pick the two entries most distant in the 768-D
   cosine space, fly origin → destination, observe the path from N+1 dims.
6. Save results JSON + 4-panel figure.

Usage
-----
  # full run (all 3355 entries — slow first time, cached after)
  python benchmarks/pepys_manifold_explorer.py

  # quick test (500 entries)
  python benchmarks/pepys_manifold_explorer.py --n 500

  # use a specific diary path
  python benchmarks/pepys_manifold_explorer.py \\
      --diary ../diary_kg/pepys/pepys_clean.txt

  # reload from embedding cache
  python benchmarks/pepys_manifold_explorer.py --cache benchmarks/pepys_embeddings.json

Requirements
------------
  All already in proteusPy env (numpy, scikit-learn, rich, matplotlib, sentence-transformers, tqdm).
  No external services required — model is loaded directly via HuggingFace.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

console = Console()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1"  # HF model id, loaded via sentence-transformers

MRL_DIMS = [64, 128, 256, 512, 768]
K_RETRIEVAL = 10

DEFAULT_DIARY = str(
    Path(__file__).parent.parent.parent / "diary_kg" / "pepys" / "pepys_clean.txt"
)
DEFAULT_CACHE = str(Path(__file__).parent / "pepys_embeddings.json")
DEFAULT_OUT_JSON = str(Path(__file__).parent / "pepys_manifold_results.json")
DEFAULT_OUT_PNG = str(Path(__file__).parent / "pepys_manifold_results.png")

# ---------------------------------------------------------------------------
# Known Pepys events with rough date ranges for retrieval ground-truth.
# Each entry: (query_text, year_start, year_end, month_start, month_end)
# We mark corpus entries in that date range as "relevant".
# ---------------------------------------------------------------------------
PEPYS_QUERIES = [
    ("The Great Plague of London, disease, death, sickness", 1665, 1666, 6, 12),
    ("Great Fire of London, fire burning city", 1666, 1666, 9, 9),
    ("King Charles II coronation, restoration monarchy", 1660, 1661, 4, 6),
    ("Theatre, playhouse, actors, entertainment, comedy", 1660, 1669, 1, 12),
    ("Navy administration, fleet, ships, Admiral Sandwich", 1660, 1665, 1, 12),
    ("Samuel Pepys wife Elizabeth, domestic life, household", 1660, 1669, 1, 12),
    ("Parliament, politics, statecraft, Council", 1660, 1669, 1, 12),
    ("Dutch War, naval battle, enemy fleet, de Ruyter", 1665, 1667, 1, 12),
    # ---- new topic types from pepys_enriched_full.txt ----
    ("Church, sermon, prayer, religion, worship, God", 1660, 1669, 1, 12),
    ("Travel, locations, streets, Thames, Westminster, Whitehall", 1660, 1669, 1, 12),
    ("Money, accounts, fees, salary, expenses, financial dealings", 1660, 1669, 1, 12),
    ("Social gathering, entertainment, friends, tavern, dining out", 1660, 1669, 1, 12),
    (
        "Emotion, fear, joy, anxiety, anger, personal feelings, health",
        1660,
        1669,
        1,
        12,
    ),
]


# ---------------------------------------------------------------------------
# Diary parser (no diary_kg dependency — parse the pipe-delimited format)
# ---------------------------------------------------------------------------
def parse_diary(path: str) -> tuple[list[str], list[datetime]]:
    """Parse a pipe-delimited diary file and return (texts, timestamps).

    Expected line format::

        TIMESTAMP | TYPE | CATEGORY | CONTENT

    The ``TYPE`` (e.g. ``pepys_domestic``, ``pepys_naval``) and ``CATEGORY``
    (e.g. ``Home``, ``Finance``, ``Health``) are prepended to the content so
    that topic-level signal is preserved in the embedding space.  Comment
    lines beginning with ``#`` and section headers are silently skipped.

    Always parses the full file; use :func:`temporally_sample` afterwards to
    subsample while preserving temporal diversity.

    :param path: Path to the diary file.
    :return: Tuple of (content strings, datetime objects).
    """
    texts: list[str] = []
    timestamps: list[datetime] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|", 3)
            if len(parts) < 4:
                continue
            ts_str, entry_type, category, content = [p.strip() for p in parts]
            content = content.strip()
            if not content or content == ".":
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                continue
            # Prepend topic and category so the embedding captures
            # the enriched classification from pepys_enriched_full.txt.
            topic_label = entry_type.replace("_", " ").strip()
            embed_text = f"{topic_label} | {category} | {content}"
            texts.append(embed_text)
            timestamps.append(ts)
    return texts, timestamps


def temporally_sample(
    texts: list[str], timestamps: list[datetime], n: int
) -> tuple[list[str], list[datetime]]:
    """Return *n* entries sampled evenly across the full time span.

    Entries are assumed to be sorted chronologically (as produced by
    :func:`parse_diary`).  We pick indices at equal intervals so that the
    sample covers the whole 1660-1669 arc rather than clustering at the start.

    :param texts: Full list of entry strings.
    :param timestamps: Corresponding datetime objects (same length, sorted).
    :param n: Desired sample size.  If >= len(texts) the full corpus is returned.
    :return: Tuple of (sampled texts, sampled timestamps).
    """
    total = len(texts)
    if n <= 0 or n >= total:
        return texts, timestamps
    indices = [round(i * (total - 1) / (n - 1)) for i in range(n)]
    return [texts[i] for i in indices], [timestamps[i] for i in indices]


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def embed_local(
    texts: list[str],
    model: str = DEFAULT_MODEL,
    batch_size: int = 64,
    checkpoint_path: str | None = None,
    checkpoint_texts: list[str] | None = None,
    checkpoint_timestamps: list[datetime] | None = None,
) -> np.ndarray:
    """Embed texts directly via sentence-transformers (no ollama required).

    Loads the model once, then encodes in batches with a progress bar.
    Writes a checkpoint to *checkpoint_path* every 200 entries so a crash
    mid-run doesn't lose all work.

    :param texts: List of strings to embed.
    :param model: HuggingFace model id (default: DEFAULT_MODEL).
    :param batch_size: Encoding batch size (tune to GPU/RAM).
    :param checkpoint_path: If set, save a partial cache every 200 entries.
    :param checkpoint_texts: Full aligned text list used when writing checkpoints.
    :param checkpoint_timestamps: Full aligned timestamp list for checkpoints.
    :return: Float32 array of shape (N, D).
    """
    console.print(f"  Loading embedder: {model} …")
    embedder = SentenceTransformer(model, trust_remote_code=True)

    vecs: list[np.ndarray] = []
    checkpoint_interval = 200

    for batch_start in tqdm(
        range(0, len(texts), batch_size),
        desc="Embedding",
        unit="batch",
    ):
        batch = texts[batch_start : batch_start + batch_size]
        # nomic-embed-text expects a task-type prefix for search/retrieval
        prefixed = [f"search_document: {t}" for t in batch]
        batch_vecs = embedder.encode(
            prefixed, convert_to_numpy=True, show_progress_bar=False
        )
        vecs.append(batch_vecs.astype(np.float32))

        n_done = batch_start + len(batch)
        if (
            checkpoint_path
            and checkpoint_texts is not None
            and checkpoint_timestamps is not None
            and n_done % checkpoint_interval == 0
        ):
            partial = np.concatenate(vecs, axis=0)
            save_cache(
                checkpoint_path,
                partial,
                checkpoint_texts[: len(partial)],
                checkpoint_timestamps[: len(partial)],
            )

    return np.concatenate(vecs, axis=0) if vecs else np.empty((0, 0), dtype=np.float32)


# ---------------------------------------------------------------------------
# Embedding cache: save / load (timestamps stored as ISO strings)
# ---------------------------------------------------------------------------
def save_cache(
    path: str, embeddings: np.ndarray, texts: list[str], timestamps: list[datetime]
) -> None:
    """Persist embeddings and metadata to a JSON cache file.

    :param path: Output JSON path.
    :param embeddings: Float32 array of shape (N, D).
    :param texts: Corpus text strings.
    :param timestamps: Corresponding datetime objects.
    """
    data = {
        "embeddings": embeddings.tolist(),
        "texts": texts,
        "timestamps": [ts.isoformat() for ts in timestamps],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    console.print(f"[dim]Embedding cache saved → {path}[/dim]")


def load_cache(path: str) -> tuple[np.ndarray, list[str], list[datetime]]:
    """Load embeddings and metadata from a JSON cache file.

    :param path: Path to the JSON cache.
    :return: Tuple of (embeddings, texts, timestamps).
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    embeddings = np.array(data["embeddings"], dtype=np.float32)
    texts = data["texts"]
    timestamps = [datetime.fromisoformat(ts) for ts in data["timestamps"]]
    console.print(f"[dim]Embedding cache loaded ← {path}  ({len(texts)} entries)[/dim]")
    return embeddings, texts, timestamps


# ---------------------------------------------------------------------------
# Intrinsic dimensionality estimators
# ---------------------------------------------------------------------------
def participation_ratio(eigenvalues: np.ndarray) -> float:
    """PR = (Σλ)² / Σλ² — effective number of active dimensions."""
    lam = eigenvalues[eigenvalues > 0]
    return float(lam.sum() ** 2 / (lam**2).sum())


def twonn_id(X: np.ndarray) -> float:
    """TwoNN intrinsic dimensionality estimator (Facco et al. 2017).

    :param X: Data matrix (N, D), L2-normalised.
    :return: Estimated intrinsic dimensionality.
    """
    nbrs = NearestNeighbors(n_neighbors=3, metric="cosine").fit(X)
    distances, _ = nbrs.kneighbors(X)
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    mask = r1 > 0
    mu = r2[mask] / r1[mask]
    return float(-len(mu) / np.sum(np.log(mu)))


def elbow_pca(eigenvalues: np.ndarray, threshold: float = 0.90) -> int:
    """Number of PCA components needed to explain *threshold* variance.

    :param eigenvalues: PCA explained variance array.
    :param threshold: Cumulative variance threshold (0–1).
    :return: Number of components.
    """
    cumvar = np.cumsum(eigenvalues) / eigenvalues.sum()
    return int(np.searchsorted(cumvar, threshold)) + 1


# ---------------------------------------------------------------------------
# Retrieval evaluation (temporal ground-truth)
# ---------------------------------------------------------------------------
def build_retrieval_pairs(
    timestamps: list[datetime],
) -> list[tuple[str, list[int]]]:
    """Build (query_text, relevant_indices) pairs from PEPYS_QUERIES.

    A corpus entry is "relevant" to a query if its timestamp falls within
    the (year_start, year_end, month_start, month_end) window.

    :param timestamps: Datetime list aligned with the corpus.
    :return: List of (query_string, [relevant_corpus_indices]).
    """
    pairs = []
    for query_text, yr0, yr1, mo0, mo1 in PEPYS_QUERIES:
        relevant = [
            i
            for i, ts in enumerate(timestamps)
            if (
                yr0 <= ts.year <= yr1
                and (yr0 < ts.year or ts.month >= mo0)
                and (ts.year < yr1 or ts.month <= mo1)
            )
        ]
        if relevant:
            pairs.append((query_text, relevant))
    return pairs


def eval_retrieval(
    E: np.ndarray,
    Q: np.ndarray,
    rel_lists: list[list[int]],
    k: int = 10,
) -> float:
    """MRR@K over a set of pre-embedded query vectors.

    :param E: Corpus embeddings (N, D), L2-normalised.
    :param Q: Query embeddings (M, D), L2-normalised.
    :param rel_lists: Relevant corpus indices for each query.
    :param k: Rank cutoff.
    :return: Mean Reciprocal Rank.
    """
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(E)), metric="cosine").fit(E)
    scores = []
    for qvec, rel in zip(Q, rel_lists):
        _, indices = nbrs.kneighbors(qvec.reshape(1, -1))
        retrieved = list(indices[0])[:k]
        rr = 0.0
        for rank, idx in enumerate(retrieved, 1):
            if idx in rel:
                rr = 1.0 / rank
                break
        scores.append(rr)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(
    E_full: np.ndarray,
    timestamps: list[datetime],
    results: dict,
    flight_obs: dict | None,
    out_path: str,
    dpi: int = 150,
) -> None:
    """Produce a 4-panel figure summarising the Pepys manifold exploration.

    Panels
    ------
    1 (top-left)    PCA-2D scatter coloured by year — temporal topology.
    2 (top-right)   MRL MRR@10 bar chart across truncation dimensions.
    3 (bottom-left) Observer height along the manifold flight path (if available).
    4 (bottom-right) Observer curvature along the flight path (if available).

    :param E_full: Full 768-D embedding matrix (N, 768), L2-normalised.
    :param timestamps: Aligned timestamps for each entry.
    :param results: Dict from main() containing metrics.
    :param flight_obs: Output of ManifoldObserver.observe_path(), or None.
    :param out_path: PNG output path.
    :param dpi: Figure DPI.
    """
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

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Pepys Diary · nomic-embed-text Manifold", fontsize=15, color="#c9d1d9", y=0.98
    )

    years = np.array([ts.year for ts in timestamps])

    # --- Panel 1: temporal PCA scatter ---
    ax = axes[0, 0]
    pca2 = PCA(n_components=2)
    coords = pca2.fit_transform(E_full)
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=years,
        cmap="plasma",
        s=4,
        alpha=0.6,
        linewidths=0,
    )
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Year", color="#c9d1d9", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8b949e", fontsize=8)
    ax.set_title("PCA-2D (coloured by year)", color="#c9d1d9", fontsize=11)
    ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.1%})", fontsize=9)
    ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.1%})", fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: MRL MRR@10 bar chart ---
    ax = axes[0, 1]
    mrl = results.get("mrl_retrieval", [])
    if mrl:
        dims = [r["dim"] for r in mrl]
        mrrs = [r["mrr"] for r in mrl]
        bars = ax.bar(
            [str(d) for d in dims],
            mrrs,
            color=["#58a6ff" if d < 512 else "#3fb950" for d in dims],
            alpha=0.85,
            width=0.6,
        )
        for bar, v in zip(bars, mrrs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.005,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#c9d1d9",
            )
        ax.set_ylim(0, max(mrrs) * 1.25 if mrrs else 1)
        ax.set_xlabel("Embedding Dimension (MRL)", fontsize=9)
        ax.set_ylabel("MRR@10", fontsize=9)
        ax.set_title(
            "Retrieval quality vs. MRL truncation", color="#c9d1d9", fontsize=11
        )
        ax.grid(True, axis="y", alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "No retrieval data",
            ha="center",
            va="center",
            color="#8b949e",
            transform=ax.transAxes,
        )
        ax.set_title("MRL MRR@10", color="#c9d1d9", fontsize=11)

    # --- Panels 3 & 4: flight observer (height / curvature) ---
    for col_idx, (key, label, color) in enumerate(
        [
            ("heights", "Observer height h (reconstruction error)", "#58a6ff"),
            ("curvatures", "Curvature κ (principal angle, deg)", "#f78166"),
        ]
    ):
        ax = axes[1, col_idx]
        if flight_obs and key in flight_obs and len(flight_obs[key]) > 0:
            vals = np.array(flight_obs[key], dtype=float)
            hops = np.arange(len(vals))
            ax.plot(hops, vals, color=color, lw=1.5, alpha=0.9)
            ax.fill_between(hops, vals, alpha=0.2, color=color)
            if key == "curvatures":
                hc = flight_obs.get("high_curvature_hops", [])
                for h in hc:
                    if h < len(vals):
                        ax.axvline(h, color="#f0e68c", lw=0.8, alpha=0.5)
            ax.set_xlabel("Hop", fontsize=9)
            ax.set_ylabel(label, fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5,
                0.5,
                "No flight data\n(run without --no-walker)",
                ha="center",
                va="center",
                color="#8b949e",
                transform=ax.transAxes,
                fontsize=9,
            )
        title_map = {
            "heights": "Flight path: observer height",
            "curvatures": "Flight path: manifold curvature",
        }
        ax.set_title(title_map[key], color="#c9d1d9", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]Figure saved → {out_path}[/green]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pepys diary manifold explorer — reads from embedding cache built by pepys_embedder.py"
    )
    p.add_argument(
        "--n",
        type=int,
        default=0,
        help="Temporally-sampled subset size drawn from the full cache (0 = all). Build cache first with pepys_embedder.py.",
    )
    p.add_argument(
        "--cache",
        default=DEFAULT_CACHE,
        help="Path to embedding cache JSON produced by pepys_embedder.py",
    )
    p.add_argument(
        "--tau",
        type=float,
        default=0.90,
        help="PCA variance threshold for intrinsic dim (default: 0.90)",
    )
    p.add_argument(
        "--k",
        type=int,
        default=10,
        help="KNN for ManifoldModel (default: 10)",
    )
    p.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete any existing cache + results before running",
    )
    p.add_argument(
        "--no-walker",
        action="store_true",
        help="Skip ManifoldWalker/ManifoldObserver flight step",
    )
    p.add_argument(
        "--out-json",
        default=DEFAULT_OUT_JSON,
        help="Path for results JSON output",
    )
    p.add_argument(
        "--out-png",
        default=DEFAULT_OUT_PNG,
        help="Path for figure PNG output",
    )
    return p.parse_args()


def main() -> None:  # noqa: C901
    args = parse_args()
    console.rule("[bold blue]Pepys Diary · nomic-embed-text Manifold Explorer")

    # -----------------------------------------------------------------------
    # Step 1: Load or embed
    # -----------------------------------------------------------------------
    cache_path = Path(args.cache)

    # --clear-cache: wipe cache and results so the next run starts fresh.
    if args.clear_cache:
        for _p in (cache_path, Path(args.out_json), Path(args.out_png)):
            if _p.exists():
                _p.unlink()
                console.print(f"[yellow]Cleared:[/yellow] {_p}")

    # Embedding is handled by pepys_embedder.py — this script is cache-only.
    if not cache_path.exists():
        console.print(
            "[red]No embedding cache found.\n"
            "Build it first with pepys_embedder.py:[/red]\n"
            f"  python benchmarks/pepys_embedder.py --output {cache_path}"
        )
        sys.exit(1)

    console.print(
        f"\n[bold]Step 1:[/bold] Loading cached embeddings from {cache_path} …"
    )
    E, texts, timestamps = load_cache(str(cache_path))

    if args.n:
        if args.n > len(texts):
            console.print(
                f"[yellow]Warning:[/yellow] --n {args.n} exceeds cache size "
                f"({len(texts)} entries) — using all cached entries."
            )
        elif args.n < len(texts):
            total = len(texts)
            n = args.n
            indices = [round(i * (total - 1) / (n - 1)) for i in range(n)]
            E = E[indices]
            texts = [texts[i] for i in indices]
            timestamps = [timestamps[i] for i in indices]
            console.print(
                f"  Temporally sampled {len(texts)} entries  "
                f"({timestamps[0].date()} → {timestamps[-1].date()})"
            )

    N, full_dim = E.shape
    console.print(f"  Corpus: {N} entries × {full_dim} dims  (dtype={E.dtype})")
    console.print(f"  Date range: {min(timestamps).date()} → {max(timestamps).date()}")

    # L2-normalise (nomic outputs should already be unit-norm, but be safe)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    E = E / np.clip(norms, 1e-8, None)

    # -----------------------------------------------------------------------
    # Step 2: Intrinsic dimensionality
    # -----------------------------------------------------------------------
    console.print("\n[bold]Step 2:[/bold] Intrinsic dimensionality …")
    n_components = min(N - 1, full_dim, 200)  # cap at 200 for speed
    pca = PCA(n_components=n_components)
    pca.fit(E)
    eigvals = pca.explained_variance_

    pr = participation_ratio(eigvals)
    d90 = elbow_pca(eigvals, 0.90)
    d95 = elbow_pca(eigvals, 0.95)
    d99 = elbow_pca(eigvals, 0.99)
    twonn = twonn_id(E)

    dim_table = Table(title="Intrinsic Dimensionality Estimates", show_header=True)
    dim_table.add_column("Estimator", style="cyan")
    dim_table.add_column("Value", justify="right", style="green")
    dim_table.add_row("Corpus size", str(N))
    dim_table.add_row("Full embedding dim", str(full_dim))
    dim_table.add_row("PCA dims for 90% variance", str(d90))
    dim_table.add_row("PCA dims for 95% variance", str(d95))
    dim_table.add_row("PCA dims for 99% variance", str(d99))
    dim_table.add_row("Participation Ratio (PR)", f"{pr:.1f}")
    dim_table.add_row("TwoNN ID estimate", f"{twonn:.1f}")
    console.print(dim_table)

    # -----------------------------------------------------------------------
    # Step 3: MRL truncation retrieval
    # -----------------------------------------------------------------------
    console.print(
        "\n[bold]Step 3:[/bold] MRL truncation retrieval (Pepys historical queries) …"
    )
    retrieval_pairs = build_retrieval_pairs(timestamps)

    if retrieval_pairs:
        query_texts = [q for q, _ in retrieval_pairs]
        rel_lists = [r for _, r in retrieval_pairs]
        console.print(
            f"  {len(query_texts)} queries; avg {np.mean([len(r) for r in rel_lists]):.0f} relevant entries/query"
        )
        try:
            Q_full = embed_local(query_texts, model=args.model)
            q_norms = np.linalg.norm(Q_full, axis=1, keepdims=True)
            Q_full = Q_full / np.clip(q_norms, 1e-8, None)
            console.print("  Query embeddings ready.")
        except Exception as exc:
            console.print(
                f"[yellow]Query embedding failed ({exc}); skipping retrieval.[/yellow]"
            )
            Q_full = None
    else:
        console.print(
            "  [yellow]No retrieval pairs built (corpus too small or dates out of range).[/yellow]"
        )
        Q_full = None

    retrieval_table = Table(
        title="MRR@10 by Embedding Dimension (MRL Truncation)", show_header=True
    )
    retrieval_table.add_column("Dimension", justify="right", style="cyan")
    retrieval_table.add_column("MRR@10", justify="right", style="green")
    retrieval_table.add_column("Variance Explained", justify="right")
    retrieval_table.add_column("PCA ID (90%)", justify="right")

    mrl_results = []
    for d in MRL_DIMS:
        if d > full_dim:
            continue
        E_d = E[:, :d]
        nd = np.linalg.norm(E_d, axis=1, keepdims=True)
        E_d_n = E_d / np.clip(nd, 1e-8, None)

        pca_d = PCA(n_components=min(N - 1, d, 200))
        pca_d.fit(E_d_n)
        var_expl = float(pca_d.explained_variance_ratio_.sum())
        id_d90 = elbow_pca(pca_d.explained_variance_, 0.90)

        if Q_full is not None and retrieval_pairs:
            Q_d = Q_full[:, :d]
            nqd = np.linalg.norm(Q_d, axis=1, keepdims=True)
            Q_d_n = Q_d / np.clip(nqd, 1e-8, None)
            mrr = eval_retrieval(E_d_n, Q_d_n, rel_lists, k=K_RETRIEVAL)
        else:
            mrr = float("nan")

        mrl_results.append(
            {"dim": d, "mrr": mrr, "var_explained": var_expl, "pca_id_90": id_d90}
        )
        retrieval_table.add_row(str(d), f"{mrr:.3f}", f"{var_expl:.1%}", str(id_d90))

    console.print(retrieval_table)

    # -----------------------------------------------------------------------
    # Step 4: ManifoldWalker flight (most-distant pair by cosine distance)
    # -----------------------------------------------------------------------
    flight_obs: dict | None = None
    flight_info: dict = {}

    if not args.no_walker:
        console.print(
            "\n[bold]Step 4:[/bold] ManifoldWalker flight (most-distant pair) …"
        )
        try:
            from proteusPy.manifold_model import ManifoldModel
            from proteusPy.manifold_observer import ManifoldObserver

            labels = np.array([ts.year for ts in timestamps])

            console.print(
                f"  Fitting ManifoldModel (k={args.k}, τ={args.tau}) on {N} entries …"
            )
            mm = ManifoldModel(
                k_graph=args.k,
                variance_threshold=args.tau,
                manifold_weight=0.8,
            )
            mm.fit(E, labels)

            # Find the two entries most distant in cosine space
            console.print("  Finding most-distant pair …")
            # Sample up to 500 random pairs to keep O(N²) manageable
            rng = np.random.default_rng(42)
            n_sample = min(N, 500)
            sample_idx = rng.choice(N, size=n_sample, replace=False)
            E_s = E[sample_idx]
            # Cosine distance matrix for the sample
            sims = E_s @ E_s.T
            np.fill_diagonal(sims, 1.0)
            i_loc, j_loc = np.unravel_index(np.argmin(sims), sims.shape)
            i_orig = int(sample_idx[i_loc])
            i_dest = int(sample_idx[j_loc])

            ts_orig = timestamps[i_orig]
            dist_cos = float(1 - sims[i_loc, j_loc])
            console.print(
                f"  Origin  : entry {i_orig}  ({ts_orig.date()})  "
                f'"{texts[i_orig][:60]}..."'
            )
            console.print(
                f"  Dest    : entry {i_dest}  ({timestamps[i_dest].date()})  "
                f'"{texts[i_dest][:60]}..."'
            )
            console.print(f"  Cosine distance: {dist_cos:.4f}")

            mm.fly_to(f"n{i_orig}")
            path = mm.fly_toward(E[i_dest], max_steps=200, patience=15)
            arrived = (
                np.linalg.norm(E[i_dest] - mm._graph.get_embedding(path[-1])) < 0.05
            )
            console.print(f"  Path length: {len(path)} hops  |  arrived: {arrived}")

            obs = ManifoldObserver(mm)
            flight_obs = obs.observe_path(path)
            mean_h = (
                float(np.mean(flight_obs["heights"]))
                if len(flight_obs["heights"])
                else 0.0
            )
            mean_c = (
                float(np.mean(flight_obs["curvatures"]))
                if len(flight_obs["curvatures"])
                else 0.0
            )
            console.print(
                f"  Observer: mean height={mean_h:.4f}  "
                f"mean curvature={mean_c:.4f}°"
            )

            flight_info = {
                "origin_idx": i_orig,
                "dest_idx": i_dest,
                "origin_date": ts_orig.isoformat(),
                "dest_date": timestamps[i_dest].isoformat(),
                "cosine_distance": round(dist_cos, 4),
                "path_length": len(path),
                "arrived": arrived,
                "mean_height": mean_h,
                "mean_curvature": mean_c,
            }

        except Exception as exc:
            console.print(f"[yellow]ManifoldWalker step failed: {exc}[/yellow]")
            import traceback

            traceback.print_exc()

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    console.rule("[bold]Summary[/bold]")
    console.print(
        f"""
  Corpus              : {N} Pepys diary entries
  Date range          : {min(timestamps).date()} → {max(timestamps).date()}
  Full embedding dim  : {full_dim}
  TwoNN ID estimate   : {twonn:.1f}
  Participation Ratio : {pr:.1f}
  PCA 90%             : {d90} dims
  PCA 95%             : {d95} dims
  PCA 99%             : {d99} dims
"""
    )

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    results = {
        "corpus_size": N,
        "date_range": [min(timestamps).isoformat(), max(timestamps).isoformat()],
        "full_dim": full_dim,
        "twonn_id": round(twonn, 2),
        "participation_ratio": round(pr, 2),
        "pca_id_90pct": d90,
        "pca_id_95pct": d95,
        "pca_id_99pct": d99,
        "mrl_retrieval": mrl_results,
        "flight": flight_info,
    }

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, cls=_NumpyEncoder)
    console.print(f"[dim]Results saved → {args.out_json}[/dim]")

    # -----------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------
    console.print("\n[bold]Generating figure …[/bold]")
    try:
        make_figure(E, timestamps, results, flight_obs, args.out_png)
    except Exception as exc:
        console.print(f"[yellow]Figure generation failed: {exc}[/yellow]")


if __name__ == "__main__":
    main()
