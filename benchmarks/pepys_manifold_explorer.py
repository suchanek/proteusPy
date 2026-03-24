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
2. Subsample if requested; embed via ollama nomic-embed-text (or nomic.ai API).
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
  All already in proteusPy env (numpy, scikit-learn, rich, matplotlib, requests, tqdm).
  Embeddings: ollama running locally with nomic-embed-text pulled, OR NOMIC_API_KEY set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import requests
from rich.console import Console
from rich.table import Table
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

console = Console()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "nomic-embed-text-4k"  # 4k-context variant handles long Pepys entries
USE_API = False
NOMIC_API_KEY = os.environ.get("NOMIC_API_KEY", "")
NOMIC_API_URL = "https://api-atlas.nomic.ai/v1/embedding/text"

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
def parse_diary(path: str, n: int = 0) -> tuple[list[str], list[datetime]]:
    """Parse a pipe-delimited diary file and return (texts, timestamps).

    Expected line format::

        TIMESTAMP | TYPE | CATEGORY | CONTENT

    The ``TYPE`` (e.g. ``pepys_domestic``, ``pepys_naval``) and ``CATEGORY``
    (e.g. ``Home``, ``Finance``, ``Health``) are prepended to the content so
    that topic-level signal is preserved in the embedding space.  Comment
    lines beginning with ``#`` and section headers are silently skipped.

    :param path: Path to the diary file.
    :param n: Maximum number of entries to return (0 = all).
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
            if n and len(texts) >= n:
                break
    return texts, timestamps


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def embed_ollama(
    texts: list[str],
    model: str = OLLAMA_MODEL,
    max_retries: int = 3,
    checkpoint_path: str | None = None,
    checkpoint_texts: list[str] | None = None,
    checkpoint_timestamps: list[datetime] | None = None,
) -> np.ndarray:
    """Embed texts via ollama (one at a time) with retry and live checkpointing.

    :param texts: List of strings to embed.
    :param model: ollama model name (default: OLLAMA_MODEL).
    :param max_retries: Retries per entry on 5xx errors before giving up.
    :param checkpoint_path: If set, write a cache checkpoint after every 50
        entries so a crash mid-run doesn't lose all work.
    :param checkpoint_texts: Aligned text list for the checkpoint cache.
    :param checkpoint_timestamps: Aligned timestamp list for the checkpoint cache.
    :return: Float32 array of shape (N, D).
    """
    vecs = []
    for i, text in enumerate(tqdm(texts, desc=f"Embedding via {model}")):
        for attempt in range(max_retries):
            try:
                r = requests.post(
                    OLLAMA_URL,
                    json={"model": model, "prompt": text},
                    timeout=120,
                )
                r.raise_for_status()
                vecs.append(r.json()["embedding"])
                break
            except Exception as exc:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # 1s, 2s backoff
                else:
                    raise exc
        # Checkpoint every 50 entries
        if (
            checkpoint_path
            and checkpoint_texts is not None
            and checkpoint_timestamps is not None
            and (i + 1) % 50 == 0
        ):
            partial = np.array(vecs, dtype=np.float32)
            save_cache(
                checkpoint_path,
                partial,
                checkpoint_texts[: len(vecs)],
                checkpoint_timestamps[: len(vecs)],
            )
    return np.array(vecs, dtype=np.float32)


def embed_nomic_api(texts: list[str]) -> np.ndarray:
    """Embed texts in batches via nomic.ai API."""
    headers = {
        "Authorization": f"Bearer {NOMIC_API_KEY}",
        "Content-Type": "application/json",
    }
    batch_size = 32
    vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding (API)"):
        batch = texts[i : i + batch_size]
        r = requests.post(
            NOMIC_API_URL,
            headers=headers,
            json={"model": "nomic-embed-text-v1", "texts": batch},
            timeout=60,
        )
        r.raise_for_status()
        vecs.extend(r.json()["embeddings"])
    return np.array(vecs, dtype=np.float32)


def embed(texts: list[str], model: str = OLLAMA_MODEL) -> np.ndarray:
    """Dispatch to the configured embedding backend.

    :param texts: Texts to embed.
    :param model: ollama model name (ignored when USE_API=True).
    :return: Float32 array of shape (N, D).
    """
    if USE_API:
        return embed_nomic_api(texts)
    return embed_ollama(texts, model=model)


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
    p = argparse.ArgumentParser(description="Pepys diary nomic-embed manifold explorer")
    p.add_argument(
        "--diary",
        default=DEFAULT_DIARY,
        help="Path to pepys_clean.txt pipe-delimited diary file",
    )
    p.add_argument(
        "--n",
        type=int,
        default=0,
        help="Max entries to use (0 = all)",
    )
    p.add_argument(
        "--cache",
        default=DEFAULT_CACHE,
        help="Path to embedding cache JSON (read if exists, write after embedding)",
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
        "--model",
        default=OLLAMA_MODEL,
        help="ollama model name (default: nomic-embed-text-4k for long-entry support)",
    )
    p.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="Truncate each entry to this many characters before embedding "
        "(0 = no truncation; use ~12000 for nomic-embed-text standard)",
    )
    p.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete any existing embedding cache before running (start fresh)",
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

    if cache_path.exists():
        console.print(
            f"\n[bold]Step 1:[/bold] Loading cached embeddings from {cache_path} …"
        )
        E, texts, timestamps = load_cache(str(cache_path))
        # Honour --n even when loading from cache
        if args.n and len(texts) > args.n:
            E = E[: args.n]
            texts = texts[: args.n]
            timestamps = timestamps[: args.n]
    else:
        diary_path = Path(args.diary)
        if not diary_path.exists():
            console.print(
                f"[red]Diary file not found: {diary_path}\n"
                "Pass --diary /path/to/pepys_clean.txt[/red]"
            )
            sys.exit(1)

        console.print(f"\n[bold]Step 1:[/bold] Parsing {diary_path} …")
        texts, timestamps = parse_diary(str(diary_path), n=args.n)
        console.print(
            f"  Loaded {len(texts)} entries  "
            f"({timestamps[0].date()} → {timestamps[-1].date()})"
        )

        if args.max_chars:
            n_long = sum(1 for t in texts if len(t) > args.max_chars)
            if n_long:
                console.print(
                    f"  Truncating {n_long} entries to {args.max_chars} chars"
                )
            texts = [t[: args.max_chars] if args.max_chars else t for t in texts]

        # Resume from partial checkpoint if it exists
        if cache_path.exists():
            console.print(
                f"  [yellow]Partial cache found — resuming from {cache_path}[/yellow]"
            )
            E_partial, texts_done, ts_done = load_cache(str(cache_path))
            n_done = len(texts_done)
            console.print(f"  Resuming from entry {n_done}/{len(texts)}")
            texts_remaining = texts[n_done:]
        else:
            E_partial, n_done = None, 0
            texts_remaining = texts

        console.print(
            f"  Embedding {len(texts_remaining)} entries with {args.model} "
            f"(this may take a while) …"
        )
        t0 = time.time()
        try:
            E_new = embed_ollama(
                texts_remaining,
                model=args.model,
                checkpoint_path=str(cache_path),
                checkpoint_texts=texts,
                checkpoint_timestamps=timestamps,
            )
        except Exception as exc:
            console.print(f"[red]Embedding failed: {exc}[/red]")
            console.print(
                "Ensure ollama is running:  ollama serve && ollama pull nomic-embed-text"
            )
            sys.exit(1)
        elapsed = time.time() - t0
        # Merge with any partial results from a previous interrupted run
        if E_partial is not None:
            E = np.concatenate([E_partial, E_new], axis=0)
        else:
            E = E_new
        console.print(
            f"  Embedded {len(texts_remaining)} new entries in {elapsed:.1f}s  "
            f"({elapsed / max(len(texts_remaining), 1):.2f}s/entry)"
        )
        save_cache(str(cache_path), E, texts, timestamps)

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
            Q_full = embed(query_texts, model=args.model)
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
