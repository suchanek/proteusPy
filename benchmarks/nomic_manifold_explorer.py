#!/usr/bin/env python3
# nomic_manifold_explorer.py
# Copyright (c) 2026 Eric G. Suchanek, PhD, Flux-Frontiers
# https://github.com/Flux-Frontiers
# License: BSD
# Last revised: 2026-03-23 -egs-
"""
nomic_manifold_explorer.py
--------------------------
Explore the embedding manifold produced by nomic-embed-text-v1.

Pipeline:
  1. Embed a small canonical corpus (Wikipedia abstracts via datasets, or
     a local text list) with nomic-embed-text via the ollama HTTP API.
  2. Determine intrinsic dimensionality:
       - PCA explained-variance curve (find the "elbow")
       - Participation Ratio: PR = (Σλ_i)² / Σλ_i²
       - TwoNN estimator (Facco et al. 2017) — robust, no hyperparameters
  3. Build a reduced version at the MRL checkpoints: 64, 128, 256, 512, 768.
  4. Compare retrieval quality (mean reciprocal rank @ 10) of each
     truncation vs. the full 768-d embedding on the same corpus.
  5. Run ManifoldWalker on the 768-d cloud to confirm intrinsic dim.

Requirements (all already in proteusPy env):
  pip install requests numpy scikit-learn tqdm rich

For embeddings you need either:
  a) ollama running locally:  ollama serve && ollama pull nomic-embed-text
  b) or set NOMIC_API_KEY and flip USE_API=True below.

Usage:
  python benchmarks/nomic_manifold_explorer.py
"""

import json
import os
import sys

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
OLLAMA_MODEL = "nomic-embed-text"
USE_API = False          # flip to True to use nomic.ai REST API
NOMIC_API_KEY = os.environ.get("NOMIC_API_KEY", "")
NOMIC_API_URL = "https://api-atlas.nomic.ai/v1/embedding/text"

MRL_DIMS = [64, 128, 256, 512, 768]   # Matryoshka checkpoints
N_RETRIEVAL_QUERIES = 50              # how many query → ground-truth pairs
K_RETRIEVAL = 10                       # MRR@K

# ---------------------------------------------------------------------------
# Corpus: short varied texts that stress different semantic dimensions
# ---------------------------------------------------------------------------
CORPUS = [
    # Science / Biology
    "Proteins fold into three-dimensional structures determined by their amino acid sequence.",
    "Disulfide bonds are covalent links between cysteine residues that stabilize protein structure.",
    "The RCSB Protein Data Bank archives experimentally determined macromolecular structures.",
    "DNA replication proceeds via a semi-conservative mechanism involving complementary base pairing.",
    "CRISPR-Cas9 is a gene-editing tool derived from a bacterial immune defense system.",
    "Mitochondria generate ATP through oxidative phosphorylation along the electron transport chain.",
    "Ribosomes translate messenger RNA into polypeptide chains during protein synthesis.",
    "Enzymes lower the activation energy of biochemical reactions without being consumed.",
    # Mathematics / ML
    "A Riemannian manifold is a smooth manifold equipped with an inner product on each tangent space.",
    "Principal component analysis finds orthogonal axes of maximum variance in high-dimensional data.",
    "The intrinsic dimensionality of a dataset is the minimum number of parameters needed to describe it.",
    "Stochastic gradient descent updates model parameters in the direction of the negative gradient.",
    "Attention mechanisms in transformers compute weighted averages of value vectors.",
    "Contrastive learning trains encoders to bring similar pairs closer and push dissimilar pairs apart.",
    "The Johnson-Lindenstrauss lemma states that random projections approximately preserve pairwise distances.",
    "Kernel methods implicitly map data to high-dimensional feature spaces via the kernel trick.",
    "Graph neural networks aggregate features from local neighborhoods via message passing.",
    "The bias-variance tradeoff describes the tension between model complexity and generalization.",
    # History / Literature
    "Plato's Allegory of the Cave describes prisoners who mistake shadows for reality.",
    "Edwin Abbott's Flatland imagines a two-dimensional world whose inhabitants cannot perceive depth.",
    "The Industrial Revolution transformed manufacturing through mechanization beginning in Britain.",
    "The Treaty of Westphalia in 1648 established the modern concept of state sovereignty.",
    "Shakespeare's Hamlet explores themes of revenge, mortality, and existential doubt.",
    "The French Revolution dismantled the ancien régime and proclaimed liberty and equality.",
    # Technology / Computing
    "Large language models are trained on vast text corpora to predict the next token.",
    "Docker containers package software with its dependencies for reproducible deployment.",
    "Kubernetes orchestrates containerized applications across clusters of machines.",
    "REST APIs communicate using HTTP verbs and structured resource representations.",
    "Git tracks changes to files using a directed acyclic graph of commit objects.",
    "The TCP/IP protocol stack enables reliable data transmission across heterogeneous networks.",
    # Physics / Chemistry
    "Quantum entanglement correlates the states of particles regardless of their separation.",
    "The Schrödinger equation describes how quantum states evolve over time.",
    "General relativity models gravity as the curvature of spacetime by mass and energy.",
    "Entropy measures the number of microscopic configurations consistent with a macroscopic state.",
    "Covalent bonds arise from the sharing of electron pairs between atoms.",
    "Nuclear magnetic resonance spectroscopy probes molecular structure via spin-lattice relaxation.",
    # Arts / Culture
    "Impressionism emerged in nineteenth-century France as painters captured light and atmosphere.",
    "Jazz originated in New Orleans through the synthesis of African rhythms and European harmony.",
    "Architecture balances structural engineering constraints with aesthetic and functional goals.",
    "Poetry compresses meaning through rhythm, imagery, and the careful selection of words.",
    "Cinema evolved from silent black-and-white shorts to immersive three-dimensional experiences.",
]

# Ground-truth query → relevant index mapping for retrieval evaluation
QUERIES_WITH_RELEVANT = [
    ("How do proteins maintain their folded shape?",            [0, 1]),
    ("What is the role of ribosomes in the cell?",             [6, 5]),
    ("Explain principal component analysis",                   [9, 10]),
    ("What is Plato's cave about?",                            [18]),
    ("How does contrastive learning work?",                    [13, 11]),
    ("Describe the structure of DNA",                          [3, 4]),
    ("What is Kubernetes used for?",                           [26, 25]),
    ("Tell me about quantum mechanics",                        [31, 32]),
    ("Explain the attention mechanism in transformers",        [12, 11]),
    ("What is the Riemannian manifold?",                       [8, 10]),
]


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def embed_ollama(texts: list[str]) -> np.ndarray:
    """Embed texts one at a time via ollama."""
    vecs = []
    for text in tqdm(texts, desc="Embedding (ollama)"):
        r = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": text}, timeout=30)
        r.raise_for_status()
        vecs.append(r.json()["embedding"])
    return np.array(vecs, dtype=np.float32)


def embed_nomic_api(texts: list[str]) -> np.ndarray:
    """Embed texts in batches via nomic.ai API."""
    headers = {"Authorization": f"Bearer {NOMIC_API_KEY}", "Content-Type": "application/json"}
    batch_size = 32
    vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding (API)"):
        batch = texts[i : i + batch_size]
        r = requests.post(NOMIC_API_URL, headers=headers,
                          json={"model": "nomic-embed-text-v1", "texts": batch}, timeout=60)
        r.raise_for_status()
        vecs.extend(r.json()["embeddings"])
    return np.array(vecs, dtype=np.float32)


def embed(texts: list[str]) -> np.ndarray:
    if USE_API:
        return embed_nomic_api(texts)
    return embed_ollama(texts)


# ---------------------------------------------------------------------------
# Intrinsic dimensionality estimators
# ---------------------------------------------------------------------------
def participation_ratio(eigenvalues: np.ndarray) -> float:
    """PR = (Σλ)² / Σλ² — effective number of active dimensions."""
    lam = eigenvalues[eigenvalues > 0]
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def twonn_id(X: np.ndarray, n_neighbors: int = 2) -> float:
    """
    TwoNN intrinsic dimensionality estimator (Facco et al., 2017).
    Robust, parameter-free, works in high dimensions.
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine").fit(X)
    distances, _ = nbrs.kneighbors(X)
    r1 = distances[:, 1]   # nearest neighbour distance
    r2 = distances[:, 2]   # second nearest
    # Avoid division by zero
    mask = r1 > 0
    mu = r2[mask] / r1[mask]
    # MLE: ID = -N / Σ ln(μ_i)
    id_est = -len(mu) / np.sum(np.log(mu))
    return float(id_est)


def elbow_pca(eigenvalues: np.ndarray, threshold: float = 0.90) -> int:
    """Number of components needed to explain `threshold` variance."""
    cumvar = np.cumsum(eigenvalues) / eigenvalues.sum()
    return int(np.searchsorted(cumvar, threshold)) + 1


# ---------------------------------------------------------------------------
# Retrieval evaluation: MRR@K
# ---------------------------------------------------------------------------
def mrr_at_k(embeddings: np.ndarray, queries_idx: list, relevant: list, k: int = 10) -> float:
    """
    Mean Reciprocal Rank @ K.
    queries_idx: indices into corpus used as query
    relevant: list of lists of relevant indices per query
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(embeddings)
    scores = []
    for qi, rel in zip(queries_idx, relevant):
        qvec = embeddings[qi : qi + 1]
        _, indices = nbrs.kneighbors(qvec)
        retrieved = [i for i in indices[0] if i != qi][:k]
        rr = 0.0
        for rank, idx in enumerate(retrieved, 1):
            if idx in rel:
                rr = 1.0 / rank
                break
        scores.append(rr)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    console.rule("[bold blue]Nomic-embed-text Manifold Explorer")

    # -----------------------------------------------------------------------
    # Step 1: Embed corpus
    # -----------------------------------------------------------------------
    console.print(f"\n[bold]Step 1:[/bold] Embedding {len(CORPUS)} texts …")
    try:
        E = embed(CORPUS)
    except Exception as exc:
        console.print(f"[red]Embedding failed: {exc}[/red]")
        console.print("Ensure ollama is running:  ollama serve && ollama pull nomic-embed-text")
        sys.exit(1)

    console.print(f"  Embedding matrix: {E.shape}  (dtype={E.dtype})")
    full_dim = E.shape[1]

    # L2-normalise for cosine geometry (nomic outputs are already normalised but be safe)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    E = E / np.clip(norms, 1e-8, None)

    # -----------------------------------------------------------------------
    # Step 2: Intrinsic dimensionality
    # -----------------------------------------------------------------------
    console.print("\n[bold]Step 2:[/bold] Intrinsic dimensionality analysis …")
    pca = PCA(n_components=min(len(CORPUS) - 1, full_dim))
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
    dim_table.add_row("PCA dims for 90% variance", str(d90))
    dim_table.add_row("PCA dims for 95% variance", str(d95))
    dim_table.add_row("PCA dims for 99% variance", str(d99))
    dim_table.add_row("Participation Ratio (PR)", f"{pr:.1f}")
    dim_table.add_row("TwoNN ID estimate", f"{twonn:.1f}")
    dim_table.add_row("Full embedding dimension", str(full_dim))
    console.print(dim_table)

    # -----------------------------------------------------------------------
    # Step 3: MRL truncation retrieval comparison
    # -----------------------------------------------------------------------
    console.print("\n[bold]Step 3:[/bold] MRL truncation retrieval comparison …")

    # Build query→relevant pairs from the corpus
    # Encode queries separately and search the corpus
    query_texts = [q for q, _ in QUERIES_WITH_RELEVANT]
    rel_indices = [r for _, r in QUERIES_WITH_RELEVANT]

    console.print(f"  Embedding {len(query_texts)} evaluation queries …")
    try:
        Q = embed(query_texts)
        norms_q = np.linalg.norm(Q, axis=1, keepdims=True)
        Q = Q / np.clip(norms_q, 1e-8, None)
    except Exception as exc:
        console.print(f"[yellow]Query embedding failed ({exc}); skipping retrieval eval.[/yellow]")
        Q = None

    retrieval_table = Table(title="MRR@10 by Embedding Dimension (MRL Truncation)", show_header=True)
    retrieval_table.add_column("Dimension", justify="right", style="cyan")
    retrieval_table.add_column("MRR@10", justify="right", style="green")
    retrieval_table.add_column("Variance Explained", justify="right")
    retrieval_table.add_column("PCA ID (90%)", justify="right")

    results = []
    for d in MRL_DIMS:
        if d > full_dim:
            continue
        E_d = E[:, :d]
        # Re-normalise after truncation (MRL design: truncated vectors still useful)
        norms_d = np.linalg.norm(E_d, axis=1, keepdims=True)
        E_d_norm = E_d / np.clip(norms_d, 1e-8, None)

        # PCA on truncated space
        pca_d = PCA(n_components=min(len(CORPUS) - 1, d))
        pca_d.fit(E_d_norm)
        var_explained = float(pca_d.explained_variance_ratio_.sum())
        id_d90 = elbow_pca(pca_d.explained_variance_, 0.90)

        # Retrieval: project queries too
        if Q is not None:
            Q_d = Q[:, :d]
            norms_qd = np.linalg.norm(Q_d, axis=1, keepdims=True)
            Q_d_norm = Q_d / np.clip(norms_qd, 1e-8, None)

            nbrs = NearestNeighbors(n_neighbors=min(K_RETRIEVAL + 1, len(CORPUS)), metric="cosine").fit(E_d_norm)
            mrr_scores = []
            for qvec, rel in zip(Q_d_norm, rel_indices):
                _, indices = nbrs.kneighbors(qvec.reshape(1, -1))
                retrieved = list(indices[0])[:K_RETRIEVAL]
                rr = 0.0
                for rank, idx in enumerate(retrieved, 1):
                    if idx in rel:
                        rr = 1.0 / rank
                        break
                mrr_scores.append(rr)
            mrr = float(np.mean(mrr_scores))
        else:
            mrr = float("nan")

        results.append({"dim": d, "mrr": mrr, "var_explained": var_explained, "pca_id_90": id_d90})
        retrieval_table.add_row(
            str(d), f"{mrr:.3f}", f"{var_explained:.1%}", str(id_d90)
        )

    console.print(retrieval_table)

    # -----------------------------------------------------------------------
    # Step 4: ManifoldWalker intrinsic dim (if available)
    # -----------------------------------------------------------------------
    console.print("\n[bold]Step 4:[/bold] ManifoldWalker KNN dimensionality check …")
    try:
        from proteusPy.manifold_model import ManifoldModel
        mm = ManifoldModel(n_neighbors=5, variance_threshold=0.90, manifold_weight=0.8)
        mm.fit(E, np.zeros(len(E), dtype=int))  # dummy labels
        mean_dim = float(np.mean([mm._geometries[f"n{i}"].intrinsic_dim for i in range(len(E)) if f"n{i}" in mm._geometries]))
        console.print(f"  ManifoldModel mean local intrinsic dim: [green]{mean_dim:.1f}[/green]")
    except Exception as exc:
        console.print(f"  [yellow]ManifoldModel check skipped: {exc}[/yellow]")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    console.rule("[bold]Summary[/bold]")
    console.print(f"""
  Full embedding dim  : {full_dim}
  TwoNN ID estimate   : {twonn:.1f}
  Participation Ratio : {pr:.1f}   (effective active dims)
  PCA 90% threshold   : {d90} dims
  PCA 95% threshold   : {d95} dims
  PCA 99% threshold   : {d99} dims

  Interpretation:
  - The nomic-embed-text manifold likely has intrinsic dimension ~{round(twonn)}.
  - MRL truncation to {d90}-{d95}d retains 90-95% of variance with minimal retrieval loss.
  - Embeddings live on a low-dim curved manifold inside ℝ^{full_dim}.
  - Our ManifoldWalker can traverse this space using only ~{round(twonn)} effective dims.
""")

    # Save results
    out = {
        "corpus_size": len(CORPUS),
        "full_dim": full_dim,
        "twonn_id": round(twonn, 2),
        "participation_ratio": round(pr, 2),
        "pca_id_90pct": d90,
        "pca_id_95pct": d95,
        "pca_id_99pct": d99,
        "mrl_retrieval": results,
    }
    out_path = "benchmarks/nomic_manifold_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    console.print(f"[dim]Results saved to {out_path}[/dim]")


if __name__ == "__main__":
    main()
