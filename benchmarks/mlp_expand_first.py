#!/usr/bin/env python3
# mlp_expand_first.py
# Copyright (c) 2026 Eric G. Suchanek, PhD, Flux-Frontiers
# https://github.com/Flux-Frontiers
# License: BSD
# Last revised: 2026-03-23 -egs-
"""
mlp_expand_first.py
--------------------
Benchmark: does projecting to N+1 dimensions in the first MLP layer
outperform the standard compress-first architecture?

Four architectures on CIFAR-10 PCA→30D:

  A  Standard compress  : 30 → 16 → 8 → 10
  B  Expand N+1 (31)    : 30 → 31 → 16 → 8 → 10
  C  Expand ×4 (120)    : 30 → 120 → 16 → 8 → 10   (Transformer FFN style)
  D  QR-init expand     : same topology as B but W₁ initialized from
                          PCA + QR tangent directions (then fine-tuned)

Hypothesis (geometric):
  B and D will outperform A on structured manifold data because the extra
  neuron gives the network room to represent the "normal" direction before
  compressing to class space.  D converges faster than B because the QR
  initialization aligns the expansion layer with the actual manifold geometry.

All architectures are trained with the same:
  optimizer : Adam lr=1e-3
  epochs    : 60
  batch     : 128
  seed      : 42

Usage:
  python benchmarks/mlp_expand_first.py [--epochs E] [--seed S] [--n-runs R]
"""

import argparse
import json
import time

import numpy as np

# Try TensorFlow first, fall back to a note if unavailable
try:
    import tensorflow as tf
    from tensorflow import keras

    HAS_TF = True
except ImportError:
    HAS_TF = False

from rich.console import Console
from rich.table import Table
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

console = Console()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-runs", type=int, default=3, help="Independent runs (different seeds)")
    p.add_argument("--pca-dim", type=int, default=30, help="PCA target dimension")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_cifar10_pca(pca_dim: int, seed: int):
    """Load CIFAR-10 and reduce to pca_dim with PCA."""
    console.print("  Loading CIFAR-10 via sklearn …")
    dataset = fetch_openml("CIFAR_10", version=1, as_frame=False, parser="auto")
    X = dataset.data.astype(np.float32) / 255.0
    y = dataset.target.astype(int)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    split = 50000
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    console.print(f"  PCA → {pca_dim}D …")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=pca_dim, random_state=seed)
    X_train = pca.fit_transform(X_train).astype(np.float32)
    X_test = pca.transform(X_test).astype(np.float32)

    var = pca.explained_variance_ratio_.sum()
    console.print(f"  PCA variance retained: {var:.1%}")
    return X_train, X_test, y_train, y_test, pca


# ---------------------------------------------------------------------------
# QR initialization
# ---------------------------------------------------------------------------


def qr_init_matrix(pca: PCA, n_in: int, n_out: int) -> np.ndarray:
    """
    Build a (n_out × n_in) weight matrix for the first layer.

    First n_in rows = PCA basis directions (tangent to data manifold).
    Row n_in (the +1 row) = the vector orthogonal to all PCA directions,
    found via QR decomposition of [V; e_{n_in+1}].

    This initializes the expansion neuron to point in the manifold normal
    direction.  All other rows (n_in+1 … n_out-1) are random Glorot.
    """
    # pca.components_[:n_in] are the top-n_in principal directions (original space).
    # Our layer input is already PCA-projected, so in PCA space the frame is I_{n_in}.
    # The normal in PCA space (after truncation) is e_{n_in} extended.

    # Build an (n_in+1 × n_in) matrix: I_{n_in} stacked with a zero row
    A = np.vstack([np.eye(n_in), np.zeros((1, n_in))])  # (n_in+1, n_in)
    # QR gives us the (n_in+1)-th basis vector orthogonal to the n_in columns
    Q, _ = np.linalg.qr(A.T, mode="complete")  # Q is (n_in, n_in+1) — wait
    # Actually: QR of A (n_in+1, n_in):
    Q, _ = np.linalg.qr(A)  # Q: (n_in+1, n_in+1)
    normal_direction = Q[:, n_in]  # last column = normal

    # Weight matrix for the expansion layer (n_out, n_in)
    W = np.zeros((n_out, n_in), dtype=np.float32)

    # Rows 0..n_in-1: identity (preserve the PCA coordinates)
    for i in range(min(n_in, n_out)):
        W[i, i] = 1.0

    # Row n_in: the normal direction (projected back to n_in-dim input)
    if n_in < n_out:
        W[n_in, :] = normal_direction[:n_in].astype(np.float32)

    # Remaining rows: Glorot uniform
    if n_out > n_in + 1:
        limit = np.sqrt(6.0 / (n_in + n_out))
        W[n_in + 1 :] = np.random.uniform(-limit, limit, (n_out - n_in - 1, n_in)).astype(
            np.float32
        )

    return W


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def build_standard(n_in: int, n_classes: int, lr: float, seed: int):
    """A: 30 → 16 → 8 → 10 (standard compress)"""
    tf.random.set_seed(seed)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(n_in,)),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ],
        name="A_standard",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_expand_n1(n_in: int, n_classes: int, lr: float, seed: int, qr_init: bool = False, pca=None):
    """B/D: 30 → 31 → 16 → 8 → 10"""
    tf.random.set_seed(seed)
    n_expand = n_in + 1

    first_layer = keras.layers.Dense(n_expand, activation="relu", name="expand_n1")

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(n_in,)),
            first_layer,
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ],
        name="D_qr_init" if qr_init else "B_expand_n1",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Build model to initialize weights, then set QR-derived W
    model.build((None, n_in))
    if qr_init and pca is not None:
        W_init = qr_init_matrix(pca, n_in, n_expand)
        b_init = np.zeros(n_expand, dtype=np.float32)
        first_layer.set_weights([W_init.T, b_init])  # Dense expects (in, out)

    return model


def build_expand_4x(n_in: int, n_classes: int, lr: float, seed: int):
    """C: 30 → 120 → 16 → 8 → 10 (Transformer FFN style ×4)"""
    tf.random.set_seed(seed)
    n_expand = n_in * 4
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(n_in,)),
            keras.layers.Dense(n_expand, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ],
        name="C_expand_4x",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def count_params(model) -> int:
    return int(model.count_params())


def train_eval(model, X_train, y_train, X_test, y_test, epochs: int, batch: int) -> dict:
    t0 = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch,
        validation_split=0.1,
        verbose=0,
    )
    train_time = time.perf_counter() - t0

    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    val_accs = history.history.get("val_accuracy", [])
    best_val = float(max(val_accs)) if val_accs else float("nan")

    # Convergence: epoch at which val_acc first exceeds 90% of its best
    threshold = 0.9 * best_val
    converge_epoch = next(
        (i + 1 for i, v in enumerate(val_accs) if v >= threshold), epochs
    )

    return {
        "test_acc": float(test_acc),
        "best_val_acc": best_val,
        "converge_epoch": converge_epoch,
        "train_time_s": round(train_time, 1),
        "n_params": count_params(model),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    if not HAS_TF:
        console.print("[red]TensorFlow not installed. Install with: pip install tensorflow[/red]")
        return

    console.rule("[bold blue]MLP Expand-First Benchmark")
    console.print(f"\n  epochs={args.epochs}  pca_dim={args.pca_dim}  "
                  f"batch={args.batch}  lr={args.lr}  n_runs={args.n_runs}\n")

    X_train, X_test, y_train, y_test, pca = load_cifar10_pca(args.pca_dim, args.seed)
    n_in = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    console.print(f"  Input dim: {n_in}  Classes: {n_classes}\n")

    architectures = {
        "A_standard":  "30→16→8→10  (standard compress)",
        "B_expand_n1": f"30→{n_in+1}→16→8→10  (N+1 expand)",
        "C_expand_4x": f"30→{n_in*4}→16→8→10  (×4 Transformer style)",
        "D_qr_init":   f"30→{n_in+1}→16→8→10  (QR-init normal, then fine-tune)",
    }

    all_results = {k: [] for k in architectures}

    for run in range(args.n_runs):
        seed = args.seed + run * 100
        console.print(f"[bold]Run {run + 1}/{args.n_runs}[/bold]  (seed={seed})")

        builders = {
            "A_standard":  lambda s=seed: build_standard(n_in, n_classes, args.lr, s),
            "B_expand_n1": lambda s=seed: build_expand_n1(n_in, n_classes, args.lr, s, qr_init=False),
            "C_expand_4x": lambda s=seed: build_expand_4x(n_in, n_classes, args.lr, s),
            "D_qr_init":   lambda s=seed: build_expand_n1(n_in, n_classes, args.lr, s, qr_init=True, pca=pca),
        }

        for key, desc in architectures.items():
            console.print(f"  Training {key}: {desc} …")
            model = builders[key]()
            metrics = train_eval(model, X_train, y_train, X_test, y_test, args.epochs, args.batch)
            all_results[key].append(metrics)
            console.print(
                f"    test_acc={metrics['test_acc']:.3f}  "
                f"best_val={metrics['best_val_acc']:.3f}  "
                f"converge@ep={metrics['converge_epoch']}  "
                f"params={metrics['n_params']:,}  "
                f"time={metrics['train_time_s']:.0f}s"
            )
            tf.keras.backend.clear_session()

    # Aggregate across runs
    summary_table = Table(title="MLP Expand-First: Mean Results Across Runs", show_header=True)
    summary_table.add_column("Architecture", style="cyan")
    summary_table.add_column("Test Acc (%)", justify="right", style="green")
    summary_table.add_column("Best Val (%)", justify="right")
    summary_table.add_column("Converge Ep", justify="right")
    summary_table.add_column("Params", justify="right")
    summary_table.add_column("Time (s)", justify="right")
    summary_table.add_column("Δ vs A", justify="right")

    # Compute baseline
    baseline_acc = float(np.mean([r["test_acc"] for r in all_results["A_standard"]]))

    aggregated = {}
    for key, desc in architectures.items():
        runs = all_results[key]
        mean_acc = float(np.mean([r["test_acc"] for r in runs]))
        mean_val = float(np.mean([r["best_val_acc"] for r in runs]))
        mean_conv = float(np.mean([r["converge_epoch"] for r in runs]))
        params = runs[0]["n_params"]
        mean_time = float(np.mean([r["train_time_s"] for r in runs]))
        delta = mean_acc - baseline_acc
        delta_str = f"{delta:+.3f}" if key != "A_standard" else "—"

        aggregated[key] = {
            "description": desc,
            "mean_test_acc": round(mean_acc, 4),
            "mean_best_val_acc": round(mean_val, 4),
            "mean_converge_epoch": round(mean_conv, 1),
            "n_params": params,
            "mean_time_s": round(mean_time, 1),
            "delta_vs_A": round(delta, 4),
        }

        summary_table.add_row(
            desc,
            f"{mean_acc * 100:.2f}",
            f"{mean_val * 100:.2f}",
            f"{mean_conv:.0f}",
            f"{params:,}",
            f"{mean_time:.0f}",
            delta_str,
        )

    console.print(summary_table)

    # Interpretation
    console.rule("[bold]Interpretation[/bold]")
    b_delta = aggregated["B_expand_n1"]["delta_vs_A"]
    d_delta = aggregated["D_qr_init"]["delta_vs_A"]
    d_conv = aggregated["D_qr_init"]["mean_converge_epoch"]
    a_conv = aggregated["A_standard"]["mean_converge_epoch"]

    if d_delta > b_delta and d_conv < a_conv:
        verdict = "QR-init confirms geometric hypothesis: faster convergence AND higher accuracy."
    elif b_delta > 0:
        verdict = "N+1 expansion improves accuracy; QR-init effect is marginal."
    else:
        verdict = "Geometric expansion did not improve on this dataset configuration."

    console.print(f"""
  Baseline (A)     : {aggregated['A_standard']['mean_test_acc']*100:.2f}%
  N+1 expand (B)   : {aggregated['B_expand_n1']['mean_test_acc']*100:.2f}%  (Δ={b_delta:+.3f})
  ×4 expand (C)    : {aggregated['C_expand_4x']['mean_test_acc']*100:.2f}%  (Δ={aggregated['C_expand_4x']['delta_vs_A']:+.3f})
  QR-init (D)      : {aggregated['D_qr_init']['mean_test_acc']*100:.2f}%  (Δ={d_delta:+.3f},  converge @ ep {d_conv:.0f} vs {a_conv:.0f})

  Verdict: {verdict}
""")

    out = {
        "config": vars(args),
        "n_in": n_in,
        "n_classes": n_classes,
        "architectures": aggregated,
        "raw_runs": {k: v for k, v in all_results.items()},
    }
    out_path = "benchmarks/mlp_expand_first_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    console.print(f"[dim]Results saved to {out_path}[/dim]")


if __name__ == "__main__":
    main()
