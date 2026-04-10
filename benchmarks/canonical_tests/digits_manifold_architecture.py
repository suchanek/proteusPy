#!/usr/bin/env python3
"""
Digits Benchmark: Manifold-Informed Architecture — Comprehensive Comparison
============================================================================

Benchmarks ALL approaches on sklearn digits — the canonical small dataset
where manifold geometry is completely transparent:

  sklearn digits: 1,797 samples, 64-dimensional (8×8 pixels), 10 classes.

Because the dataset is small enough for complete evaluation, all methods
are assessed with 5-fold stratified cross-validation, giving statistically
robust results without subsampling.

Six approaches compared
-----------------------
  1. Standard MLP (128→64)          — fixed-width baseline neural network
  2. Manifold MLP (2d→d)            — bottleneck at intrinsic dimensionality d
  3. Wide Manifold MLP (4d→2d→d)    — progressive compression to manifold dim
  4. PCA→dD + MLP (2d→d)            — global PCA projection + nonlinear head
  5. ManifoldModel (τ=0.90)         — zero learned parameters, pure geometry
  6. Euclidean KNN (k=7)            — classic Euclidean baseline

Three phases
------------
Phase 1 — Manifold Discovery
    Local PCA over --discovery-samples points (k=--k-pca neighbors each).
    Global and per-class intrinsic dimensionality reported.
    Bottleneck d = max per-class max intrinsic dim at τ=--tau.

Phase 2 — Architecture Summary
    All six architectures are described with parameter counts.

Phase 3 — 5-Fold CV
    All methods run on the same folds.  Aggregate mean ± std accuracy,
    timing, and geometry stats are printed and saved to
    ``digits_manifold_architecture_results.json``.  A four-panel
    matplotlib figure is saved alongside (``digits_manifold_architecture_results.png``).

Part of proteusPy, https://github.com/suchanek/proteusPy
Author: Eric G. Suchanek, PhD
Affiliation: Flux-Frontiers
Last Revision: 2026-03-28

Usage
-----
    python benchmarks/canonical_tests/digits_manifold_architecture.py
    python benchmarks/canonical_tests/digits_manifold_architecture.py --epochs 50 --trials 3 --tau 0.85
"""

# ---------------------------------------------------------------------------
# BSD 2-Clause License
#
# Copyright (c) 2026, Eric G. Suchanek, PhD — Flux-Frontiers
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ---------------------------------------------------------------------------

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA as skPCA
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# TensorFlow setup
# ---------------------------------------------------------------------------

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

# On macOS with tensorflow-metal, Metal GPUs appear as GPU type devices.
# Confirm by checking for the metal plugin in registered devices.
_all_devices = tf.config.list_physical_devices()
metal = sys.platform == "darwin" and len(gpus) > 0

DEVICE_INFO = {
    "tensorflow_version": tf.__version__,
    "device_used": "Metal GPU" if metal else ("GPU" if gpus else "CPU"),
    "physical_devices": [d.name for d in _all_devices],
}
print(f"TensorFlow {tf.__version__} | Device: {DEVICE_INFO['device_used']}")
print(f"Physical devices: {[d.name for d in _all_devices]}")

# Explicit device strategy: route to Metal GPU when available, else CPU.
if gpus:
    STRATEGY = tf.distribute.OneDeviceStrategy("/GPU:0")
else:
    STRATEGY = tf.distribute.OneDeviceStrategy("/CPU:0")

from tensorflow import keras  # noqa: E402

# ---------------------------------------------------------------------------
# proteusPy path injection
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from proteusPy.manifold_model import ManifoldModel  # noqa: E402

# ---------------------------------------------------------------------------
# Phase 1: Manifold Discovery
# ---------------------------------------------------------------------------


def discover_dimensionality(X, n_samples=500, k=50, variance_thresholds=(0.95, 0.90, 0.85)):
    """Discover intrinsic dimensionality of the data manifold via local PCA.

    Samples n_samples random points, computes local PCA at each,
    and returns statistics on intrinsic dimensionality.

    :param X: Data matrix of shape (n_points, n_dims).
    :param n_samples: Number of random points to sample.
    :param k: Neighborhood size for local PCA.
    :param variance_thresholds: Iterable of τ values to report.
    :returns: Dict mapping each τ to a statistics dict.
    """
    n_points, ndim = X.shape
    sample_idx = np.random.choice(n_points, size=min(n_samples, n_points), replace=False)

    results = {tau: [] for tau in variance_thresholds}

    for idx in sample_idx:
        point = X[idx]
        # k nearest neighbors
        dists = np.linalg.norm(X - point, axis=1)
        knn_idx = np.argpartition(dists, k)[:k]
        neighbors = X[knn_idx]

        # Local PCA — cast to float64 to avoid Apple BLAS float32 overflow
        centered = (neighbors - neighbors.mean(axis=0)).astype(np.float64)
        cov = np.einsum("ij,ik->jk", centered, centered) / (len(neighbors) - 1)
        eigenvalues = np.linalg.eigvalsh(cov)[::-1]
        eigenvalues = np.maximum(eigenvalues, 0.0)

        total = eigenvalues.sum()
        if total > 0:
            cumulative = np.cumsum(eigenvalues) / total
            for tau in variance_thresholds:
                d = int(np.searchsorted(cumulative, tau) + 1)
                results[tau].append(d)

    report = {}
    for tau in variance_thresholds:
        dims = results[tau]
        report[tau] = {
            "mean": np.mean(dims),
            "std": np.std(dims),
            "median": np.median(dims),
            "min": int(np.min(dims)),
            "max": int(np.max(dims)),
        }
    return report


def discover_per_class_dimensionality(X, y, k=50, tau=0.90, n_samples_per_class=50):
    """Discover intrinsic dimensionality per class.

    :param X: Data matrix of shape (n_points, n_dims).
    :param y: Class labels of shape (n_points,).
    :param k: Neighborhood size for local PCA.
    :param tau: Variance threshold.
    :param n_samples_per_class: Number of random points to sample per class.
    :returns: Dict mapping class label to a statistics dict.
    """
    classes = sorted(set(y))
    class_dims = {}

    for c in classes:
        X_c = X[y == c]
        n_sample = min(n_samples_per_class, len(X_c))
        sample_idx = np.random.choice(len(X_c), size=n_sample, replace=False)

        dims = []
        for idx in sample_idx:
            point = X_c[idx]
            dists = np.linalg.norm(X_c - point, axis=1)
            k_use = min(k, len(X_c) - 1)
            knn_idx = np.argpartition(dists, k_use)[:k_use]
            neighbors = X_c[knn_idx]

            # cast to float64 to avoid Apple BLAS float32 overflow
            centered = (neighbors - neighbors.mean(axis=0)).astype(np.float64)
            cov = np.einsum("ij,ik->jk", centered, centered) / (len(neighbors) - 1)
            eigenvalues = np.linalg.eigvalsh(cov)[::-1]
            eigenvalues = np.maximum(eigenvalues, 0.0)

            total = eigenvalues.sum()
            if total > 0:
                cumulative = np.cumsum(eigenvalues) / total
                d = int(np.searchsorted(cumulative, tau) + 1)
                dims.append(d)

        class_dims[c] = {
            "mean": np.mean(dims),
            "std": np.std(dims),
            "min": int(np.min(dims)),
            "max": int(np.max(dims)),
        }

    return class_dims


# ---------------------------------------------------------------------------
# Phase 2: Model Builders
# ---------------------------------------------------------------------------


def build_standard_model(input_dim, n_classes, lr=0.001):
    """Standard architecture: input → 128 → 64 → n_classes. Common default.

    :param input_dim: Dimensionality of input features.
    :param n_classes: Number of output classes.
    :param lr: Adam learning rate.
    :returns: Compiled Keras model.
    """
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_manifold_model(input_dim, n_classes, intrinsic_dim, lr=0.001):
    """Manifold-informed architecture: input → 2d → d → output.

    The bottleneck width matches the discovered intrinsic dimensionality.
    The layer before it is 2d to give the network room to learn the
    projection down to the manifold.

    :param input_dim: Dimensionality of input features.
    :param n_classes: Number of output classes.
    :param intrinsic_dim: Discovered intrinsic dimensionality d.
    :param lr: Adam learning rate.
    :returns: Compiled Keras model.
    """
    d = max(intrinsic_dim, n_classes)  # at least as wide as output
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(2 * d, activation="relu"),
            keras.layers.Dense(d, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_manifold_observer_model(input_dim, n_classes, intrinsic_dim, lr=0.001):
    """Manifold observer architecture: input → d+1 → d+1 → output.

    Mirrors the ManifoldObserver's extrinsic (N+1)-dimensional vantage:
    one extra dimension above the manifold.

    :param input_dim: Dimensionality of input features.
    :param n_classes: Number of output classes.
    :param intrinsic_dim: Discovered intrinsic dimensionality d.
    :param lr: Adam learning rate.
    :returns: Compiled Keras model.
    """
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(intrinsic_dim + 1, activation="relu"),
            keras.layers.Dense(intrinsic_dim + 1, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_wide_manifold_model(input_dim, n_classes, intrinsic_dim, lr=0.001):
    """Wider manifold-informed: input → 4d → 2d → d → output.

    Three hidden layers with progressive compression toward the
    manifold bottleneck. More capacity for learning the projection.

    :param input_dim: Dimensionality of input features.
    :param n_classes: Number of output classes.
    :param intrinsic_dim: Discovered intrinsic dimensionality d.
    :param lr: Adam learning rate.
    :returns: Compiled Keras model.
    """
    d = max(intrinsic_dim, n_classes)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(4 * d, activation="relu"),
            keras.layers.Dense(2 * d, activation="relu"),
            keras.layers.Dense(d, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_pca_model(n_classes, intrinsic_dim, lr=0.001):
    """PCA pre-projected model: d → 2d → d → output.

    Input is already PCA-projected to intrinsic_dim dimensions.
    The network only needs to learn the nonlinear classification
    in the manifold subspace, not the projection itself.

    :param n_classes: Number of output classes.
    :param intrinsic_dim: Discovered intrinsic dimensionality d (also input dim).
    :param lr: Adam learning rate.
    :returns: Compiled Keras model.
    """
    d = max(intrinsic_dim, n_classes)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(intrinsic_dim,)),
            keras.layers.Dense(2 * d, activation="relu"),
            keras.layers.Dense(d, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_pca_intrinsic_dim_model(n_classes, intrinsic_dim, lr=0.001):
    """Intrinsic-dim-only model: d → d → output.

    Minimal head on top of PCA projection — just one hidden layer at
    intrinsic dimensionality then output.

    :param n_classes: Number of output classes.
    :param intrinsic_dim: Discovered intrinsic dimensionality d (also input dim).
    :param lr: Adam learning rate.
    :returns: Compiled Keras model.
    """
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(intrinsic_dim,)),
            keras.layers.Dense(intrinsic_dim, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def count_params(model):
    """Return total number of trainable parameters in a Keras model.

    :param model: Compiled Keras model.
    :returns: Integer parameter count.
    """
    return sum(int(np.prod(w.shape)) for w in model.trainable_weights)


# ---------------------------------------------------------------------------
# Phase 3: Fold Runner (handles both Keras and sklearn)
# ---------------------------------------------------------------------------


def run_fold(
    build_fn, X_train, y_train, X_test, y_test, epochs, batch_size, fold, is_sklearn=False
):
    """Run one CV fold — works for both Keras models and sklearn classifiers.

    :param build_fn: Zero-arg callable that returns a fresh model or classifier.
    :param X_train: Training features.
    :param y_train: Training labels.
    :param X_test: Test features.
    :param y_test: Test labels.
    :param epochs: Training epochs (Keras only).
    :param batch_size: Batch size (Keras only).
    :param fold: Fold index (0-based), recorded in result.
    :param is_sklearn: If True, treat as a sklearn estimator.
    :returns: Dict of per-fold metrics.
    """
    if is_sklearn:
        clf = build_fn()
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0
        t1 = time.perf_counter()
        acc = float(clf.score(X_test, y_test))
        pred_time = time.perf_counter() - t1
        # geometry stats for ManifoldModel
        geometry = None
        if hasattr(clf, "geometry_summary"):
            geometry = clf.geometry_summary()
        return {
            "fold": fold,
            "n_params": 0,
            "test_loss": None,
            "test_acc": acc,
            "wall_time": fit_time + pred_time,
            "fit_time": fit_time,
            "pred_time": pred_time,
            "convergence_epoch": None,
            "geometry": geometry,
        }
    else:
        # Keras model
        with STRATEGY.scope():
            model = build_fn()
        n_params = count_params(model)
        t0 = time.perf_counter()
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=0,
        )
        wall_time = time.perf_counter() - t0
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        conv_epoch = None
        for i, a in enumerate(history.history["accuracy"]):
            if a >= 0.95:
                conv_epoch = i
                break
        return {
            "fold": fold,
            "n_params": n_params,
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "wall_time": wall_time,
            "fit_time": wall_time,
            "pred_time": None,
            "convergence_epoch": conv_epoch,
            "geometry": None,
            "train_acc": [float(a) for a in history.history["accuracy"]],
            "val_acc": [float(a) for a in history.history["val_accuracy"]],
            "train_loss": [float(v) for v in history.history["loss"]],
            "val_loss": [float(v) for v in history.history["val_loss"]],
        }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(all_results, intrinsic_dim, save_path, elapsed=None):
    """Save a four-panel comparison figure.

    :param all_results: Dict mapping architecture name → list of fold result dicts.
    :param intrinsic_dim: Discovered bottleneck dimension d.
    :param save_path: Filesystem path for the PNG output.
    :param elapsed: Optional total wall time in seconds (for figure title).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    # Color palette — sklearn methods get distinct colours
    _palette = [
        "steelblue",
        "firebrick",
        "forestgreen",
        "darkorchid",
        "darkorange",
        "deeppink",
        "saddlebrown",
        "teal",
    ]
    names = list(all_results.keys())
    color_map = {n: _palette[i % len(_palette)] for i, n in enumerate(names)}

    # Separate Keras vs sklearn results
    keras_names = [n for n in names if any(r.get("train_acc") for r in all_results[n])]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    elapsed_str = f"  |  run time: {elapsed:.0f}s" if elapsed is not None else ""
    fig.suptitle(
        f"Digits: Manifold Architecture Comparison (d={intrinsic_dim}){elapsed_str}",
        fontsize=14,
        fontweight="bold",
    )

    # ------------------------------------------------------------------
    # Top-left: Per-fold accuracy — box plot for all methods
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    per_fold_accs = []
    box_labels = []
    box_colors = []
    for name in names:
        accs = [r["test_acc"] for r in all_results[name]]
        per_fold_accs.append(accs)
        short = name.split("(")[0].strip()
        box_labels.append(short)
        box_colors.append(color_map[name])

    bp = ax.boxplot(per_fold_accs, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(box_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("CV Fold Accuracy")
    ax.set_title("Per-Fold Accuracy Distribution (5-Fold CV)")
    ax.grid(True, alpha=0.3, axis="y")

    # ------------------------------------------------------------------
    # Top-right: Training loss curves (Keras methods only)
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    for name in keras_names:
        results = all_results[name]
        loss_curves = [r["train_loss"] for r in results if r.get("train_loss")]
        if not loss_curves:
            continue
        # Pad to same length before stacking
        max_len = max(len(c) for c in loss_curves)
        padded = [c + [c[-1]] * (max_len - len(c)) for c in loss_curves]
        losses = np.array(padded)
        epochs_ax = np.arange(1, losses.shape[1] + 1)
        color = color_map[name]
        short = name.split("(")[0].strip()
        ax.plot(epochs_ax, losses.mean(0), "-", label=short, linewidth=2, color=color)
        ax.fill_between(
            epochs_ax,
            losses.mean(0) - losses.std(0),
            losses.mean(0) + losses.std(0),
            alpha=0.15,
            color=color,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss (Keras models, mean ± std across folds×trials)")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Bottom-left: Final test accuracy bar chart with error bars
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    means = [np.mean([r["test_acc"] for r in all_results[n]]) for n in names]
    stds = [np.std([r["test_acc"] for r in all_results[n]]) for n in names]
    bar_colors = [color_map[n] for n in names]
    short_names = [n.split("(")[0].strip() for n in names]
    bars = ax.bar(short_names, means, yerr=stds, color=bar_colors, alpha=0.8, capsize=5)
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{m:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=8,
        )
    ax.set_ylabel("Mean CV Test Accuracy")
    ax.set_title("Final Test Accuracy (mean ± std)")
    ax.set_xticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    acc_floor = max(0.85, min(means) - 0.05)
    ax.set_ylim(acc_floor, 1.01)

    # ------------------------------------------------------------------
    # Bottom-right: Parameter count (log scale), sklearn at position 1
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    param_counts = []
    param_labels = []
    for name in names:
        n_params = all_results[name][0]["n_params"]
        param_counts.append(max(n_params, 1))  # avoid log(0)
        short = name.split("(")[0].strip()
        param_labels.append(short)

    bars_p = ax.bar(
        param_labels,
        param_counts,
        color=bar_colors,
        alpha=0.8,
    )
    for bar, name, p in zip(bars_p, names, param_counts):
        actual = all_results[name][0]["n_params"]
        label = "geometric" if actual == 0 else f"{actual:,}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.3,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_ylabel("Parameters")
    ax.set_title("Parameter Count (log scale)\nsklearn methods: non-parametric")
    ax.set_yscale("log")
    ax.set_xticks(range(len(param_labels)))
    ax.set_xticklabels(param_labels, rotation=30, ha="right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Digits: Manifold-Informed Architecture — Comprehensive Comparison"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs for Keras models")
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Random-seed trials per fold for Keras models",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Adam learning rate")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size (small — dataset is tiny)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.90,
        help="Variance threshold τ for intrinsic dimensionality",
    )
    parser.add_argument(
        "--discovery-samples",
        type=int,
        default=200,
        help="Points to sample for dimensionality discovery (out of 1797)",
    )
    parser.add_argument("--k-pca", type=int, default=30, help="Neighborhood size for local PCA")
    parser.add_argument(
        "--k-graph",
        type=int,
        default=15,
        help="Neighborhood size for ManifoldModel graph construction",
    )
    parser.add_argument("--k-vote", type=int, default=7, help="Voting neighbors for ManifoldModel")
    parser.add_argument("--plot", action="store_true", default=True)
    args = parser.parse_args()
    t_start = time.perf_counter()

    # -----------------------------------------------------------------------
    # Load and scale data
    # -----------------------------------------------------------------------

    print("\nLoading sklearn digits...")
    data = load_digits()
    X, y = data.data.astype("float32"), data.target

    scaler = StandardScaler()
    # nan_to_num: constant pixels (std=0) produce NaN after scaling → zero them
    X = np.nan_to_num(scaler.fit_transform(X)).astype("float32")

    input_dim = X.shape[1]
    n_classes = len(set(y))

    print(f"  Dataset: {X.shape[0]} samples, {input_dim} dims, {n_classes} classes")
    print("  Evaluation: 5-fold stratified cross-validation")

    # -----------------------------------------------------------------------
    # Phase 1: Discover intrinsic dimensionality
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 1: MANIFOLD DISCOVERY")
    print("=" * 70)

    print(f"\nSampling {args.discovery_samples} points, k={args.k_pca} neighbors...")
    t0 = time.perf_counter()
    dim_report = discover_dimensionality(
        X,
        n_samples=args.discovery_samples,
        k=args.k_pca,
        variance_thresholds=(0.95, 0.90, 0.85, 0.80),
    )
    discovery_time = time.perf_counter() - t0
    print(f"Discovery time: {discovery_time:.1f}s\n")

    print(f"{'τ':>6} {'Mean d':>8} {'Std':>6} {'Min':>5} {'Max':>5} {'Noise %':>8}")
    print("-" * 45)
    for tau in sorted(dim_report.keys(), reverse=True):
        r = dim_report[tau]
        noise_pct = 100 * (1 - r["mean"] / input_dim)
        print(
            f"{tau:>6.2f} {r['mean']:>8.1f} {r['std']:>6.1f}"
            f" {r['min']:>5} {r['max']:>5} {noise_pct:>7.1f}%"
        )

    # Per-class dimensionality
    print(f"\nPer-class intrinsic dimensionality (τ={args.tau}):")
    class_dims = discover_per_class_dimensionality(
        X, y, k=args.k_pca, tau=args.tau, n_samples_per_class=50
    )
    for c in sorted(class_dims.keys()):
        cd = class_dims[c]
        print(f"  Digit {c}: d = {cd['mean']:.1f} ± {cd['std']:.1f}  [{cd['min']}, {cd['max']}]")

    # Bottleneck = max of per-class maxima — accommodates the hardest manifold
    global_dim = int(round(dim_report[args.tau]["mean"]))
    intrinsic_dim = max(cd["max"] for cd in class_dims.values())
    d = intrinsic_dim
    print(
        f"\n>> Global intrinsic dim (mean): {global_dim}"
        f"  |  Max per-class max: {intrinsic_dim}"
        f"  →  using d = {d} (τ={args.tau})"
    )

    # -----------------------------------------------------------------------
    # Phase 2: Architecture summary
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 2: ARCHITECTURE SUMMARY")
    print("=" * 70)

    # Build one instance of each Keras architecture to print param counts
    _sample_builds = {
        "Standard (128→64)": lambda: build_standard_model(input_dim, n_classes, lr=args.lr),
        f"Wide Manifold (4d→2d→d, d={d})": lambda: build_wide_manifold_model(
            input_dim, n_classes, d, lr=args.lr
        ),
        f"Manifold (2d→d, d={d})": lambda: build_manifold_model(
            input_dim, n_classes, d, lr=args.lr
        ),
        f"PCA→{d}D + MLP (2d→d)": lambda: build_pca_model(n_classes, d, lr=args.lr),
        f"Intrinsic Dim (PCA→{d}D→output)": lambda: build_pca_intrinsic_dim_model(
            n_classes, d, lr=args.lr
        ),
    }

    for name, bfn in _sample_builds.items():
        model = bfn()
        n_params = count_params(model)
        print(f"\n{name}:")
        print(f"  Parameters: {n_params:,}")
        for layer in model.layers:
            if hasattr(layer, "units"):
                print(f"  {layer.name}: → {layer.units}")

    print(f"\nManifoldModel (τ={args.tau}): 0 learned parameters")
    print("  The manifold IS the model. Pure geometry.")
    print(f"\nEuclidean KNN (k={args.k_vote}): 0 learned parameters")
    print("  Classic non-parametric baseline.")

    # -----------------------------------------------------------------------
    # Phase 3: 5-Fold CV
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 3: 5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 70)
    print(f"  Folds: 5  |  Keras trials per fold: {args.trials}  |  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}  |  LR: {args.lr}  |  τ: {args.tau}")

    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Build the architectures dict now that we know d.
    # Each entry: (build_fn, is_sklearn, needs_pca)
    # needs_pca=True means the fold's training data must be PCA-projected.
    architectures = {
        f"Euclidean KNN (k={args.k_vote})": (
            lambda: KNeighborsClassifier(n_neighbors=args.k_vote, metric="euclidean"),
            True,
            False,
        ),
        f"ManifoldModel (τ={args.tau})": (
            lambda: ManifoldModel(
                k_graph=args.k_graph,
                k_pca=args.k_pca,
                k_vote=args.k_vote,
                variance_threshold=args.tau,
            ),
            True,
            False,
        ),
        "Standard (128→64)": (
            lambda: build_standard_model(input_dim, n_classes, lr=args.lr),
            False,
            False,
        ),
        f"Wide Manifold (4d→2d→d, d={d})": (
            lambda: build_wide_manifold_model(input_dim, n_classes, d, lr=args.lr),
            False,
            False,
        ),
        f"Manifold (2d→d, d={d})": (
            lambda: build_manifold_model(input_dim, n_classes, d, lr=args.lr),
            False,
            False,
        ),
        f"PCA→{d}D + MLP (2d→d)": (
            lambda: build_pca_model(n_classes, d, lr=args.lr),
            False,
            True,
        ),
        f"Intrinsic Dim (PCA→{d}D→output)": (
            lambda: build_pca_intrinsic_dim_model(n_classes, d, lr=args.lr),
            False,
            True,
        ),
    }

    all_results = {}

    fold_splits = list(skf.split(X, y))

    for name, (build_fn, is_sklearn, needs_pca) in architectures.items():
        print(f"\n{name}...")
        fold_results = []

        for fold_i, (train_idx, test_idx) in enumerate(fold_splits):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # PCA is fit on the training fold only (no data leakage)
            if needs_pca:
                pca_fold = skPCA(n_components=d)
                # cast to float64 for PCA matmul to avoid Apple BLAS float32 overflow
                X_tr_use = pca_fold.fit_transform(X_tr.astype(np.float64)).astype("float32")
                X_te_use = pca_fold.transform(X_te.astype(np.float64)).astype("float32")
            else:
                X_tr_use, X_te_use = X_tr, X_te

            if is_sklearn:
                result = run_fold(
                    build_fn,
                    X_tr_use,
                    y_tr,
                    X_te_use,
                    y_te,
                    args.epochs,
                    args.batch_size,
                    fold_i,
                    is_sklearn=True,
                )
                fold_results.append(result)
            else:
                for trial in range(args.trials):
                    np.random.seed(trial * 100 + fold_i)
                    tf.random.set_seed(trial * 100 + fold_i)
                    result = run_fold(
                        build_fn,
                        X_tr_use,
                        y_tr,
                        X_te_use,
                        y_te,
                        args.epochs,
                        args.batch_size,
                        fold_i,
                        is_sklearn=False,
                    )
                    fold_results.append(result)

        all_results[name] = fold_results

        # Single summary line per method
        mean_acc = np.mean([r["test_acc"] for r in fold_results])
        std_acc = np.std([r["test_acc"] for r in fold_results])
        total_time = sum(r["wall_time"] for r in fold_results)
        extra = ""
        if name.startswith("ManifoldModel"):
            geoms = [r["geometry"] for r in fold_results if r.get("geometry")]
            if geoms:
                mean_id = np.mean([g["mean_intrinsic_dim"] for g in geoms if g])
                extra = f"  id={mean_id:.1f}"
        print(f"  {mean_acc:.4f} ± {std_acc:.4f}  ({total_time:.1f}s total{extra})")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------

    elapsed = time.perf_counter() - t_start

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY — 5-FOLD STRATIFIED CV")
    print("=" * 70)
    print(f"Dataset: sklearn digits ({input_dim}D, {n_classes} classes, 1797 samples)")
    print(f"Intrinsic dimensionality: d = {intrinsic_dim} (τ={args.tau})")
    print(
        f"Noise dimensions: {100 * (1 - intrinsic_dim / input_dim):.1f}%"
        f"  ({input_dim - intrinsic_dim} of {input_dim} dims suppressed)"
    )
    print(f"Epochs: {args.epochs}  |  Trials/fold (Keras): {args.trials}")
    print(f"Device: {DEVICE_INFO['device_used']}  |  Total time: {elapsed:.1f}s")
    print("-" * 70)

    # Determine best mean accuracy for marker
    best_acc = max(np.mean([r["test_acc"] for r in all_results[n]]) for n in all_results)

    print(f"\n{'Architecture':<40} {'Mean Acc':>10} {'Std':>8} {'Params':>10} {'Time (s)':>10}")
    print("-" * 80)

    for name in all_results:
        results = all_results[name]
        accs = [r["test_acc"] for r in results]
        times = [r["wall_time"] for r in results]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_time = np.mean(times)
        n_params = results[0]["n_params"]
        params_str = "non-param" if n_params == 0 else f"{n_params:,}"
        marker = "  << BEST" if abs(mean_acc - best_acc) < 1e-9 else ""
        print(
            f"{name:<40} {mean_acc:>10.4f} {std_acc:>8.4f}"
            f" {params_str:>10} {mean_time:>9.2f}s{marker}"
        )

    # ManifoldModel geometry footnote
    print("\n" + "-" * 70)
    print("MANIFOLD GEOMETRY (ManifoldModel folds)")
    print("-" * 70)
    for name in all_results:
        if name.startswith("ManifoldModel"):
            geoms = [r["geometry"] for r in all_results[name] if r.get("geometry")]
            if geoms:
                mean_ids = [g["mean_intrinsic_dim"] for g in geoms if g]
                ambient = geoms[0].get("ambient_dim", input_dim)
                noise_pct = 100 * (1 - np.mean(mean_ids) / ambient)
                print(
                    f"  {name}: mean intrinsic dim = {np.mean(mean_ids):.1f}"
                    f" / {ambient}  ({noise_pct:.0f}% noise suppressed)"
                )

    # Parameter efficiency
    print("-" * 70)
    print("PARAMETER EFFICIENCY (accuracy per 1K parameters, Keras models only):")
    for name in all_results:
        results = all_results[name]
        n_params = results[0]["n_params"]
        if n_params > 0:
            mean_acc = np.mean([r["test_acc"] for r in results])
            eff = mean_acc / n_params * 1000
            print(f"  {name}: {eff:.4f} acc/Kparam  ({mean_acc:.4f} / {n_params:,})")

    # Winner callout
    print("-" * 70)
    best_name = max(
        all_results,
        key=lambda n: np.mean([r["test_acc"] for r in all_results[n]]),
    )
    best_mean = np.mean([r["test_acc"] for r in all_results[best_name]])
    std_name = "Standard (128→64)"
    std_mean = np.mean([r["test_acc"] for r in all_results[std_name]])

    if best_name != std_name:
        delta = best_mean - std_mean
        print(f">> WINNER: {best_name}")
        print(f"   {best_mean:.4f} vs {std_mean:.4f} (standard MLP)")
        print(f"   Delta: +{delta:.4f} ({delta * 100:.2f} pp)")
        n_params_best = all_results[best_name][0]["n_params"]
        n_params_std = all_results[std_name][0]["n_params"]
        if n_params_best == 0:
            print("   Uses ZERO learned parameters — pure manifold geometry")
        elif n_params_best < n_params_std:
            reduction = 100 * (1 - n_params_best / n_params_std)
            print(
                f"   With {reduction:.0f}% FEWER parameters ({n_params_best:,} vs {n_params_std:,})"
            )
        elif n_params_best > n_params_std:
            increase = 100 * (n_params_best / n_params_std - 1)
            print(
                f"   With {increase:.0f}% more parameters ({n_params_best:,} vs {n_params_std:,})"
            )
    else:
        print(f">> Standard architecture wins: {std_mean:.4f}")

    print("=" * 70)

    # -----------------------------------------------------------------------
    # Save results JSON
    # -----------------------------------------------------------------------

    save_data = {
        "device": DEVICE_INFO,
        "dataset": "digits",
        "n_samples": int(X.shape[0]),
        "input_dim": input_dim,
        "n_classes": n_classes,
        "intrinsic_dim": intrinsic_dim,
        "global_intrinsic_dim_mean": global_dim,
        "tau": args.tau,
        "epochs": args.epochs,
        "trials": args.trials,
        "n_folds": n_folds,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "k_pca": args.k_pca,
        "k_graph": args.k_graph,
        "k_vote": args.k_vote,
        "total_elapsed_s": elapsed,
        "dimensionality_report": {str(k): v for k, v in dim_report.items()},
        "per_class_dims": {str(k): v for k, v in class_dims.items()},
        "results": {name: results for name, results in all_results.items()},
    }

    results_path = Path(__file__).resolve().parent / "digits_manifold_architecture_results.json"
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------

    if args.plot:
        plot_path = str(
            Path(__file__).resolve().parent / "digits_manifold_architecture_results.png"
        )
        plot_results(all_results, intrinsic_dim, plot_path, elapsed=elapsed)


if __name__ == "__main__":
    main()
