#!/usr/bin/env python3
"""
CIFAR-10 Benchmark: Manifold-Informed Architecture vs Standard Architecture
===========================================================================

Extends the MNIST manifold-architecture experiment to CIFAR-10:
3,072-dimensional color images (32×32×3), 50K training / 10K test samples,
10 classes.

The hypothesis: if CIFAR-10 images live on a d-dimensional manifold inside
3,072-dimensional pixel space, then an MLP whose bottleneck matches d should
achieve comparable accuracy to a standard over-parameterised architecture
while using fewer parameters on the noise dimensions.

Three phases
------------
Phase 1 — Manifold Discovery
    Local PCA over --discovery-samples random training points (k=--k-pca
    neighbors each).  Intrinsic dimensionality d is set to the maximum
    per-class intrinsic dim at τ=--tau (default 0.90), accommodating the
    hardest class.

Phase 2 — Architecture Comparison
    Five architectures are built and summarised:

    - Standard (256→128):           input → 256 → 128 → 10
    - Manifold (2d→d):              input → 2d  → d   → 10
    - Wide Manifold (4d→2d→d):      input → 4d  → 2d  → d → 10
    - PCA→dD + MLP (2d→d):          PCA-projected input → 2d → d → 10
    - Intrinsic Dim (PCA→dD→output): PCA-projected input → d  → 10

    The PCA models receive input already compressed to d dimensions by
    global sklearn PCA; the remaining models operate in raw pixel space.

Phase 3 — Training and Evaluation
    All architectures are trained with Adam for --epochs epochs across
    --trials independent random seeds.  Aggregate results are printed and
    saved to ``cifar10_architecture_results.json`` alongside a four-panel
    matplotlib figure (``cifar10_architecture_results.png``).

Part of proteusPy, https://github.com/suchanek/proteusPy
Author: Eric G. Suchanek, PhD
Affiliation: Flux-Frontiers

Usage
-----
    python benchmarks/canonical_tests/cifar10_manifold_architecture.py [--epochs 50] [--trials 5]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
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

# On macOS with tensorflow-metal, Metal GPUs appear as GPU-type devices.
# "METAL" does NOT appear in device names — correct detection: darwin + GPU present.
metal = sys.platform == "darwin" and len(gpus) > 0
DEVICE_INFO = {
    "tensorflow_version": tf.__version__,
    "device_used": "Metal GPU" if metal else ("GPU" if gpus else "CPU"),
}
print(f"TensorFlow {tf.__version__} | Device: {DEVICE_INFO['device_used']}")

from tensorflow import keras  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ---------------------------------------------------------------------------
# Phase 1: Manifold Discovery
# ---------------------------------------------------------------------------


def discover_dimensionality(X, n_samples=500, k=50, variance_thresholds=(0.95, 0.90, 0.85)):
    """Discover intrinsic dimensionality of the data manifold via local PCA.

    For high-dimensional data like CIFAR-10 (3072D), we use the covariance
    of the k-nearest neighbors rather than the full covariance matrix.
    """
    n_points, ndim = X.shape
    sample_idx = np.random.choice(n_points, size=min(n_samples, n_points), replace=False)

    results = {tau: [] for tau in variance_thresholds}

    for i, idx in enumerate(sample_idx):
        if (i + 1) % 100 == 0:
            print(f"  Local PCA: {i + 1}/{len(sample_idx)}", flush=True)

        point = X[idx]
        # k nearest neighbors
        dists = np.linalg.norm(X - point, axis=1)
        knn_idx = np.argpartition(dists, k)[:k]
        neighbors = X[knn_idx]

        # Local PCA via SVD (more efficient for k << ndim)
        centered = neighbors - neighbors.mean(axis=0)
        # Use SVD on the data matrix instead of eigendecomposition on the
        # covariance matrix — this is O(k^2 * ndim) instead of O(ndim^3)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        eigenvalues = (s ** 2) / (len(neighbors) - 1)

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
    """Discover intrinsic dimensionality per class."""
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

            centered = neighbors - neighbors.mean(axis=0)
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
            eigenvalues = (s ** 2) / (len(neighbors) - 1)

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
# CIFAR-10 class names
# ---------------------------------------------------------------------------

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ---------------------------------------------------------------------------
# Phase 2: Model Builders
# ---------------------------------------------------------------------------


def build_standard_model(input_dim, n_classes, lr=0.001):
    """Standard MLP for CIFAR-10: 3072 → 256 → 128 → 10.

    Scaled up from MNIST's 128→64 since CIFAR-10 is 4× the dimensionality
    and significantly more complex (color, texture, shape).
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_manifold_model(input_dim, n_classes, intrinsic_dim, lr=0.001):
    """Manifold-informed architecture: input → 2d → d → output.

    The bottleneck width matches the discovered intrinsic dimensionality.
    """
    d = max(intrinsic_dim, n_classes)
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(2 * d, activation="relu"),
        keras.layers.Dense(d, activation="relu"),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])
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
    """
    d = max(intrinsic_dim, n_classes)
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(4 * d, activation="relu"),
        keras.layers.Dense(2 * d, activation="relu"),
        keras.layers.Dense(d, activation="relu"),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_pca_model(n_classes, intrinsic_dim, lr=0.001):
    """PCA pre-projected model: d → 2d → d → output.

    Input is already PCA-projected to intrinsic_dim dimensions.
    """
    d = max(intrinsic_dim, n_classes)
    model = keras.Sequential([
        keras.layers.Input(shape=(intrinsic_dim,)),
        keras.layers.Dense(2 * d, activation="relu"),
        keras.layers.Dense(d, activation="relu"),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_pca_intrinsic_dim_model(n_classes, intrinsic_dim, lr=0.001):
    """PCA pre-projected model: d → d → output.

    Input is already PCA-projected to intrinsic_dim dimensions.
    The network only needs to learn the nonlinear classification
    in the manifold subspace, not the projection itself.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(intrinsic_dim,)),
        keras.layers.Dense(intrinsic_dim, activation="relu"),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def count_params(model):
    return sum(int(np.prod(w.shape)) for w in model.trainable_weights)


# ---------------------------------------------------------------------------
# Phase 3: Benchmark
# ---------------------------------------------------------------------------


def run_trial(build_fn, X_train, y_train, X_test, y_test, epochs, batch_size, trial):
    """Train a model and return metrics."""
    model = build_fn()
    n_params = count_params(model)

    t0 = time.perf_counter()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=0,
    )
    wall_time = time.perf_counter() - t0

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    # Convergence epoch (first epoch hitting 40% train accuracy — lower
    # threshold than MNIST since CIFAR-10 MLPs top out around 55%)
    conv_epoch = None
    for i, acc in enumerate(history.history["accuracy"]):
        if acc >= 0.40:
            conv_epoch = i
            break

    return {
        "trial": trial,
        "n_params": n_params,
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "wall_time": wall_time,
        "convergence_epoch": conv_epoch,
        "train_acc": [float(a) for a in history.history["accuracy"]],
        "val_acc": [float(a) for a in history.history["val_accuracy"]],
        "train_loss": [float(v) for v in history.history["loss"]],
        "val_loss": [float(v) for v in history.history["val_loss"]],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(all_results, intrinsic_dim, save_path, elapsed=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    colors = {
        "Standard (256→128)": "steelblue",
        f"Manifold (2d→d, d={intrinsic_dim})": "firebrick",
        f"Wide Manifold (4d→2d→d, d={intrinsic_dim})": "forestgreen",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    elapsed_str = f"  |  run time: {elapsed:.0f}s" if elapsed is not None else ""
    fig.suptitle(
        f"CIFAR-10: Manifold-Informed Architecture (d={intrinsic_dim}) vs Standard\n"
        f"3,072D color images → manifold discovery → architecture{elapsed_str}",
        fontsize=14, fontweight="bold",
    )

    # Validation accuracy curves
    ax = axes[0, 0]
    for name, results in all_results.items():
        accs = np.array([r["val_acc"] for r in results])
        epochs = np.arange(1, accs.shape[1] + 1)
        color = colors.get(name, "gray")
        ax.plot(epochs, accs.mean(0), "-", label=name, linewidth=2, color=color)
        ax.fill_between(epochs, accs.mean(0) - accs.std(0),
                        accs.mean(0) + accs.std(0), alpha=0.15, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Validation Accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Training loss curves
    ax = axes[0, 1]
    for name, results in all_results.items():
        losses = np.array([r["train_loss"] for r in results])
        epochs = np.arange(1, losses.shape[1] + 1)
        color = colors.get(name, "gray")
        ax.plot(epochs, losses.mean(0), "-", label=name, linewidth=2, color=color)
        ax.fill_between(epochs, losses.mean(0) - losses.std(0),
                        losses.mean(0) + losses.std(0), alpha=0.15, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Final test accuracy comparison
    ax = axes[1, 0]
    names = list(all_results.keys())
    means = [np.mean([r["test_acc"] for r in all_results[n]]) for n in names]
    stds = [np.std([r["test_acc"] for r in all_results[n]]) for n in names]
    bar_colors = [colors.get(n, "gray") for n in names]
    short_names = [n.split("(")[0].strip() for n in names]
    bars = ax.bar(short_names, means, yerr=stds, color=bar_colors, alpha=0.8, capsize=5)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{m:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Final Test Accuracy")
    ax.grid(True, alpha=0.3, axis="y")
    # CIFAR-10 MLP accuracy typically 40-55%
    ax.set_ylim(0.3, 0.65)

    # Parameter efficiency
    ax = axes[1, 1]
    param_counts = [all_results[n][0]["n_params"] for n in names]
    bars = ax.bar(short_names, param_counts, color=bar_colors, alpha=0.8)
    for bar, p, m in zip(bars, param_counts, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{p:,}\nacc={m:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Parameters")
    ax.set_title("Parameter Count (lower is better at same accuracy)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _generate_synthetic_cifar10(
    n_train=50000, n_test=10000, ambient_dim=3072, intrinsic_dim=35,
    n_classes=10, seed=42,
):
    """Generate synthetic data mimicking CIFAR-10's manifold structure.

    Each class lives on a ~intrinsic_dim-dimensional manifold embedded
    in ambient_dim-dimensional space via a random linear projection plus
    nonlinear warping. Classes share some subspace overlap to make
    classification non-trivial (targeting ~55% MLP accuracy, similar to
    CIFAR-10 with a flat MLP).
    """
    rng = np.random.RandomState(seed)

    # Shared random projection matrix: all classes embed through this,
    # but each class uses a different subspace + offset
    rng.randn(ambient_dim, ambient_dim).astype("float32") * 0.01

    X_all, y_all = [], []
    n_total = n_train + n_test
    per_class = n_total // n_classes

    # Class centers are close together (not well-separated)
    centers = rng.randn(n_classes, ambient_dim).astype("float32") * 0.5

    for c in range(n_classes):
        # Each class gets its own subspace basis, but with partial overlap:
        # 50% of basis vectors are shared across neighboring classes
        raw_basis = rng.randn(intrinsic_dim, ambient_dim).astype("float32")

        # Mix in basis vectors from neighboring classes for overlap
        (c + 1) % n_classes
        neighbor_basis = rng.randn(intrinsic_dim, ambient_dim).astype("float32")
        rng_state = rng.get_state()  # save state
        n_shared = intrinsic_dim // 3
        raw_basis[:n_shared] = 0.6 * raw_basis[:n_shared] + 0.4 * neighbor_basis[:n_shared]
        rng.set_state(rng_state)  # restore state

        # Orthonormalize
        Q, _ = np.linalg.qr(raw_basis.T)
        basis = Q.T[:intrinsic_dim]  # (intrinsic_dim, ambient_dim)

        # Latent coordinates — class-specific spread and structure
        spread = 1.5 + rng.rand() * 1.0
        z = rng.randn(per_class, intrinsic_dim).astype("float32") * spread

        # Multi-scale nonlinear warping for realistic manifold curvature
        z_warped = (
            z
            + 0.5 * np.sin(z * 1.2)
            + 0.2 * np.cos(z * 2.5 + c)
            + 0.1 * np.sin(z[:, :1] * z[:, 1:2])  # cross-dimension interaction (broadcast)
        )

        # Project to ambient space through class-specific basis
        X_c = z_warped @ basis + centers[c]

        # Substantial ambient noise (makes classification harder)
        X_c += rng.randn(per_class, ambient_dim).astype("float32") * 0.8

        X_all.append(X_c)
        y_all.append(np.full(per_class, c, dtype=np.int64))

    X_all = np.concatenate(X_all)
    y_all = np.concatenate(y_all)

    # Shuffle
    perm = rng.permutation(len(X_all))
    X_all, y_all = X_all[perm], y_all[perm]

    return (
        X_all[:n_train],
        y_all[:n_train],
        X_all[n_train:n_train + n_test],
        y_all[n_train:n_train + n_test],
    )


def main():
    parser = argparse.ArgumentParser(
        description="CIFAR-10: Manifold-Informed Architecture vs Standard"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tau", type=float, default=0.90,
                        help="Variance threshold for intrinsic dim")
    parser.add_argument("--discovery-samples", type=int, default=500,
                        help="Points to sample for dimensionality discovery")
    parser.add_argument("--k-pca", type=int, default=50,
                        help="Neighborhood size for local PCA")
    parser.add_argument("--plot", action="store_true", default=True)
    args = parser.parse_args()
    t_start = time.perf_counter()

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------

    try:
        print("\nLoading CIFAR-10...")
        (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
        # Flatten: (N, 32, 32, 3) → (N, 3072)
        X_train = X_train.reshape(-1, 3072).astype("float32")
        X_test = X_test.reshape(-1, 3072).astype("float32")
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        dataset_label = "CIFAR-10"
        is_synthetic = False
    except Exception as e:
        print(f"\nCIFAR-10 download unavailable ({e.__class__.__name__})")
        print("Generating synthetic CIFAR-10-like data (3072D, 10 classes, manifold structure)...")
        X_train, y_train, X_test, y_test = _generate_synthetic_cifar10()
        dataset_label = "Synthetic CIFAR-10-like"
        is_synthetic = True

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    input_dim = X_train.shape[1]
    n_classes = len(set(y_train))
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    if is_synthetic:
        print(f"  Classes: {n_classes}, Input dim: {input_dim}")
        print(f"  (Synthetic data with ~35D intrinsic manifold embedded in {input_dim}D)")
    else:
        print(f"  Classes: {n_classes} ({', '.join(CIFAR10_CLASSES)})")
        print(f"  Input dim: {input_dim} (32×32×3 color images)")

    # -----------------------------------------------------------------------
    # Phase 1: Discover intrinsic dimensionality
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 1: MANIFOLD DISCOVERY")
    print("=" * 70)

    print(f"\nSampling {args.discovery_samples} points, k={args.k_pca} neighbors...")
    print("(Using SVD-based local PCA — efficient for high-dimensional data)")
    t0 = time.perf_counter()
    dim_report = discover_dimensionality(
        X_train, n_samples=args.discovery_samples, k=args.k_pca,
        variance_thresholds=(0.95, 0.90, 0.85, 0.80),
    )
    discovery_time = time.perf_counter() - t0
    print(f"\nDiscovery time: {discovery_time:.1f}s\n")

    print(f"{'τ':>6} {'Mean d':>8} {'Std':>6} {'Min':>5} {'Max':>5} {'Noise %':>8}")
    print("-" * 45)
    for tau in sorted(dim_report.keys(), reverse=True):
        r = dim_report[tau]
        noise_pct = 100 * (1 - r["mean"] / input_dim)
        print(f"{tau:>6.2f} {r['mean']:>8.1f} {r['std']:>6.1f} {r['min']:>5} {r['max']:>5} {noise_pct:>7.1f}%")

    # Per-class dimensionality
    print(f"\nPer-class intrinsic dimensionality (τ={args.tau}):")
    class_dims = discover_per_class_dimensionality(
        X_train, y_train, k=args.k_pca, tau=args.tau, n_samples_per_class=50
    )
    for c in sorted(class_dims.keys()):
        cd = class_dims[c]
        label = CIFAR10_CLASSES[c] if c < len(CIFAR10_CLASSES) else str(c)
        print(f"  {label:>12}: d = {cd['mean']:.1f} ± {cd['std']:.1f}  [{cd['min']}, {cd['max']}]")

    # Use max of per-class maxima as the bottleneck — accommodates the hardest sample
    global_dim = int(round(dim_report[args.tau]["mean"]))
    intrinsic_dim = max(cd["max"] for cd in class_dims.values())
    # Clamp d to n_classes — need at least that many dims to separate classes.
    d = max(intrinsic_dim, n_classes)
    print(f"\n>> Global intrinsic dim (mean): {global_dim}  |  Max per-class max: {intrinsic_dim}")
    print(f"   Using d = {d} (max of local-PCA={intrinsic_dim}, n_classes={n_classes})")
    print(f"   d = {d / input_dim * 100:.1f}% of ambient dimensions")

    # -----------------------------------------------------------------------
    # Phase 2: Build architectures
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 2: ARCHITECTURE COMPARISON")
    print("=" * 70)

    # PCA projection (needed for intrinsic-dim and PCA models)
    from sklearn.decomposition import PCA as skPCA

    pca = skPCA(n_components=d)
    X_train_pca = pca.fit_transform(X_train).astype("float32")
    X_test_pca = pca.transform(X_test).astype("float32")
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA to {d}D captures {var_explained * 100:.1f}% of global variance")

    # Each entry: (build_fn, X_tr, X_te) — raw or PCA-projected as appropriate
    architectures = {
        "Standard (256→128)": (
            lambda: build_standard_model(input_dim, n_classes, lr=args.lr),
            X_train, X_test,
        ),
        f"Wide Manifold (4d→2d→d, d={d})": (
            lambda: build_wide_manifold_model(input_dim, n_classes, d, lr=args.lr),
            X_train, X_test,
        ),
        f"Manifold (2d→d, d={d})": (
            lambda: build_manifold_model(input_dim, n_classes, d, lr=args.lr),
            X_train, X_test,
        ),
        f"PCA→{d}D + MLP (2d→d)": (
            lambda: build_pca_model(n_classes, d, lr=args.lr),
            X_train_pca, X_test_pca,
        ),
        f"Intrinsic Dim (PCA→{d}D→output)": (
            lambda: build_pca_intrinsic_dim_model(n_classes, d, lr=args.lr),
            X_train_pca, X_test_pca,
        ),
    }

    # Show architecture details
    for name, (build_fn, _, _) in architectures.items():
        model = build_fn()
        n_params = count_params(model)
        print(f"\n{name}:")
        print(f"  Parameters: {n_params:,}")
        for layer in model.layers:
            if hasattr(layer, "units"):
                print(f"  {layer.name}: → {layer.units}")

    # -----------------------------------------------------------------------
    # Phase 3: Train and compare
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 3: TRAINING")
    print("=" * 70)

    all_results = {}

    for name, (build_fn, X_tr, X_te) in architectures.items():
        print(f"\n{name}")
        trial_results = []

        for trial in range(args.trials):
            np.random.seed(trial * 42)
            tf.random.set_seed(trial * 42)

            result = run_trial(
                build_fn, X_tr, y_train, X_te, y_test,
                epochs=args.epochs, batch_size=args.batch_size, trial=trial,
            )

            conv_str = f"conv@{result['convergence_epoch']}" if result["convergence_epoch"] is not None else "no conv"
            print(f"  Trial {trial + 1}/{args.trials}: "
                  f"acc={result['test_acc']:.4f}  "
                  f"loss={result['test_loss']:.4f}  "
                  f"{conv_str}  "
                  f"time={result['wall_time']:.1f}s")
            trial_results.append(result)

        all_results[name] = trial_results

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Dataset: {dataset_label} ({input_dim}D, {n_classes} classes)")
    print(f"Intrinsic dimensionality: d = {d} (local-PCA max: {intrinsic_dim}, global mean: {global_dim}, τ={args.tau})")
    print(f"Noise dimensions: {100 * (1 - d / input_dim):.1f}%")
    print(f"Epochs: {args.epochs}, Trials: {args.trials}")
    print(f"Device: {DEVICE_INFO['device_used']}")
    print("-" * 70)

    col_w = 22
    header = f"{'Metric':<25}"
    for name in all_results:
        short = name.split("(")[0].strip()
        header += f"{short:>{col_w}}"
    print(header)
    print("-" * 70)

    for label, key, fmt in [
        ("Test Accuracy", "test_acc", ".4f"),
        ("Test Loss", "test_loss", ".4f"),
        ("Parameters", "n_params", ",d"),
        ("Wall Time (s)", "wall_time", ".1f"),
    ]:
        row = f"{label:<25}"
        for name, results in all_results.items():
            vals = [r[key] for r in results]
            if fmt == ",d":
                row += f"{vals[0]:>{col_w},}"
            else:
                m, s = np.mean(vals), np.std(vals)
                row += f"  {m:{fmt}} ± {s:{fmt}}  "
        print(row)

    # Convergence
    row = f"{'Epochs to 40%':<25}"
    for name, results in all_results.items():
        convs = [r["convergence_epoch"] for r in results if r["convergence_epoch"] is not None]
        if convs:
            row += f"  {np.mean(convs):.1f} ± {np.std(convs):.1f} ({len(convs)}/{len(results)})  "
        else:
            row += f"{'N/A':>{col_w}}"
    print(row)

    # Parameter efficiency
    print("-" * 70)
    print("PARAMETER EFFICIENCY (accuracy per 1K parameters):")
    for name, results in all_results.items():
        mean_acc = np.mean([r["test_acc"] for r in results])
        n_params = results[0]["n_params"]
        eff = mean_acc / n_params * 1000
        print(f"  {name}: {eff:.4f} acc/Kparam  ({mean_acc:.4f} / {n_params:,})")

    # Winner
    print("-" * 70)
    best_name = max(all_results, key=lambda n: np.mean([r["test_acc"] for r in all_results[n]]))
    best_acc = np.mean([r["test_acc"] for r in all_results[best_name]])
    std_name = "Standard (256→128)"
    std_acc = np.mean([r["test_acc"] for r in all_results[std_name]])

    if best_name != std_name:
        delta = best_acc - std_acc
        print(f">> MANIFOLD-INFORMED WINS: {best_name}")
        print(f"   {best_acc:.4f} vs {std_acc:.4f} (standard)")
        print(f"   Delta: +{delta:.4f} ({delta * 100:.2f}%)")

        best_params = all_results[best_name][0]["n_params"]
        std_params = all_results[std_name][0]["n_params"]
        if best_params < std_params:
            reduction = 100 * (1 - best_params / std_params)
            print(f"   With {reduction:.0f}% FEWER parameters ({best_params:,} vs {std_params:,})")
        elif best_params > std_params:
            increase = 100 * (best_params / std_params - 1)
            print(f"   With {increase:.0f}% more parameters ({best_params:,} vs {std_params:,})")
    else:
        print(f">> Standard architecture wins: {std_acc:.4f}")

    print("=" * 70)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------

    save_data = {
        "device": DEVICE_INFO,
        "dataset": "cifar10" if not is_synthetic else "synthetic_cifar10",
        "input_dim": input_dim,
        "n_classes": n_classes,
        "class_names": CIFAR10_CLASSES,
        "intrinsic_dim": intrinsic_dim,
        "tau": args.tau,
        "epochs": args.epochs,
        "trials": args.trials,
        "dimensionality_report": {
            str(k): v for k, v in dim_report.items()
        },
        "per_class_dims": {
            str(k): {
                **v,
                "class_name": CIFAR10_CLASSES[k] if k < len(CIFAR10_CLASSES) else str(k),
            }
            for k, v in class_dims.items()
        },
        "results": {
            name: results for name, results in all_results.items()
        },
    }

    results_path = Path(__file__).resolve().parent / "cifar10_architecture_results.json"
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if args.plot:
        plot_path = str(Path(__file__).resolve().parent / "cifar10_architecture_results.png")
        plot_results(all_results, d, plot_path, elapsed=time.perf_counter() - t_start)


if __name__ == "__main__":
    main()
