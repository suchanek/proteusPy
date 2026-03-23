#!/usr/bin/env python3
"""
CIFAR-10 Benchmark: ManifoldModel — The Ultimate Test
=====================================================

The ManifoldModel on CIFAR-10: can pure geometry — zero learned parameters —
classify 3,072-dimensional color images?

This is the hardest test of the WaveRider thesis. CIFAR-10 has:
  - 3,072 ambient dimensions (32×32×3 pixels)
  - 10 visually complex classes (animals, vehicles)
  - High intra-class variability (many poses, backgrounds, lighting)
  - Significant inter-class overlap (cats vs dogs, trucks vs automobiles)

The ManifoldModel discovers the manifold structure via local PCA, builds a
knowledge graph with manifold-weighted edges, and classifies via graph-walk
+ manifold-projected voting. No neural network. No backpropagation. No
learned weights. Just geometry.

Because CIFAR-10 has 50K training samples in 3072D, we subsample for
ManifoldModel fitting (full O(n²·d) distance computation would be
prohibitive at 50K). We compare:
  - ManifoldModel on subsampled data
  - Euclidean KNN on the same subsample (apples-to-apples)
  - Euclidean KNN on full training data (upper bound)

Part of proteusPy, https://github.com/suchanek/proteusPy
Author: Eric G. Suchanek, PhD

Usage
-----
    python benchmarks/cifar10_manifold_model.py [--n-train 5000] [--tau 0.90]
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# TensorFlow setup (for dataset loading only)
# ---------------------------------------------------------------------------

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ---------------------------------------------------------------------------
# Bootstrap ManifoldModel imports
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_tnd_spec = importlib.util.spec_from_file_location(
    "proteusPy.turtleND",
    Path(__file__).resolve().parent.parent / "proteusPy" / "turtleND.py",
)
_tnd_mod = importlib.util.module_from_spec(_tnd_spec)
sys.modules["proteusPy.turtleND"] = _tnd_mod
_tnd_spec.loader.exec_module(_tnd_mod)

_gr_spec = importlib.util.spec_from_file_location(
    "proteusPy.graph_reasoner",
    Path(__file__).resolve().parent.parent / "proteusPy" / "graph_reasoner.py",
)
_gr_mod = importlib.util.module_from_spec(_gr_spec)
sys.modules["proteusPy.graph_reasoner"] = _gr_mod
_gr_spec.loader.exec_module(_gr_mod)

_mm_spec = importlib.util.spec_from_file_location(
    "proteusPy.manifold_model",
    Path(__file__).resolve().parent.parent / "proteusPy" / "manifold_model.py",
)
_mm_mod = importlib.util.module_from_spec(_mm_spec)
sys.modules["proteusPy.manifold_model"] = _mm_mod
_mm_spec.loader.exec_module(_mm_mod)

ManifoldModel = _mm_mod.ManifoldModel


# ---------------------------------------------------------------------------
# CIFAR-10 class names
# ---------------------------------------------------------------------------

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_cifar10():
    """Load CIFAR-10, falling back to synthetic if download fails."""
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        from tensorflow import keras
        print("Loading CIFAR-10...")
        (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
        X_train = X_train.reshape(-1, 3072).astype("float64")
        X_test = X_test.reshape(-1, 3072).astype("float64")
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        return X_train, y_train, X_test, y_test, False
    except Exception as e:
        print(f"CIFAR-10 download unavailable ({e})")
        print("Generating synthetic CIFAR-10-like data...")
        return *_generate_synthetic_cifar10(), True


def _generate_synthetic_cifar10(
    n_train=50000, n_test=10000, ambient_dim=3072, intrinsic_dim=35,
    n_classes=10, seed=42,
):
    """Generate synthetic data mimicking CIFAR-10's manifold structure.

    Designed to be realistically hard: classes share ~60% of their subspace,
    high ambient noise, and inter-class overlap. Target: ~50-60% KNN accuracy
    (similar to real CIFAR-10 with flat MLPs).
    """
    rng = np.random.RandomState(seed)
    X_all, y_all = [], []
    n_total = n_train + n_test
    per_class = n_total // n_classes

    # Small class centers — classes are NOT well-separated
    centers = rng.randn(n_classes, ambient_dim).astype("float64") * 0.1

    # Shared global basis — all classes project through overlapping subspaces
    global_basis = rng.randn(intrinsic_dim * 2, ambient_dim).astype("float64")

    for c in range(n_classes):
        # Each class uses a mix of shared + unique basis vectors
        # ~60% shared across all classes, ~40% class-specific
        n_shared = int(intrinsic_dim * 0.6)
        n_unique = intrinsic_dim - n_shared

        shared_part = global_basis[:n_shared]
        unique_part = rng.randn(n_unique, ambient_dim).astype("float64")
        raw_basis = np.vstack([shared_part, unique_part])

        # Orthonormalize
        Q, _ = np.linalg.qr(raw_basis.T)
        basis = Q.T[:intrinsic_dim]

        # Class-specific spread and structure
        spread = 2.0 + rng.rand() * 1.5
        z = rng.randn(per_class, intrinsic_dim).astype("float64") * spread

        # Nonlinear warping
        z_warped = (
            z + 0.3 * np.sin(z * 1.2) + 0.15 * np.cos(z * 2.5 + c)
        )

        # Project to ambient space
        X_c = z_warped @ basis + centers[c]

        # Ambient noise — calibrated for ~40-50% KNN accuracy
        # (similar to real CIFAR-10 with flat classifiers)
        X_c += rng.randn(per_class, ambient_dim).astype("float64") * 0.6

        X_all.append(X_c)
        y_all.append(np.full(per_class, c, dtype=np.int64))

    X_all = np.concatenate(X_all)
    y_all = np.concatenate(y_all)
    perm = rng.permutation(len(X_all))
    X_all, y_all = X_all[perm], y_all[perm]

    return X_all[:n_train], y_all[:n_train], X_all[n_train:n_train + n_test], y_all[n_train:n_train + n_test]


# ---------------------------------------------------------------------------
# Stratified subsample
# ---------------------------------------------------------------------------


def stratified_subsample(X, y, n_samples, seed=42):
    """Subsample maintaining class balance."""
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    per_class = n_samples // len(classes)
    indices = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        chosen = rng.choice(c_idx, size=min(per_class, len(c_idx)), replace=False)
        indices.extend(chosen)
    indices = np.array(indices)
    rng.shuffle(indices)
    return X[indices], y[indices]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="CIFAR-10: ManifoldModel — The Ultimate Test"
    )
    parser.add_argument("--n-train", type=int, default=2000,
                        help="Training subsample size for ManifoldModel (default 2000)")
    parser.add_argument("--n-test", type=int, default=1000,
                        help="Test subsample size (default 1000)")
    parser.add_argument("--tau", type=float, default=0.90,
                        help="Variance threshold for intrinsic dim")
    parser.add_argument("--k-graph", type=int, default=15,
                        help="Graph neighbors per node")
    parser.add_argument("--k-pca", type=int, default=50,
                        help="PCA neighborhood size")
    parser.add_argument("--k-vote", type=int, default=7,
                        help="Voting neighbors for prediction")
    parser.add_argument("--manifold-weight", type=float, default=0.8,
                        help="Manifold vs Euclidean distance blend")
    args = parser.parse_args()

    print("=" * 70)
    print("CIFAR-10: ManifoldModel — THE ULTIMATE TEST")
    print("Zero learned parameters. The manifold IS the model.")
    print("=" * 70)

    # Load data
    X_train_full, y_train_full, X_test_full, y_test_full, is_synthetic = load_cifar10()

    # Normalize
    scaler = StandardScaler()
    X_train_full = scaler.fit_transform(X_train_full)
    X_test_full = scaler.transform(X_test_full)

    # Subsample for ManifoldModel (O(n²·d) is prohibitive at 50K)
    X_train, y_train = stratified_subsample(
        X_train_full, y_train_full, args.n_train, seed=42
    )
    X_test, y_test = stratified_subsample(
        X_test_full, y_test_full, args.n_test, seed=99
    )

    input_dim = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    dataset_label = "Synthetic CIFAR-10-like" if is_synthetic else "CIFAR-10"

    print(f"\nDataset: {dataset_label}")
    print(f"  Full: {X_train_full.shape[0]} train, {X_test_full.shape[0]} test")
    print(f"  Subsample: {X_train.shape[0]} train, {X_test.shape[0]} test")
    print(f"  Dimensions: {input_dim}, Classes: {n_classes}")
    if not is_synthetic:
        print(f"  Classes: {', '.join(CIFAR10_CLASSES)}")

    # -----------------------------------------------------------------------
    # Experiment 1: ManifoldModel at multiple tau values
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPERIMENT: ManifoldModel vs Euclidean KNN")
    print("=" * 70)

    tau_values = [0.95, 0.90, 0.85]
    results = {}

    # Baseline: Euclidean KNN on same subsample
    print(f"\nEuclidean KNN (k={args.k_vote}) on subsample...")
    t0 = time.perf_counter()
    knn = KNeighborsClassifier(n_neighbors=args.k_vote, metric="euclidean")
    knn.fit(X_train, y_train)
    knn_acc = knn.score(X_test, y_test)
    knn_time = time.perf_counter() - t0
    results["Euclidean KNN (subsample)"] = {
        "accuracy": knn_acc,
        "time": knn_time,
        "params": 0,
    }
    print(f"  Accuracy: {knn_acc:.4f}  Time: {knn_time:.1f}s")

    # Baseline: Euclidean KNN on full data
    print(f"\nEuclidean KNN (k={args.k_vote}) on FULL training data...")
    t0 = time.perf_counter()
    knn_full = KNeighborsClassifier(n_neighbors=args.k_vote, metric="euclidean")
    knn_full.fit(X_train_full, y_train_full)
    knn_full_acc = knn_full.score(X_test, y_test)
    knn_full_time = time.perf_counter() - t0
    results["Euclidean KNN (full 50K)"] = {
        "accuracy": knn_full_acc,
        "time": knn_full_time,
        "params": 0,
    }
    print(f"  Accuracy: {knn_full_acc:.4f}  Time: {knn_full_time:.1f}s")

    # ManifoldModel at each tau
    for tau in tau_values:
        name = f"ManifoldModel (tau={tau})"
        print(f"\n{name}...")
        print(f"  Building: k_graph={args.k_graph}, k_pca={args.k_pca}, "
              f"k_vote={args.k_vote}, manifold_weight={args.manifold_weight}")

        model = ManifoldModel(
            k_graph=args.k_graph,
            k_pca=args.k_pca,
            k_vote=args.k_vote,
            variance_threshold=tau,
            manifold_weight=args.manifold_weight,
        )

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0
        print(f"  Fit time: {fit_time:.1f}s")

        # Geometry report
        summary = model.geometry_summary()
        print(f"  Nodes: {summary['n_nodes']}, Edges: {summary['n_edges']}")
        print(f"  Intrinsic dim: {summary['mean_intrinsic_dim']:.1f} ± "
              f"{summary['std_intrinsic_dim']:.1f} "
              f"[{summary['min_intrinsic_dim']}, {summary['max_intrinsic_dim']}]")
        noise_pct = 100 * (1 - summary['mean_intrinsic_dim'] / input_dim)
        print(f"  Noise dimensions: {noise_pct:.1f}%")

        # Predict
        print(f"  Predicting {len(X_test)} test samples...")
        t0 = time.perf_counter()
        preds = model.predict(X_test)
        pred_time = time.perf_counter() - t0
        acc = float(np.mean(preds == y_test))
        total_time = fit_time + pred_time
        print(f"  Predict time: {pred_time:.1f}s")
        print(f"  Accuracy: {acc:.4f}  Total time: {total_time:.1f}s")

        results[name] = {
            "accuracy": acc,
            "time": total_time,
            "fit_time": fit_time,
            "pred_time": pred_time,
            "params": 0,
            "geometry": summary,
        }

        # Per-class accuracy
        print("  Per-class accuracy:")
        for c in range(n_classes):
            mask = y_test == c
            if mask.sum() > 0:
                c_acc = float(np.mean(preds[mask] == y_test[mask]))
                label = CIFAR10_CLASSES[c] if c < len(CIFAR10_CLASSES) else str(c)
                print(f"    {label:>12}: {c_acc:.4f} ({mask.sum()} samples)")

    # -----------------------------------------------------------------------
    # Experiment 2: PCA → ManifoldModel pipeline
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: PCA → ManifoldModel Pipeline")
    print("Global PCA strips ambient noise, then ManifoldModel discovers")
    print("local manifold structure in the reduced space.")
    print("=" * 70)

    from sklearn.decomposition import PCA as skPCA

    # Try multiple PCA dimensions
    pca_dims = [30, 50, 100]

    for pca_d in pca_dims:
        print(f"\n--- PCA to {pca_d}D ---")
        pca = skPCA(n_components=pca_d)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        var_explained = pca.explained_variance_ratio_.sum()
        print(f"  Variance explained: {var_explained * 100:.1f}%")

        # KNN baseline on PCA data
        knn_pca = KNeighborsClassifier(n_neighbors=args.k_vote, metric="euclidean")
        knn_pca.fit(X_train_pca, y_train)
        knn_pca_acc = knn_pca.score(X_test_pca, y_test)
        pca_knn_name = f"PCA→{pca_d}D + KNN"
        results[pca_knn_name] = {
            "accuracy": knn_pca_acc, "time": 0, "params": 0,
        }
        print(f"  Euclidean KNN on PCA-{pca_d}D: {knn_pca_acc:.4f}")

        # ManifoldModel on PCA data
        for tau in [0.90, 0.85]:
            mm_pca_name = f"PCA→{pca_d}D + ManifoldModel (tau={tau})"
            print(f"  ManifoldModel (tau={tau}) on PCA-{pca_d}D...")

            mm_pca = ManifoldModel(
                k_graph=args.k_graph,
                k_pca=min(args.k_pca, len(X_train_pca) - 1),
                k_vote=args.k_vote,
                variance_threshold=tau,
                manifold_weight=args.manifold_weight,
            )

            t0 = time.perf_counter()
            mm_pca.fit(X_train_pca, y_train)
            fit_t = time.perf_counter() - t0

            summary = mm_pca.geometry_summary()
            print(f"    Intrinsic dim: {summary['mean_intrinsic_dim']:.1f}/{pca_d} "
                  f"({100 * (1 - summary['mean_intrinsic_dim']/pca_d):.0f}% noise)")

            t0 = time.perf_counter()
            preds = mm_pca.predict(X_test_pca)
            pred_t = time.perf_counter() - t0
            acc = float(np.mean(preds == y_test))
            print(f"    Accuracy: {acc:.4f} (fit={fit_t:.1f}s, pred={pred_t:.1f}s)")

            results[mm_pca_name] = {
                "accuracy": acc,
                "time": fit_t + pred_t,
                "params": 0,
                "geometry": summary,
            }

    # -----------------------------------------------------------------------
    # Fly demo (using PCA-reduced data for efficiency)
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FLY MODE DEMO: Navigating the CIFAR-10 manifold (PCA-50D)")
    print("=" * 70)

    pca_fly = skPCA(n_components=50)
    X_fly = pca_fly.fit_transform(X_train)

    fly_model = ManifoldModel(
        k_graph=args.k_graph, k_pca=args.k_pca, k_vote=args.k_vote,
        variance_threshold=0.90, manifold_weight=args.manifold_weight,
    )
    fly_model.fit(X_fly, y_train)

    start_class, end_class = 0, 9
    start_idx = np.where(y_train == start_class)[0][0]
    end_idx = np.where(y_train == end_class)[0][0]

    start_label = CIFAR10_CLASSES[start_class] if not is_synthetic else str(start_class)
    end_label = CIFAR10_CLASSES[end_class] if not is_synthetic else str(end_class)

    fly_model.fly_to(f"n{start_idx}")
    print(f"\nStarting at n{start_idx} ({start_label})")
    geom = fly_model.get_geometry(f"n{start_idx}")
    print(f"  Local intrinsic dim: {geom.intrinsic_dim}")

    print(f"\nFlying toward {end_label} (n{end_idx})...")
    path = fly_model.fly_toward(X_fly[end_idx], max_steps=20)
    for step, node_id in enumerate(path):
        idx = int(node_id[1:])
        node_geom = fly_model.get_geometry(node_id)
        dist = np.linalg.norm(X_fly[idx] - X_fly[end_idx])
        class_id = y_train[idx]
        label = CIFAR10_CLASSES[class_id] if class_id < len(CIFAR10_CLASSES) and not is_synthetic else str(class_id)
        print(f"  Step {step + 1:>2}: {node_id} ({label:>12}, "
              f"intrinsic_dim={node_geom.intrinsic_dim:>2}, "
              f"dist={dist:.2f})")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Dataset: {dataset_label} ({input_dim}D, {n_classes} classes)")
    print(f"Training subsample: {args.n_train}, Test: {args.n_test}")
    print(f"{'Method':<35} {'Accuracy':>10} {'Params':>10} {'Time':>10}")
    print("-" * 70)

    best_acc = max(v["accuracy"] for v in results.values())
    for name, r in results.items():
        marker = " << BEST" if r["accuracy"] == best_acc else ""
        print(f"{name:<35} {r['accuracy']:>10.4f} {r['params']:>10} {r['time']:>9.1f}s{marker}")

    # ManifoldModel vs KNN delta
    best_mm_name = max(
        (n for n in results if n.startswith("ManifoldModel")),
        key=lambda n: results[n]["accuracy"],
    )
    mm_acc = results[best_mm_name]["accuracy"]
    knn_sub_acc = results["Euclidean KNN (subsample)"]["accuracy"]
    delta = mm_acc - knn_sub_acc

    print("-" * 70)
    if delta > 0:
        print(f">> ManifoldModel BEATS Euclidean KNN (same data): "
              f"{mm_acc:.4f} vs {knn_sub_acc:.4f} (+{delta:.4f})")
    elif delta == 0:
        print(f">> ManifoldModel TIES Euclidean KNN: {mm_acc:.4f}")
    else:
        print(f">> Euclidean KNN leads by {-delta:.4f}: "
              f"{knn_sub_acc:.4f} vs {mm_acc:.4f}")

    # Geometry summary
    print("\nMANIFOLD GEOMETRY ACROSS tau VALUES:")
    for name, r in results.items():
        if "geometry" in r:
            g = r["geometry"]
            noise = 100 * (1 - g["mean_intrinsic_dim"] / input_dim)
            print(f"  {name}: d = {g['mean_intrinsic_dim']:.1f}/{input_dim} "
                  f"({noise:.1f}% noise)")

    print("=" * 70)

    # Save results
    save_data = {
        "dataset": "cifar10" if not is_synthetic else "synthetic_cifar10",
        "input_dim": input_dim,
        "n_classes": n_classes,
        "n_train_subsample": args.n_train,
        "n_test_subsample": args.n_test,
        "results": {},
    }
    for name, r in results.items():
        save_r = {k: v for k, v in r.items() if k != "geometry"}
        if "geometry" in r:
            save_r["geometry"] = r["geometry"]
        save_data["results"][name] = save_r

    results_path = "benchmarks/cifar10_manifold_model_results.json"
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
