#!/usr/bin/env python3
"""
Digits Benchmark: ManifoldModel vs Standard KNN
================================================

Tests the new ManifoldModel architecture — where the manifold IS the model —
against standard Euclidean KNN on the sklearn digits dataset.

The ManifoldModel:
  1. Explores the data manifold (fit) — discovers local geometry, builds graph
  2. Navigates the graph (predict) — classifies via manifold-aware graph walk
  3. Can fly the embedded space — interactive turtle navigation

Part of proteusPy, https://github.com/suchanek/proteusPy
Author: Eric G. Suchanek, PhD
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Bootstrap imports without heavy proteusPy dependencies
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


def main():
    print("=" * 70)
    print("DIGITS BENCHMARK: ManifoldModel vs Standard KNN")
    print("The manifold IS the model. No weights. Just geometry.")
    print("=" * 70)

    # Load data
    data = load_digits()
    X, y = data.data.astype("float64"), data.target
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} dims, {len(set(y))} classes")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Cross-validation
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    methods = {
        "Euclidean KNN (k=7)": lambda: KNeighborsClassifier(n_neighbors=7, metric="euclidean"),
        "ManifoldModel (tau=0.95)": lambda: ManifoldModel(
            k_graph=15, k_pca=50, k_vote=7, variance_threshold=0.95
        ),
        "ManifoldModel (tau=0.90)": lambda: ManifoldModel(
            k_graph=15, k_pca=50, k_vote=7, variance_threshold=0.90
        ),
        "ManifoldModel (tau=0.85)": lambda: ManifoldModel(
            k_graph=15, k_pca=50, k_vote=7, variance_threshold=0.85
        ),
        "ManifoldModel (tau=0.80)": lambda: ManifoldModel(
            k_graph=15, k_pca=50, k_vote=7, variance_threshold=0.80
        ),
    }

    results = {}
    geom_reports = {}

    for name, make_clf in methods.items():
        fold_accs = []
        fold_times = []
        all_dims = []

        print(f"\n{name}...")
        for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            clf = make_clf()
            t0 = time.perf_counter()
            clf.fit(X_tr, y_tr)
            acc = clf.score(X_te, y_te)
            elapsed = time.perf_counter() - t0

            fold_accs.append(acc)
            fold_times.append(elapsed)

            if hasattr(clf, "intrinsic_dim") and clf.intrinsic_dim is not None:
                all_dims.append(clf.intrinsic_dim)

            print(f"  Fold {fold_i + 1}: {acc:.4f} ({elapsed:.2f}s)")

        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        mean_time = np.mean(fold_times)
        results[name] = (mean_acc, std_acc, mean_time)

        if all_dims:
            geom_reports[name] = float(np.mean(all_dims))

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<35} {'Accuracy':>12} {'Time':>10}")
    print("-" * 70)

    best_acc = max(v[0] for v in results.values())
    for name, (mean_acc, std_acc, mean_time) in results.items():
        marker = " << BEST" if mean_acc == best_acc else ""
        print(f"{name:<35} {mean_acc:.4f} +/- {std_acc:.4f} {mean_time:>8.2f}s{marker}")

    if geom_reports:
        print("\n" + "-" * 70)
        print("MANIFOLD GEOMETRY")
        print("-" * 70)
        for name, mean_d in geom_reports.items():
            noise_pct = 100 * (1 - mean_d / X.shape[1])
            print(f"  {name}: intrinsic dim = {mean_d:.1f}/{X.shape[1]} ({noise_pct:.0f}% noise)")

    # Fly demo
    print("\n" + "=" * 70)
    print("FLY MODE DEMO")
    print("=" * 70)

    model = ManifoldModel(k_graph=15, k_pca=50, k_vote=7, variance_threshold=0.90)
    model.fit(X, y)

    # Start at digit 0, fly toward digit 9
    start_idx = np.where(y == 0)[0][0]
    target_idx = np.where(y == 9)[0][0]

    model.fly_to(f"n{start_idx}")
    print(f"\nStarting at node n{start_idx} (digit {y[start_idx]})")

    geom = model.get_geometry(f"n{start_idx}")
    print(f"  Local intrinsic dim: {geom.intrinsic_dim}")

    path = model.fly_toward(X[target_idx], max_steps=15)
    print(f"\nFlying toward digit {y[target_idx]}...")
    for step, node_id in enumerate(path):
        idx = int(node_id[1:])  # strip 'n'
        node_geom = model.get_geometry(node_id)
        dist = np.linalg.norm(X[idx] - X[target_idx])
        print(
            f"  Step {step + 1}: {node_id} (digit {y[idx]}, "
            f"intrinsic_dim={node_geom.intrinsic_dim}, "
            f"dist_to_target={dist:.2f})"
        )

    summary = model.geometry_summary()
    print(f"\nModel summary: {summary['n_nodes']} nodes, {summary['n_edges']} edges")
    print(f"  Mean intrinsic dim: {summary['mean_intrinsic_dim']:.1f} / {summary['ambient_dim']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
