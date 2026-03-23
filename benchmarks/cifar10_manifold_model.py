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
# TensorBoard logger
# ---------------------------------------------------------------------------


class TBLogger:
    """Thin tf.summary wrapper for benchmark logging.

    Falls back to a no-op when TensorFlow is unavailable so the script runs
    regardless of whether TF is installed.

    :param logdir: root directory for the TensorBoard event files
    """

    def __init__(self, logdir: str | None) -> None:
        self._writer = None
        if logdir is None or not _TF_AVAILABLE:
            return
        import tensorflow as tf
        run_dir = (
            Path(logdir)
            / time.strftime("%Y%m%d_%H%M%S")
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        self._writer = tf.summary.create_file_writer(str(run_dir))
        print(f"  TensorBoard logdir → {run_dir}")
        print("  Launch with:  tensorboard --logdir " + str(Path(logdir)))

    # ------------------------------------------------------------------
    def scalar(self, tag: str, value: float, step: int = 0) -> None:
        if self._writer is None:
            return
        import tensorflow as tf
        with self._writer.as_default():
            tf.summary.scalar(tag, float(value), step=step)

    def text(self, tag: str, value: str, step: int = 0) -> None:
        if self._writer is None:
            return
        import tensorflow as tf
        with self._writer.as_default():
            tf.summary.text(tag, value, step=step)

    def flush(self) -> None:
        if self._writer:
            self._writer.flush()

    def close(self) -> None:
        if self._writer:
            self._writer.close()
            self._writer = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def log_result(self, method_step: int, name: str, r: dict) -> None:
        """Log a single benchmark result entry.

        :param method_step: integer step index (one per method)
        :param name: method name (used as tag suffix)
        :param r: result dict with accuracy, time, optional geometry
        """
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        self.scalar(f"{safe}/accuracy", r["accuracy"], step=method_step)
        if r.get("time", 0) > 0:
            self.scalar(f"{safe}/time_total", r["time"], step=method_step)
        if "fit_time" in r:
            self.scalar(f"{safe}/time_fit",  r["fit_time"],  step=method_step)
            self.scalar(f"{safe}/time_pred", r["pred_time"], step=method_step)
        if "geometry" in r:
            g = r["geometry"]
            self.scalar(f"{safe}/geom/intrinsic_dim_mean", g["mean_intrinsic_dim"], step=method_step)
            self.scalar(f"{safe}/geom/intrinsic_dim_std",  g["std_intrinsic_dim"],  step=method_step)
            self.scalar(f"{safe}/geom/intrinsic_dim_min",  g["min_intrinsic_dim"],  step=method_step)
            self.scalar(f"{safe}/geom/intrinsic_dim_max",  g["max_intrinsic_dim"],  step=method_step)
            noise = 1.0 - g["mean_intrinsic_dim"] / g["ambient_dim"]
            self.scalar(f"{safe}/geom/noise_fraction", noise, step=method_step)

    def log_per_class(self, method_step: int, name: str,
                      class_names: list[str], accs: list[float]) -> None:
        """Log per-class accuracy scalars.

        :param method_step: step index for this method
        :param name: method name
        :param class_names: list of class label strings
        :param accs: per-class accuracy values (same order)
        """
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        for lbl, acc in zip(class_names, accs):
            self.scalar(f"{safe}/class/{lbl}", acc, step=method_step)

    def log_fly_step(self, step: int, dist: float, intrinsic_dim: int) -> None:
        """Log one step of a fly_toward trajectory.

        :param step: trajectory step index
        :param dist: Euclidean distance to goal
        :param intrinsic_dim: local intrinsic dimension at this node
        """
        self.scalar("fly/distance_to_goal", dist,          step=step)
        self.scalar("fly/intrinsic_dim",    intrinsic_dim, step=step)

    def log_summary_text(self, dataset_label: str, results: dict) -> None:
        """Write a markdown results table as a TensorBoard text entry.

        :param dataset_label: dataset name for the title
        :param results: full results dict
        """
        lines = [
            f"## {dataset_label} — ManifoldModel Benchmark\n",
            "| Method | Accuracy | Time (s) |",
            "| --- | ---: | ---: |",
        ]
        for name, r in results.items():
            lines.append(
                f"| {name} | {r['accuracy']:.4f} | {r.get('time', 0):.2f} |"
            )
        self.text("summary/results_table", "\n".join(lines), step=0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(results: dict, dataset_label: str = "CIFAR-10",
                 save_path: str | None = None) -> None:
    """Plot benchmark results: accuracy bar chart + accuracy-vs-time scatter.

    :param results: dict mapping method name → result dict (accuracy, time, …)
    :param dataset_label: title label for the dataset
    :param save_path: if given, save figure to this path instead of showing
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # --- colour palette by method family ------------------------------------
    def _color(name: str) -> str:
        if name.startswith("PCA") and "ManifoldModel" in name:
            return "#2196F3"   # blue
        if name.startswith("ManifoldModel"):
            return "#4CAF50"   # green
        if name.startswith("PCA") and "KNN" in name:
            return "#FF9800"   # orange
        return "#9E9E9E"       # grey  (plain KNN baselines)

    names = list(results.keys())
    accs  = [results[n]["accuracy"] for n in names]
    times = [results[n].get("time", 0) for n in names]
    colors = [_color(n) for n in names]

    # Short display labels (keep τ info, drop redundant prefix)
    def _label(n: str) -> str:
        n = n.replace("ManifoldModel", "MM")
        n = n.replace("Euclidean ", "")
        n = n.replace("(subsample)", "(sub)")
        n = n.replace("(full 50K)", "(50K)")
        return n

    labels = [_label(n) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"ManifoldModel Benchmark — {dataset_label}\n"
        "Zero learned parameters  |  Pure geometry",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # ── Panel 1: horizontal accuracy bars ──────────────────────────────────
    ax = axes[0]
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, accs, color=colors, edgecolor="white", linewidth=0.6,
                   height=0.72)

    # Annotate values
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{acc:.3f}", va="center", ha="left", fontsize=8.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Test Accuracy", fontsize=10)
    ax.set_title("Accuracy by Method", fontsize=11)
    ax.set_xlim(0, max(accs) * 1.18)
    ax.axvline(x=max(accs), color="#333", linestyle="--", linewidth=0.8,
               alpha=0.5, label=f"Best: {max(accs):.3f}")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="lower right")

    # Legend patches
    legend_items = [
        mpatches.Patch(color="#4CAF50", label="ManifoldModel (raw)"),
        mpatches.Patch(color="#2196F3", label="PCA + ManifoldModel"),
        mpatches.Patch(color="#FF9800", label="PCA + KNN"),
        mpatches.Patch(color="#9E9E9E", label="KNN baseline"),
    ]
    ax.legend(handles=legend_items, fontsize=8, loc="lower right")

    # ── Panel 2: accuracy vs log(time) scatter ──────────────────────────────
    ax2 = axes[1]
    for _, acc, t, c, lbl in zip(names, accs, times, colors, labels):
        t_plot = max(t, 1e-3)   # avoid log(0)
        ax2.scatter(t_plot, acc, color=c, s=90, zorder=3, edgecolors="white",
                    linewidths=0.6)
        ax2.annotate(lbl, (t_plot, acc), fontsize=7.5,
                     textcoords="offset points", xytext=(6, 2),
                     color="#333")

    ax2.set_xscale("log")
    ax2.set_xlabel("Total Time  (s, log scale)", fontsize=10)
    ax2.set_ylabel("Test Accuracy", fontsize=10)
    ax2.set_title("Accuracy vs. Time", fontsize=11)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(True, which="both", linestyle="--", alpha=0.35)

    # Add legend to scatter too
    ax2.legend(handles=legend_items, fontsize=8, loc="lower right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

# ---------------------------------------------------------------------------
# TensorFlow setup (for dataset loading only)
# ---------------------------------------------------------------------------

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

_TF_AVAILABLE = False
DEVICE_INFO: dict = {"device_used": "N/A"}

try:
    import tensorflow as _tf

    _gpus = _tf.config.list_physical_devices("GPU")
    for _gpu in _gpus:
        try:
            _tf.config.experimental.set_memory_growth(_gpu, True)
        except RuntimeError:
            pass

    # On macOS with tensorflow-metal, Metal GPUs appear as GPU-type devices.
    # The string "METAL" does NOT appear in device names or types — correct
    # detection is: macOS platform + at least one GPU present.
    _metal = sys.platform == "darwin" and len(_gpus) > 0

    DEVICE_INFO = {
        "tensorflow_version": _tf.__version__,
        "device_used": "Metal GPU" if _metal else ("GPU" if _gpus else "CPU"),
        "physical_devices": [d.name for d in _tf.config.list_physical_devices()],
    }
    print(f"TensorFlow {_tf.__version__} | Device: {DEVICE_INFO['device_used']}")
    _TF_AVAILABLE = True

except ImportError:
    pass

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
    if not _TF_AVAILABLE:
        print("TensorFlow not available — using synthetic CIFAR-10-like data.")
        return *_generate_synthetic_cifar10(), True
    try:
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
    parser.add_argument("--plot", action="store_true",
                        help="Show results plot after benchmarking")
    parser.add_argument("--plot-file", type=str, default=None,
                        help="Save plot to this path (PNG/PDF/SVG) instead of showing")
    parser.add_argument("--plot-only", type=str, default=None,
                        help="Load results from JSON and plot, skip benchmarking")
    parser.add_argument("--tb-logdir", type=str, default="runs/cifar10_manifold",
                        help="TensorBoard log root (default: runs/cifar10_manifold)")
    parser.add_argument("--no-tb", action="store_true",
                        help="Disable TensorBoard logging")
    args = parser.parse_args()

    # --plot-only: load existing JSON and plot without running benchmarks
    if args.plot_only:
        with open(args.plot_only) as f:
            saved = json.load(f)
        plot_results(
            saved["results"],
            dataset_label=saved.get("dataset", "CIFAR-10"),
            save_path=args.plot_file,
        )
        return

    W = 72

    def _hdr(title: str) -> None:
        print(f"\n{'─' * W}")
        print(f"  {title}")
        print(f"{'─' * W}")

    def _row(label: str, value: str, indent: int = 2) -> None:
        pad = " " * indent
        print(f"{pad}{label:<28} {value}")

    _hdr("CIFAR-10  |  ManifoldModel Benchmark  |  Zero learned parameters")

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

    _hdr("Dataset")
    _row("Name", dataset_label)
    _row("Full train / test",
         f"{X_train_full.shape[0]:,} / {X_test_full.shape[0]:,}")
    _row("Subsample train / test",
         f"{X_train.shape[0]:,} / {X_test.shape[0]:,}")
    _row("Dimensions", str(input_dim))
    _row("Classes", str(n_classes))
    if not is_synthetic:
        _row("Class names", ", ".join(CIFAR10_CLASSES))

    tb = TBLogger(None if args.no_tb else args.tb_logdir)
    method_step = 0   # incremented after each method is evaluated

    # -----------------------------------------------------------------------
    # Experiment 1: ManifoldModel at multiple tau values
    # -----------------------------------------------------------------------

    _hdr("Experiment 1  |  ManifoldModel vs Euclidean KNN")

    tau_values = [0.95, 0.90, 0.85]
    results = {}

    # Baseline: Euclidean KNN on same subsample
    print(f"\n  KNN (k={args.k_vote}) — subsample")
    t0 = time.perf_counter()
    knn = KNeighborsClassifier(n_neighbors=args.k_vote, metric="euclidean")
    knn.fit(X_train, y_train)
    knn_acc = knn.score(X_test, y_test)
    knn_time = time.perf_counter() - t0
    results["Euclidean KNN (subsample)"] = {
        "accuracy": knn_acc, "time": knn_time, "params": 0,
    }
    tb.log_result(method_step, "Euclidean KNN (subsample)", results["Euclidean KNN (subsample)"])
    method_step += 1
    _row("accuracy", f"{knn_acc:.4f}", indent=4)
    _row("time", f"{knn_time:.2f}s", indent=4)

    # Baseline: Euclidean KNN on full data
    print(f"\n  KNN (k={args.k_vote}) — full 50K")
    t0 = time.perf_counter()
    knn_full = KNeighborsClassifier(n_neighbors=args.k_vote, metric="euclidean")
    knn_full.fit(X_train_full, y_train_full)
    knn_full_acc = knn_full.score(X_test, y_test)
    knn_full_time = time.perf_counter() - t0
    results["Euclidean KNN (full 50K)"] = {
        "accuracy": knn_full_acc, "time": knn_full_time, "params": 0,
    }
    tb.log_result(method_step, "Euclidean KNN (full 50K)", results["Euclidean KNN (full 50K)"])
    method_step += 1
    _row("accuracy", f"{knn_full_acc:.4f}", indent=4)
    _row("time", f"{knn_full_time:.2f}s", indent=4)

    # ManifoldModel at each tau
    for tau in tau_values:
        name = f"ManifoldModel (tau={tau})"
        print(f"\n  {name}")
        _row("k_graph / k_pca / k_vote",
             f"{args.k_graph} / {args.k_pca} / {args.k_vote}", indent=4)
        _row("manifold_weight", str(args.manifold_weight), indent=4)

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
        _row("fit time", f"{fit_time:.2f}s", indent=4)

        summary = model.geometry_summary()
        _row("nodes / edges",
             f"{summary['n_nodes']:,} / {summary['n_edges']:,}", indent=4)
        _row("intrinsic dim",
             f"{summary['mean_intrinsic_dim']:.1f} ± {summary['std_intrinsic_dim']:.1f}"
             f"  [{summary['min_intrinsic_dim']}, {summary['max_intrinsic_dim']}]",
             indent=4)
        noise_pct = 100 * (1 - summary["mean_intrinsic_dim"] / input_dim)
        _row("noise dims", f"{noise_pct:.1f}%", indent=4)

        t0 = time.perf_counter()
        preds = model.predict(X_test)
        pred_time = time.perf_counter() - t0
        acc = float(np.mean(preds == y_test))
        total_time = fit_time + pred_time
        _row("predict time", f"{pred_time:.2f}s", indent=4)
        _row("accuracy", f"{acc:.4f}", indent=4)
        _row("total time", f"{total_time:.2f}s", indent=4)

        results[name] = {
            "accuracy": acc,
            "time": total_time,
            "fit_time": fit_time,
            "pred_time": pred_time,
            "params": 0,
            "geometry": summary,
        }
        tb.log_result(method_step, name, results[name])

        print("    per-class accuracy:")
        class_names_log, class_accs_log = [], []
        for c in range(n_classes):
            mask = y_test == c
            if mask.sum() > 0:
                c_acc = float(np.mean(preds[mask] == y_test[mask]))
                lbl = CIFAR10_CLASSES[c] if c < len(CIFAR10_CLASSES) else str(c)
                print(f"      {lbl:>12}  {c_acc:.4f}  (n={mask.sum()})")
                class_names_log.append(lbl)
                class_accs_log.append(c_acc)
        tb.log_per_class(method_step, name, class_names_log, class_accs_log)
        method_step += 1

    # -----------------------------------------------------------------------
    # Experiment 2: PCA → ManifoldModel pipeline
    # -----------------------------------------------------------------------

    _hdr("Experiment 2  |  PCA → ManifoldModel pipeline")

    from sklearn.decomposition import PCA as skPCA

    pca_dims = [30, 50, 100]

    for pca_d in pca_dims:
        print(f"\n  PCA → {pca_d}D")
        pca = skPCA(n_components=pca_d)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        var_explained = pca.explained_variance_ratio_.sum()
        _row("variance explained", f"{var_explained * 100:.1f}%", indent=4)

        knn_pca = KNeighborsClassifier(n_neighbors=args.k_vote, metric="euclidean")
        knn_pca.fit(X_train_pca, y_train)
        knn_pca_acc = knn_pca.score(X_test_pca, y_test)
        pca_knn_name = f"PCA\u2192{pca_d}D + KNN"
        results[pca_knn_name] = {"accuracy": knn_pca_acc, "time": 0, "params": 0}
        tb.log_result(method_step, pca_knn_name, results[pca_knn_name])
        method_step += 1
        _row("KNN accuracy", f"{knn_pca_acc:.4f}", indent=4)

        for tau in [0.90, 0.85]:
            mm_pca_name = f"PCA\u2192{pca_d}D + ManifoldModel (tau={tau})"
            print(f"    ManifoldModel  tau={tau}")

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
            noise_pct = 100 * (1 - summary["mean_intrinsic_dim"] / pca_d)
            _row("intrinsic dim",
                 f"{summary['mean_intrinsic_dim']:.1f}/{pca_d}  ({noise_pct:.0f}% noise)",
                 indent=6)

            t0 = time.perf_counter()
            preds = mm_pca.predict(X_test_pca)
            pred_t = time.perf_counter() - t0
            acc = float(np.mean(preds == y_test))
            _row("accuracy", f"{acc:.4f}  (fit={fit_t:.2f}s  pred={pred_t:.2f}s)",
                 indent=6)

            results[mm_pca_name] = {
                "accuracy": acc,
                "time": fit_t + pred_t,
                "fit_time": fit_t,
                "pred_time": pred_t,
                "params": 0,
                "geometry": summary,
            }
            tb.log_result(method_step, mm_pca_name, results[mm_pca_name])
            method_step += 1

    # -----------------------------------------------------------------------
    # Fly demo (using PCA-reduced data for efficiency)
    # -----------------------------------------------------------------------

    _hdr("Fly-mode demo  |  Navigating the CIFAR-10 manifold (PCA-50D)")

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
    end_label   = CIFAR10_CLASSES[end_class]   if not is_synthetic else str(end_class)

    fly_model.fly_to(f"n{start_idx}")
    geom = fly_model.get_geometry(f"n{start_idx}")
    print(f"\n  Start  n{start_idx} ({start_label})  intrinsic_dim={geom.intrinsic_dim}")
    print(f"  Goal   n{end_idx}   ({end_label})")
    print(f"\n  {'Step':>4}  {'node':<8}  {'class':>12}  {'d_int':>5}  {'dist':>8}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*12}  {'─'*5}  {'─'*8}")

    path = fly_model.fly_toward(X_fly[end_idx], max_steps=20)
    for step, node_id in enumerate(path):
        idx = int(node_id[1:])
        node_geom = fly_model.get_geometry(node_id)
        dist = np.linalg.norm(X_fly[idx] - X_fly[end_idx])
        class_id = y_train[idx]
        lbl = (CIFAR10_CLASSES[class_id]
               if class_id < len(CIFAR10_CLASSES) and not is_synthetic
               else str(class_id))
        print(f"  {step + 1:>4}  {node_id:<8}  {lbl:>12}  "
              f"{node_geom.intrinsic_dim:>5}  {dist:>8.3f}")
        tb.log_fly_step(step, float(dist), node_geom.intrinsic_dim)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    _hdr("Results summary")
    print(f"\n  Dataset : {dataset_label}  ({input_dim}D, {n_classes} classes)")
    print(f"  Train   : {args.n_train}   Test: {args.n_test}\n")

    col_w = (38, 10, 10)
    hdr_fmt  = f"  {{:<{col_w[0]}}} {{:>{col_w[1]}}} {{:>{col_w[2]}}}"
    row_fmt  = f"  {{:<{col_w[0]}}} {{:>{col_w[1]}.4f}} {{:>{col_w[2]}.2f}}s"
    sep = "  " + "─" * (sum(col_w) + 4)

    print(hdr_fmt.format("Method", "Accuracy", "Time"))
    print(sep)

    best_acc = max(v["accuracy"] for v in results.values())
    for name, r in results.items():
        marker = "  *" if r["accuracy"] == best_acc else ""
        print(row_fmt.format(name, r["accuracy"], r["time"]) + marker)

    print(sep)

    best_mm_name = max(
        (n for n in results if n.startswith("ManifoldModel")),
        key=lambda n: results[n]["accuracy"],
    )
    mm_acc      = results[best_mm_name]["accuracy"]
    knn_sub_acc = results["Euclidean KNN (subsample)"]["accuracy"]
    delta       = mm_acc - knn_sub_acc

    if delta > 0:
        print(f"\n  ManifoldModel beats KNN (same data):  "
              f"{mm_acc:.4f} vs {knn_sub_acc:.4f}  (+{delta:.4f})")
    elif delta == 0:
        print(f"\n  ManifoldModel ties KNN:  {mm_acc:.4f}")
    else:
        print(f"\n  KNN leads ManifoldModel:  "
              f"{knn_sub_acc:.4f} vs {mm_acc:.4f}  ({delta:+.4f})")

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

    tb.log_summary_text(dataset_label, results)
    tb.flush()
    tb.close()

    if args.plot or args.plot_file:
        plot_results(
            results,
            dataset_label=dataset_label,
            save_path=args.plot_file,
        )


if __name__ == "__main__":
    main()
