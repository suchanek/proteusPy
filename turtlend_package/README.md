# TurtleND — The N-Dimensional Turtle & ManifoldModel

**The manifold IS the model.**

An N-dimensional turtle graphics system extended into a manifold-aware classifier.
No learned weights. No neural network. Just geometry.

## What's Here

```
turtlend_package/
├── src/                          # Core implementation (2,365 LOC)
│   ├── turtle3D.py               # 3D turtle with orient/roll/pitch/yaw (684 LOC)
│   ├── turtleND.py               # N-dimensional turtle via Givens rotations (403 LOC)
│   ├── manifold_walker.py        # Manifold-aware navigation with local PCA (484 LOC)
│   └── manifold_model.py         # ManifoldModel classifier (794 LOC)
│
├── tests/                        # Full test suite
│   ├── test_turtle3d.py          # Turtle3D unit tests
│   ├── test_turtleND.py          # TurtleND unit tests
│   ├── test_manifold_walker.py   # ManifoldWalker tests
│   └── test_manifold_model.py    # ManifoldModel tests
│
├── benchmarks/                   # Benchmark scripts & results
│   ├── mnist_manifold_model.py   # MNIST benchmark (the headline result)
│   ├── mnist_manifold_architecture.py
│   ├── cifar10_manifold_model.py
│   ├── cifar10_manifold_architecture.py
│   ├── digits_manifold_model.py
│   ├── digits_manifold_knn.py
│   ├── iris_manifold_adam_walker.py
│   ├── iris_adam_vs_manifold.py
│   └── results/                  # Saved benchmark results (JSON + PNG)
│       ├── mnist_manifold_model_results.json
│       ├── cifar10_manifold_model_results.json
│       ├── iris_benchmark_results.json
│       └── iris_benchmark_results.png
│
└── docs/                         # Documentation & writeups
    ├── the_turtle_who_learned_to_fly.md   # Narrative journey document
    └── manifold_walker_spec/              # Technical specification
        ├── manifold_walker_spec.md
        └── manifold_walker_spec.tex
```

## The Architecture Stack

```
Turtle3D          →  3D coordinate frame, orient/roll/pitch/yaw
    ↓
TurtleND          →  N-dimensional generalization via Givens rotations
    ↓
ManifoldWalker    →  Local PCA discovers tangent spaces, projects gradients
    ↓
ManifoldModel     →  "Explore then navigate" — the manifold IS the model
```

## Key Results

### MNIST (784D, 10 classes, real handwritten digits)

| Method | Accuracy | Params | Notes |
|--------|----------|--------|-------|
| Euclidean KNN (5K subsample) | 88.65% | 0 | Same training data |
| **ManifoldModel (tau=0.8)** | **89.65%** | **0** | **+1.0% over KNN** |
| ManifoldModel (tau=0.95) | 89.55% | 0 | Higher variance retained |
| Euclidean KNN (full 60K) | 93.75% | 0 | 12x more training data |

**ManifoldModel beats Euclidean KNN on the same data** — by discovering that MNIST's
784 dimensions are really ~14-dimensional manifolds (98.2% of dimensions are noise).

### Manifold Geometry Discovery (MNIST)

| tau | Mean Intrinsic Dim | Ambient Dim | Noise Dimensions |
|-----|-------------------|-------------|-----------------|
| 0.95 | 29.9 | 784 | 96.2% |
| 0.90 | 22.4 | 784 | 97.1% |
| 0.85 | 17.7 | 784 | 97.7% |
| 0.80 | 14.3 | 784 | 98.2% |

### Digits (64D, 10 classes, sklearn)

ManifoldModel also beats KNN on the smaller digits dataset.

## How ManifoldModel Works

1. **Explore phase**: Build a KNN graph over training data. At each node, compute
   local PCA to discover the tangent space (intrinsic dimensionality). Store the
   local basis and label.

2. **Navigate phase (prediction)**: For a test point, find its nearest graph node.
   Project the distance vector onto the local tangent basis. Compute manifold-projected
   distances. Classify by weighted KNN in the projected space.

The key insight: projecting distances onto locally-discovered tangent spaces
strips away noise dimensions, making the distance metric *faithful to the data manifold*.

## Dependencies

- numpy
- scipy
- scikit-learn (for benchmarks and PCA)
- matplotlib (optional, for benchmark plots)

## Running Tests

```bash
cd turtlend_package
python -m pytest tests/ -v
```

## Running Benchmarks

```bash
# Quick: sklearn digits (seconds)
python benchmarks/digits_manifold_model.py

# Full: MNIST (minutes, downloads data)
python benchmarks/mnist_manifold_model.py
```

## Origin

Developed in [proteusPy](https://github.com/suchanek/proteusPy) — a Python package
for protein structure analysis using turtle geometry. The turtle was born to build
disulfide bonds by walking dihedral angles. It grew into a general N-dimensional
navigator. Then it learned that the manifold *is* the model.

## The Story

See [docs/the_turtle_who_learned_to_fly.md](docs/the_turtle_who_learned_to_fly.md)
for the full narrative — from Logo turtles on a Commodore 64 to manifold classification
in 784 dimensions.
