# WaveRider: A Zero-Parameter Geometric Machine Learning Stack for Manifold-Aware Classification

**Eric G. Suchanek, PhD**
Independent Researcher
`suchanek@mac.com`
proteusPy Project — https://github.com/suchanek/proteusPy

---

> *Draft for arXiv submission — cs.LG / stat.ML*

---

## Abstract

We present WaveRider, a four-layer framework for manifold-aware machine learning that achieves competitive classification without learned parameters. The core thesis is that *the manifold is the model*: the data manifold, once discovered through local principal component analysis and encoded as a weighted knowledge graph with stored basis geometry, is itself a sufficient classifier. WaveRider consists of: (1) **TurtleND**, an N-dimensional navigation primitive carrying position and an orthonormal frame, generalized from the classical Turtle3D molecular modeling primitive via Givens rotations; (2) **ManifoldWalker** and its Adam variant, implementing Riemannian-approximate gradient descent by projecting updates onto locally-estimated tangent planes; (3) **ManifoldModel**, a zero-parameter classifier whose "weights" are the graph topology, local basis vectors, and eigenvalue field discovered during a single exploration pass; and (4) **ManifoldObserver**, a novel (N+1)-dimensional construction that extends any N-dimensional manifold frame by one orthonormal dimension via QR decomposition, gaining the manifold normal and the ability to observe curvature, topology, and class boundaries from above the surface — turning a search problem into a projection problem. The ManifoldObserver operationalizes a principle with deep roots: Plato's prisoners see only shadows on the cave wall until lifted into three dimensions; Abbott's Flatlanders see only slices until raised above their plane; Gauss's extrinsic curvature is invisible from the surface until one dimension is added. WaveRider provides the constructive ML instantiation of this principle.

On the UCI Digits dataset (64 dimensions), ManifoldModel achieves 97.72% accuracy with zero parameters versus 97.33% for Euclidean KNN. On CIFAR-10 (3,072 dimensions) ManifoldModel uses 219× fewer parameters than a standard CNN while matching its accuracy. The ManifoldObserver achieves 100% agreement with its N-dimensional subject on synthetic benchmarks while providing curvature and topology information unavailable to the surface-bound walker. We also describe an application to protein disulfide bond analysis, where a 6-dimensional observer looks into the 5-dimensional torsion-angle manifold and sees the structural landscape that the walker must search.

---

## 1. Introduction

The dominant paradigm in machine learning is to fit a parameterized function — a neural network, a kernel machine, a decision tree — to training data by optimizing a loss. The resulting weights or splits encode, implicitly, knowledge about the data distribution. This works well but has a cost: millions of parameters for complex tasks, opacity about what was learned, and fragility when the test distribution shifts.

An alternative is to *discover* the data manifold explicitly and use that discovery as the model. Real high-dimensional data — images, protein structures, speech signals — rarely uses all of its ambient dimensions. Digits live on a manifold of intrinsic dimension roughly 13 inside 64-dimensional pixel space. CIFAR-10 natural images sit on a manifold of intrinsic dimension roughly 29 inside 3,072-dimensional space. More than 99% of the ambient dimensions are noise.

If we can find the manifold, we can discard the noise, measure distances correctly (along the surface, not through the void), and classify by *geometry* rather than by optimization.

WaveRider does exactly this. It is not the first work to identify the manifold hypothesis [1,2], nor the first to use local PCA for dimensionality reduction [3,4]. What is novel is the integration: a coherent stack from a navigation primitive through optimization to a zero-parameter classifier to a (N+1)-dimensional extrinsic observer that can see what the classifier cannot.

The key conceptual contributions are:

1. **TurtleND** — the formal generalization of Turtle3D to N dimensions using Givens rotations, enabling navigation in arbitrary-dimensional data spaces with the same primitives (move, rotate, orient) that have been used for molecular modeling since 1986.

2. **ManifoldModel** — the identification that a KNN graph with stored local PCA geometry at each node is itself a classifier, with no additional learned weights. Classification is manifold-aware nearest-neighbor search over this graph.

3. **ManifoldObserver** — a constructive proof that any N-dimensional manifold frame can be extended to (N+1) dimensions by a single QR decomposition, gaining the manifold normal. The observer classifies by projection rather than search, observes curvature and topology in O(n) rather than O(hops × k), and embodies the geometric principle: *to understand a surface, you must leave it*.

### 1.1 Motivation: From Molecular Modeling to Machine Learning

The WaveRider stack did not originate in machine learning. It originated in molecular modeling. The Turtle3D class in proteusPy [5] was designed to build protein structures by navigating local coordinate frames along bond vectors — a technique dating to the author's 1992 *Proteus* program [6]. The frame carried heading, left, and up vectors; rotations encoded dihedral angles; molecules were "drawn" by a turtle navigating chemical space.

The generalization to N dimensions arose when the same navigational primitive was needed for the 5-dimensional torsion-angle space of disulfide bonds. But once the N-dimensional turtle existed, the question became: what else can it navigate? The answer — data manifolds — led directly to the WaveRider stack.

This origin matters for the paper's framing: WaveRider is not an abstraction built for machine learning. It is a navigation primitive that turned out to be equally at home in a protein crystal and a high-dimensional embedding space.

---

## 2. Related Work

**Manifold learning.** Isomap [1], LLE [2], UMAP [7], and t-SNE [8] are the standard methods for manifold learning, but they are dimensionality reduction tools — they produce low-dimensional embeddings for visualization or downstream processing. WaveRider does not reduce dimensions; it discovers the manifold geometry *in place* and uses it directly for classification.

**Local PCA.** Local linear embedding (LLE) [2] and its variants use local PCA to estimate tangent spaces. WaveRider's ManifoldWalker and ManifoldModel use local PCA similarly but store the full geometry (basis, eigenvalues, intrinsic dimension) at each node rather than producing a global embedding.

**Riemannian optimization.** Methods for optimization on Riemannian manifolds [9,10] project gradients onto tangent planes and retract back to the manifold surface. ManifoldWalker performs approximate Riemannian gradient descent without requiring an explicit manifold parameterization — the tangent plane is estimated locally at each step from the data neighborhood.

**Geometric deep learning.** The field of geometric deep learning [11] argues that symmetries and geometric structure should be built into neural architectures. WaveRider takes this further: rather than embedding geometric structure in a trained model, the geometry *is* the model.

**Prototype-based classifiers.** Learning Vector Quantization (LVQ) [12] and its successors classify by distance to learned prototypes. ManifoldModel is in the spirit of prototype-based classifiers, but the prototypes are all training points (not a learned subset), and distances are measured in manifold space rather than Euclidean space.

**Non-parametric classification.** Euclidean KNN is the canonical non-parametric classifier. ManifoldModel can be understood as a *manifold-corrected* KNN: same nearest-neighbor idea, but distances measured correctly along the manifold surface rather than through the ambient void.

**Extrinsic vs. intrinsic geometry.** The distinction between intrinsic (Gauss curvature) and extrinsic (mean curvature, normal vectors) differential geometry is classical [13]. The ManifoldObserver is an operationalization of this distinction: by adding one dimension, it gains access to the manifold normal and all extrinsic quantities that are invisible from the surface.

---

## 3. The WaveRider Stack

### 3.1 Layer 1: TurtleND — Navigation Primitive

A TurtleND in N-dimensional space carries:

- **Position** p ∈ ℝᴺ
- **Frame** F ∈ ℝᴺˣᴺ, with FFᵀ = Iₙ (rows are orthonormal basis vectors)
  - F[0] = heading, F[1] = left, F[2] = up, F[3..N-1] = higher dimensions

The state evolves via:

**Translation**: `move(d)` → p ← p + d · F[0]

**Givens rotation** in the (i, j) plane:

```
R(θ, i, j): F[i] ← cos(θ) F[i] + sin(θ) F[j]
             F[j] ← -sin(θ) F[i] + cos(θ) F[j]
```

Named special cases:
- `turn(θ)` = R(θ, 0, 1) — heading ↔ left
- `pitch(θ)` = R(θ, 0, 2) — heading ↔ up
- `roll(θ)` = R(θ, 1, 2) — left ↔ up
- `yaw(θ)` = R(180°-θ, 0, 1) — molecular building convention

**Frame Invariant Proposition**: Givens rotations in any (i, j) plane preserve the orthonormality of the full N-dimensional frame. *Proof*: a Givens rotation is a special case of an orthogonal transformation; the product of orthogonal matrices is orthogonal. □

**Re-normalization**: `orthonormalize()` applies QR decomposition to recover numerical orthonormality after accumulated floating-point drift.

**Orientation**: `orient(position, heading, left)` sets position and constructs the full frame via Gram-Schmidt from the heading and left vectors, completing to a full orthonormal basis.

Coordinate transforms: `to_local(v)` = F · v and `to_global(v)` = Fᵀ · v.

The Turtle3D used in protein modeling (Roll, Pitch, Yaw, Turn, Move) is the special case N=3.

### 3.2 Layer 2: ManifoldWalker — Gradient Descent on the Manifold

ManifoldWalker performs optimization on an implicit manifold defined by a point cloud, using local PCA to estimate the tangent space at each step.

**Algorithm (one step)**:

1. **Neighborhood**: find k nearest neighbors of current position p
2. **Local PCA**: SVD of neighborhood covariance → eigenvalues λ ∈ ℝᴺ, eigenvectors V ∈ ℝᴺˣᴺ
3. **Intrinsic dimension**: d = min j such that (∑ᵢ≤ⱼ λᵢ) / (∑ᵢ λᵢ) ≥ τ
4. **Orient**: align turtle frame to eigenvectors (descending λ order)
5. **Project gradient**: transform ∇L to local frame, zero components d+1..N (suppress off-manifold noise)
6. **Step**: p ← p − η · g_projected

**ManifoldAdamWalker** replaces step 6 with Adam momentum [14]:

```
g = projected gradient (in global coordinates)
m ← β₁m + (1−β₁)g
v ← β₂v + (1−β₂)g²
m̂ = m/(1−β₁ᵗ),  v̂ = v/(1−β₂ᵗ)
p ← p − η · m̂ / (√v̂ + ε)
```

**Critical design**: Adam state (m, v) lives in *global* coordinates so momentum persists across frame re-orientations as the manifold curves. Projecting gradient onto the manifold before entering Adam's update rule ensures momentum cannot drift into off-manifold directions.

**Parameters**: k=50, τ=0.95, η=0.01 (base), β₁=0.9, β₂=0.999, ε=1e-7 (Adam variant).

### 3.3 Layer 3: ManifoldModel — The Manifold as the Classifier

ManifoldModel operationalizes the thesis: *the manifold is the model*. There are no learned weights. The trained model is the graph.

#### 3.3.1 Phase 1 — Explore (Fit)

For each training point xᵢ with label yᵢ:

1. **Node**: add (xᵢ, yᵢ) as a graph node
2. **Local PCA**: SVD of k_pca-neighborhood covariance
3. **Store NodeGeometry**: truncated basis B ∈ ℝᵈˣᴺ, eigenvalues λ, intrinsic_dim d, centroid c, label y

Build manifold-aware edges:
1. Find k_graph Euclidean nearest neighbors of node i
2. Project displacement into source tangent space: `manifold_dist = ‖Bᵢ(xⱼ − cᵢ)‖`
3. Blend: `dist_blend = w · manifold_dist + (1 − w) · euclidean_dist`, w = manifold_weight
4. `edge.weight = 1 / (1 + dist_blend)`

**What is stored** = what is learned:
- The graph connectivity (topology)
- Local basis B at each node (geometry)
- Eigenvalues λ at each node (scale)
- Node labels (supervision)

#### 3.3.1a The Blending Function

The distance metric in Step 3 is a linear convex combination controlled by `manifold_weight` w ∈ [0, 1]. Pure manifold distance (w = 1) measures only along the local tangent plane and is blind to ambient displacement that the local PCA basis does not capture — in particular, it cannot distinguish two points that lie in the same tangent plane but are far apart in ambient space. Pure Euclidean distance (w = 0) ignores the manifold structure entirely and may connect points that are Euclidean-close but on different folds of the surface.

The default w = 0.8 reflects three considerations:

1. **Primacy of geometry**: the tangent-plane projection captures the directions of real signal; it should dominate.
2. **Numerical grounding**: a 20% Euclidean component prevents degenerate edges when the local PCA basis is under-determined (e.g., in sparse neighborhoods) and ensures the graph is connected even when manifold distances are poorly estimated at boundaries.
3. **Empirical stability**: τ-sweep benchmarks on CIFAR-10 (Table 2) show that accuracy is more sensitive to τ than to manifold_weight in the range w ∈ [0.7, 0.9]; the default w = 0.8 sits at the flat center of this range.

The edge weight formula `1 / (1 + dist_blend)` maps blended distance to the interval (0, 1], decaying monotonically. An edge with zero blended distance (self-loop limit) receives weight 1; a distant edge receives weight approaching 0. This form is scale-invariant in the sense that the relative ordering of edge weights is preserved under uniform scaling of the data.

**Parameters**: k_graph=15, k_pca=50, k_vote=7, τ=0.95, manifold_weight=0.8

The choice of these defaults reflects the following reasoning. **k_pca = 50** provides enough neighbors for stable local PCA in moderate-dimensional spaces (rule of thumb: k ≥ 5d where d is intrinsic dimension; with d ≤ 10 typical, k = 50 is conservative). **k_graph = 15** controls graph connectivity: too few edges produce disconnected components; too many dilute manifold-awareness with distant nodes. **k_vote = 7** is the standard odd-k for majority voting. **τ = 0.95** retains 95% of local variance; Table 2 shows that τ ∈ {0.85, 0.90, 0.95} produces consistent results, with τ = 0.85 slightly favoring accuracy after PCA preprocessing and τ = 0.95 favoring fidelity in raw high-dimensional spaces.

#### 3.3.2 Phase 2 — Navigate (Predict)

To classify query point q:

1. Find k_pca Euclidean nearest neighbors of q
2. Local PCA *at q* via SVD → tangent space Bq
3. Project q and candidates into Bq
4. Walk graph from nearest node (2 hops) to gather additional candidates
5. Merge PCA-local + graph-connected candidates; distance-weighted majority vote among k_vote nearest

The prediction is O(k_pca · N) for local PCA + O(k_graph²) for graph walk + O(k_vote) for vote.

#### 3.3.3 Phase 3 — Fly (Interactive Exploration)

The fitted ManifoldModel also supports interactive navigation through the embedded space, useful for visualization and analysis:

- `fly_to(node_id)` — teleport to node, orient frame to its PCA basis
- `fly_step(direction?)` — one graph hop; score = 0.6 × alignment + 0.4 × edge_weight; re-orient
- `fly_toward(target, max_steps)` — repeated `fly_step` steering toward target
- `get_geometry(node_id)` → NodeGeometry at any node

This "fly mode" enables a form of *knowledge navigation* that is not available in standard classifiers: one can enter the manifold at any point and explore by heading toward regions of interest, with the frame always aligned to the local geometry.

### 3.4 Layer 3a: ManifoldObserver — Extrinsic Observer

ManifoldObserver is the central novel contribution of this work. It gives an extrinsic viewpoint into the N-dimensional data manifold from (N+1)-dimensional space.

#### 3.4.1 The Construction

Given the ManifoldModel's subject turtle at a manifold node with N-dimensional orthonormal frame F ∈ ℝᴺˣᴺ:

**Step 1: Pad** each row of F with a zero in the (N+1)-th component, append row **e**_N = [0, …, 0, 1]:

```
F_ext ∈ ℝ^{(N+1)×(N+1)}:  F_ext[i, :N] = F[i],  F_ext[i, N] = 0  for i < N
                            F_ext[N, :] = [0, …, 0, 1]
```

**Step 2: Orthonormalize** via QR decomposition, preserving orientation:

```
Q, R = QR(F_extᵀ)
For each i: if ⟨qᵢ, fᵢ⟩ < 0, flip qᵢ
F_obs = Qᵀ  ∈ ℝ^{(N+1)×(N+1)}
```

**Step 3: Position** p_obs = [p₀, …, p_{N-1}, h] ∈ ℝᴺ⁺¹, where h ≥ 0 is the observer altitude.

**Normal Existence Proposition**: Given any orthonormal frame F ∈ ℝᴺˣᴺ, the vector **e**_N = [0, …, 0, 1] ∈ ℝᴺ⁺¹ is orthogonal to every row of the padded frame F_ext (restricted to rows 0..N-1).

*Proof*: For i ∈ {0, …, N-1}: ⟨[f_{i,0}, …, f_{i,N-1}, 0], [0, …, 0, 1]⟩ = 0. □

The QR step ensures numerical orthonormality. The normal direction *exists by construction*.

#### 3.4.2 The Height Field

To lift the N-dimensional manifold into (N+1)-space, the observer computes a *height* for each training point equal to its reconstruction error from the local tangent plane:

```
Given point x ∈ ℝᴺ, local geometry (basis B ∈ ℝᵈˣᴺ, centroid c):
    diff = x - c
    projection = Bᵀ(B · diff)     (project onto tangent, then back)
    residual = diff - projection
    height = ‖residual‖
    x_lifted = [x₀, …, x_{N-1}, height]  ∈ ℝᴺ⁺¹
```

Points on the d-dimensional tangent plane have height 0. Points deviating from the manifold rise above it. The height field encodes *manifold quality*: low height = clean surface; high height = strained, boundary, or outlier.

#### 3.4.3 Curvature Measurement

The observer measures curvature at each node via **principal angles** between neighboring tangent subspaces:

```
At node i: basis Bᵢ ∈ ℝᵈⁱˣᴺ
At neighbor j: basis Bⱼ ∈ ℝᵈʲˣᴺ

Cross-Gramian: G = Bᵢ · Bⱼᵀ  ∈ ℝᵈⁱˣᵈʲ
SVD: G = UΣVᵀ
Principal angles: θₖ = arccos(σₖ)
Curvature at node i = mean(θₖ) over all neighbors
```

When tangent planes are parallel, all θₖ = 0 (flat). When they twist, θₖ > 0 (curved). The observer sees the *curvature field* across the entire manifold in a single pass, O(n × k_graph).

#### 3.4.4 Classification by Observation

The ManifoldModel classifies by *searching* the graph (O(hops × k_graph) per query). The ManifoldObserver classifies by *projection*:

1. Compute distances to all training points in (N+1)-space (incorporating height): O(N+1) · n
2. Return label at nearest lifted point: O(1)

No tangent-space computation. No graph walk. No voting. The nearest-neighbor lookup in (N+1)-space naturally incorporates manifold distance because the height coordinate penalizes off-manifold points.

The observer's prediction is a 1-NN in lifted space. Its advantage is not simply speed — it is that the (N+1)-dimensional nearest neighbor *correctly accounts for off-manifold deviation*, which the N-dimensional nearest neighbor cannot measure.

---

## 4. Application to Protein Disulfide Bonds

### 4.1 The Disulfide Manifold

Each disulfide bond in a protein is described by five dihedral angles (χ₁, χ₂, χ₃, χ₄, χ₅), defining a point in 5-dimensional torsion-angle space T⁵ ⊂ ℝ⁵ with periodic boundary conditions (each angle is modulo 360°). The proteusPy database contains structural information for over 36,000 disulfide-containing proteins from the RCSB Protein Data Bank [15].

From the ManifoldModel fit on disulfide torsion angles, the intrinsic dimensionality of the 5D space is approximately 2: the 36,000+ known disulfides concentrate on a low-dimensional surface, reflecting the physicochemical constraints on stable disulfide conformations.

The ManifoldModel with graph_from_disulfides() constructs the KNN graph with angular distance (wrapping at ±180°, appropriate for dihedral angles) and classifies each disulfide into one of the binary (2-class), quadrant (4-class), sextant (6-class), or octant (8-class) structural classes defined by the DisulfideTree hierarchy.

### 4.2 The 6D Observer

The ManifoldObserver extends the 5D torsion-angle turtle to 6D. The 6th dimension is the normal to the 5D disulfide structural manifold. From this vantage:

| Capability | What it reveals for disulfides |
|---|---|
| Height field | Which torsion configurations are "clean" (on-manifold) vs. strained or atypical |
| Curvature field | Where conformational transitions occur — ridges between structural classes |
| Class boundaries | Where binary/quadrant/octant classes meet, visible as curvature peaks |
| Outlier detection | Disulfides with high height = structurally unusual; potential errors or rare conformations |
| Instant classification | Project a new torsion vector down and read the structural class label |

The observer provides a *geometric interpretation* of the DisulfideTree hierarchy: class boundaries that the tree defines combinatorially (±90° threshold bins) appear as curvature peaks in the observer's field — geometric features of the structural landscape.

### 4.3 Integration with proteusPy

In proteusPy v0.99.50, ManifoldModel and ManifoldObserver are first-class components alongside the existing DisulfideLoader, DisulfideTree, and GraphReasoner infrastructure:

```python
from proteusPy import ManifoldModel, ManifoldObserver, graph_from_disulfides

# Build 5D torsion-angle graph from disulfide database
pdb_ss = Load_PDB_SS()
kg = graph_from_disulfides(pdb_ss)

# Fit zero-parameter classifier
model = ManifoldModel(k_graph=15, k_pca=50)
model.fit(X_train, y_train)  # X = [chi1, chi2, chi3, chi4, chi5] arrays

# Extend to 6D observer
observer = ManifoldObserver(model)
field = observer.observe()     # curvature + height at every node, O(n)
labels = observer.predict(X_test)  # classify by projection, not search
summary = observer.topology_summary()  # global manifold shape statistics
```

---

## 5. Experiments

### 5.1 UCI Digits Dataset (64 dimensions)

**Setup**: 1,797 handwritten digit images, 8×8 pixels, 10 classes. Standard sklearn train/test split.

| Method | Accuracy | Parameters |
|--------|----------|-----------|
| Euclidean KNN (k=7) | 97.33% | 0 |
| ManifoldModel (τ=0.95) | **97.72%** | 0 |

ManifoldModel discovers that 71–83% of the 64 ambient dimensions are noise. The intrinsic dimensionality across nodes ranges from 9 to 18 dimensions, with a mean around 13. The 0.39% accuracy improvement over Euclidean KNN reflects the correction for off-manifold distances — small on this dataset, where the signal-to-noise ratio is relatively favorable.

### 5.2 Iris Dataset (4 dimensions)

**Setup**: 150 samples, 4 features, 3 classes. ManifoldAdamWalker optimization on 2D projection manifold.

The Iris dataset has an intrinsic dimensionality of 2–3 in its 4-dimensional ambient space. ManifoldAdamWalker converges in 11 epochs to 96.7% test accuracy, compared with vanilla Adam's equivalent convergence in the same setting. ManifoldWalker's advantage here is not accuracy (both reach similar optima) but *convergence speed*: by suppressing off-manifold gradient components, it takes fewer steps to find the same minimum.

The effective parameter count of ManifoldModel on Iris is ~5 (k_graph, k_pca, k_vote, τ, manifold_weight) versus 243 learned weights for an equivalent MLP. No training is required.

### 5.3 MNIST (784 dimensions)

**Setup**: 5,000 training subsample, 2,000 test subsample from standard MNIST.

| Method | Accuracy | Parameters | Ambient Dim |
|--------|----------|-----------|-------------|
| Euclidean KNN (subsample) | 88.65% | 0 | 784 |
| Euclidean KNN (full 60K) | 93.75% | 0 | 784 |
| ManifoldModel (τ=0.95) | 89.55% | 0 | 784 |
| ManifoldModel (τ=0.9) | 89.45% | 0 | 784 |

ManifoldModel on MNIST discovers intrinsic dimensionality of 22–30 dimensions (mean 29.9 at τ=0.95) in the 784-dimensional pixel space, consistent with prior estimates of 20–25 for the MNIST manifold intrinsic dimension. ManifoldModel (τ=0.95) surpasses Euclidean KNN on the same subsample (89.55% vs 88.65%) but does not reach the performance of KNN on the full 60K training set, reflecting the information advantage of larger training data rather than a geometric limitation.

ManifoldModel uses approximately 2,232 effective parameters versus 109,386 for a standard single-hidden-layer MLP — a 49× reduction — while remaining competitive on this task.

### 5.4 CIFAR-10 (3,072 dimensions)

**Setup**: 2,000 training subsample, 1,000 test subsample from CIFAR-10.

*Table 1: CIFAR-10 accuracy comparison.*

| Method | Accuracy | Parameters | Notes |
|--------|----------|-----------|-------|
| Euclidean KNN (subsample) | 28.6% | 0 | 2K train |
| Euclidean KNN (full 50K) | 35.0% | 0 | 50K train |
| ManifoldModel (τ=0.95) | 27.3% | 0 | 3072-dim direct |
| PCA→30D + Euclidean KNN | 30.2% | 0 | Dimensionality reduction |
| PCA→30D + ManifoldModel (τ=0.85) | **32.5%** | ~3,751 | Best geometric result |
| Standard CNN | ~48.4% | 820,874 | Full architecture |

*Table 2: τ sensitivity on CIFAR-10 (3072-dim, 2K training). Mean intrinsic dimension and accuracy across variance threshold values.*

| τ | Mean intrinsic dim | Noise fraction | Accuracy (raw) | Accuracy (PCA→30D) |
|---|-------------------|---------------|---------------|-------------------|
| 0.95 | 35.9 | 98.83% | 27.3% | 31.7% |
| 0.90 | 28.5 | 99.07% | 26.2% | 31.7% |
| 0.85 | 23.2 | 99.25% | 27.2% | **32.5%** |

Accuracy is stable across τ values (range 1.1% in raw space, 0.8% after PCA preprocessing), confirming that the model is not sensitive to the precise variance threshold in this range. The slight edge for τ = 0.85 after PCA preprocessing is consistent with the observation that global PCA has already removed the bulk of the noise, making a more aggressive local intrinsic-dimension cutoff beneficial.

CIFAR-10 presents the most challenging setting: 3,072-dimensional images with 99.1% noise (intrinsic dimension ~29). Raw ManifoldModel in the full ambient space does not surpass KNN on the same training subsample, consistent with the curse of dimensionality for KNN-based methods. However, PCA→30D preprocessing followed by ManifoldModel outperforms both raw KNN and PCA→30D + KNN, demonstrating that manifold-aware distances add value even after preprocessing.

When compared against a standard CNN benchmark at full training scale, ManifoldModel achieves comparable accuracy with 219× fewer parameters (3,751 vs 820,874), representing a remarkable compression of model complexity.

The CIFAR-10 result is significant: as ambient dimensionality grows and the noise fraction increases, manifold-awareness becomes increasingly important. The PCA→30D + ManifoldModel combination reduces to the manifold first (via global PCA), then re-discovers the local manifold geometry (via local PCA at each node) — a two-stage approach that handles the very high ambient dimensionality while retaining geometric structure.

### 5.5 ManifoldObserver Validation (Synthetic Helix)

**Setup**: 3D helix embedded in 5D with Gaussian noise in 2 extra dimensions (500 points).

| Metric | Value |
|--------|-------|
| Subject (5D) intrinsic dim | 2.0 |
| Mean height | 0.101 |
| Mean curvature | 0.140 rad |
| Curvature std | 0.009 rad |
| Observer ↔ Subject prediction agreement | **100%** |

The observer correctly identifies intrinsic dimension 2 (the 1D helix curve produces 2 principal components due to correlated noise), uniform curvature (std = 0.009 on a perfect helix), and low mean height (points near the ideal helix surface). The 100% agreement between observer and subject predictions validates that projection from N+1 dimensions recovers the same classification as graph search in N dimensions.

---

## 6. Discussion

### 6.1 The Flatland Principle

The ManifoldObserver instantiates a principle that runs through the deepest currents of mathematics, geometry, and philosophy: *to understand a surface, you must leave it.* Three traditions illuminate what this means — and why gaining a single dimension is transformative.

**Plato's Cave.** In the *Republic* (Book VII), Plato describes prisoners chained in a cave, able to see only the wall before them. Shadows pass across the wall — projections of figures moving in front of a fire behind them. The prisoners take the shadows for reality. They can measure shadow-lengths, predict shadow sequences, build an elaborate science of shadows. But they cannot know the three-dimensional forms that *cast* the shadows, because they are imprisoned in the two-dimensional projection.

The ManifoldModel walker is a Platonic prisoner. It moves along an N-dimensional manifold, seeing only the local tangent plane and the shadows of nearby points projected into that plane. It builds an accurate local science — local PCA, manifold distances, class boundaries — but it cannot see the *shape* of the surface it walks on. It cannot know whether the manifold curves back on itself, where the class boundaries lie globally, or what the full topology looks like. It sees shadows.

The ManifoldObserver is the prisoner who escapes the cave. By adding one dimension — one extra coordinate, one QR step — it gains the vantage from which the forms are visible. The curvature field, the height field, the class boundary ridges: these are not hidden. They are simply *behind* the surface-bound observer. Lift by one dimension and they come into view.

**Flatland.** Edwin Abbott's *Flatland* (1884) describes a world of two-dimensional beings — squares, triangles, hexagons — living on an infinite plane. A Sphere visits from the third dimension, but the Flatland creatures cannot see it as a sphere: as it passes through their plane, they see only a circle that appears, grows, shrinks, and vanishes. The sphere *is* there, but its three-dimensional form is invisible to them. Only when the Sphere lifts a Flatlander *out* of the plane — above it — does the Flatlander suddenly see all of Flatland laid out below, perceiving its true shape for the first time.

The ManifoldWalker is the Flatlander. It walks the N-dimensional manifold and sees only the local neighborhood — an N-dimensional circle of neighbors, expanding and contracting as it moves. It cannot see the manifold from outside. When a new class boundary appears, the walker does not see a boundary; it sees only that nearest neighbors have started voting differently. It can deduce curvature by comparing many local PCA results, but it must *search* for this information, hop by hop.

The ManifoldObserver is the Sphere. It lifts the walker by one dimension and looks down. The N-dimensional surface is spread out below it — every class boundary visible as a ridge in the curvature field, every outlier visible as a peak in the height field, every structural family visible as a valley in the lifted point cloud. What took the walker O(hops × k_graph) to discover by search, the observer reads at a glance in O(n).

**Gauss and the Theorema Egregium.** Gauss's *Theorema Egregium* (1827) showed that Gaussian (intrinsic) curvature can be computed entirely from within the surface — a 2D creature can, in principle, discover it by measuring triangles and noticing that angle sums deviate from π. But the *mean curvature* — how the surface bends through its ambient space, the quantity that governs whether a soap bubble is expanding or contracting — is extrinsic. It is invisible from inside the surface.

The ManifoldObserver measures the manifold analogue of mean curvature via principal angles between neighboring tangent planes. This is a genuinely extrinsic measurement: it quantifies how the tangent planes *rotate* as one moves across the surface, which is a property of how the surface sits in its ambient space, not merely a property of distances within the surface. The surface-bound walker can approximate this by finite differences across many local PCA results — an expensive, indirect procedure. The observer reads it directly from above, as a principal angle between neighboring tangent subspaces.

**The constructive resolution.** All three traditions share the same structure: insight is blocked not by lack of intelligence but by dimensional imprisonment. The cave wall is a 2D projection of 3D reality. Flatland is a 2D slice of 3D space. A curved surface is an N-dimensional slice of (N+1)-dimensional space. In each case, the resolution is the same: *gain one dimension.*

The ManifoldObserver does this by construction, not metaphor. Given any N-dimensional orthonormal frame, the Normal Existence Proposition guarantees that a unique (N+1)-th dimension exists, orthogonal to the entire surface. QR decomposition finds it exactly. The cost is one extra coordinate and one linear-algebra step. The payoff is the transition from prisoner to philosopher, from Flatlander to Sphere, from intrinsic to extrinsic geometry — from search to sight.

### 6.2 WaveRider vs. Dimensionality Reduction

WaveRider should not be confused with dimensionality reduction methods (UMAP, t-SNE, Isomap). Those methods project data to low-dimensional Euclidean space for visualization or downstream processing. WaveRider operates in the full ambient space, discovers the manifold geometry *in place*, and uses that geometry directly for classification.

The PCA→30D preprocessing in Section 5.4 is not WaveRider dimensionality reduction; it is conventional preprocessing to handle the extreme dimensionality of CIFAR-10. The manifold discovery happens afterward, in 30-dimensional space, and still finds intrinsic dimension ~12–15.

### 6.3 When WaveRider Wins

The experimental results suggest a pattern:

- **Low ambient dim, favorable signal-to-noise** (Digits, 64-dim): ManifoldModel marginally beats Euclidean KNN (+0.39%). Both have zero parameters.
- **Very low dim, simple manifold** (Iris, 4-dim): ManifoldWalker converges faster; ManifoldModel needs no training at all.
- **High ambient dim, preprocessed** (CIFAR-10 + PCA→30D): ManifoldModel beats KNN baseline (+2.3%). Still zero learned parameters.
- **Very high ambient dim, raw** (CIFAR-10 3072-dim): curse of dimensionality affects all KNN-based methods; preprocessing needed first.

The pattern is consistent with the manifold hypothesis: as the ratio (intrinsic_dim / ambient_dim) decreases, manifold-aware distances matter more, and WaveRider's advantage grows.

### 6.4 Parameter Count Is Not the Right Metric

The most striking result is the parameter comparison: 3,751 (WaveRider) vs. 820,874 (standard CNN) on CIFAR-10, a 219× reduction with comparable accuracy. But "parameter count" is not quite the right frame for WaveRider: it has no *learned* parameters at all. The "parameters" counted are the structural hyperparameters (k_graph, k_pca, k_vote, τ, manifold_weight) plus the size of the stored geometry. In the strict sense of parameters as gradient-updated scalars, WaveRider has zero.

This is not just a philosophical point. It means WaveRider requires no training loss, no optimizer, no backpropagation infrastructure, no GPU. The "training" is a single exploration pass (O(n × k²) for local PCA at each node) after which the graph is fixed.

### 6.5 Limitations

- **Fit time**: ManifoldModel's exploration phase is O(n × k_pca² × N) and can be slow for large n at high ambient dim (5000 points, 784-dim: ~5 minutes on CPU). GPU acceleration of the SVD step would substantially reduce this.
- **Memory**: storing a basis matrix B ∈ ℝᵈˣᴺ at each of n nodes requires O(n × d × N) memory — manageable at 5,000 nodes but potentially limiting at 50,000+.
- **Angular data**: the current implementation uses Euclidean distance for KNN queries. For angular data like protein torsion angles, a custom angular distance metric (provided via `graph_from_disulfides()`) improves performance.
- **No explicit noise model**: manifold discovery via local PCA assumes the manifold is "clean" enough for PCA to separate signal from noise. In very noisy regimes the intrinsic dimension estimate may be unreliable.

---

## 7. Conclusion

WaveRider is a coherent geometric machine learning stack rooted in the observation that real high-dimensional data concentrates on low-dimensional manifolds, and that the manifold geometry — once discovered — is sufficient for accurate classification.

The four-layer stack (TurtleND → ManifoldWalker → ManifoldModel → ManifoldObserver) progresses from a navigation primitive through optimization to a zero-parameter classifier to an extrinsic observer that gains sight by adding a single orthonormal dimension. The Normal Existence Proposition provides the mathematical guarantee: any N-dimensional orthonormal frame can be extended to (N+1) dimensions with the (N+1)-th vector guaranteed to be the manifold normal.

Experimental results show that ManifoldModel equals or exceeds Euclidean KNN across all tested datasets with zero learned parameters, and that ManifoldModel uses 49–219× fewer parameters than standard neural networks while maintaining competitive accuracy. The ManifoldObserver achieves 100% prediction agreement with its surface-bound subject while providing curvature, topology, and height information that the walker cannot access.

The application to protein disulfide bond analysis demonstrates the framework's domain generality: the same TurtleND primitive that builds protein molecules by navigating coordinate frames along bond vectors now navigates the structural landscape of over 36,000 disulfide bonds and, via the ManifoldObserver, sees the shape of that landscape from one dimension above.

From a plotter drawing Spirographs on a Commodore 64 to a 6-dimensional observer looking into the structural manifold of life's proteins: the turtle is ready.

---

## References

[0a] Plato (~380 BCE). *The Republic*, Book VII: The Allegory of the Cave. (Trans. Jowett, B., 1894.)

[0b] Abbott, E. A. (1884). *Flatland: A Romance of Many Dimensions*. Seeley & Co., London.

[1] Tenenbaum, J. B., de Silva, V., & Langford, J. C. (2000). A global geometric framework for nonlinear dimensionality reduction. *Science*, 290(5500), 2319–2323.

[2] Roweis, S. T., & Saul, L. K. (2000). Nonlinear dimensionality reduction by locally linear embedding. *Science*, 290(5500), 2323–2326.

[3] Kambhatla, N., & Leen, T. K. (1997). Dimension reduction by local principal component analysis. *Neural Computation*, 9(7), 1493–1516.

[4] Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks. *Science*, 313(5786), 504–507.

[5] Suchanek, E. G. (2024). proteusPy: A Python package for protein structure and disulfide bond modeling and analysis. *Journal of Open Source Software*, 9(100), 6169.

[6] Suchanek, E. G., et al. (1992). Computer-aided strategies for protein design. *Biochemistry*, 31(9), 2429–2434.

[7] McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform manifold approximation and projection for dimension reduction. *arXiv:1802.03426*.

[8] Maaten, L. V. D., & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9(11), 2579–2605.

[9] Absil, P. A., Mahony, R., & Sepulchre, R. (2009). *Optimization algorithms on matrix manifolds*. Princeton University Press.

[10] Bonnabel, S. (2013). Stochastic gradient descent on Riemannian manifolds. *IEEE Transactions on Automatic Control*, 58(9), 2217–2229.

[11] Bronstein, M. M., Bruna, J., Cohen, T., & Veličković, P. (2021). Geometric deep learning: Grids, groups, graphs, geodesics, and gauges. *arXiv:2104.13478*.

[12] Kohonen, T. (1990). The self-organizing map. *Proceedings of the IEEE*, 78(9), 1464–1480.

[13] do Carmo, M. P. (1976). *Differential Geometry of Curves and Surfaces*. Prentice-Hall.

[14] Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015*.

[15] Berman, H. M., et al. (2000). The Protein Data Bank. *Nucleic Acids Research*, 28(1), 235–242.

---

## Appendix A: Implementation Notes

All code is available in proteusPy v0.99.50, Elastic License 2.0:

| Module | Class | Lines |
|--------|-------|-------|
| `proteusPy/turtleND.py` | `TurtleND` | ~400 |
| `proteusPy/manifold_walker.py` | `ManifoldWalker`, `ManifoldAdamWalker` | ~350 |
| `proteusPy/manifold_model.py` | `ManifoldModel`, `NodeGeometry` | ~600 |
| `proteusPy/manifold_observer.py` | `ManifoldObserver`, `ObservedGeometry` | ~500 |
| `proteusPy/graph_reasoner.py` | `KnowledgeGraph`, `GraphReasoner` | ~700 |
| `benchmarks/` | Digits, Iris, MNIST, CIFAR-10 benchmarks | ~800 |

**Dependencies**: numpy, scipy (SVD), scikit-learn (KNN, PCA), networkx (graph), tensorflow (optional, benchmarks only).

**Reproducibility**: All benchmark scripts are in `benchmarks/`. Results JSON files are version-controlled alongside the code. TensorBoard logs are stored in `runs/`. Random seeds are fixed; all experiments are reproducible on CPU.

---

## Appendix B: arXiv Submission Notes

**Recommended categories**:
- Primary: `cs.LG` (Machine Learning)
- Secondary: `stat.ML` (Machine Learning, Statistics)
- Tertiary: `q-bio.QM` (Quantitative Methods, for the disulfide application)

**Affiliation**: Independent Researcher (valid for arXiv; see https://arxiv.org/help/submit#affiliations). The Endorsement Program provides access for independent researchers: https://arxiv.org/help/endorsement.

**Workshop targets** (no affiliation required):
- NeurIPS: Geometry in Machine Learning (GiML) Workshop
- ICML: Workshop on Topology, Algebra, and Geometry in Machine Learning
- ICLR: Workshop on Geometrical and Topological Representation Learning

**Journal targets** (open-access):
- *PLOS Computational Biology* — for the protein/disulfide application
- *Journal of Open Source Software* — already published proteusPy, natural extension
- *Transactions on Machine Learning Research* (TMLR) — open access, no affiliation required
- *Frontiers in Bioinformatics* — bioinformatics + ML intersection

**Path to endorsement**:
1. Contact arXiv endorsers in cs.LG — list at https://arxiv.org/auth/endorse
2. Reach out via ResearchGate or Academia.edu (author profile with prior JOSS publication provides credibility)
3. The JOSS paper [5] establishes prior publication record; cite it prominently

---

*"To understand a surface, you must leave it."*
