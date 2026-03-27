# Temporal Momentum Hypothesis
**Author:** Eric G. Suchanek, PhD
**Date:** 2026-03-27
**Status:** Axis inversion confirmed 2026-03-27 · Temporal momentum next

---

## Core Observation

The current temporal flight primitives in `TurtleND` treat time steps as potentially reversible — the local arrow of time can flip.  But there are two distinct arrows in play, and they behave differently:

| Arrow | Scope | Reversible? | Analogue |
|---|---|---|---|
| **Global arrow of time** | Entire trajectory | No — entropic, causal | Adam's m_t momentum accumulation |
| **Local arrow of time** | Single step | Yes — adaptive, curvature-aware | Adam's per-step gradient direction |

The question: **should we maintain a global temporal momentum term** analogous to Adam's first-moment estimator, rather than letting the global arrow flip?

---

## The Adam Analogy

In `ManifoldAdamWalker`, the Adam optimizer maintains:

```
m_t = β₁ · m_{t-1} + (1 − β₁) · g_t        # first moment (momentum)
v_t = β₂ · v_{t-1} + (1 − β₂) · g_t²       # second moment (adaptive scale)
θ_{t+1} = θ_t − α · m̂_t / (√v̂_t + ε)
```

Key insight: **m_t accumulates a directional bias over the full history of gradients.** Local gradient g_t can point anywhere — even "backward" relative to prior steps — but m_t damps that noise and preserves the global trajectory direction.

This is not accidental. It is the mechanism that makes Adam stable on curved loss surfaces.

---

## Proposed Temporal Analogue

Define a **temporal momentum term** for `TurtleND`:

```
τ_t = β_τ · τ_{t-1} + (1 − β_τ) · Δt_t
```

Where:
- `τ_t` — accumulated temporal momentum at step t
- `β_τ` — temporal momentum decay (analogous to Adam β₁, e.g. 0.9)
- `Δt_t` — local time displacement at step t (signed — the local arrow)

The **effective time step** used for navigation becomes:

```
Δt_effective = τ_t / (1 − β_τ^t)    # bias-corrected, like Adam
```

This allows:
- **Local adaptivity**: `Δt_t` can be small, large, or even locally "backward" in a curved temporal manifold
- **Global forward bias**: `τ_t` accumulates history and enforces causal directionality
- **The global arrow never reverses**: even if a single `Δt_t` is negative (local reversal), the momentum term keeps the integrated trajectory pointing forward

---

## Physical Justification

The second law of thermodynamics is a **global** statement, not a local one.
Microscopically (locally), time-reversible dynamics are common.
Macroscopically (globally), entropy increases monotonically.

The same structure holds here:
- Local: the navigator can follow geodesics that curve "back" through the temporal manifold
- Global: the integrated trajectory must remain forward-pointing in physical time

This is exactly what the momentum term enforces.

---

## Implications for WaveRider

### Navigation (Sulu / TurtleND)
Sulu does not reverse the warp drive. He adjusts **heading within forward momentum.**
The warp field has memory — it carries the ship's prior trajectory as inertia.
That inertia is `τ_t`.

### Manifold Walker coupling
If temporal position is a coordinate in the manifold, then `ManifoldAdamWalker` already handles this correctly — provided the temporal axis is treated as an **ordinary dimension with a momentum-carrying gradient**, not a special axis with a hard-coded arrow.

The question becomes: do we need a separate temporal momentum accumulator, or does folding the temporal axis into Adam's standard m_t give us the right behavior for free?

**Hypothesis:** Folding it in directly is correct and sufficient — the Adam machinery already enforces forward temporal bias globally through accumulated momentum, without any special-casing.

---

## Empirical Results — 2026-03-27

### τ-reversal experiment (`pepys_temporal_flight.py`, 6450 entries, 768D→769D, α=1.0, k=10)

| Mode | Mono | Kendall τ | Net span |
|---|---|---|---|
| Semantic | 47.3% | +0.049 | +4.0 yr |
| Temporal → (fixed) | 55.3% | **+0.457** | −0.2 yr |
| Temporal ← | 50.7% | −0.306 | +2.6 yr |
| Mixed 50/50 | 47.3% | −0.165 | +3.2 yr |

τ-reversal symmetry residual: **0.1507** (non-zero due to non-uniform entry density across years).

### Finding 1 — Axis inversion confirmed

`orient_in_time` was pointing toward `+e_t` (positive z-score = later dates).
Empirically, forward-in-time motion in the KNN graph corresponds to the **negative** z-score direction.
**Fix applied:** `orient_in_time` now points toward `e_t[time_axis] = -1.0`.
After fix: τ(forward) = +0.457, τ(backward) = −0.306. ✓

### Finding 2 — Temporal wandering persists

Even with the correct axis, temporal forward flight shows:
- monotonicity = 55.3% (barely above chance 50%)
- net span = −0.2 yr over 151 hops (mean |Δt| = 0.82 yr per hop)

The turtle follows the correct global direction but oscillates locally — each greedy
KNN step picks the neighbor most aligned with `-e_t`, but semantic clustering
in the graph pulls the path sideways through time.  The result is chronological
drift rather than chronological march.

**Root cause:** no temporal memory.  Each step is independent.  A hop that goes
slightly backward in time is not penalised by any accumulated forward bias.
This is exactly the gap the temporal momentum term is designed to fill.

---

## Open Questions

1. **Is the local arrow of time correct as-is?**
   **Resolved 2026-03-27.** The axis was inverted. Fixed in `orient_in_time`. Local
   step direction is now correct.

2. **Should `τ_t` be a separate accumulator, or just the temporal component of Adam's m_t?**
   Still open. A separate accumulator in `TemporalFlyer.fly_step` is the lowest-risk
   first implementation — doesn't touch `ManifoldAdamWalker`. Fold in later if it works.

3. **What is the right β_τ?**
   High β_τ (→ 1) = strong global temporal inertia, slow to respond to curvature.
   Low β_τ (→ 0) = purely local, no global memory. Probably wrong.
   β_τ = 0.9 (Adam default) is the starting point. Sweep [0.5, 0.7, 0.9, 0.95].

4. **Next implementation: temporal momentum in `TemporalFlyer.fly_step`**

   Add a `tau` accumulator to `temporal_flight` and `mixed_flight`:

   ```python
   # initialise before flight loop
   tau = 0.0
   beta_tau = 0.9
   t = 0

   # inside fly_step, after choosing best_idx:
   delta_t = fyears[best_idx] - fyears[current]   # signed time displacement
   t += 1
   tau = beta_tau * tau + (1 - beta_tau) * delta_t
   tau_corrected = tau / (1 - beta_tau ** t)       # bias correction

   # penalise the step score if tau_corrected < 0 (global arrow reversing)
   # or blend tau_corrected into the heading direction on the time axis
   ```

   Expected outcome: monotonicity rises from 55% toward 70–80%; net span becomes
   consistently positive; wandering visibly reduced in the figure.

5. **Chapter hook (still valid):**
   Spock notes the anomaly — temporal position is drifting backward in the local frame.
   Sulu holds course. "The momentum keeps us moving forward, Mr. Spock. The warp field remembers where we've been."
   McCoy: "Are you telling me the *ship* has a memory now?"

---

## Related Files

- `proteusPy/TurtleND.py` — temporal primitives added in commit `7b5ba3ef`
- `proteusPy/ManifoldAdamWalker.py` — Adam momentum implementation to study
- `waverider_missions/STORY_ARC.md` — narrative arc; temporal chapter TBD
- `waverider_missions/waverider_trek_ch5.md` — candidate chapter for this theme
