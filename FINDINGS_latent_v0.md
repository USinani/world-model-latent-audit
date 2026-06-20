# Findings — latent v0 (closeout)

**Bounded result:** Latent v0 closes with a bounded mechanism-level negative: the hidden-mass
consequence delta was well-posed in the auditor, recoverable from rendered observations at matched
12-D capacity, but not preserved by this reconstruction-trained AE stack in `z`.

**Formal label:** `VOID-FIDELITY` (terminal for latent v0).
**Scientific interpretation:** `Branch C / available consequence not preserved by this
reconstruction-trained AE stack`. The formal label is exactly `VOID-FIDELITY`; the binding evidence is
the **negative matched-pair latent delta** (`R²(Δz_next → Δqdd_window) = −0.175`) set against the
**matched-capacity, matched-probe raw-observation delta** (`R²(Δraw_obs_next → Δqdd_window) = 0.716`,
PCA→12) and the analytic oracle (`R²(Δtrue_pose → Δqdd_window) = 0.992`). So the consequence was
available in the observation and was not preserved in the latent — not merely "the observation was too
coarse." (See the Addendum below for the full ceiling diagnostic.)

This memo is the single-page closeout. The gate-by-gate build is in
[LOG.md](LOG.md) (LW-11 section); the claim ledger is in [README.md](README.md); the canonical run
is [results/20260619_193900_33faabb_latent/](results/20260619_193900_33faabb_latent/).

## The question latent v0 asked

When analytic state `(q, q̇, τ)` and the physics equation are hidden behind an autoencoder, can a
learned latent transition-consistency signal `c_z` do the analytic physics-residual `r`'s job —
catch a consequence-only physics shift (hidden mass change) that latent input-detectors are blind to
— **without** the latent secretly recovering the privileged state? Reproducing `r`'s column requires
the physics consequence to first be present in the latent that `c_z` consumes.

## Decisive LW-11 facts (verified; 48×48 full run)

| gate | requirement | result | status |
|------|-------------|--------|:------:|
| GATE B (no input leak) | AUROC(d_z),(u_z) clean-vs-shift ∈ [0.45, 0.55] | 0.500 / 0.500; matched-pair max\|Δ\| = 0.0 | **PASS** |
| RECON (output window) | recon ≥1.5× over mean-frame; shift ≤1.5× clean | 0.00516 vs baseline 0.01840 (**3.6×**); shift/clean ratio 0.99 | **PASS** |
| **GATE A** (z_next carries `qdd_window`) | pooled **and** physics_shift `R²(z_next+a → qdd_window)` > 0.60 | clean 0.444 / physics_shift 0.406 / **pooled 0.425** | **FAIL** |
| **DELTA GATE** (Δ consequence in latent) | mean `R²(Δz_next → Δqdd_window)` > 0.30 (paired nuisance) | mean **−0.175** (per-seed −0.152 / −0.185 / −0.187; non-jittered −0.162) | **FAIL** |

The **detector table was NOT interpreted** — per the pre-registered rule, with Gate A failing the
missed-catastrophe table is recorded in `metrics.json` for the file only and carries no verdict.

The decisive number is the **negative** matched-pair delta: even after removing the render-noise
confound (paired nuisance), and with the matched `obs_t`/action cancelling in the difference,
`Δz_next` predicts `Δqdd_window` *worse than its own mean* (R² = −0.175). The literal clean→heavy
consequence that `c_z` would have to key on is **absent from the latent difference**.

## Confounds eliminated (each ruled out by a specific gate or fix)

- **Not input leak.** Gate B passed: input-only detectors (`d_z`, `u_z`) cannot separate clean from
  physics_shift (AUROC 0.500, matched-pair Δ = 0.0), by the shared nominal `obs_t` construction.
- **Not a native-`dt` timescale mismatch.** LW-10 corrected the Gate A target to window-scale
  acceleration `qdd_window = (q̇_{t+S·dt} − q̇_t)/(S·dt)`, aligned to the observation timescale.
- **Not a raw-vs-latent probe mismatch.** LW-11 reads Gate A on `z_next` — the exact representation
  `c_z` consumes — not on raw pixels; raw `obs_next+a` is kept only as a diagnostic.
- **Not unpaired render jitter.** The delta gate uses paired nuisance (clean and heavy renders share
  the per-pair jitter seed), and the non-jittered diagnostic agrees (−0.162), so the negative delta
  is not a render-noise artifact.
- **Not detector threshold tuning.** The 5% calibration and the `missed()` logic were unchanged
  throughout LW-09/10/11; the 0.60 bar was never softened after seeing a failing number.

Recon competence and capacity were also checked: recon passed (3.6× over baseline, physics_shift
comparable), and the capacity-matched probe showed `z_next+a` (0.429) ≥ PCA-matched `obs_next+a`
(0.345) — i.e. the latent is not simply a lossier projection than the pixels. Neither rescued the
two binding gates.

## Scope limit (read this before citing)

This is **not** a general claim that learned latents cannot support physics auditability, and **not**
a general claim about all reconstruction objectives. It is a bounded result about **this
reconstruction-trained AE** and **this observation/window family** (48×48 grayscale stacked windows;
frame interval 0.08s / `frame_stride=8`; ZOH action).

**Bounded mechanism hypothesis (this stack only):** under this reconstruction objective and
architecture, the learned latent allocated its limited capacity to reconstruction-relevant visual
structure rather than to the low-variance but audit-relevant consequence delta. This is a bounded
mechanism hypothesis for this stack, not a general claim about all reconstruction objectives or all
learned latents.

## Next variable (named, not implemented)

The next variable to change is the **representation objective**: a predictive / JEPA-style /
audit-preserving latent objective instead of reconstruction, so the latent is trained to keep the
transition consequence rather than only to reconstruct frames. This is out of scope for latent v0 and
requires a new charter — it changes the question, not just the rig.

## Addendum: raw-observation consequence ceiling

**Purpose.** One isolated diagnostic to settle the remaining causal ambiguity in the
`VOID-FIDELITY / consequence-not-encoded` reading: *was the matched hidden-mass consequence delta
present and recoverable in the rendered observations before the reconstruction-trained AE stack failed
to preserve it, or was it absent/unresolved before the AE ever saw it?* This is **not** a new gate and
**not** a new experiment phase; **no detector table was interpreted**; the formal LW-11 label remains
**`VOID-FIDELITY`**. No AE/dynamics retraining (the probe is AE-free).

**Method.** Target-aligned (not raw pixel MSE): the capacity-matched raw-observation consequence probe
`R²(Δraw_obs_next → Δqdd_window)`, where `Δraw_obs_next = raw_obs_next_heavy − raw_obs_next_clean` and
`Δqdd_window = qdd_window_heavy − qdd_window_clean`, on the **same** matched clean/heavy pairs and
**paired-nuisance** convention as the LW-11 delta gate, using the full flattened `obs_next` stack
reduced to `latent_dim=12` by PCA and the same 3-seed nonlinear probe. The reconstruction was anchored
by reproducing the committed LW-11 capacity-matched obs probe: **0.349** (committed 0.345; per-seed
0.372/0.395/0.282 vs 0.373/0.397/0.266) — faithful.

The raw-observation and latent delta comparisons used the same 3-seed nonlinear probe policy (the
identical `_multiseed_r2`/`_nl_fit` MLP with `PROBE_SEEDS=(7,17,27)` and the same `Δqdd_window`
targets that the committed LW-11 latent delta gate used), so the 0.891 gap is a matched-capacity,
matched-probe comparison.

**Results (48×48; `frame_stride=8`; PCA→12; seeds 7/17/27):**

| quantity | value |
|---|---|
| **Primary: `R²(Δraw_obs_next → Δqdd_window)` PCA-12** | **mean 0.716** (std 0.007; per-seed 0.706 / 0.718 / 0.722) |
| Latent delta (committed LW-11) for comparison | mean −0.175 (per-seed −0.152 / −0.185 / −0.187) |
| `pixel_minus_latent_gap` | **0.891** |
| `pca12_explained_variance_ratio_sum(Δraw_obs_next)` | 0.327 |
| Analytic `Δqdd_window` non-degeneracy | mean‖·‖ 9.459; total var 140.10; per-joint var [12.56, 127.53] |
| Oracle `R²(Δtrue_pose_window → Δqdd_window)` (reference, not a gate) | 0.992 |
| Diagnostic only (no branch rule): `pixel_delta_r2_pca24` | 0.791 (EVR-24 0.501) |
| Diagnostic only: curvature-delta `(f₂−2f₁+f₀)` PCA-12 | 0.357 |

The 12-D bottleneck retains only ~33% of the `Δraw_obs_next` variance (EVR-12 = 0.327) yet still
recovers `Δqdd_window` at R² 0.716 — so at the **matched** observation capacity the consequence lives
in retained directions; capacity is not the limiter.

**Pre-registered interpretation — Branch C.** The analytic delta is non-degenerate (total var 140.10)
and recoverable from true kinematics (oracle 0.992); the capacity-matched raw observation carries it
(`pixel_mean = 0.716 > 0.30`) with `pixel_mean − latent_mean = 0.891 ≥ 0.30`. Therefore:

> The rendered observation made the matched hidden-mass consequence delta recoverable, but this
> reconstruction-trained AE stack at this capacity failed to preserve it in `z`. This strengthens the
> representation-objective hypothesis, while still not proving a general claim about reconstruction
> objectives.

**Wording guard (held).** This is **not** evidence that "reconstruction objectives discard physical
consequences." The bounded statement is only: *this reconstruction-trained AE stack at this capacity
failed to preserve an available consequence.* The broader objective claim is what the 2×2 transfer
matrix exists to test; it is **not** assumed here.

**Status notes.** Formal LW-11 label unchanged (`VOID-FIDELITY`); not a new gate; not a new experiment
phase; no detector table interpreted. The diagnostic script was temporary and uncommitted.
