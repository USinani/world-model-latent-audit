# Latent_Wedge — build log

Chronological, gate-by-gate. One log, no control-plane sprawl.

> **Closeout:** latent v0 is terminal at `VOID-FIDELITY` (LW-11). Single-page findings memo:
> `FINDINGS_latent_v0.md`.

## M0 — port the validated substrate (numpy-only)

- Sourced the canonical v3 tree from `Archives/` by **content**, not filename: the complete
  tree (4 experiments + FINDINGS + CHARTER + failure-mode `metrics.json`) corresponds to
  phase-close commit `72866ed`. The `a86ee25` stamp in the old results folder is the **v2**
  commit label — recorded as a label error, not a wrong tree.
- Ported verbatim: `arm.py` (added only `joint_positions` for rendering — pure geometry,
  no dynamics change), `data.py`, `config.py` (latent/render fields additive), and
  `tests/test_invariants.py`.
- Ported with numpy swaps: `model.py` (`MLPRegressor → NumpyMLP`), `experiment_failure_modes.py`
  (`roc_auc_score → metrics_np.roc_auc`).
- `numpy_mlp.py`: generalised the 003B stable-MLP recipe to arbitrary `(in, …, out)` with
  Adam (to match the validated sklearn convergence at `max_iter=150`), grad-clip, weight
  decay, finiteness guards. `metrics_np.py`: tie-aware Mann-Whitney `roc_auc` + `spearman`.
- Physics self-audit: ALL PASS (energy drift 2.6e-7; damping dissipates; grav=∇V 2e-9;
  residual≈0 when nominal==true). numpy-swap audit: ALL PASS.

## M1 — PARITY GATE (STOP if fail)

- Pre-registered tolerance written into `experiment_failure_modes.py` **before** the run:
  each `r` cell within ±8 absolute points of v3, AND crossover intact (r best on
  physics_shift, u/d best on extrapolation).
- Full run result vs v3 `72866ed`:
  - extrapolation `r` 57.8% (v3 57.4%, |Δ|=0.4 pt) — OK
  - physics_shift `r` 0.0% (v3 0.0%) — OK
  - id_holdout `r` 58.4% (v3 62.6%, |Δ|=4.1 pt) — OK
  - crossover preserved — OK
  - **PARITY PASS.** The numpy port is not a confound; latent work is licensed.
- (u/d columns differ in magnitude from v3, as expected for a different MLP implementation;
  the gate is on the `r` column + crossover, which are what license the comparison.)

## M2 — render.py

- Forward-kinematics grayscale raster; vectorised + chunked (a per-frame Python loop was
  ~4 min/quick — unacceptable; vectorised core brought quick to ~30 s).
- Anti-laundering: stacked window (`frame_stack=3`, `frame_stride=8` so `q̇` is inferable
  from frame deltas, never handed over) + fixed background texture + per-pixel jitter.
- Renderer uses kinematics only (mass/damping invisible), so physics_shift is consequence-only
  in pixels too — mirroring the state-space design.

## M3 — latent_model.py

- `LatentAE`: one `NumpyMLP` as D→enc→L→dec→D, trained to reconstruct, **linear bottleneck**
  (so codes are linearly decodable and the probe R² measures real content, not tanh saturation).
- `LatentEnsemble`: bootstrap `(z_t, a_t) → Δz`, Mahalanobis on standardised inputs.
- Both train on **clean ID only**.

## M3.5 — RECON GATE (VOID if fail)

- Baseline-anchored: AE recon must beat the mean-frame baseline by ≥1.5×, AND physics_shift
  recon must stay ≤1.5× the clean recon (else `c_z` would be partly an AE-OOD artefact).
- Result: recon 0.0043 vs baseline 0.0176 (**4.1×**); physics_shift/clean ratio **1.00**.
  **GATE PASS** — the latent result is not VOID.
- Tuning needed to reach a *competent* (not vacuous) encoder: thicker links (arm prominence),
  larger frame stride (velocity visibility), linear bottleneck. This is methodological AE
  quality, decided by recon + probe gates — **not** tuned toward the `c_z` verdict.

## M4/M5 — latent harness, outputs, verdict

- Three detectors only (`u_z`, `d_z`, `c_z`); `r` and true `q` withheld for labels/post-hoc.
- Catastrophe `eps = ||pred − encode(clean render of TRUE next state)||` (privileged);
  `c_z = ||encode(observed next frame) − pred||` (observation-only). Distinct by construction.
- Dual probe (decided when the linear mean sat just below the floor): **linear** probe is the
  laundering **ceiling** (R²=0.25 ≪ 0.9 → not laundered); **nonlinear** probe is the vacuity
  **floor** (R²=0.58 → state present, just nonlinearly coded). q2 (relative elbow angle) and
  velocities are not linear readouts — expected, not a bug.
- `c_z`–`d_z` rank-corr early-kill: ρ=0.13 (≤0.8) → `c_z` IS orthogonal to `d_z`. So the RED
  is **not** "`c_z` is `d_z` in disguise"; it is the stronger result below.

## Verdict — RED at the detector level (attribution provisional)

`c_z` is the mirror image of `r`: it fires on extrapolation (20% missed) and misses
physics_shift (52% missed), Spearman vs `r` ≈ −0.12. The *hypothesised* mechanism: a learned
latent consistency check has no off-support guarantee and collapses toward novelty detection,
so it cannot inherit the part of `r` that mattered (quietness on extrapolation). The
detector-level RED is robust; whether its CAUSE is the deep one (privileged structure needed)
or a representational shortfall is decided by the M-verify check below.

## M-verify — adversarial q̈ (acceleration) recoverability check

Why: a mass shift's footprint is in the **acceleration** (the velocity change over the step).
If `c_z` misses physics_shift only because the AE/observation under-encodes that dimension,
the RED would be a fixable artifact, not the deep boundary. So regress realized
`q̈ = (qd_next − qd)/dt` (clean ID, train/test split) from four predictor sets with a matched
nonlinear probe (linear shown for reference):

| predictor | nonlinear R²(→q̈) | linear R² |
|-----------|-----------------:|----------:|
| frames + a | 0.302 | −0.67 |
| **z + a**  | **0.284** | 0.09 |
| z | 0.305 | 0.08 |
| frames | 0.312 | −0.70 |

Caveat baked into the design: `render.py` back-extrapolates the stacked window at **constant
velocity** (`q − k·stride·dt·qd`), so the window has ~zero within-window curvature; `q̈` enters
only via `(inputs, action) + physics`, hence `a_t` is concatenated to every predictor. The
raw `frames+a` linear probe going strongly negative confirms that high-D probe is
ill-conditioned, so `z+a` (well-conditioned 12-D) is the fair test.

Conclusion: **`which_red = inconclusive_observation_limit`.** `q̈` is only ~0.28–0.31
recoverable from *either* `z` or the raw frames — right at the 0.30 floor. The acceleration the
shift lives in is barely present in the observation itself, so we **cannot** cleanly attribute
the RED to the deep cause yet. This is the stress-test doing its job: the detector RED stands,
but the deep-boundary claim is **not** established.

Recommended next step (not done here — would be a new render/data change): give the window real
curvature (forward multi-step window, or back-integrate with the true dynamics) so `q̈` is
carried, re-verify recoverability, then re-read whether `c_z` still misses physics_shift while
`q̈` is recoverable from `z` (→ deep boundary) or not (→ representational fix).

## LW-09 — curvature output window + pre-registered gates (new commit after a30c142)

The M-verify check left the LW-07/08 result unreadable: the constant-velocity window did not
carry `q̈`, and physics_shift is an acceleration-level consequence, so `c_z` was never given a
fair test. LW-09 fixes the rig and re-decides under pre-registered gates.

**What changed (rig):**
- `obs_next = [f_t, f_{t+1}, f_{t+2}]` is now a **real forward window** simulated under the
  regime physics (nominal for clean, heavy for physics_shift) with zero-order-hold action, so
  the window carries genuine curvature/acceleration. `obs_t = [f_{t-2}, f_{t-1}, f_t]` is a
  **real backward window under NOMINAL physics**, built **once** from the shared input states and
  reused for the matched clean/physics_shift pair (so hidden mass cannot leak into the input).
- New `windows.py`: vectorised RK4 (`rk4_batch`/`qddot_batch`) reading the audited `arm._consts()`,
  guarded by a runtime parity assert vs scalar `arm.step_rk4`. `render.render_window` renders
  explicit poses (no back-extrapolation).
- `eps` now compares the prediction against the **non-jittered regime-specific** `obs_next_true`
  (heavy for physics_shift), never against nominal. Detectors, 5% calibration, and `missed()` are
  unchanged (no threshold tuning).

**GATE B — no hidden-physics leak into `obs_t` (PASS):** AUROC(d_z)=0.500, AUROC(u_z)=0.500,
matched-pair max|Δ|=0.0 (exact, by shared-`obs_t` construction; the gate is the regression guard).

**RECON GATE (on the output window) — PASS:** recon 0.0044 vs mean-frame baseline 0.0176 (**4.0×**);
physics_shift/clean recon ratio **0.99**.

**GATE A — output observation carries `q̈` (FAIL → VOID).** Hard gate: pooled AND physics_shift
`nonlinear R²(obs_next + a → q̈) > 0.60`.

| predictor → realized q̈ | clean | physics_shift | pooled |
|---|---:|---:|---:|
| **obs_next + a** (gate) | 0.318 | 0.335 | **0.326** |
| obs_next | 0.313 | — | — |
| obs_t + a (diagnostic) | 0.316 | — | — |
| z_next + a | 0.383 | — | — |
| z_next | 0.311 | — | — |
| z_t + a | 0.338 | — | — |

Everything clusters at ~0.31–0.38; action and the added curvature buy little, and forward
(`obs_next`) ≈ backward (`obs_t`). In **state space** `q̈ = f(q, q̇, τ)` is 100% determined, so the
~2/3 loss is the 24×24 pixel render + AE bottleneck attenuating the consequence — not physics.
The leading cause is a scale mismatch: the window baseline is `frame_stride·dt = 0.08 s` (needed so
velocity is visible at 24×24), while the target `q̈ = (q̇_next − q̇)/dt` is the **instantaneous**
single-`dt` acceleration; the coarse window cannot pin the fine-scale `q̈`, and higher resolution
would be needed to have both.

**FINAL LABEL: VOID — output observation still does not carry `q̈` (pooled R²=0.33 < 0.60 floor).**
Per the pre-registered rule the detector table is **NOT interpreted** (recorded for the file only:
`c_z` looks low-miss everywhere, i.e. a universal post-transition error detector — but that reading
is not licensed while GATE A fails). The 0.60 bar was **not** softened after seeing 0.33.

This supersedes the LW-07/08 "RED": that result was likewise unreadable, and LW-09 now says so
honestly with the gate that proves it. Recommended next step (a further rig change, out of scope
here): align the `q̈` target to the window scale (or raise render resolution / shrink stride with a
larger image) so the consequence clears the 0.60 bar, then re-read `c_z`.

## LW-10 — window-scale q̈ gate (two new commits after eecf41a)

**Why the target was corrected.** LW-09 gated on the **instantaneous** `q̈ = (q̇_{t+dt} − q̇_t)/dt`
at native `dt = 0.01 s`. But `obs_next` is a rendered window spanning `frame_stride·dt = 0.08 s`; a
multi-frame observation can carry **window-scale** average acceleration, not necessarily the
single-native-`dt` derivative. The LW-09 gate target was mis-specified relative to what the
observation represents. This is a measurement-definition correction, not a threshold change — the
0.60 bar is unchanged. (Committed `8cf3c58` *before* rerunning, with no detector reinterpretation.)

**Corrected target (primary):** `qdd_window = (q̇_{t+S·dt} − q̇_t)/(S·dt)`, `S = frame_stride`.
Diagnostics kept: `qdd_instant = (q̇_{t+dt} − q̇_t)/dt`, pose-curvature
`(q_{t+2S·dt} − 2q_{t+S·dt} + q_t)/(S·dt)²`.

**q̈ recoverability (nonlinear R², clean):**

| predictor | window (primary) | instant (diag) | pose-curv (diag) |
|---|---:|---:|---:|
| obs_next + a | 0.394 | 0.318 | 0.456 |
| z_next + a | 0.530 | 0.383 | 0.524 |
| obs_next | 0.370 | — | — |
| z_next | 0.442 | — | — |

The correction **helped** (window > instant everywhere: z_next+a 0.530 vs 0.383), confirming the
timescale mismatch was real — but did not lift the raw-observation probe to the bar.

**Capacity-matched comparison (PCA→12d on clean-training obs_next, same probe as z_next, 3 seeds):**

| predictor | mean | std | per-seed |
|---|---:|---:|---|
| obs_next + a (PCA-12) | 0.336 | 0.016 | 0.319 / 0.358 / 0.333 |
| z_next + a | 0.520 | 0.008 | 0.530 / 0.519 / 0.510 |

gap = −0.183, margin 0.125, probe-noise floor 0.024 → **consequence_reached_latent = True** (z is
*richer* than a 12-d linear projection of the pixels, as expected for a nonlinear AE).

**GATE A (raw `obs_next+a → qdd_window` > 0.60):** clean 0.394 / physics_shift 0.422 / pooled 0.408
→ **FAIL**. **GATE B (no input leak):** AUROC d_z/u_z = 0.500/0.500, matched-pair Δ = 0.0 → **PASS**.
**RECON (output window):** 4.0× over baseline, shift/clean ratio 0.99 → **PASS**.

**FINAL LABEL: VOID-FIDELITY** — even *window-scale* acceleration is not recoverable from the
observation to the 0.60 bar (raw obs_next+a ≈ 0.41). Per the pre-registered rule the detector table
is **NOT interpreted** (saved for record only); the 0.60 bar was not softened. A render/window
fidelity change is **justified but explicitly not performed in this task** (forbidden: no resolution
/ AE / window-span change).

**Honest caveat surfaced by this run:** the *latent* already carries window-scale q̈ at ≈0.52
(z_next+a 0.530; capacity-matched z 0.520 ≫ PCA-obs 0.336), i.e. the consequence demonstrably
reaches z. The binding failure is the **observation-side gate**: the 24×24 render + a coarse
0.08 s window do not let even a fair probe pull window-scale q̈ from the *pixels* up to 0.60. So the
next move is observation fidelity (higher resolution or a q̈-scale window), after which `c_z` can
finally be read. Not done here by task constraint.

## LW-11 — latent-surface Gate A + matched-pair delta gate, then read once (two new commits after 151aad0)

**Terminal step of latent v0.** Two commits: `33faabb` (pre-registration: Gate A surface + delta
gate, code only, still 24×24, no detector reinterpretation) then the 48×48 fidelity change + rerun.

**Why the surface was corrected.** LW-10 gated Gate A on **raw `obs_next` pixels** as the fairness
upper bound, but its own capacity-matched probe showed `z_next+a` (0.520) **exceeded** PCA-matched
`obs_next+a` (0.336): the nonlinear AE *concentrates* the consequence, so raw pixels were the wrong
surface. LW-11 moves the primary Gate A to the representation `c_z` actually consumes (`z_next`),
adds a **required matched-pair delta gate** (the literal clean→shift difference `c_z` keys on), makes
the **smallest** observation-fidelity change (`img_size 24 → 48`), and reads the verdict **once**. The
0.60 bar, Gate B, recon gate, detector semantics, and 5% calibration are all unchanged.

**Pre-registered gate flow (read `c_z` only if all pass):** recon → Gate B (AUROC d_z,u_z ∈
[0.45,0.55]) → Gate A (`z_next+a → qdd_window` > 0.60 on physics_shift AND pooled) → delta gate
(`Δz_next → Δqdd_window` mean R² > 0.30, **paired nuisance**) → capacity (`z_next+a ≥` matched
`obs_next+a − max(margin, noise)`) → `c_z` catches physics_shift? Labels: VOID / LEAK-INVALID /
VOID-FIDELITY / VOID-CONSEQUENCE-NOT-ENCODED / AE-LIMIT / DEEP RED / GREEN. Terminal constraint: a
single 48px run; if Gate A fails the terminal label is **VOID-FIDELITY** (no further escalation).

**Delta gate (new), PAIRED nuisance.** `Δz_next = z_shift − z_clean` isolates the hidden-mass
consequence, so each clean/shift pair is rendered from the **same** per-pair jitter seed (texture
already fixed) — the poses differ (nominal vs heavy), the noise does not. Dedicated renders, separate
from the packs' detector renders (untouched). Train deltas from `train_ae` (nominal vs heavy), test
deltas from the packs' matched poses; 3-seed nonlinear probe + per-joint + Spearman of |Δ|; a
non-jittered delta reported as the cleanest-possible diagnostic.

**48×48 full-run result.**

GATE B (no input leak): AUROC d_z/u_z = 0.500/0.500, matched-pair max|Δ| = 0.0 → **PASS**.
RECON (output window): 0.00516 vs mean-frame baseline 0.01840 (**3.6×**), shift/clean ratio 0.99 → **PASS**.

GATE A — `z_next+a → qdd_window` (hard gate, > 0.60 on physics_shift AND pooled):

| surface → qdd_window (nonlinear R²) | clean | physics_shift | pooled |
|---|---:|---:|---:|
| **z_next + a** (gate) | 0.444 | 0.406 | **0.425** |
| raw obs_next + a (diag) | 0.434 | 0.483 | 0.459 |

→ **FAIL** (pooled 0.42, physics_shift 0.41 < 0.60). At 48px the z-surface and the raw-pixel surface
both sit ~0.42–0.46 — the surface correction no longer separates them, and neither clears the bar.

DELTA GATE — `Δz_next → Δqdd_window` (mean R² > 0.30, paired nuisance): mean **−0.175** ± 0.016
(per-seed −0.152 / −0.185 / −0.187), per-joint −0.165 / −0.14, Spearman(|Δ|) 0.443, non-jittered
−0.162 → **FAIL**. The matched clean→shift consequence does **not** linearly/nonlinearly survive into
`Δz_next` at this fidelity (negative R² = worse than predicting the mean): the literal signal `c_z`
must key on is absent from the latent difference.

Capacity-matched (PCA→12d, 3 seeds): obs_next+a 0.345 ± 0.057 vs z_next+a 0.429 ± 0.012, gap −0.084
→ consequence_reached_latent = True (kept for record; not load-bearing once Gate A fails).
Laundering: linear R²(z→q,q̇) 0.236, nonlinear 0.566 (lossy-but-informative band, unchanged story).

**FINAL LABEL: VOID-FIDELITY** (Gate A fails first in the order). The detector table is **NOT
interpreted** (saved for record only); the 0.60 bar was not softened. Per the terminal constraint the
single 48px run is the read: no sixth gate, no further in-task fidelity escalation.

**What 48px taught us (honest).** The minimal fidelity bump did **not** clear the bar; if anything it
*lowered* the z-surface number vs the 24px run (z_next+a window 0.444 vs LW-10's 0.530) while the test
set grew. So the LW-10 hope — "the latent already carries window-scale q̈ at ≈0.52, only the pixels
lag" — does **not** robustly hold once the surface is read on a larger test set at higher resolution
and the *matched-pair delta* (not the absolute recoverability) is required: the consequence does not
reliably reach the latent **difference**. This is the terminal verdict of latent v0
(`VOID-FIDELITY / consequence-not-encoded`: formal label `VOID-FIDELITY`, with the binding evidence
the negative matched-pair delta): with this **reconstruction-trained AE** and this observation/window
family, the physics-consequence channel cannot be cleanly read from the learned latent, so `c_z`
cannot be fairly evaluated against `r`.

**Scope limit (read before citing).** This is **not** a general claim that learned latents cannot
support physics auditability — it is a bounded result about this reconstruction-trained AE and this
observation/window family (48×48 grayscale; frame interval 0.08s / `frame_stride=8`). The next
variable to change is the **representation objective** (predictive / JEPA-style / audit-preserving
latent objective instead of reconstruction); that changes the question and needs a new charter. Full
closeout in `FINDINGS_latent_v0.md`.

## Notes / open threads

- Seeds fixed; full run ~1.5–2 min at 24px, ~7 min at 48px on CPU. `--quick` for smoke.
- Possible follow-ups (not done; would need a new charter): multi-step latent rollouts for `c_z`,
  or an explicitly physics-structured latent — both change the question and are out of scope here.
