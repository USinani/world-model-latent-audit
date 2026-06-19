# Latent_Wedge — build log

Chronological, gate-by-gate. One log, no control-plane sprawl.

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

## Notes / open threads

- Seeds fixed; full run ~1.5–2 min on CPU. `--quick` for smoke.
- Possible follow-ups (not done; would need a new charter): multi-step latent rollouts for `c_z`,
  or an explicitly physics-structured latent — both change the question and are out of scope here.
