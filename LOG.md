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

## Notes / open threads

- Seeds fixed; full run ~1.5–2 min on CPU. `--quick` for smoke.
- Possible follow-ups (not done; would need a new charter): multi-step latent rollouts for `c_z`,
  or an explicitly physics-structured latent — both change the question and are out of scope here.
