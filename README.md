# Latent_Wedge

A lean, self-contained experiment that asks **one** question:

> When analytic state `(q, q̇, τ)` and the physics equation are **hidden** behind an
> autoencoder, can a learned latent transition-consistency signal `c_z` recover the
> analytic physics-residual `r`'s job — catching a *consequence-only* physics shift
> that latent input-detectors are blind to — **without** the latent secretly
> recovering the privileged state?

If yes, physics-auditability transfers to latent (V-JEPA/Dreamer-shaped) world models.
If no, physics-violation detection needs privileged state — a hard limit worth naming.

The full spec is `Latent_Wedge_CHARTER.md` (one page; the single source of truth).
This folder deliberately carries **only** experiment-level discipline (this README
ledger + `LOG.md` + sanity tests). No program-level control plane is replicated here.

## What is reused vs new

Ported **verbatim** from the validated toy-wedge v3 tree (phase-close commit `72866ed`;
the `a86ee25` stamp in the old results folder was a v2 label error, corrected here):

- `arm.py` — 2-link Spong dynamics, ground-truth world *and* nominal model, `torque_residual`.
- `data.py` — rollout / dataset construction (ID vs OOD by mass/damping multiplier).
- `config.py` — the run dataclass (latent/render fields are additive; base fields unchanged).
- `tests/test_invariants.py` — the four physics audits.

Ported with **numpy-only swaps** (no sklearn / scipy):

- `model.py` — bootstrap dynamics ensemble; member is `NumpyMLP` instead of `MLPRegressor`.
- `experiment_failure_modes.py` — the v3 state-space harness; `roc_auc_score → metrics_np.roc_auc`.
  This file **is** the M1 parity gate.

New (the minimal added surface):

- `numpy_mlp.py` — one stabilised MLP backbone (Adam, He/Xavier init, grad-clip, weight
  decay, finiteness guards) reused for the ensemble member, the AE, and the latent dynamics.
  Ports the recipe proven in `World_Models/src/world_models/models/mlp.py` (003B).
- `metrics_np.py` — pure-numpy `roc_auc` (tie-aware Mann-Whitney) and `spearman`.
- `render.py` — forward-kinematics grayscale frames; stacked window (`q̇` inferable, not
  handed over) + fixed texture + pixel jitter (anti-laundering nuisance).
- `latent_model.py` — `LatentAE` (linear-bottleneck MLP autoencoder) + `LatentEnsemble`
  (bootstrap latent dynamics `(z_t, a_t) → z_{t+1}`).
- `experiment_latent_failure_modes.py` — the v3 harness in latent space (the wedge).

## How to run

```bash
python tests/test_invariants.py            # physics self-audit (must pass first)
python tests/test_numpy_swaps.py           # roc_auc / spearman / MLP / AE round-trip guards
python experiment_failure_modes.py         # M1 PARITY GATE (state space) — must PASS before latent
python experiment_latent_failure_modes.py  # the latent wedge (writes results/*_latent/)
# add --quick to either experiment for a fast smoke run
```

Dependencies: `numpy`, `matplotlib` only (see `requirements.txt`).

## Gated pipeline (each gate blocks the next)

```
physics audit ─▶ M1 PARITY GATE ─▶ render+AE+ensemble ─▶ M3.5 RECON GATE ─▶ latent table ─▶ verdict
                (r reproduces v3?)                       (AE competent?)     (c_z vs r's column)
                   STOP if FAIL                            VOID if FAIL
```

## Ledger (claims, status, evidence)

| id | claim | status | evidence |
|----|-------|--------|----------|
| LW-01 | Physics substrate ported verbatim is correct | **PASS** | `tests/test_invariants.py`: energy drift 2.6e-7, damping dissipates, grav=∇V 2e-9, residual≈0 |
| LW-02 | numpy swaps (roc_auc/spearman/MLP) are correct | **PASS** | `tests/test_numpy_swaps.py` ALL PASS |
| LW-03 | **M1 parity**: numpy harness reproduces v3 `r` column within ±8 pts + crossover | **PASS** | r: extrap 57.8% (v3 57.4%), physics_shift 0.0% (0.0%), id_holdout 58.4% (62.6%); crossover intact |
| LW-04 | **M3.5 recon gate**: AE beats mean-frame baseline & physics_shift recon comparable to clean | **PASS** | recon 0.0043 vs baseline 0.0176 (4.1×); physics_shift/clean ratio 1.00 |
| LW-05 | Latent not laundered (linear R²(z→q,q̇) < 0.9) | **PASS** | linear R² = 0.25 |
| LW-06 | Latent not vacuous (nonlinear probe recovers state) | **PASS** | nonlinear R² = 0.58 (state present, nonlinearly coded) |
| LW-07 | **Hinge**: `c_z` reproduces `r`'s whole column | **RED** (detector level) | mirror of `r`: fires on extrapolation (20% missed), misses physics_shift (52% missed), Spearman vs `r` = −0.12 |
| LW-08 | **q̈ adversarial check**: is the deep boundary established (mass-shift signature usable in `z`)? | **INCONCLUSIVE** | realized `q̈` only ~0.28–0.31 recoverable from `z+a` AND raw `frames+a` (floor 0.30) → attribution observation-limited; `which_red = inconclusive_observation_limit` |

## Latest result (full run; see `results/<latest>_latent/`)

Missed-catastrophe (lower = safer; thresholds calibrated on clean at 5% total false alarm):

| regime | #cat | u_z | d_z | **c_z** | PORTFOLIO |
|--------|-----:|----:|----:|--------:|----------:|
| clean | 400 | 81.5% | 95.8% | 50.5% | 68.8% |
| extrapolation | 3701 | 28.1% | 41.6% | **20.0%** | 12.3% |
| physics_shift | 393 | 80.2% | 97.2% | **51.9%** | 69.7% |
| id_holdout | 274 | 72.3% | 92.0% | 42.7% | 59.1% |

Laundering / vacuity: linear R²(z→q,q̇)=**0.25** (≪ 0.9 ceiling → not laundered);
nonlinear R²=**0.58** (> 0.3 floor → not vacuous). Latent is *informative-but-nonlinear*.
Spearman `c_z` vs withheld `r` = **−0.12**; Spearman `c_z` vs `d_z` = **0.13**.

Adversarial q̈ check (acceleration recoverability, the mass-shift's footprint):
realized `q̈` is only **~0.28–0.31** recoverable (nonlinear R²) from `z+a`, raw `frames+a`,
`z`, and `frames` alike — right at the 0.30 floor. `which_red = inconclusive_observation_limit`.

### Verdict: RED at the detector level — deep-boundary attribution PROVISIONAL

`c_z` does **not** reproduce `r`'s column. It is the mirror image of `r`:

- `r` (analytic) catches physics_shift (0% missed) and is quiet on extrapolation (57% missed).
- `c_z` (learned latent) **fires on extrapolation** (20% missed — it catches the regime where
  *inputs* are novel) and **misses physics_shift** (52% missed). It correlates ≈0 with `r`.

This detector-level RED is robust, and it is **not** "`c_z` is `d_z` in disguise" (`c_z`–`d_z`
ρ=0.13). The *interpretation* — that a learned consistency check has no off-support guarantee
and collapses toward novelty detection — is the leading hypothesis, **but the adversarial q̈
check did not confirm it**: the acceleration the shift lives in is barely present in the
observation itself (~0.30 from even the raw frames), because `render.py` back-extrapolates the
stacked window at constant velocity (no within-window curvature). So part of `c_z`'s miss on
physics_shift may be a representational shortfall rather than the deep limit. **The deep-boundary
claim is not yet established.** Next step (a new render/data change, not done here): give the
window real curvature so `q̈` is carried, then re-read whether `c_z` still misses physics_shift
while `q̈` is recoverable from `z` (→ deep boundary) or not (→ representational fix).

### Boundary paragraph (required output)

This is **post-transition detection, not pre-action prevention**: every detector here fires
*after* a step has been observed (`c_z` needs `image_{t+1}`), so it can audit a world model's
predictions but cannot stop a catastrophe before it happens. The wedge's robust finding is that
in this setup, a latent transition-consistency signal `c_z` does not reproduce the analytic
residual `r`'s column — it fires on novel inputs and misses the consequence-only physics shift,
the opposite of `r`. The deeper claim that this is *fundamental* to learned latents (rather than
fixable by a richer observation that carries acceleration) is the open question the q̈ check
flags: with a constant-velocity window the shift's signature is only ~0.30-recoverable even from
raw pixels, so the boundary is *indicated but not proven*. Either way, physics-auditability does
**not** transfer for free to latent world models on this evidence; whether it can transfer with
a better observation is the next experiment.
