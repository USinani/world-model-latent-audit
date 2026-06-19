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
- `render.py` — forward-kinematics grayscale frames; `render_window` rasterises explicit
  stacked poses + fixed texture + pixel jitter (anti-laundering nuisance).
- `windows.py` (LW-09) — vectorised RK4 (`rk4_batch`/`qddot_batch`, anchored to `arm._consts()`
  with a runtime parity assert vs `arm.step_rk4`) building **real** forward/backward stacked
  windows, so the output observation carries genuine curvature/acceleration.
- `latent_model.py` — `LatentAE` (linear-bottleneck MLP autoencoder) + `LatentEnsemble`
  (bootstrap latent dynamics `(z_t, a_t) → z_{t+1}`).
- `experiment_latent_failure_modes.py` — the v3 harness in latent space (the wedge); LW-09 adds
  the curvature output window and pre-registered GATE A (obs carries q̈) / GATE B (no input leak).

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
physics audit ─▶ M1 PARITY GATE ─▶ render+AE+ensemble ─▶ RECON GATE ─▶ GATE B (no input leak)
                (r reproduces v3?)                       (AE competent?)   ─▶ GATE A (obs carries q̈)
                   STOP if FAIL          VOID if FAIL        VOID if FAIL    ─▶ detector table ─▶ label
                                                                              (VOID/AE-LIMIT/DEEP RED/GREEN)
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
| LW-07 | **Hinge**: `c_z` reproduces `r`'s whole column | **VOID/unreadable** | observation under-encoded `q̈` (constant-velocity window); test was never fair — superseded by LW-09 |
| LW-08 | **q̈ adversarial check**: is the deep boundary established (mass-shift signature usable in `z`)? | **INCONCLUSIVE** | realized `q̈` only ~0.28–0.31 recoverable from `z+a` AND raw `frames+a` (floor 0.30) → observation-limited; motivated the LW-09 rig fix |
| LW-09 | **Curvature window + pre-registered gates**: make the latent audit interpretable | **VOID** | GATE B PASS (AUROC d_z/u_z = 0.500, leak Δ=0) and RECON PASS (4.0×, ratio 0.99), but **GATE A FAIL**: real curvature output window still yields only `q̈` R²≈0.33 from `obs_next+a` (and ~0.31–0.38 from every predictor incl. `z_next+a`) vs 0.60 floor. State-space `q̈` is 100% determined, so the loss is the 24×24 render + AE bottleneck. Detector table not interpreted; 0.60 not softened |
| LW-10 | **Window-scale q̈ gate**: correct Gate A target to the observation timescale (`qdd_window` over `frame_stride·dt`), then re-read | **VOID-FIDELITY** | Correction helped (window z_next+a 0.53 vs instant 0.38) and the latent carries it (capacity-matched z 0.52 ≫ PCA-obs 0.34, consequence reached latent), but **GATE A still FAIL**: raw `obs_next+a → qdd_window` = 0.39/0.42/0.41 (clean/shift/pooled) < 0.60. GATE B PASS (0.500/0.500), RECON PASS (4.0×). Detector table not interpreted; 0.60 unchanged. Observation-side fidelity (resolution / q̈-scale window) is the justified next step — not performed here by task constraint |

## Latest result — LW-10 (full run; see `results/<latest>_latent/`)

LW-10 corrects the Gate A target from the native single-`dt` derivative to **window-scale**
acceleration `qdd_window = (q̇_{t+S·dt} − q̇_t)/(S·dt)` (`S = frame_stride`), the timescale the
rendered window actually represents. Window semantics, Gate B, recon, detector semantics, and the
0.60 bar are unchanged. AE-LIMIT vs DEEP RED is decided on a **capacity-matched** probe (PCA→12d on
clean-training `obs_next`, same nonlinear probe as `z_next`, 3 seeds), never the underpowered raw
high-dim obs probe.

| gate | requirement | result | status |
|------|-------------|--------|:------:|
| GATE B (no input leak) | AUROC(d_z),(u_z) clean-vs-shift ∈ [0.45, 0.55] | 0.500 / 0.500; matched-pair Δ = 0.0 | **PASS** |
| RECON (output window) | recon ≥1.5× over mean-frame; shift ≤1.5× clean | 4.0× over baseline; shift/clean ratio 0.99 | **PASS** |
| **GATE A** (obs carries `qdd_window`) | pooled **and** physics_shift `R²(obs_next+a → qdd_window)` > 0.60 | clean 0.394 / physics_shift 0.422 / **pooled 0.408** | **FAIL** |

q̈ recoverability (nonlinear R², clean):

| predictor | window (primary) | instant (diag) | pose-curv (diag) |
|---|---:|---:|---:|
| obs_next + a | 0.394 | 0.318 | 0.456 |
| z_next + a | 0.530 | 0.383 | 0.524 |
| obs_next | 0.370 | — | — |
| z_next | 0.442 | — | — |

Capacity-matched 3-seed `qdd_window` (PCA→12d obs vs latent): obs_next+a **0.336 ± 0.016**,
z_next+a **0.520 ± 0.008**; gap −0.183, noise floor 0.024 → **consequence_reached_latent = True**.

### Final label: VOID-FIDELITY — even window-scale q̈ is not recoverable to the bar

The correction **helped** — window-scale beats instantaneous on every predictor (`z_next+a` 0.530 vs
0.383) — confirming the LW-09 timescale mismatch was real. And the **latent already carries** the
window-scale consequence (`z_next+a` 0.530; capacity-matched z 0.520 ≫ PCA-obs 0.336). But the
binding gate is the **observation-side** probe: raw `obs_next+a → qdd_window` ≈ 0.41 (clean/shift/
pooled 0.39/0.42/0.41) stays below the unchanged 0.60 bar. The 24×24 render + a coarse 0.08 s window
do not let even a fair probe pull window-scale q̈ from the *pixels* up to 0.60.

Per the pre-registered rule, with GATE A failing the **detector table is NOT interpreted** (saved in
`metrics.json` for the record only). The 0.60 bar was **not** softened. The justified next step is an
**observation-fidelity** change (higher render resolution, or a q̈-scale window) — explicitly **not
performed in this task** by constraint — after which `c_z` can finally be read.

### Boundary paragraph (required output)

The honest result of this phase is that **the rig, not the detector, is the current limit**: a
learned latent transition-consistency signal `c_z` cannot be fairly tested for reproducing the
analytic residual `r` until the observation actually carries the consequence, and at this pixel +
autoencoder fidelity the acceleration footprint of a mass change survives only ~1/3 — below the
pre-registered interpretability bar. Fixing the window to carry curvature was **necessary but not
sufficient**; a richer observation (higher render resolution, or a `q̈`-scale window) is required
before any GREEN / DEEP RED / AE-LIMIT verdict on latent physics-auditability can be earned. The
gates did their job: they refused to let an unreadable rig masquerade as a scientific boundary.
