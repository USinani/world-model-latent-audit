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
  the curvature output window and pre-registered GATE A (obs carries q̈) / GATE B (no input leak);
  LW-11 moves GATE A to the `z_next` surface and adds a matched-pair delta-consequence gate.

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
physics audit ─▶ M1 PARITY GATE ─▶ render+AE+ensemble ─▶ RECON ─▶ GATE B ─▶ GATE A ─▶ DELTA GATE ─▶ capacity ─▶ c_z?
                (r reproduces v3?)                       (AE ok?) (no leak) (z_next   (Δz_next     (reached    │
                   STOP if FAIL          VOID if FAIL      VOID    LEAK-     carries   carries Δq̈?) latent?)   │
                                                                  INVALID    q̈?)       VOID-CONSEQ  AE-LIMIT    ▼
                                                                            VOID-FID   -NOT-ENCODED            DEEP RED / GREEN
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
| LW-10 | **Window-scale q̈ gate**: correct Gate A target to the observation timescale (`qdd_window` over `frame_stride·dt`), then re-read | **VOID-FIDELITY** | Correction helped (window z_next+a 0.53 vs instant 0.38) and the latent carries it (capacity-matched z 0.52 ≫ PCA-obs 0.34, consequence reached latent), but **GATE A still FAIL** on the raw-pixel surface: raw `obs_next+a → qdd_window` = 0.39/0.42/0.41 (clean/shift/pooled) < 0.60. GATE B PASS, RECON PASS. Showed raw pixels were the wrong fairness surface (z > matched obs) → motivated LW-11 |
| LW-11 | **Latent-surface Gate A + matched-pair delta gate** (terminal, 48px, read once): Gate A on `z_next+a → qdd_window`, required `Δz_next → Δqdd_window` > 0.30 (paired nuisance), `img_size 24→48` | **VOID-FIDELITY** | At 48px the z-surface no longer beats raw pixels and **neither clears the bar**: `z_next+a → qdd_window` = 0.444/0.406/0.425 (clean/shift/pooled) < 0.60 → GATE A FAIL. DELTA gate FAIL (mean R² = **−0.175**: the matched clean→shift consequence is absent from `Δz_next`). GATE B PASS (0.500/0.500), RECON PASS (3.6×). Detector table not interpreted; 0.60 not softened. **Terminal verdict of latent v0**: with this observation/AE/window family the physics-consequence channel cannot be cleanly read from the learned latent, so `c_z` cannot be fairly tested against `r` |

## Latest result — LW-11 (terminal, 48px full run; see `results/<latest>_latent/`)

LW-11 is the **terminal step of latent v0**. It moves Gate A to the surface `c_z` actually consumes
(`z_next`, not raw pixels — LW-10 showed the AE *concentrates* the consequence so z exceeded matched
obs), adds a **required matched-pair delta gate** (`Δz_next → Δqdd_window` with **paired nuisance** so
the delta isolates the hidden-mass consequence, not render noise), makes the **smallest** fidelity
change (`img_size 24 → 48`), and reads the verdict **once**. The 0.60 bar, Gate B, recon, detector
semantics, and 5% calibration are unchanged. Terminal constraint: a single 48px run; if Gate A fails
the label is **VOID-FIDELITY** with no further escalation.

| gate | requirement | result | status |
|------|-------------|--------|:------:|
| GATE B (no input leak) | AUROC(d_z),(u_z) clean-vs-shift ∈ [0.45, 0.55] | 0.500 / 0.500; matched-pair Δ = 0.0 | **PASS** |
| RECON (output window) | recon ≥1.5× over mean-frame; shift ≤1.5× clean | 3.6× over baseline; shift/clean ratio 0.99 | **PASS** |
| **GATE A** (z_next carries `qdd_window`) | pooled **and** physics_shift `R²(z_next+a → qdd_window)` > 0.60 | clean 0.444 / physics_shift 0.406 / **pooled 0.425** | **FAIL** |
| **DELTA GATE** (Δ consequence in latent) | mean `R²(Δz_next → Δqdd_window)` > 0.30 (paired nuisance) | mean **−0.175** ± 0.016; non-jittered −0.162 | **FAIL** |

q̈ recoverability (nonlinear R², clean) — surface comparison at 48px:

| predictor | window (primary) | instant (diag) | pose-curv (diag) |
|---|---:|---:|---:|
| z_next + a (gate surface) | 0.444 | 0.277 | 0.392 |
| obs_next + a (diag) | 0.434 | 0.400 | 0.503 |
| obs_next | 0.446 | — | — |
| z_next | 0.365 | — | — |

Capacity-matched 3-seed `qdd_window` (PCA→12d obs vs latent): obs_next+a **0.345 ± 0.057**,
z_next+a **0.429 ± 0.012**; gap −0.084 → consequence_reached_latent = True (kept for record only).
Laundering unchanged in story: linear R²(z→q,q̇) 0.236, nonlinear 0.566 (lossy-but-informative).

### Final label: VOID-FIDELITY — even on the latent surface the consequence does not clear the bar

The LW-10 surface correction does **not** survive contact with a larger test set at higher
resolution. At 48px the z-surface (`z_next+a → qdd_window` = 0.42 pooled) **no longer beats** the raw
pixels (0.46 pooled) and **neither clears 0.60**. More decisively, the **matched-pair delta gate
fails hard**: `Δz_next → Δqdd_window` mean R² = **−0.175** (worse than predicting the mean), i.e. the
literal clean→shift difference `c_z` keys on is **absent** from the latent difference, even with
paired nuisance removing the render-noise confound. So the LW-10 hope ("the latent already carries the
consequence at ≈0.52, only the pixels lag") does not robustly hold once the *required* signal — the
matched delta — is read.

Per the pre-registered rule, with GATE A failing the **detector table is NOT interpreted** (saved in
`metrics.json` for the record only). The 0.60 bar was **not** softened, and the single 48px run is the
terminal read by constraint (no sixth gate, no further in-task fidelity escalation).

### Boundary paragraph (required output)

The terminal, honest result of latent v0 is that **with this observation / autoencoder / window
family, the physics-consequence channel cannot be cleanly read from the learned latent** — so a
learned transition-consistency signal `c_z` cannot be fairly tested for reproducing the analytic
residual `r`. The minimal fidelity bump (24→48px) did not rescue it, and the matched-pair delta — the
sharpest fair form of the question, isolating the hidden-mass consequence from render noise — comes
back **negative**: the consequence does not reach the latent *difference*. The gates did their job:
they refused to let an unreadable rig masquerade as a GREEN, a DEEP RED, or an AE-LIMIT. Any next move
(a different observation/AE/window, or an explicitly physics-structured latent) **changes the
question** and requires a new charter — it is out of scope for latent v0.
