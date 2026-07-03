# World-Model Latent Audit — charter & build spec (the one spec; supersedes prior packets)

## Hinge (one question)

> When analytic state `(q, q̇, τ)` and the physics equation are HIDDEN, can a learned
> latent transition-consistency signal `c_z` recover the physics-residual `r`'s job —
> catching a consequence-only physics shift that latent input-detectors are blind to —
> **without** the latent secretly recovering the privileged state?

If yes: physics-auditability transfers to V-JEPA/Dreamer-shaped latent world models.
If no: physics-violation detection needs privileged state — a hard limit worth naming.

## The acceptance gate that matters (do not soften)

`c_z` is a genuine `r`-substitute ONLY if it reproduces `r`'s WHOLE v3 column:

1. **catches physics_shift** (low missed-catastrophe where `u_z`/`d_z` are blind), AND
2. **stays QUIET on extrapolation** (high missed-catastrophe there, like `r`'s 57%),
   so it is not just a second input-OOD detector redundant with `d_z`, AND
3. **correlates with the withheld analytic `r`** (post-hoc Spearman).

Failing #2 or #3 means `c_z` is `d_z` in disguise — that is a RED result.

## The anti-laundering requirement (the packet's blind spot)

A 2-link single frame is a 2-DOF manifold; a small AE will invert it to true state and
`c_z` becomes the analytic residual in disguise — making a green result uninformative
about the hard case. Therefore:

- Observe a **stacked 2–4 frame window** (velocity inferable from frames; `q̇` stays
  unprivileged), plus **mild visual nuisance** (light background texture / pixel jitter)
  so the AE cannot trivially invert to clean state.
- **Report a linear-probe R²** fitting `z → (q, q̇)`. If R² ≈ 1 the latent *is* the
  state; a green `c_z` then only proves the easy case — say so. The informative regime
  is `c_z` working *while* the latent is lossy/non-invertible. **This R² is a required
  reported number.**

## Build (minimal new surface — reuse the proven wedge)

Reuse from the toy wedge: `arm.py`, `data.py` (ground truth), the `model.py` ensemble
pattern, and the `experiment_failure_modes.py` harness/metrics.
Add only:
- `render.py` — 2-link state → small grayscale frame(s); stacked-window + nuisance.
- `latent_model.py` — AE (image→`z`), ensemble latent dynamics `(z_t, a_t)→z_{t+1}`.
- `experiment_latent_failure_modes.py` — the v3 harness in latent space.

## Detectors (exactly three; no analytic `r` inside any detector)

```
u_z = latent ensemble disagreement (input-based)
d_z = Mahalanobis distance of (z_t, a_t)   (input-based)
c_z = || encoder(image_{t+1}) - predicted_z_{t+1} ||   (transition consistency)
```
Withheld for labels/post-hoc only: true `q`, analytic `r`. NEVER a detector input.

## Regimes (identical to v3)

```
clean         : ID frames/actions, nominal physics            (calibration)
extrapolation : wide states/actions, NOMINAL physics          -> u_z/d_z catch, c_z QUIET
physics_shift : same frames/actions as clean, HEAVY world     -> u_z/d_z blind, c_z catches
id_holdout    : fresh ID, nominal physics, model still wrong   -> expect all weak (the hole)
```
Calibrate all thresholds on clean at 5% total false alarm (jointly, for any OR fusion).

## Forbidden (until `c_z` lands a number)

No MPC, no controller, no contact, no extra detector beyond `u_z/d_z/c_z`, no
VAE/diffusion/transformer expansion, no "agent" language, and **no program-level control
plane replicated inside this folder** (one LOG + one README ledger + sanity tests only).

## Required outputs

1. One table: missed-catastrophe by regime × detector.
2. One plot: detector score distributions, clean vs physics_shift.
3. AE reconstruction-quality sanity check.
4. **Linear-probe R² (`z → q, q̇`)** — the laundering check.
5. One boundary paragraph: post-transition detection, not pre-action prevention.

## Green / red

- **GREEN:** `c_z` catches physics_shift, stays quiet on extrapolation, correlates with
  withheld `r`, AND the latent is meaningfully lossy (R² < ~0.9). Physics-auditability
  transfers to latent world models.
- **RED:** `c_z` blind to physics_shift, OR fires on extrapolation too (just `d_z`), OR
  only works because the latent recovered true state (R² ≈ 1). Names the hard boundary.

## Scan-first (≤45 min, then stop)

Before writing code, extract from last week's World_Models build only: (a) documented
limitations as "avoid" bullets, (b) reusable render/AE/latent-dynamics code, (c) why it
stalled. Write 5 bullets max. If it becomes an open-ended audit, the trap has reopened.
