# Findings — latent v0 (closeout)

**Bounded result:** A reconstruction-trained autoencoder latent did not preserve the
hidden-mass consequence delta required for residual-like self-verification.

**Formal label:** `VOID-FIDELITY` (terminal for latent v0).
**Scientific interpretation:** `VOID-FIDELITY / consequence-not-encoded` — the formal label is
exactly `VOID-FIDELITY`, but the binding evidence is the **negative matched-pair delta**, so the
result is read as "the physics consequence was not encoded into the latent," not merely "the
observation was too coarse."

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

This is **not** a general claim that learned latents cannot support physics auditability. It is a
bounded result about **this reconstruction-trained AE** and **this observation/window family**
(48×48 grayscale stacked windows; frame interval 0.08s / `frame_stride=8`; ZOH action). The training
objective was image reconstruction; nothing in that objective rewards preserving the second-order
mass-consequence delta, and the result is that the delta was not preserved.

## Next variable (named, not implemented)

The next variable to change is the **representation objective**: a predictive / JEPA-style /
audit-preserving latent objective instead of reconstruction, so the latent is trained to keep the
transition consequence rather than only to reconstruct frames. This is out of scope for latent v0 and
requires a new charter — it changes the question, not just the rig.
