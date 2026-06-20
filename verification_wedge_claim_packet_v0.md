# Verification Wedge — claim packet v0

*A controlled wedge for testing whether learned world-model latents preserve audit-relevant physical
consequences under matched hidden-cause shift.*

This is a critique artifact, not a paper. It is deliberately rough, fork-neutral (it does not assume
the result is novel-as-finding versus valuable-as-method), and bounded to two banked results. It is
meant to be attacked. Every number traces to a source file in this repository (pointers in §7).

---

## Abstract

I present a controlled wedge for testing whether learned world-model latents preserve audit-relevant
physical consequences under matched hidden-cause shift, and a first result in which a
reconstruction-trained latent does not preserve a consequence that is well-posed in the auditor and
recoverable from rendered observations at matched capacity.

The wedge uses a 2-link arm with an analytic physics auditor. A hidden cause (a mass change) is
applied so that two observation windows share the same input but differ only in their physical
*consequence* (the resulting acceleration). The question is whether a learned latent keeps that
consequence difference, since any residual-style self-verification in latent space depends on it.

---

## 1. Why this exists

Self-verifying world models are often proposed as a route to auditable agents: if a model can flag its
own physics violations, downstream control can trust it more. That promise rests on an unstated
assumption — that the learned latent actually *contains* the physical consequence a verifier would
check.

The wedge isolates that assumption. It holds the input fixed and changes only the hidden cause, so the
only thing distinguishing the two futures is the consequence. If a learned latent drops that
consequence, a latent-space verifier cannot recover it regardless of how the verifier is built. The
wedge is intentionally tiny (a 2-link arm) so the auditor is analytic and every claim can be checked
against ground-truth dynamics.

---

## 2. Finding 1 — verifier channels fail differently

In the state-space setting (privileged analytic state available), the numpy harness reproduces the
validated v3 failure-mode table within tolerance, including the channel **crossover**: the analytic
residual `r` catches the matched law/consequence violation (physics_shift: 0.0% missed) while the
input-novelty channels (ensemble disagreement `u`, density `d`) catch input-visible novelty
(extrapolation), and vice versa.

Bounded statement: *for the tested failure modes, verifier channels fail differently — input-novelty
channels cover input-visible novelty, while the analytic residual covers matched law/consequence
violation. The portfolio covers those two causes but leaves confident in-distribution error open.*
This is not a claim that the portfolio is necessary and sufficient.

Parity evidence (state space): `r` missed-catastrophe extrapolation 57.8% (v3 57.4%), physics_shift
0.0% (v3 0.0%), id_holdout 58.4% (v3 62.6%); crossover intact. (Source: README ledger LW-03.)

---

## 3. Finding 2 — a reconstruction-trained latent drops the consequence

Move the auditor behind pixels: render the arm to 48×48 grayscale stacked windows, train an
autoencoder (reconstruction objective) and a latent dynamics model, and ask whether the
hidden-mass consequence delta survives into the latent the verifier consumes.

It does not. With the input held fixed and only the hidden cause changed (clean vs heavy, paired
nuisance), the matched consequence delta is **not** recoverable from the latent difference:
`R²(Δz_next → Δqdd_window) = −0.175` (worse than predicting the mean). The formal label of this run is
**VOID-FIDELITY** — the latent-space verifier comparison is not readable, because the signal it would
need is absent from the latent.

Crucially, this is **not** because the consequence is ill-posed or invisible in the observation (§5
localizes where it is lost). The interpretation is Branch C: *the rendered observation made the matched
hidden-mass consequence delta recoverable, but this reconstruction-trained AE stack at this capacity
failed to preserve it in `z`.*

---

## 4. The causal chain (where the consequence survives, and where it dies)

Same matched clean/heavy pairs, same window-scale acceleration target `Δqdd_window`, same 3-seed
nonlinear probe policy throughout. Capacity is matched at 12 dimensions (the latent width); the raw
observation is reduced to 12 PCA components so the pixel and latent probes are a matched-capacity,
matched-probe comparison.

| stage | probe | R² to `Δqdd_window` | reading |
|---|---|---:|---|
| auditor | `Δtrue_pose_window` | **0.992** | consequence is well-posed in the auditor |
| observation | `Δraw_obs_next` (PCA-12) | **0.716** | recoverable from the rendered observation at matched capacity |
| latent | `Δz_next` | **−0.175** | not preserved in the learned latent |

`pixel_minus_latent_gap = 0.716 − (−0.175) = 0.891`. The consequence is present and recoverable right
up to the encoder, then disappears. (Sources: FINDINGS addendum table; metrics.json delta_gate.)

---

## 5. Why the result is credible (confound elimination)

| candidate confound | ruled out by |
|---|---|
| hidden cause leaked into the input | Gate B: input-only detectors cannot separate clean vs shift (AUROC `d`=0.500, `u`=0.500; matched-pair max\|Δ\|=0) |
| wrong-timescale target | target is window-scale acceleration aligned to the observation window, not the native single-step derivative |
| raw-vs-latent probe mismatch | the latent gate and the raw-observation probe use the identical 3-seed nonlinear probe and the same 12-D capacity |
| unpaired render noise | clean/heavy renders share a per-pair jitter seed (paired nuisance); the non-jittered diagnostic agrees |
| detector threshold tuning | the 5% calibration was fixed and never adjusted after seeing a result |
| AE latent dimensionality as sole limiter | EVR-12 = 0.327 yet pixel R² = 0.716 — a 12-D bottleneck can preserve the consequence in principle; the failure is specific to what this reconstruction-trained AE stack preserved in `z`, not proof that all AE capacity/architecture forms are irrelevant |
| consequence ill-posed / degenerate | analytic `Δqdd_window` total variance = 140.10; oracle recovers it at R² 0.992 |

---

## 6. What this does not claim (scope limits)

- Not a general claim about reconstruction objectives, and not a general claim about learned latents.
- Bounded to **this** reconstruction-trained AE and **this** observation/window family (48×48
  grayscale stacked windows; frame interval 0.08s / `frame_stride=8`).
- The mechanism is a bounded hypothesis for this stack: under this reconstruction objective and
  architecture, the latent allocated its limited capacity to reconstruction-relevant visual structure
  rather than the low-variance but audit-relevant consequence delta. (Explicitly **not**
  "reconstruction objectives discard physical consequences.")
- The formal label is unchanged: **VOID-FIDELITY**. This was not a new gate or a new experiment phase;
  no detector table was interpreted in the latent setting; the transfer matrix has not been started.

---

## 7. Reviewer questions (please attack these)

1. Is this known under another name? (If so, which literature?)
2. Is "matched hidden-cause consequence delta" a meaningful audit variable, or an artifact of the rig?
3. Does the `0.992 → 0.716 → −0.175` chain localize the failure correctly (auditor → observation → latent)?
4. What remaining confound would invalidate the Branch C interpretation?
5. Is the right next variable the representation objective, or should the next test change the
   consequence variable / observation design first?

---

## 8. Reproducibility / commit pointers

All in the `Latent_Wedge` repository:

- LW-11 terminal run + canonical metrics: commit `583974e`, `results/20260619_193900_33faabb_latent/`
  (`metrics.json` holds `gate_b`, `delta_gate`, `gate_a`).
- Raw-observation consequence-ceiling addendum (the 0.992/0.716/−0.175 chain): commit `a3c0221`.
- Branch C final verification (matched-capacity, matched-probe): commit `d54b9a6`.
- Single-source memo: `FINDINGS_latent_v0.md`; claim ledger: `README.md`; build log: `LOG.md`.

Numbers in this packet were read from `FINDINGS_latent_v0.md` (interpreted claims) and `metrics.json`
(run numbers); README LW-03 for the state-space parity. No number was taken from memory.

---

This note is intended to be attacked. The goal is not to defend the current framing, but to find the
sharpest version of the claim before deciding whether the next chapter should test representation
objective, observation design, or the consequence variable itself.
