# Reproducing

Pure-numpy module. No GPU, no sklearn, no scipy. Two dependencies: `numpy`, `matplotlib`.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # numpy>=1.24, matplotlib>=3.6
```

Run everything from the repository root.

## Sanity tests (run first)

```bash
python tests/test_invariants.py    # physics self-audit (energy/damping/gravity/residual)
python tests/test_numpy_swaps.py   # roc_auc / spearman / MLP / AE round-trip guards
```

## Experiments

```bash
python experiment_failure_modes.py         # M1 PARITY GATE (state space) — must PASS before latent
python experiment_latent_failure_modes.py  # the latent wedge (writes results/*_latent/)
# add --quick to either for a fast smoke run
```

The parity gate must reproduce the v3 state-space `r` column (within ±8 pts, crossover intact) before
the latent run is meaningful.

## Canonical results

The two canonical runs are tracked in the repo:

- State-space parity: `results/20260619_103342_nogit_failuremodes/metrics.json`
- Latent LW-11 terminal run: `results/20260619_193900_33faabb_latent/metrics.json`
  (holds `gate_a`, `gate_b`, `delta_gate`).

## Reading the result

The latent v0 verdict is **VOID-FIDELITY**. Per the pre-registered rule, when Gate A fails the
**detector table is not interpreted** — it is saved in `metrics.json` for the record only. So do not
read a detector comparison out of the latent run; the honest result is that the physics-consequence
channel is not cleanly readable from this reconstruction-trained latent. See `FINDINGS_latent_v0.md`
for the closeout and `docs/verification_wedge_claim_packet_v0.md` for the full argument.
