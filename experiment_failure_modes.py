"""
experiment_failure_modes.py  (v3 — does a PORTFOLIO have to exist?)

The portfolio thesis only holds if there is a regime where input-based detectors
(u, d) BEAT the physics residual r, AND a regime where r beats them. v0.2/v0.3
(mine and a collaborator's) never found the former, because their OOD was always
'heavier' -- which is INPUT-QUIET (a heavy arm moves sluggishly, so input-distance
does NOT rise). So u/d never won a cell, and 'portfolio' had no empirical basis.

Fix: induce input-visibility with an INDEPENDENT knob -- wide start states /
velocities / torques under CORRECT (nominal) physics. This cleanly separates two
orthogonal failure modes:

  EXTRAPOLATION   wide inputs, nominal physics
                  -> model errs by extrapolating; physics is OBEYED so r is BLIND
                     (r~=0); only input-based u/d can see 'you're off-support'.
  PHYSICS-SHIFT   ID inputs, heavy world
                  -> familiar inputs, violated law; u/d blind, only r sees it.

Residual uses CORRECT nominal params here (damp_mismatch forced to 1.0) so that
under nominal physics r is exactly ~0 -- isolating the failure modes.

PORTED from the wedge v3 tree (phase-close commit 72866ed). The ONLY changes vs
v3 are numpy-only swaps: sklearn.metrics.roc_auc_score -> metrics_np.roc_auc, and
the ensemble member is NumpyMLP (via model.py). This file IS the M1 parity gate:
run it and compare the table to the pre-registered v3 baseline.
"""
from __future__ import annotations
import sys, os, json, subprocess, datetime
from pathlib import Path
import numpy as np
# Use a writable matplotlib cache inside the module dir (some sandboxes have a
# read-only HOME). Must be set before importing matplotlib.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplcache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from arm import TwoLinkArm
from data import build_datasets, rollout
from model import EnsembleDynamics
from metrics_np import roc_auc

RESULTS = Path(__file__).parent / "results"
SHIFT_M = 2.4

# Pre-registered v3 baseline (phase-close 72866ed failure-mode metrics.json) and
# the M1 parity tolerance. Written BEFORE the numpy run, not fitted after.
V3_BASELINE = {
    "extrapolation": {"u": 0.0764, "d": 0.0832, "r": 0.5744},
    "physics_shift": {"u": 0.6386, "d": 0.6255, "r": 0.0000},
    "id_holdout":    {"u": 0.4085, "d": 0.3617, "r": 0.6255},
}
PARITY_TOL = 0.08   # each r cell within +-8 absolute points of v3


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=Path(__file__).parent, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def step_under(arm, q, qd, tau, dt):
    q2, qd2 = arm.step_rk4(q, qd, tau, dt)
    return np.concatenate([q, qd, tau, q2, qd2])


def wide_rollout(arm, cfg, rng):
    """Nominal physics, but start state / velocity / torque sampled WIDE -> inputs
    leave the training support (genuine input-visibility, independent of mass)."""
    q  = rng.uniform(-1.5, 1.5, size=2)
    qd = rng.uniform(-2.5, 2.5, size=2)
    phase = rng.uniform(0, 2*np.pi, size=2); freq = rng.uniform(1.0, 5.0, size=2)
    amp = rng.uniform(8.0, 11.0, size=2)
    rec = []
    for k in range(cfg.steps):
        t = k*cfg.dt
        tau = np.clip(amp*(np.sin(freq*t+phase) + 0.5*np.sin(2.3*freq*t+1.7*phase)), -12, 12)
        q2, qd2 = arm.step_rk4(q, qd, tau, cfg.dt)
        rec.append(np.concatenate([q, qd, tau, q2, qd2]))
        q, qd = q2, qd2
        if not np.all(np.isfinite(q)):
            break
    return np.array(rec)


def residual(res_nom, true_arm, q, qd, tau):
    qdd = true_arm.qddot(q, qd, tau)         # true instantaneous accel (no finite diff)
    tau_pred = res_nom.M(q) @ qdd + res_nom.Cqd(q, qd) + res_nom.grav(q) + res_nom.b*qd
    return float(np.linalg.norm(tau_pred - tau))


def detectors(model, res_nom, true_arm, data):
    _, P = model.predict_all(data[:, 0:6]); u = P.var(0).sum(1)
    d = model.input_distance(data)
    r = np.array([residual(res_nom, true_arm, row[0:2], row[2:4], row[4:6]) for row in data])
    eps = model.true_error_z(data)
    return dict(u=u, d=d, r=r), eps


def run(cfg: Config):
    cfg.apply_quick()
    cfg.damp_mismatch = 1.0                  # correct residual params -> isolate failure modes
    train, test_id, _ = build_datasets(cfg)
    model = EnsembleDynamics(cfg).fit(train)
    rng = np.random.default_rng(cfg.seed + 11)

    nom   = TwoLinkArm(m1=1.0, m2=1.0, b1=cfg.nominal_b, b2=cfg.nominal_b)
    heavy = TwoLinkArm(m1=SHIFT_M, m2=SHIFT_M, b1=cfg.nominal_b, b2=cfg.nominal_b)
    res_nom = TwoLinkArm(m1=1.0, m2=1.0, b1=cfg.nominal_b, b2=cfg.nominal_b)

    SA = test_id[:, 0:6]
    n_wide = max(8, cfg.n_test_id)
    regimes = {
        "clean":        (nom,   np.array([step_under(nom, r[0:2], r[2:4], r[4:6], cfg.dt) for r in SA])),
        "extrapolation":(nom,   np.vstack([wide_rollout(nom, cfg, rng) for _ in range(n_wide)])),
        "physics_shift":(heavy, np.array([step_under(heavy, r[0:2], r[2:4], r[4:6], cfg.dt) for r in SA])),
        "id_holdout":   (nom,   np.vstack([rollout(nom, cfg, rng) for _ in range(max(6, cfg.n_test_id//2))])),
    }
    S = {name: detectors(model, res_nom, arm, data) for name, (arm, data) in regimes.items()}

    # calibrate on clean
    sc, eps_c = S["clean"]
    eps_thr = np.quantile(eps_c, 0.95)
    single_thr = {k: np.quantile(sc[k], 0.95) for k in "udr"}     # each 5% FA alone

    # jointly-calibrated OR: common quantile q s.t. total FA(any channel)==5% on clean
    def or_fire(scores, q):
        t = {k: np.quantile(sc[k], q) for k in "udr"}
        return np.maximum.reduce([(scores[k] > t[k]).astype(float) for k in "udr"]).astype(bool), t
    lo, hi = 0.5, 0.9999
    for _ in range(40):
        mid = (lo+hi)/2
        fa = or_fire(sc, mid)[0].mean()
        lo, hi = (mid, hi) if fa > 0.05 else (lo, mid)
    or_q = (lo+hi)/2
    _, or_thr = or_fire(sc, or_q)

    def missed(scores, eps, kind):
        cata = eps > eps_thr
        if cata.sum() == 0:
            return None
        if kind == "PORTFOLIO":
            fired = np.maximum.reduce([(scores[k] > or_thr[k]).astype(float) for k in "udr"]).astype(bool)
        else:
            fired = scores[kind] > single_thr[kind]
        return float(1 - fired[cata].mean())

    chans = ["u", "d", "r", "PORTFOLIO"]
    table, aurocs = {}, {}
    for name, (scores, eps) in S.items():
        table[name] = {"n_cata": int((eps > eps_thr).sum()), **{c: missed(scores, eps, c) for c in chans}}
        if name not in ("clean", "id_holdout"):
            lab = np.r_[np.zeros(len(eps_c)), np.ones(len(eps))]
            aurocs[name] = {k: float(roc_auc(lab, np.r_[sc[k], scores[k]])) for k in "udr"}

    parity = _parity_check(table)
    out = dict(eps_thr=float(eps_thr), or_false_alarm=float(or_fire(sc, or_q)[0].mean()),
               missed_catastrophe=table, detect_auroc=aurocs, parity=parity)
    _print(out); _save(cfg, out)
    return out


def _parity_check(table):
    """M1 gate: each r cell within +-8 absolute points of v3 AND qualitative
    crossover preserved (r lowest-miss on physics_shift; u/d lowest on extrapolation)."""
    cells = {}
    ok = True
    for reg, ref in V3_BASELINE.items():
        got_r = table[reg]["r"]
        if got_r is None:
            cells[reg] = {"v3_r": ref["r"], "got_r": None, "abs_diff": None,
                          "within_tol": False, "note": "no catastrophes in regime (not assessable)"}
            ok = False
            continue
        diff = abs(got_r - ref["r"])
        within = diff <= PARITY_TOL
        ok = ok and within
        cells[reg] = {"v3_r": ref["r"], "got_r": got_r, "abs_diff": diff, "within_tol": bool(within)}
    ex = table["extrapolation"]; ps = table["physics_shift"]

    def _safe(x):
        return 1.0 if x is None else x
    cross_r_shift = _safe(ps["r"]) < min(_safe(ps["u"]), _safe(ps["d"]))   # r best on physics_shift
    cross_ud_extra = min(_safe(ex["u"]), _safe(ex["d"])) < _safe(ex["r"])  # u/d best on extrapolation
    qualitative = bool(cross_r_shift and cross_ud_extra)
    ok = ok and qualitative
    return {"tol_abs": PARITY_TOL, "cells": cells,
            "crossover_r_best_on_physics_shift": bool(cross_r_shift),
            "crossover_ud_best_on_extrapolation": bool(cross_ud_extra),
            "qualitative_pattern_intact": qualitative,
            "PASS": bool(ok)}


def _fmt(x): return " n/a " if x is None else f"{x:5.1%}"


def _print(o):
    chans = ["u", "d", "r", "PORTFOLIO"]
    print("\n" + "="*74)
    print("FAILURE-MODE SEPARATION — missed-catastrophe (lower=safer), 5% total FA")
    print("="*74)
    print(f"{'regime':14s} {'#cat':>5s} " + " ".join(f"{c:>10s}" for c in chans))
    for reg, row in o["missed_catastrophe"].items():
        print(f"{reg:14s} {row['n_cata']:5d} " + " ".join(f"{_fmt(row[c]):>10s}" for c in chans))
    print(f"\n(portfolio OR false-alarm on clean = {o['or_false_alarm']:.1%}, matched to single channels)")
    p = o["parity"]
    print("\nM1 PARITY GATE (vs v3 phase-close 72866ed, tol +-8 pts on r):")
    for reg, c in p["cells"].items():
        got = "n/a" if c["got_r"] is None else f"{c['got_r']:.3f}"
        dif = "n/a" if c["abs_diff"] is None else f"{c['abs_diff']:.3f}"
        print(f"  {reg:14s} r: v3={c['v3_r']:.3f} got={got} "
              f"|diff|={dif} -> {'OK' if c['within_tol'] else 'OUT'}")
    print(f"  crossover (r best on physics_shift)   : {p['crossover_r_best_on_physics_shift']}")
    print(f"  crossover (u/d best on extrapolation) : {p['crossover_ud_best_on_extrapolation']}")
    print(f"  => PARITY {'PASS' if p['PASS'] else 'FAIL'}")


def _save(cfg, o):
    tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + git_hash() + "_failuremodes"
    d = RESULTS / tag; d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=list))
    (d / "metrics.json").write_text(json.dumps(o, indent=2))
    tbl = o["missed_catastrophe"]; regs = list(tbl); chans = ["u", "d", "r", "PORTFOLIO"]
    M = np.array([[ (tbl[r][c] if tbl[r][c] is not None else np.nan) for c in chans] for r in regs])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    im = ax.imshow(M, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(chans))); ax.set_xticklabels(chans)
    ax.set_yticks(range(len(regs))); ax.set_yticklabels(regs)
    for i in range(len(regs)):
        for j in range(len(chans)):
            v = M[i, j]
            ax.text(j, i, "n/a" if np.isnan(v) else f"{v:.0%}", ha="center", va="center", fontsize=10)
    ax.set_title("Missed-catastrophe (green=safe). State-space parity run\n"
                 "(numpy port of wedge v3; M1 gate).")
    fig.colorbar(im, label="fraction missed"); fig.tight_layout()
    fig.savefig(d / "result.png", dpi=130); plt.close(fig)
    print(f"\nsaved -> results/{tag}/")


if __name__ == "__main__":
    run(Config(quick="--quick" in sys.argv))
