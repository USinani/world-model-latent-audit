"""
experiment_latent_failure_modes.py — the v3 failure-mode harness, re-expressed in
LATENT space. This is the actual wedge: when (q, qd, tau) and the physics equation
are HIDDEN behind an autoencoder, can a learned latent transition-consistency
signal c_z recover the analytic residual r's job — catching a consequence-only
physics shift that latent input-detectors are blind to — WITHOUT the latent
secretly recovering the privileged state?

Pipeline (everything pure numpy):
  states --render--> stacked frames (+nuisance) --AE--> z
  clean ID only:  train AE,  train latent-dynamics ensemble (z_t,a_t)->z_{t+1}
  four regimes (identical structure to state-space v3): clean / extrapolation /
  physics_shift / id_holdout, but observed through pixels.

Detectors (exactly three; analytic r and true q are NEVER detector inputs):
  u_z = latent ensemble disagreement on (z_t, a_t)          [input-based]
  d_z = Mahalanobis distance of (z_t, a_t)                  [input-based]
  c_z = || encode(obs_{t+1}) - predicted_z_{t+1} ||         [transition consistency]

Privileged-but-withheld (labels / post-hoc only, NEVER a detector input):
  eps  = || predicted_z_{t+1} - encode(CLEAN render of TRUE next state) ||
         (the catastrophe definition; uses the true next state + a clean render)
  r    = analytic torque residual (the state-space detector), for Spearman only.

Gates and bands (pre-registered in the plan, not fitted after):
  RECON GATE  : AE recon must beat the mean-frame/blur baseline by a margin AND
                physics_shift recon must stay comparable to clean, else VOID.
  R^2 band    : informative iff 0.3 <= R^2(z->q,qd) <= 0.9 (floor + ceiling).
  c_z vs d_z  : rank-correlation early kill — if c_z ~= d_z the third channel is
                not orthogonal regardless of the table.
"""
from __future__ import annotations
import sys, os, json, subprocess, datetime
from pathlib import Path
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplcache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from arm import TwoLinkArm
from data import build_datasets, rollout
import render
from latent_model import LatentAE, LatentEnsemble
from numpy_mlp import NumpyMLP
from metrics_np import roc_auc, spearman
from experiment_failure_modes import wide_rollout, step_under, residual

RESULTS = Path(__file__).parent / "results"
SHIFT_M = 2.4

# Pre-registered gate thresholds (written before the run).
RECON_BEAT_FACTOR = 1.5      # AE recon must be <= baseline / 1.5 (beat mean-frame by >=1.5x)
RECON_SHIFT_RATIO = 1.5      # physics_shift recon must be <= 1.5x the clean recon
R2_FLOOR, R2_CEIL = 0.30, 0.90
CZDZ_RHO_MAX = 0.80          # if rank-corr(c_z,d_z) > this, c_z is not an orthogonal channel


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=Path(__file__).parent, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


# --------------------------------------------------------------------------
# rendering helpers (states = rows [q(2), qd(2), tau(2), q_next(2), qd_next(2)])
# --------------------------------------------------------------------------

def obs_at_t(rows, cfg, rng, jitter=None):
    states = rows[:, 0:4]
    return render.render_batch(states, cfg, rng, jitter=jitter)


def obs_at_tp1(rows, cfg, rng, jitter=None):
    states = rows[:, 6:10]
    return render.render_batch(states, cfg, rng, jitter=jitter)


# --------------------------------------------------------------------------
# linear probe R^2 (laundering check)
# --------------------------------------------------------------------------

def linear_probe_r2(z_tr, s_tr, z_te, s_te):
    """Fit linear z -> state on clean train, report R^2 per target on clean holdout.
    Standardise z (tanh latents can be near-constant on dead dims) and use a small
    ridge so the probe is well-posed and never overflows."""
    zm, zs = z_tr.mean(0), z_tr.std(0) + 1e-8
    Ztr = (z_tr - zm) / zs
    Zte = (z_te - zm) / zs
    Xtr = np.hstack([Ztr, np.ones((len(Ztr), 1))])
    Xte = np.hstack([Zte, np.ones((len(Zte), 1))])
    lam = 1e-3
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
        W = np.linalg.solve(A, Xtr.T @ s_tr)
        pred = Xte @ W
    ss_res = ((s_te - pred) ** 2).sum(0)
    ss_tot = ((s_te - s_te.mean(0)) ** 2).sum(0) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return r2                         # (4,) per [q1,q2,qd1,qd2]


def nonlinear_probe_r2(z_tr, s_tr, z_te, s_te, cfg):
    """Small MLP probe z -> state, fit on clean train, R^2 on clean holdout.
    This is NOT the laundering check (linear is). It only disambiguates the FLOOR:
    a truly vacuous latent fails even a nonlinear probe; an informative-but-
    nonlinearly-coded latent (e.g. relative angle q2) is recovered here while the
    linear probe stays low. Targets are z-scored so R^2 is comparable across dims."""
    sm, ss = s_tr.mean(0), s_tr.std(0) + 1e-8
    probe = NumpyMLP((z_tr.shape[1], 64, 64, s_tr.shape[1]), activation="tanh",
                     l2=1e-4, max_iter=300, batch_size=256, lr=1e-3, seed=7,
                     standardize_x=True, standardize_y=False)
    probe.fit(z_tr, (s_tr - sm) / ss)
    pred = probe.predict(z_te) * ss + sm
    ss_res = ((s_te - pred) ** 2).sum(0)
    ss_tot = ((s_te - s_te.mean(0)) ** 2).sum(0) + 1e-12
    return 1.0 - ss_res / ss_tot


# threshold for "qdd is recoverable" (mean R^2 over the two joints)
ACCEL_RECOVER_FLOOR = 0.30


def _which_red(fa_mean, zz_mean):
    """Decide which RED from acceleration recoverability.

    The decisive question is 'is qdd in z?', because that directly answers whether
    the bottleneck ate the dimension the shift lives in. We key on z+a FIRST: a
    well-conditioned 12-D probe is a fair test, whereas the raw frames+a probe is a
    high-D / finite-sample underestimate (its linear readout goes wildly negative),
    so frames+a is only used to disambiguate the case where z does NOT carry qdd."""
    if zz_mean >= ACCEL_RECOVER_FLOOR:
        return ("deep_boundary",
                "qdd IS recoverable from z+action (the bottleneck did not eat acceleration), "
                "yet c_z still misses physics_shift: the information is in the latent; the "
                "learned consistency check simply cannot use it without the analytic law. "
                "This is the deep RED.")
    if fa_mean >= ACCEL_RECOVER_FLOOR and zz_mean < 0.6 * fa_mean:
        return ("ae_capacity_artifact",
                "qdd recoverable from frames+action but NOT from z+action: the AE bottleneck "
                "compressed away the acceleration dimension. Fix = widen latent_dim, rerun "
                "before claiming a boundary.")
    return ("inconclusive_observation_limit",
            "qdd is not cleanly recoverable from z+action OR frames+action. The constant-"
            "velocity stacked window may not carry enough acceleration signal (curved back-"
            "extrapolation would fix it), or the frames probe is underpowered. Rerun before "
            "claiming a deep boundary.")


def accel_recoverability(f_tr, z_tr, a_tr, qdd_tr, f_te, z_te, a_te, qdd_te, cfg):
    """Regress realized acceleration qdd from four predictor sets, on clean ID.
    `frames+a` is the observation upper bound; `z+a` is what c_z actually has."""
    sets = {
        "frames+a": (np.hstack([f_tr, a_tr]), np.hstack([f_te, a_te])),
        "z+a":      (np.hstack([z_tr, a_tr]), np.hstack([z_te, a_te])),
        "z":        (z_tr, z_te),
        "frames":   (f_tr, f_te),
    }
    res = {}
    for name, (Xtr, Xte) in sets.items():
        nl = nonlinear_probe_r2(Xtr, qdd_tr, Xte, qdd_te, cfg)
        lin = linear_probe_r2(Xtr, qdd_tr, Xte, qdd_te)
        res[name] = {"nonlinear_mean": float(np.mean(nl)),
                     "nonlinear_per_joint": [float(x) for x in nl],
                     "linear_mean": float(np.mean(lin)),
                     "linear_per_joint": [float(x) for x in lin]}
    fa = res["frames+a"]["nonlinear_mean"]
    zz = res["z+a"]["nonlinear_mean"]
    label, note = _which_red(fa, zz)
    res["recover_floor"] = ACCEL_RECOVER_FLOOR
    res["frames_plus_a_mean"] = fa
    res["z_plus_a_mean"] = zz
    res["which_red"] = label
    res["which_red_note"] = note
    return res


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def run(cfg: Config):
    cfg.apply_quick()
    rng = np.random.default_rng(cfg.seed + 11)

    # ---- clean ID data (state space) ----
    train, test_id, _ = build_datasets(cfg)

    # subsample clean ID transitions to the numpy training budget
    sub_rng = np.random.default_rng(cfg.seed + 5)
    n_dyn = min(cfg.max_dyn_samples, len(train))
    dyn_idx = sub_rng.choice(len(train), size=n_dyn, replace=False)
    train_dyn = train[dyn_idx]

    # ---- train AE on CLEAN ID frames only (pool obs_t and obs_{t+1}) ----
    ae_rng = np.random.default_rng(cfg.seed + 21)
    n_ae = min(cfg.max_ae_frames // 2, len(train))
    ae_idx = sub_rng.choice(len(train), size=n_ae, replace=False)
    train_ae = train[ae_idx]
    F_tr = np.vstack([obs_at_t(train_ae, cfg, ae_rng), obs_at_tp1(train_ae, cfg, ae_rng)])
    ae = LatentAE(cfg, F_tr.shape[1], seed=cfg.seed).fit(F_tr)

    # ---- train latent-dynamics ensemble on CLEAN ID transitions only ----
    dyn_rng = np.random.default_rng(cfg.seed + 31)
    z_t_tr   = ae.encode(obs_at_t(train_dyn, cfg, dyn_rng))
    z_tp1_tr = ae.encode(obs_at_tp1(train_dyn, cfg, dyn_rng))
    A_tr     = train_dyn[:, 4:6]
    ens = LatentEnsemble(cfg).fit(z_t_tr, A_tr, z_tp1_tr)

    # mean-frame / blur baselines for the recon gate (on clean ID frames)
    mean_frame = F_tr.mean(0, keepdims=True)
    base_meanframe = float(np.mean((F_tr - mean_frame) ** 2))

    # ---- regimes (identical structure to state-space v3) ----
    nom   = TwoLinkArm(m1=1.0, m2=1.0, b1=cfg.nominal_b, b2=cfg.nominal_b)
    heavy = TwoLinkArm(m1=SHIFT_M, m2=SHIFT_M, b1=cfg.nominal_b, b2=cfg.nominal_b)
    res_nom = TwoLinkArm(m1=1.0, m2=1.0, b1=cfg.nominal_b, b2=cfg.nominal_b)

    SA = test_id[:, 0:6]
    n_wide = max(8, cfg.n_test_id)
    regimes = {
        "clean":         (nom,   np.array([step_under(nom, r[0:2], r[2:4], r[4:6], cfg.dt) for r in SA])),
        "extrapolation": (nom,   np.vstack([wide_rollout(nom, cfg, rng) for _ in range(n_wide)])),
        "physics_shift": (heavy, np.array([step_under(heavy, r[0:2], r[2:4], r[4:6], cfg.dt) for r in SA])),
        "id_holdout":    (nom,   np.vstack([rollout(nom, cfg, rng) for _ in range(max(6, cfg.n_test_id // 2))])),
    }

    S, recon = {}, {}
    for ridx, (name, (true_arm, rows)) in enumerate(regimes.items()):
        r_rng = np.random.default_rng(cfg.seed + 41 + 13 * ridx)
        o_t   = obs_at_t(rows, cfg, r_rng)                       # jittered observation at t
        o_tp1 = obs_at_tp1(rows, cfg, r_rng)                     # jittered observation at t+1
        o_tp1_clean = obs_at_tp1(rows, cfg, r_rng, jitter=0.0)   # privileged clean render of TRUE next state

        z_t        = ae.encode(o_t)
        z_tp1_obs  = ae.encode(o_tp1)
        z_tp1_true = ae.encode(o_tp1_clean)                      # privileged target
        A          = rows[:, 4:6]

        mean_next, _ = ens.predict_all(z_t, A)
        eps = np.linalg.norm(mean_next - z_tp1_true, axis=1)     # privileged catastrophe signal
        u_z = ens.disagreement(z_t, A)
        d_z = ens.input_distance(z_t, A)
        c_z = np.linalg.norm(z_tp1_obs - mean_next, axis=1)      # observation-only consistency
        r_an = np.array([residual(res_nom, true_arm, row[0:2], row[2:4], row[4:6]) for row in rows])

        S[name] = dict(u_z=u_z, d_z=d_z, c_z=c_z, eps=eps, r=r_an)
        recon[name] = float(ae.recon_error(o_t).mean())

    # ---- calibrate on clean ----
    sc = S["clean"]; eps_c = sc["eps"]
    eps_thr = np.quantile(eps_c, 0.95)
    chans = ["u_z", "d_z", "c_z"]
    single_thr = {k: np.quantile(sc[k], 0.95) for k in chans}

    def or_fire(scores, q):
        t = {k: np.quantile(sc[k], q) for k in chans}
        return np.maximum.reduce([(scores[k] > t[k]).astype(float) for k in chans]).astype(bool), t
    lo, hi = 0.5, 0.9999
    for _ in range(40):
        mid = (lo + hi) / 2
        fa = or_fire(sc, mid)[0].mean()
        lo, hi = (mid, hi) if fa > 0.05 else (lo, mid)
    or_q = (lo + hi) / 2
    _, or_thr = or_fire(sc, or_q)

    def missed(scores, eps, kind):
        cata = eps > eps_thr
        if cata.sum() == 0:
            return None
        if kind == "PORTFOLIO":
            fired = np.maximum.reduce([(scores[k] > or_thr[k]).astype(float) for k in chans]).astype(bool)
        else:
            fired = scores[kind] > single_thr[kind]
        return float(1 - fired[cata].mean())

    cols = chans + ["PORTFOLIO"]
    table, aurocs = {}, {}
    for name, scores in S.items():
        eps = scores["eps"]
        table[name] = {"n_cata": int((eps > eps_thr).sum()),
                       **{c: missed(scores, eps, c) for c in cols}}
        if name not in ("clean", "id_holdout"):
            lab = np.r_[np.zeros(len(eps_c)), np.ones(len(eps))]
            aurocs[name] = {k: float(roc_auc(lab, np.r_[sc[k], scores[k]])) for k in chans}

    # ---- linear-probe R^2 (laundering check) on clean ID ----
    pr_rng = np.random.default_rng(cfg.seed + 51)
    te_cap = min(cfg.max_dyn_samples, len(test_id))
    te_idx = np.random.default_rng(cfg.seed + 6).choice(len(test_id), size=te_cap, replace=False)
    test_probe = test_id[te_idx]
    f_probe_tr = obs_at_t(train_ae, cfg, pr_rng);  z_probe_tr = ae.encode(f_probe_tr); s_probe_tr = train_ae[:, 0:4]
    f_probe_te = obs_at_t(test_probe, cfg, pr_rng); z_probe_te = ae.encode(f_probe_te); s_probe_te = test_probe[:, 0:4]
    r2_per = linear_probe_r2(z_probe_tr, s_probe_tr, z_probe_te, s_probe_te)
    r2_mean = float(np.mean(r2_per))
    r2_nl_per = nonlinear_probe_r2(z_probe_tr, s_probe_tr, z_probe_te, s_probe_te, cfg)
    r2_nl_mean = float(np.mean(r2_nl_per))

    # ---- adversarial check: is realized acceleration recoverable? ------------
    # The physics_shift signature is a MASS change, whose footprint lives in the
    # acceleration (the velocity change over the step). If qdd is recoverable from
    # the frames but NOT from z, the bottleneck ate the relevant dimension and the
    # RED is partly an AE-capacity artifact; if it is recoverable from z and c_z
    # still misses physics_shift, the RED is the deep result. NOTE: render.py uses
    # CONSTANT-velocity back-extrapolation, so qdd has no within-window curvature
    # signature -- it enters only via (inputs, action) + physics, hence the action
    # a_t is concatenated to every predictor for a fair upper bound.
    qdd_tr = (train_ae[:, 8:10] - train_ae[:, 2:4]) / cfg.dt
    qdd_te = (test_probe[:, 8:10] - test_probe[:, 2:4]) / cfg.dt
    accel = accel_recoverability(f_probe_tr, z_probe_tr, train_ae[:, 4:6], qdd_tr,
                                 f_probe_te, z_probe_te, test_probe[:, 4:6], qdd_te, cfg)

    # ---- post-hoc correlations on the pooled non-clean set ----
    pool = ["extrapolation", "physics_shift", "id_holdout"]
    cz_pool = np.concatenate([S[n]["c_z"] for n in pool])
    dz_pool = np.concatenate([S[n]["d_z"] for n in pool])
    r_pool  = np.concatenate([S[n]["r"] for n in pool])
    rho_cz_r  = spearman(cz_pool, r_pool)
    rho_cz_dz = spearman(cz_pool, dz_pool)

    # ---- recon gate ----
    recon_clean = recon["clean"]
    recon_shift = recon["physics_shift"]
    recon_gate = {
        "ae_recon_clean": recon_clean,
        "mean_frame_baseline": base_meanframe,
        "beat_factor_required": RECON_BEAT_FACTOR,
        "beats_baseline": bool(recon_clean <= base_meanframe / RECON_BEAT_FACTOR),
        "recon_physics_shift": recon_shift,
        "shift_ratio": float(recon_shift / (recon_clean + 1e-12)),
        "shift_ratio_max": RECON_SHIFT_RATIO,
        "shift_comparable": bool(recon_shift <= RECON_SHIFT_RATIO * recon_clean),
        "per_regime": recon,
    }
    recon_gate["PASS"] = bool(recon_gate["beats_baseline"] and recon_gate["shift_comparable"]
                              and not ae.training_failed and not ens.training_failed)

    # ---- verdict ----
    verdict = _verdict(table, r2_mean, r2_nl_mean, rho_cz_r, rho_cz_dz, recon_gate)

    out = dict(
        eps_thr=float(eps_thr),
        or_false_alarm=float(or_fire(sc, or_q)[0].mean()),
        missed_catastrophe=table,
        detect_auroc=aurocs,
        linear_probe_r2={"per_target_q1_q2_qd1_qd2": [float(x) for x in r2_per],
                         "mean": r2_mean, "band": [R2_FLOOR, R2_CEIL],
                         "role": "laundering ceiling: R2>0.9 => latent recovered privileged state"},
        nonlinear_probe_r2={"per_target_q1_q2_qd1_qd2": [float(x) for x in r2_nl_per],
                            "mean": r2_nl_mean,
                            "role": "vacuity floor: low even here => latent truly discarded state"},
        spearman={"c_z_vs_r": rho_cz_r, "c_z_vs_d_z": rho_cz_dz, "rho_max_for_orthogonal": CZDZ_RHO_MAX},
        accel_recoverability=accel,
        recon_gate=recon_gate,
        training_failed={"ae": bool(ae.training_failed), "ensemble": bool(ens.training_failed)},
        verdict=verdict,
    )
    out["verdict"]["which_red"] = accel["which_red"] if verdict["label"] == "RED" else "n/a"
    _print(out); _save(cfg, out, S, eps_thr)
    return out


def _verdict(table, r2_lin, r2_nl, rho_cz_r, rho_cz_dz, recon_gate):
    if not recon_gate["PASS"]:
        return {"label": "VOID",
                "reason": "recon gate failed (AE did not beat baseline, or physics_shift recon "
                          "degraded, or training failed) — a training failure must not masquerade "
                          "as a scientific boundary."}
    ex = table["extrapolation"]; ps = table["physics_shift"]
    cz_catches_shift = (ps["c_z"] is not None) and ps["c_z"] < 0.5
    cz_quiet_extrap = (ex["c_z"] is not None) and ex["c_z"] > 0.5
    cz_corr_r = (rho_cz_r == rho_cz_r) and rho_cz_r > 0.2
    cz_distinct_dz = (rho_cz_dz == rho_cz_dz) and rho_cz_dz <= CZDZ_RHO_MAX
    # laundering ceiling on the LINEAR probe; vacuity floor on the NONLINEAR probe.
    laundered = r2_lin > R2_CEIL
    vacuous = r2_nl < R2_FLOOR
    latent_ok = (not laundered) and (not vacuous)
    green = bool(cz_catches_shift and cz_quiet_extrap and cz_corr_r and cz_distinct_dz and latent_ok)
    reasons = []
    if not cz_catches_shift: reasons.append("c_z does not catch physics_shift")
    if not cz_quiet_extrap:  reasons.append("c_z fires on extrapolation (not quiet) -> behaves like a "
                                            "universal error detector; cannot separate novelty from "
                                            "law-violation without privileged state (the boundary result)")
    if not cz_corr_r:        reasons.append("c_z does not track the withheld analytic r (Spearman near 0)")
    if not cz_distinct_dz:   reasons.append(f"c_z ~= d_z (rho={rho_cz_dz:.2f}) -> not an orthogonal channel")
    if laundered:            reasons.append(f"linear R2={r2_lin:.2f} > {R2_CEIL}: latent recovered privileged state (laundering)")
    if vacuous:              reasons.append(f"nonlinear R2={r2_nl:.2f} < {R2_FLOOR}: latent truly discarded the state (vacuous)")
    if (not laundered) and r2_lin < R2_FLOOR and not vacuous:
        reasons.append(f"latent is informative-but-nonlinear (linear R2={r2_lin:.2f} low, nonlinear R2={r2_nl:.2f} high): "
                       "state is present but not a linear readout -> NOT laundered, NOT vacuous")
    return {"label": "GREEN" if green else "RED",
            "cz_catches_physics_shift": bool(cz_catches_shift),
            "cz_quiet_on_extrapolation": bool(cz_quiet_extrap),
            "cz_tracks_r": bool(cz_corr_r),
            "cz_distinct_from_dz": bool(cz_distinct_dz),
            "latent_not_laundered": bool(not laundered),
            "latent_not_vacuous": bool(not vacuous),
            "reasons": reasons or ["all green conditions met"]}


def _fmt(x): return " n/a " if x is None else f"{x:5.1%}"


def _print(o):
    cols = ["u_z", "d_z", "c_z", "PORTFOLIO"]
    print("\n" + "=" * 78)
    print("LATENT FAILURE-MODE SEPARATION — missed-catastrophe (lower=safer), 5% total FA")
    print("=" * 78)
    print(f"{'regime':14s} {'#cat':>5s} " + " ".join(f"{c:>10s}" for c in cols))
    for reg, row in o["missed_catastrophe"].items():
        print(f"{reg:14s} {row['n_cata']:5d} " + " ".join(f"{_fmt(row[c]):>10s}" for c in cols))
    print(f"\n(portfolio OR false-alarm on clean = {o['or_false_alarm']:.1%})")
    rg = o["recon_gate"]
    print("\nRECON GATE:")
    print(f"  AE recon (clean)={rg['ae_recon_clean']:.5f}  mean-frame baseline={rg['mean_frame_baseline']:.5f}"
          f"  beats(>= {rg['beat_factor_required']}x)={rg['beats_baseline']}")
    print(f"  recon physics_shift={rg['recon_physics_shift']:.5f}  ratio={rg['shift_ratio']:.2f}"
          f" (<= {rg['shift_ratio_max']})={rg['shift_comparable']}  => GATE {'PASS' if rg['PASS'] else 'FAIL (VOID)'}")
    lp = o["linear_probe_r2"]; nlp = o["nonlinear_probe_r2"]
    print(f"\nLINEAR PROBE     R^2(z->q,qd) mean={lp['mean']:.3f}  (laundering ceiling {R2_CEIL})  "
          f"per-target={[round(x,2) for x in lp['per_target_q1_q2_qd1_qd2']]}")
    print(f"NONLINEAR PROBE  R^2(z->q,qd) mean={nlp['mean']:.3f}  (vacuity floor {R2_FLOOR})      "
          f"per-target={[round(x,2) for x in nlp['per_target_q1_q2_qd1_qd2']]}")
    sp = o["spearman"]
    print(f"SPEARMAN  c_z vs r={sp['c_z_vs_r']:.3f}   c_z vs d_z={sp['c_z_vs_d_z']:.3f} "
          f"(orthogonal iff <= {sp['rho_max_for_orthogonal']})")
    ac = o["accel_recoverability"]
    print(f"\nACCEL RECOVERABILITY  R^2(. -> realized qdd)  (recover floor {ac['recover_floor']})")
    for name in ("frames+a", "z+a", "z", "frames"):
        a = ac[name]
        print(f"  {name:9s} nonlinear={a['nonlinear_mean']:.3f} "
              f"per-joint={[round(x, 2) for x in a['nonlinear_per_joint']]}   linear={a['linear_mean']:.3f}")
    print(f"  => WHICH RED: {ac['which_red']} — {ac['which_red_note']}")
    v = o["verdict"]
    print(f"\nVERDICT: {v['label']}")
    for rsn in v["reasons"]:
        print(f"  - {rsn}")


def _save(cfg, o, S, eps_thr):
    tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + git_hash() + "_latent"
    d = RESULTS / tag; d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=list))
    (d / "metrics.json").write_text(json.dumps(o, indent=2))

    # heatmap
    tbl = o["missed_catastrophe"]; regs = list(tbl); cols = ["u_z", "d_z", "c_z", "PORTFOLIO"]
    M = np.array([[(tbl[r][c] if tbl[r][c] is not None else np.nan) for c in cols] for r in regs])
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    im = ax.imshow(M, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
    ax.set_yticks(range(len(regs))); ax.set_yticklabels(regs)
    for i in range(len(regs)):
        for j in range(len(cols)):
            v = M[i, j]
            ax.text(j, i, "n/a" if np.isnan(v) else f"{v:.0%}", ha="center", va="center", fontsize=10)
    ax.set_title(f"Latent missed-catastrophe (green=safe). VERDICT: {o['verdict']['label']}")
    fig.colorbar(im, label="fraction missed"); fig.tight_layout()
    fig.savefig(d / "latent_missed_catastrophe.png", dpi=130); plt.close(fig)

    # score distributions clean vs physics_shift
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, k in zip(axes, ["u_z", "d_z", "c_z"]):
        ax.hist(S["clean"][k], bins=30, alpha=0.6, density=True, label="clean")
        ax.hist(S["physics_shift"][k], bins=30, alpha=0.6, density=True, label="physics_shift")
        ax.set_title(k); ax.legend(fontsize=8)
    fig.suptitle("Detector score distributions: clean vs physics_shift")
    fig.tight_layout()
    fig.savefig(d / "score_distributions.png", dpi=130); plt.close(fig)

    print(f"\nsaved -> results/{tag}/")


if __name__ == "__main__":
    run(Config(quick="--quick" in sys.argv))
