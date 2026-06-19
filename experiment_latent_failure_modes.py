"""
experiment_latent_failure_modes.py — the v3 failure-mode harness, re-expressed in
LATENT space. This is the actual wedge: when (q, qd, tau) and the physics equation
are HIDDEN behind an autoencoder, can a learned latent transition-consistency
signal c_z recover the analytic residual r's job — catching a consequence-only
physics shift that latent input-detectors are blind to — WITHOUT the latent
secretly recovering the privileged state?

LW-09 change (why this file was rewritten): the physics_shift consequence is an
ACCELERATION-level effect (a mass change). The earlier constant-velocity stacked
window carried position+velocity but ~zero curvature, so the consequence was not in
the output observation and the c_z test was not interpretable. Here:

  - obs_t  = [f_{t-2}, f_{t-1}, f_t]  REAL backward window under NOMINAL physics,
             SHARED between the matched clean/physics_shift pair (no hidden-mass leak).
  - obs_next = [f_t, f_{t+1}, f_{t+2}] REAL forward window under the REGIME physics,
             zero-order-hold action -> genuine curvature/acceleration in the pixels.

Detectors (exactly three; analytic r and true q are NEVER detector inputs):
  u_z = latent ensemble disagreement on (z_t, a_t)          [input-based]
  d_z = Mahalanobis distance of (z_t, a_t)                  [input-based]
  c_z = || encode(obs_next) - predicted_z_next ||           [transition consistency]
        predicted_z_next is produced from (z_t, a_t) only.

Privileged-but-withheld (labels / post-hoc only, NEVER a detector input):
  eps  = || predicted_z_next - encode(non-jittered regime-specific obs_next_true) ||
         (the catastrophe definition; heavy obs_next_true for physics_shift, so the
          shift is scored against its OWN output, never against nominal).
  r    = analytic torque residual (the state-space detector), for Spearman only.

Pre-registered gates (written before the run; not fitted after):
  GATE A : the OUTPUT observation must carry q-double-dot. nonlinear R^2(obs_next+a
           -> qdd) reported for clean / physics_shift / pooled; hard gate is
           pooled > 0.60 AND physics_shift clears the same floor, else VOID.
  GATE B : no hidden physics leaked into obs_t. AUROC(d_z),(u_z) for clean vs
           physics_shift must lie in [0.45,0.55] (and matched-pair score delta ~0),
           else VOID.
  RECON  : AE recon must beat the mean-frame baseline AND physics_shift output recon
           must stay comparable to clean, else VOID.
  AE-cap : if qdd reaches the frames (obs_next+a) but NOT the latent (z_next+a) by
           more than AE_LIMIT_MARGIN, the label is AE-LIMIT (never DEEP RED).

Final label: VOID / AE-LIMIT / DEEP RED / GREEN (see _final_label).
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
import windows
from latent_model import LatentAE, LatentEnsemble
from numpy_mlp import NumpyMLP
from metrics_np import roc_auc, spearman
from experiment_failure_modes import wide_rollout, residual

RESULTS = Path(__file__).parent / "results"
SHIFT_M = 2.4

# Pre-registered gate thresholds (written before the run).
RECON_BEAT_FACTOR = 1.5      # AE recon must be <= baseline / 1.5 (beat mean-frame by >=1.5x)
RECON_SHIFT_RATIO = 1.5      # physics_shift output recon must be <= 1.5x the clean recon
R2_FLOOR, R2_CEIL = 0.30, 0.90
CZDZ_RHO_MAX = 0.80          # if rank-corr(c_z,d_z) > this, c_z is not an orthogonal channel
GATE_A_QDD = 0.60            # nonlinear R^2(obs_next+a -> qdd): pooled AND physics_shift must exceed
GATE_B_AUROC_BAND = (0.45, 0.55)  # input-only detectors must NOT separate clean vs shift
AE_LIMIT_MARGIN = 0.125      # z_next+a within this of obs_next+a -> consequence reached the latent


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=Path(__file__).parent, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


# --------------------------------------------------------------------------
# probes
# --------------------------------------------------------------------------

def linear_probe_r2(z_tr, s_tr, z_te, s_te):
    """Fit linear z -> target on train, report R^2 per target on holdout.
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
    return 1.0 - ss_res / ss_tot


def _nl_fit(Xtr, Ytr, seed=7):
    """Fit a small MLP probe Xtr -> Ytr (z-scored targets). Returns (probe, sm, ss)."""
    sm, ss = Ytr.mean(0), Ytr.std(0) + 1e-8
    probe = NumpyMLP((Xtr.shape[1], 64, 64, Ytr.shape[1]), activation="tanh",
                     l2=1e-4, max_iter=300, batch_size=256, lr=1e-3, seed=seed,
                     standardize_x=True, standardize_y=False)
    probe.fit(Xtr, (Ytr - sm) / ss)
    return probe, sm, ss


def _nl_eval(probe, sm, ss, Xte, Yte):
    pred = probe.predict(Xte) * ss + sm
    ss_res = ((Yte - pred) ** 2).sum(0)
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-12
    return 1.0 - ss_res / ss_tot


def nonlinear_probe_r2(z_tr, s_tr, z_te, s_te, cfg):
    """Small MLP probe z -> target, fit on train, R^2 on holdout. For the z->(q,qd)
    laundering check this disambiguates the FLOOR: a vacuous latent fails even here."""
    probe, sm, ss = _nl_fit(z_tr, s_tr, seed=7)
    return _nl_eval(probe, sm, ss, z_te, s_te)


def _qdd_probe(Xtr, qdd_tr, evals, seed=7):
    """Fit one nonlinear probe Xtr->qdd, evaluate on several (Xte,Yte) sets."""
    probe, sm, ss = _nl_fit(Xtr, qdd_tr, seed=seed)
    out = {}
    for k, (Xte, Yte) in evals.items():
        r2 = _nl_eval(probe, sm, ss, Xte, Yte)
        out[k] = {"nonlinear_mean": float(np.mean(r2)),
                  "nonlinear_per_joint": [float(x) for x in r2]}
    return out


# LW-10: probe-parity seeds for the capacity-matched obs_next vs z_next comparison.
PROBE_SEEDS = (7, 17, 27)


def _velocity_after(arm, q, qd, a, n_steps, dt):
    """Velocity after integrating n_steps native-dt RK4 sub-steps under `arm` (ZOH a).
    Reuses the exact batched integrator that builds the window (windows.rk4_batch), so
    the qdd targets sit on the same dynamics as the rendered poses."""
    Q, QD = np.asarray(q, float).copy(), np.asarray(qd, float).copy()
    a = np.asarray(a, float)
    for _ in range(int(n_steps)):
        Q, QD = windows.rk4_batch(arm, Q, QD, a, dt)
    return QD


def _pca_fit(X, k):
    """PCA on X (rows=samples), keep k components. Returns (mean, components (k,D))."""
    mu = X.mean(0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:k]


def _pca_transform(X, mu, comps):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return (X - mu) @ comps.T


def _multiseed_r2(Xtr, ytr, Xte, yte, seeds=PROBE_SEEDS):
    """Capacity-matched probe repeated over seeds; returns mean/std/per_seed of R^2."""
    vals = []
    for s in seeds:
        probe, sm, ss = _nl_fit(Xtr, ytr, seed=s)
        vals.append(float(np.mean(_nl_eval(probe, sm, ss, Xte, yte))))
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "per_seed": [float(v) for v in vals]}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def run(cfg: Config):
    cfg.apply_quick()
    rng = np.random.default_rng(cfg.seed + 11)
    dt = cfg.dt

    nom   = TwoLinkArm(m1=1.0, m2=1.0, b1=cfg.nominal_b, b2=cfg.nominal_b)
    heavy = TwoLinkArm(m1=SHIFT_M, m2=SHIFT_M, b1=cfg.nominal_b, b2=cfg.nominal_b)
    res_nom = TwoLinkArm(m1=1.0, m2=1.0, b1=cfg.nominal_b, b2=cfg.nominal_b)

    # ---- clean ID data (state space), subsampled to the numpy budget ----
    train, test_id, _ = build_datasets(cfg)
    sub_rng = np.random.default_rng(cfg.seed + 5)
    n_dyn = min(cfg.max_dyn_samples, len(train))
    train_dyn = train[sub_rng.choice(len(train), size=n_dyn, replace=False)]
    n_ae = min(cfg.max_ae_frames // 2, len(train))
    train_ae = train[sub_rng.choice(len(train), size=n_ae, replace=False)]

    def back(arm, rows, r):
        return render.render_window(windows.backward_window(arm, rows[:, 0:2], rows[:, 2:4], rows[:, 4:6], cfg), cfg, r)

    def fwd(arm, rows, r, jitter=None):
        poses, qd_next = windows.forward_window(arm, rows[:, 0:2], rows[:, 2:4], rows[:, 4:6], cfg)
        return render.render_window(poses, cfg, r, jitter=jitter), qd_next

    # ---- train AE on CLEAN (nominal) windows: pool real obs_t and obs_next ----
    ae_rng = np.random.default_rng(cfg.seed + 21)
    F_obs_t_tr = back(nom, train_ae, ae_rng)
    poses_next_ae, qd_next_ae = windows.forward_window(nom, train_ae[:, 0:2], train_ae[:, 2:4], train_ae[:, 4:6], cfg)
    F_obs_next_tr = render.render_window(poses_next_ae, cfg, ae_rng)
    F_tr = np.vstack([F_obs_t_tr, F_obs_next_tr])
    ae = LatentAE(cfg, F_tr.shape[1], seed=cfg.seed).fit(F_tr)

    # ---- latent-dynamics ensemble on CLEAN (nominal) transitions (z_t,a)->z_next ----
    dyn_rng = np.random.default_rng(cfg.seed + 31)
    z_t_tr = ae.encode(back(nom, train_dyn, dyn_rng))
    z_next_tr = ae.encode(fwd(nom, train_dyn, dyn_rng)[0])
    A_tr = train_dyn[:, 4:6]
    ens = LatentEnsemble(cfg).fit(z_t_tr, A_tr, z_next_tr)

    mean_frame = F_tr.mean(0, keepdims=True)
    base_meanframe = float(np.mean((F_tr - mean_frame) ** 2))

    # ---- regime inputs ----
    # clean & physics_shift SHARE the input states (and so share obs_t): the only
    # difference is the regime physics used to roll the OUTPUT window forward.
    SA_full = test_id[:, 0:6]
    cap = min(cfg.max_dyn_samples, len(SA_full))
    SA = SA_full[np.random.default_rng(cfg.seed + 7).choice(len(SA_full), size=cap, replace=False)]
    obs_t_shared = back(nom, SA, np.random.default_rng(cfg.seed + 71))  # shared, nominal -> no leak

    def _cap(rows, seed):
        c = min(cfg.max_dyn_samples, len(rows))
        return rows[np.random.default_rng(seed).choice(len(rows), size=c, replace=False)]
    n_wide = max(8, cfg.n_test_id)
    rows_ex = _cap(np.vstack([wide_rollout(nom, cfg, rng) for _ in range(n_wide)]), cfg.seed + 8)
    rows_id = _cap(np.vstack([rollout(nom, cfg, rng) for _ in range(max(6, cfg.n_test_id // 2))]), cfg.seed + 9)

    S_stride = int(cfg.frame_stride)

    def build_pack(rows, fwd_arm, true_arm, obs_t, seed):
        rr = np.random.default_rng(seed)
        poses, qd_next = windows.forward_window(fwd_arm, rows[:, 0:2], rows[:, 2:4], rows[:, 4:6], cfg)
        obs_next = render.render_window(poses, cfg, rr)               # jittered observed
        obs_next_true = render.render_window(poses, cfg, rr, jitter=0.0)  # privileged non-jittered
        # qd_window = velocity after the FIRST rendered frame interval (S native dt steps),
        # under the SAME regime physics that rolls the output window -> window-scale qdd target.
        qd_window = _velocity_after(fwd_arm, rows[:, 0:2], rows[:, 2:4], rows[:, 4:6], S_stride, cfg.dt)
        return dict(q=rows[:, 0:2], qd=rows[:, 2:4], a=rows[:, 4:6], true_arm=true_arm,
                    obs_t=obs_t, obs_next=obs_next, obs_next_true=obs_next_true,
                    qd_next=qd_next, qd_window=qd_window, poses=poses)

    # order matches v3 table: clean / extrapolation / physics_shift / id_holdout
    packs = {}
    packs["clean"]         = build_pack(SA, nom, nom, obs_t_shared, cfg.seed + 101)
    packs["extrapolation"] = build_pack(rows_ex, nom, nom, back(nom, rows_ex, np.random.default_rng(cfg.seed + 103)), cfg.seed + 113)
    packs["physics_shift"] = build_pack(SA, heavy, heavy, obs_t_shared, cfg.seed + 102)
    packs["id_holdout"]    = build_pack(rows_id, nom, nom, back(nom, rows_id, np.random.default_rng(cfg.seed + 104)), cfg.seed + 114)

    # ---- detectors per regime ----
    S, recon = {}, {}
    for name, p in packs.items():
        z_t = ae.encode(p["obs_t"])
        z_next_obs = ae.encode(p["obs_next"])
        z_next_true = ae.encode(p["obs_next_true"])
        mean_next, _ = ens.predict_all(z_t, p["a"])
        eps = np.linalg.norm(mean_next - z_next_true, axis=1)         # privileged catastrophe signal
        u_z = ens.disagreement(z_t, p["a"])
        d_z = ens.input_distance(z_t, p["a"])
        c_z = np.linalg.norm(z_next_obs - mean_next, axis=1)          # observation-only consistency
        r_an = np.array([residual(res_nom, p["true_arm"], p["q"][i], p["qd"][i], p["a"][i])
                         for i in range(len(p["q"]))])
        S[name] = dict(u_z=u_z, d_z=d_z, c_z=c_z, eps=eps, r=r_an)
        recon[name] = float(ae.recon_error(p["obs_next"]).mean())     # recon on the OUTPUT window
        p["z_t"], p["z_next"] = z_t, z_next_obs

    # ---- calibrate on clean (5% total false alarm), unchanged ----
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

    # ---- GATE B: no hidden physics leaked into obs_t (matched clean/shift pair) ----
    cl, ps = S["clean"], S["physics_shift"]
    lab_b = np.r_[np.zeros(len(cl["d_z"])), np.ones(len(ps["d_z"]))]
    auroc_dz = float(roc_auc(lab_b, np.r_[cl["d_z"], ps["d_z"]]))
    auroc_uz = float(roc_auc(lab_b, np.r_[cl["u_z"], ps["u_z"]]))
    delta_dz = float(np.max(np.abs(cl["d_z"] - ps["d_z"])))           # aligned: same SA order
    delta_uz = float(np.max(np.abs(cl["u_z"] - ps["u_z"])))
    band = GATE_B_AUROC_BAND
    gate_b = {"auroc_dz": auroc_dz, "auroc_uz": auroc_uz, "band": list(band),
              "max_abs_delta_dz": delta_dz, "max_abs_delta_uz": delta_uz,
              "PASS": bool(band[0] <= auroc_dz <= band[1] and band[0] <= auroc_uz <= band[1])}

    # ---- GATE A (LW-10: WINDOW-scale qdd) + diagnostics + capacity-matched AE check ----
    # obs_next is a rendered window spanning frame_stride*dt, so the gate target is the
    # WINDOW-scale average acceleration over the first rendered frame interval, aligned to
    # the observation timescale -- NOT the native single-dt derivative (which the coarse
    # window cannot represent). Instantaneous qdd and pose-curvature are kept as diagnostics.
    S_dt = S_stride * dt
    a_ae = train_ae[:, 4:6]
    z_obs_next_tr = ae.encode(F_obs_next_tr)
    z_obs_t_tr = ae.encode(F_obs_t_tr)
    Pc, Ph = packs["clean"], packs["physics_shift"]

    # training targets (clean / nominal)
    qd_window_ae = _velocity_after(nom, train_ae[:, 0:2], train_ae[:, 2:4], a_ae, S_stride, dt)
    qddw_ae = (qd_window_ae - train_ae[:, 2:4]) / S_dt                       # PRIMARY: window-scale
    qddi_ae = (qd_next_ae - train_ae[:, 2:4]) / dt                          # diagnostic: instantaneous
    qddc_ae = (poses_next_ae[:, 2] - 2 * poses_next_ae[:, 1] + poses_next_ae[:, 0]) / (S_dt ** 2)  # pose-curv

    def qddw(P): return (P["qd_window"] - P["qd"]) / S_dt
    def qddi(P): return (P["qd_next"] - P["qd"]) / dt
    def qddc(P): return (P["poses"][:, 2] - 2 * P["poses"][:, 1] + P["poses"][:, 0]) / (S_dt ** 2)

    def H(*xs): return np.hstack(xs)

    def r2c(Xtr, ytr, Xte, yte):
        return _qdd_probe(Xtr, ytr, {"clean": (Xte, yte)})["clean"]["nonlinear_mean"]

    # --- GATE A predictor: raw obs_next + a -> qdd_window  (clean / physics_shift / pooled) ---
    on_w = _qdd_probe(H(F_obs_next_tr, a_ae), qddw_ae, {
        "clean":         (H(Pc["obs_next"], Pc["a"]), qddw(Pc)),
        "physics_shift": (H(Ph["obs_next"], Ph["a"]), qddw(Ph)),
        "pooled":        (np.vstack([H(Pc["obs_next"], Pc["a"]), H(Ph["obs_next"], Ph["a"])]),
                          np.vstack([qddw(Pc), qddw(Ph)])),
    })
    gate_a = {"floor": GATE_A_QDD, "target": "qdd_window = (qd_{t+S*dt} - qd_t)/(S*dt)",
              "clean": on_w["clean"]["nonlinear_mean"],
              "physics_shift": on_w["physics_shift"]["nonlinear_mean"],
              "pooled": on_w["pooled"]["nonlinear_mean"]}
    gate_a["PASS"] = bool(gate_a["pooled"] > GATE_A_QDD and gate_a["physics_shift"] > GATE_A_QDD)

    # --- recoverability tables on clean (single-seed, for the record) ---
    window_tbl = {
        "obs_next+a": gate_a["clean"],
        "z_next+a":   r2c(H(z_obs_next_tr, a_ae), qddw_ae, H(Pc["z_next"], Pc["a"]), qddw(Pc)),
        "obs_next":   r2c(F_obs_next_tr,          qddw_ae, Pc["obs_next"],           qddw(Pc)),
        "z_next":     r2c(z_obs_next_tr,          qddw_ae, Pc["z_next"],             qddw(Pc)),
    }
    instant_tbl = {  # diagnostic (native single-dt derivative; what LW-09 gated on)
        "obs_next+a": r2c(H(F_obs_next_tr, a_ae), qddi_ae, H(Pc["obs_next"], Pc["a"]), qddi(Pc)),
        "z_next+a":   r2c(H(z_obs_next_tr, a_ae), qddi_ae, H(Pc["z_next"], Pc["a"]), qddi(Pc)),
    }
    posecurv_tbl = {  # diagnostic (second pose difference over the window)
        "obs_next+a": r2c(H(F_obs_next_tr, a_ae), qddc_ae, H(Pc["obs_next"], Pc["a"]), qddc(Pc)),
        "z_next+a":   r2c(H(z_obs_next_tr, a_ae), qddc_ae, H(Pc["z_next"], Pc["a"]), qddc(Pc)),
    }

    # --- capacity-matched comparison for AE-LIMIT vs DEEP RED (probe parity) ---
    # PCA on CLEAN-TRAINING obs_next only -> ~latent_dim, then the SAME nonlinear probe as
    # z_next; both repeated over 3 seeds (mean/std). The AE-LIMIT/DEEP RED decision uses THIS
    # capacity-matched comparison, never the raw high-dim obs_next probe (which underestimates).
    k = int(cfg.latent_dim)
    pca_mu, pca_comps = _pca_fit(F_obs_next_tr, k)
    obs_pca_tr = _pca_transform(F_obs_next_tr, pca_mu, pca_comps)
    obs_pca_te = _pca_transform(Pc["obs_next"], pca_mu, pca_comps)
    capm_obs = _multiseed_r2(H(obs_pca_tr, a_ae), qddw_ae, H(obs_pca_te, Pc["a"]), qddw(Pc))
    capm_z   = _multiseed_r2(H(z_obs_next_tr, a_ae), qddw_ae, H(Pc["z_next"], Pc["a"]), qddw(Pc))
    noise_floor = float(capm_obs["std"] + capm_z["std"])
    gap = float(capm_obs["mean"] - capm_z["mean"])
    # consequence reached the latent iff z_next is NOT meaningfully worse than capacity-matched
    # obs_next (within the AE margin OR within probe noise).
    reached = bool(gap <= AE_LIMIT_MARGIN or gap <= noise_floor)

    accel = {"recover_floor": GATE_A_QDD, "S_dt": float(S_dt),
             "qdd_window": window_tbl, "qdd_instant_diagnostic": instant_tbl,
             "qdd_pose_curv_diagnostic": posecurv_tbl,
             "gate_a_per_regime": {kk: on_w[kk]["nonlinear_mean"]
                                   for kk in ("clean", "physics_shift", "pooled")}}
    capacity = {"target": "qdd_window", "pca_components": k,
                "capacity_matched_obs_next+a": capm_obs, "z_next+a": capm_z,
                "raw_obs_next+a_diagnostic": window_tbl["obs_next+a"],
                "margin": AE_LIMIT_MARGIN, "probe_noise_floor": noise_floor, "gap": gap,
                "consequence_reached_latent": reached}

    # ---- laundering / vacuity probe on z_t (q, qd) ----
    z_probe_tr = z_obs_t_tr; s_probe_tr = train_ae[:, 0:4]
    z_probe_te = Pc["z_t"];  s_probe_te = SA[:, 0:4]
    r2_per = linear_probe_r2(z_probe_tr, s_probe_tr, z_probe_te, s_probe_te)
    r2_mean = float(np.mean(r2_per))
    r2_nl_per = nonlinear_probe_r2(z_probe_tr, s_probe_tr, z_probe_te, s_probe_te, cfg)
    r2_nl_mean = float(np.mean(r2_nl_per))

    # ---- post-hoc correlations on the pooled non-clean set ----
    pool = ["extrapolation", "physics_shift", "id_holdout"]
    cz_pool = np.concatenate([S[n]["c_z"] for n in pool])
    dz_pool = np.concatenate([S[n]["d_z"] for n in pool])
    r_pool = np.concatenate([S[n]["r"] for n in pool])
    rho_cz_r = spearman(cz_pool, r_pool)
    rho_cz_dz = spearman(cz_pool, dz_pool)

    # ---- recon gate (on the OUTPUT window, where physics_shift actually differs) ----
    recon_clean = recon["clean"]; recon_shift = recon["physics_shift"]
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

    # ---- final 4-way label ----
    verdict = _final_label(recon_gate, gate_a, gate_b, capacity, table,
                           r2_mean, r2_nl_mean, rho_cz_r, rho_cz_dz)

    out = dict(
        eps_thr=float(eps_thr),
        or_false_alarm=float(or_fire(sc, or_q)[0].mean()),
        missed_catastrophe=table,
        detect_auroc=aurocs,
        gate_a=gate_a,
        gate_b=gate_b,
        accel_recoverability=accel,
        ae_capacity=capacity,
        linear_probe_r2={"per_target_q1_q2_qd1_qd2": [float(x) for x in r2_per],
                         "mean": r2_mean, "band": [R2_FLOOR, R2_CEIL],
                         "role": "laundering ceiling: R2>0.9 => latent recovered privileged state"},
        nonlinear_probe_r2={"per_target_q1_q2_qd1_qd2": [float(x) for x in r2_nl_per],
                            "mean": r2_nl_mean,
                            "role": "vacuity floor: low even here => latent truly discarded state"},
        spearman={"c_z_vs_r": rho_cz_r, "c_z_vs_d_z": rho_cz_dz, "rho_max_for_orthogonal": CZDZ_RHO_MAX},
        recon_gate=recon_gate,
        training_failed={"ae": bool(ae.training_failed), "ensemble": bool(ens.training_failed)},
        verdict=verdict,
    )
    _print(out); _save(cfg, out, S, eps_thr)
    return out


def _final_label(recon_gate, gate_a, gate_b, capacity, table,
                 r2_lin, r2_nl, rho_cz_r, rho_cz_dz):
    """LW-10 label rule. Gate B (leak) and Gate A (window-scale qdd fidelity) are
    interpretation gates; only when both pass AND the consequence demonstrably reached
    the latent (capacity-matched comparison, with probe noise) do we read c_z."""
    # --- rig-validity / unreadable conditions ---
    if not recon_gate["PASS"]:
        return {"label": "VOID", "interpret_table": False,
                "reasons": ["recon gate failed (AE did not beat baseline, or physics_shift "
                            "output recon degraded, or training failed) — a training failure "
                            "must not masquerade as a scientific boundary."]}
    if not gate_b["PASS"]:
        return {"label": "LEAK-INVALID / VOID", "interpret_table": False,
                "reasons": [f"GATE B failed: hidden physics leaked into obs_t "
                            f"(AUROC d_z={gate_b['auroc_dz']:.2f}, u_z={gate_b['auroc_uz']:.2f} "
                            f"outside {gate_b['band']}). Detector table NOT interpreted."]}
    if not gate_a["PASS"]:
        return {"label": "VOID-FIDELITY", "interpret_table": False,
                "reasons": [f"GATE A failed on WINDOW-scale q-double-dot (pooled R2={gate_a['pooled']:.2f}, "
                            f"physics_shift R2={gate_a['physics_shift']:.2f}, floor {gate_a['floor']}): even "
                            "window-scale acceleration is not recoverable from the observation. A "
                            "render/window-fidelity change is justified (NOT performed in this task). "
                            "Detector table NOT interpreted."]}

    # --- both interpretation gates pass: read the result ---
    reached = capacity["consequence_reached_latent"]
    r2o = capacity["capacity_matched_obs_next+a"]["mean"]
    r2z = capacity["z_next+a"]["mean"]
    nf = capacity["probe_noise_floor"]; gap = capacity["gap"]
    ps_cz = table["physics_shift"]["c_z"]
    cz_catches = (ps_cz is not None) and ps_cz < 0.5
    laundered = r2_lin > R2_CEIL
    vacuous = r2_nl < R2_FLOOR

    common = {"interpret_table": True,
              "gate_a_pass": True, "gate_b_pass": True,
              "consequence_reached_latent": bool(reached),
              "capacity_matched_obs_R2": r2o, "z_next_R2": r2z,
              "gap": gap, "probe_noise_floor": nf,
              "cz_catches_physics_shift": bool(cz_catches),
              "cz_tracks_r_spearman": rho_cz_r,
              "cz_vs_dz_spearman": rho_cz_dz,
              "latent_not_laundered": bool(not laundered),
              "latent_not_vacuous": bool(not vacuous)}

    if not reached:
        return {"label": "AE-LIMIT",
                "reasons": [f"window-scale q-double-dot reaches the (capacity-matched) frames "
                            f"R2={r2o:.2f} but NOT the latent (z_next+a R2={r2z:.2f}, gap={gap:.2f} > "
                            f"max(margin {AE_LIMIT_MARGIN}, noise {nf:.2f})): the AE bottleneck discarded "
                            "the acceleration. c_z cannot be blamed for a consequence the encoder threw "
                            "away. Fix = widen latent_dim, rerun."],
                **common}
    if cz_catches:
        return {"label": "GREEN",
                "reasons": [f"consequence reached the latent (z_next+a R2={r2z:.2f} ~ capacity-matched "
                            f"obs_next+a R2={r2o:.2f}, gap={gap:.2f}) AND c_z catches physics_shift "
                            f"({table['physics_shift']['c_z']:.0%} missed): learned transition consistency "
                            "recovers the analytic residual's role once the observation carries the "
                            "consequence."],
                **common}
    return {"label": "DEEP RED",
            "reasons": [f"consequence reached the latent (z_next+a R2={r2z:.2f} ~ capacity-matched "
                        f"obs_next+a R2={r2o:.2f}, gap={gap:.2f}) yet c_z still MISSES physics_shift "
                        f"({table['physics_shift']['c_z']:.0%} missed): learned transition consistency "
                        "fails despite the consequence being accessible in z — auditability needs "
                        "privileged structure, not just a learned latent."],
            **common}


def _fmt(x): return " n/a " if x is None else f"{x:5.1%}"


def _print(o):
    cols = ["u_z", "d_z", "c_z", "PORTFOLIO"]
    v = o["verdict"]
    ga, gb = o["gate_a"], o["gate_b"]
    ac = o["accel_recoverability"]; cap = o["ae_capacity"]
    print("\n" + "=" * 78)
    print("LW-10 LATENT FAILURE-MODE (window-scale q-double-dot gate)")
    print("=" * 78)

    print(f"\nGATE A — obs carries WINDOW-scale q-double-dot  (hard gate: pooled & physics_shift > {ga['floor']})")
    print(f"  target: {ga['target']}")
    print(f"  nonlinear R2(obs_next+a -> qdd_window):  clean={ga['clean']:.3f}  "
          f"physics_shift={ga['physics_shift']:.3f}  pooled={ga['pooled']:.3f}")
    print(f"  => GATE A {'PASS' if ga['PASS'] else 'FAIL (VOID-FIDELITY)'}")

    print(f"\nGATE B — no hidden physics leaked into obs_t  (AUROC clean vs physics_shift in {gb['band']})")
    print(f"  AUROC d_z={gb['auroc_dz']:.3f}  u_z={gb['auroc_uz']:.3f}  "
          f"max|delta| d_z={gb['max_abs_delta_dz']:.2e} u_z={gb['max_abs_delta_uz']:.2e}  "
          f"=> GATE B {'PASS' if gb['PASS'] else 'FAIL (LEAK-INVALID)'}")

    rg = o["recon_gate"]
    print("\nRECON GATE (output window):")
    print(f"  recon clean={rg['ae_recon_clean']:.5f} vs baseline={rg['mean_frame_baseline']:.5f} "
          f"beats(>= {rg['beat_factor_required']}x)={rg['beats_baseline']}; "
          f"shift ratio={rg['shift_ratio']:.2f} (<= {rg['shift_ratio_max']})={rg['shift_comparable']} "
          f"=> {'PASS' if rg['PASS'] else 'FAIL (VOID)'}")

    w = ac["qdd_window"]; di = ac["qdd_instant_diagnostic"]; dc = ac["qdd_pose_curv_diagnostic"]
    print("\nq-double-dot RECOVERABILITY  R^2(. -> target)  (clean):")
    print(f"  [WINDOW ]  obs_next+a={w['obs_next+a']:.3f}  z_next+a={w['z_next+a']:.3f}  "
          f"obs_next={w['obs_next']:.3f}  z_next={w['z_next']:.3f}")
    print(f"  [instant]  obs_next+a={di['obs_next+a']:.3f}  z_next+a={di['z_next+a']:.3f}   (diagnostic)")
    print(f"  [posecrv]  obs_next+a={dc['obs_next+a']:.3f}  z_next+a={dc['z_next+a']:.3f}   (diagnostic)")

    co = cap["capacity_matched_obs_next+a"]; cz = cap["z_next+a"]
    print(f"\nCAPACITY-MATCHED (PCA->{cap['pca_components']}d) 3-seed qdd_window  [AE-LIMIT vs DEEP RED basis]:")
    print(f"  obs_next+a  mean={co['mean']:.3f} std={co['std']:.3f} per_seed={[round(x,3) for x in co['per_seed']]}")
    print(f"  z_next+a    mean={cz['mean']:.3f} std={cz['std']:.3f} per_seed={[round(x,3) for x in cz['per_seed']]}")
    print(f"  gap={cap['gap']:.3f}  margin={cap['margin']}  noise_floor={cap['probe_noise_floor']:.3f} "
          f"-> consequence_reached_latent={cap['consequence_reached_latent']}")

    interp = v.get("interpret_table", True)
    print("\nDETECTOR TABLE — missed-catastrophe (lower=safer), 5% total FA"
          + ("" if interp else "   [NOT INTERPRETED: gates failed]"))
    print(f"{'regime':14s} {'#cat':>5s} " + " ".join(f"{c:>10s}" for c in cols))
    for reg, row in o["missed_catastrophe"].items():
        print(f"{reg:14s} {row['n_cata']:5d} " + " ".join(f"{_fmt(row[c]):>10s}" for c in cols))
    print(f"(portfolio OR false-alarm on clean = {o['or_false_alarm']:.1%})")

    lp = o["linear_probe_r2"]; nlp = o["nonlinear_probe_r2"]; sp = o["spearman"]
    print(f"\nLINEAR PROBE     R^2(z->q,qd) mean={lp['mean']:.3f}  (laundering ceiling {R2_CEIL})  "
          f"per-target={[round(x,2) for x in lp['per_target_q1_q2_qd1_qd2']]}")
    print(f"NONLINEAR PROBE  R^2(z->q,qd) mean={nlp['mean']:.3f}  (vacuity floor {R2_FLOOR})      "
          f"per-target={[round(x,2) for x in nlp['per_target_q1_q2_qd1_qd2']]}")
    print(f"SPEARMAN  c_z vs r={sp['c_z_vs_r']:.3f}   c_z vs d_z={sp['c_z_vs_d_z']:.3f} "
          f"(orthogonal iff <= {sp['rho_max_for_orthogonal']})")

    print(f"\nFINAL LABEL: {v['label']}")
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
            val = M[i, j]
            ax.text(j, i, "n/a" if np.isnan(val) else f"{val:.0%}", ha="center", va="center", fontsize=10)
    ax.set_title(f"LW-10 latent missed-catastrophe (green=safe). FINAL LABEL: {o['verdict']['label']}")
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
