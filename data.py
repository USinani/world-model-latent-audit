"""
data.py — rollout generation. Each trajectory samples ONE (mass, damping) pair
and holds it fixed; the param multiplier range is what defines ID vs OOD. The
shift is PHYSICAL (mass/damping), not visual — that is the point of the test.

Row layout: [q(2), qd(2), tau(2), q_next(2), qd_next(2)]  -> 10 columns.

PORTED VERBATIM from the validated wedge v3 tree (phase-close commit 72866ed).
"""
from __future__ import annotations
import numpy as np
from arm import TwoLinkArm
from config import Config


def _smooth_torque(t, phase, freq, amp, tau_max):
    tau = amp * (np.sin(freq*t + phase) + 0.5*np.sin(2.3*freq*t + 1.7*phase))
    return np.clip(tau, -tau_max, tau_max)


def rollout(arm: TwoLinkArm, cfg: Config, rng):
    q = rng.uniform(-0.5, 0.5, size=2)
    qd = np.zeros(2)
    phase = rng.uniform(0, 2*np.pi, size=2)
    freq = rng.uniform(1.0, 4.0, size=2)
    amp = rng.uniform(2.0, cfg.tau_max, size=2)
    rec = []
    for k in range(cfg.steps):
        tau = _smooth_torque(k*cfg.dt, phase, freq, amp, cfg.tau_max)
        q2, qd2 = arm.step_rk4(q, qd, tau, cfg.dt)
        rec.append(np.concatenate([q, qd, tau, q2, qd2]))
        q, qd = q2, qd2
        if not np.all(np.isfinite(q)):
            break
    return np.array(rec)


def make_set(n, lo, hi, cfg: Config, rng):
    blocks = []
    for _ in range(n):
        m = cfg.nominal_m * rng.uniform(lo, hi, size=2)
        b = cfg.nominal_b * rng.uniform(lo, hi, size=2)
        arm = TwoLinkArm(m1=m[0], m2=m[1], b1=b[0], b2=b[1])
        blocks.append(rollout(arm, cfg, rng))
    return np.vstack(blocks)


def build_datasets(cfg: Config):
    rng = np.random.default_rng(cfg.seed)
    train = make_set(cfg.n_train, cfg.id_lo, cfg.id_hi, cfg, rng)
    test_id = make_set(cfg.n_test_id, cfg.id_lo, cfg.id_hi, cfg, rng)
    test_ood = make_set(cfg.n_test_ood, cfg.ood_lo, cfg.ood_hi, cfg, rng)
    return train, test_id, test_ood
