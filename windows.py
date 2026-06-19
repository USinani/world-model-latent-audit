"""
windows.py — build REAL stacked-frame windows by integrating the arm forward and
backward in time, so an observation window carries genuine curvature (acceleration)
rather than the constant-velocity back-extrapolation used in render.render_batch.

Why this exists (LW-09): the mass-shift consequence lives in the acceleration, i.e.
the within-window second difference of pose. A constant-velocity window has ~zero
second difference, so the consequence is invisible and the latent c_z test is not
interpretable. Here the output window [f_t, f_{t+1}, f_{t+2}] is produced by actually
stepping the dynamics under the regime physics (zero-order-hold action), which puts
the curvature into the pixels.

Integrator: a VECTORISED RK4 that reproduces TwoLinkArm.qddot in batch form. To stay
anchored to the single audited physics source (arm.py), it reads the arm's own
inertial constants via arm._consts() and public params, and a one-time runtime parity
assert checks rk4_batch against the scalar arm.step_rk4 (raise on mismatch). Editing
arm.py is out of scope for this task, so the batch form lives here.
"""
from __future__ import annotations
import numpy as np
from arm import TwoLinkArm, G

_PARITY_CHECKED: set = set()


def qddot_batch(arm: TwoLinkArm, Q, QD, TAU):
    """Vectorised TwoLinkArm.qddot. Q,QD,TAU: (N,2). Returns (N,2)."""
    a1, a2, a3, lc1, lc2 = arm._consts()
    c2 = np.cos(Q[:, 1]); s2 = np.sin(Q[:, 1])
    # mass matrix (N,2,2)
    m00 = a1 + 2 * a2 * c2; m01 = a3 + a2 * c2; m11 = np.full_like(m00, a3)
    M = np.stack([np.stack([m00, m01], axis=1),
                  np.stack([m01, m11], axis=1)], axis=1)            # (N,2,2)
    # Coriolis term C(q,qd) @ qd  -> (N,2)
    cqd0 = -a2 * s2 * QD[:, 1] * QD[:, 0] - a2 * s2 * (QD[:, 0] + QD[:, 1]) * QD[:, 1]
    cqd1 = a2 * s2 * QD[:, 0] * QD[:, 0]
    Cqd = np.stack([cqd0, cqd1], axis=1)
    # gravity (N,2)
    g0 = (arm.m1 * lc1 + arm.m2 * arm.l1) * G * np.cos(Q[:, 0]) + arm.m2 * lc2 * G * np.cos(Q[:, 0] + Q[:, 1])
    g1 = arm.m2 * lc2 * G * np.cos(Q[:, 0] + Q[:, 1])
    grav = np.stack([g0, g1], axis=1)
    b = np.array([arm.b1, arm.b2])
    rhs = TAU - Cqd - grav - b * QD - arm.mu_c * np.sign(QD)        # (N,2)
    return np.linalg.solve(M, rhs[..., None])[..., 0]


def rk4_batch(arm: TwoLinkArm, Q, QD, TAU, h):
    """One vectorised RK4 step of size h on (Q,QD) with held torque TAU."""
    def acc(q, qd):
        return qddot_batch(arm, q, qd, TAU)
    k1q = QD;                 k1v = acc(Q, QD)
    k2q = QD + 0.5 * h * k1v; k2v = acc(Q + 0.5 * h * k1q, QD + 0.5 * h * k1v)
    k3q = QD + 0.5 * h * k2v; k3v = acc(Q + 0.5 * h * k2q, QD + 0.5 * h * k2v)
    k4q = QD + h * k3v;       k4v = acc(Q + h * k3q, QD + h * k3v)
    Qn = Q + (h / 6.0) * (k1q + 2 * k2q + 2 * k3q + k4q)
    QDn = QD + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return Qn, QDn


def _assert_parity(arm: TwoLinkArm, dt: float):
    """One-time guard: batched RK4 must match the audited scalar arm.step_rk4."""
    key = (id(arm), round(dt, 12))
    if key in _PARITY_CHECKED:
        return
    rng = np.random.default_rng(123)
    Q = rng.uniform(-1.5, 1.5, size=(7, 2))
    QD = rng.uniform(-2.5, 2.5, size=(7, 2))
    TAU = rng.uniform(-6.0, 6.0, size=(7, 2))
    for h in (dt, -dt):
        Qn, QDn = rk4_batch(arm, Q, QD, TAU, h)
        for i in range(len(Q)):
            q2, qd2 = arm.step_rk4(Q[i], QD[i], TAU[i], h)
            if not (np.allclose(Qn[i], q2, atol=1e-9, rtol=0)
                    and np.allclose(QDn[i], qd2, atol=1e-9, rtol=0)):
                raise AssertionError(
                    "rk4_batch diverged from arm.step_rk4 — batched physics is not "
                    "faithful to the audited source; refusing to proceed.")
    _PARITY_CHECKED.add(key)


def forward_window(arm: TwoLinkArm, q, qd, a, cfg):
    """Real forward window under `arm`, zero-order-hold action a.
    Returns (poses (N,F,2), qd_next_1dt (N,2)).
    poses are the arm at t, t+stride*dt, ..., t+(F-1)*stride*dt; qd_next_1dt is the
    velocity after a single native dt step (for qdd = (qd_next - qd)/dt)."""
    _assert_parity(arm, cfg.dt)
    q = np.asarray(q, float).copy(); qd = np.asarray(qd, float).copy()
    a = np.asarray(a, float)
    N = len(q); F = cfg.frame_stack; stride = int(getattr(cfg, "frame_stride", 1))
    poses = np.empty((N, F, 2))
    poses[:, 0] = q
    Q, QD = q.copy(), qd.copy()
    qd_next_1dt = None
    substep = 0
    for f in range(1, F):
        for _ in range(stride):
            Q, QD = rk4_batch(arm, Q, QD, a, cfg.dt)
            substep += 1
            if substep == 1:
                qd_next_1dt = QD.copy()
        poses[:, f] = Q
    if qd_next_1dt is None:                          # F == 1 fallback
        _, qd_next_1dt = rk4_batch(arm, q.copy(), qd.copy(), a, cfg.dt)
    return poses, qd_next_1dt


def backward_window(arm: TwoLinkArm, q, qd, a, cfg):
    """Real backward window under `arm` (intended NOMINAL), zero-order-hold action a.
    Returns poses (N,F,2) ending at the current pose: t-(F-1)*stride*dt, ..., t."""
    _assert_parity(arm, cfg.dt)
    q = np.asarray(q, float).copy(); qd = np.asarray(qd, float).copy()
    a = np.asarray(a, float)
    N = len(q); F = cfg.frame_stack; stride = int(getattr(cfg, "frame_stride", 1))
    poses = np.empty((N, F, 2))
    poses[:, F - 1] = q
    Q, QD = q.copy(), qd.copy()
    for f in range(F - 2, -1, -1):
        for _ in range(stride):
            Q, QD = rk4_batch(arm, Q, QD, a, -cfg.dt)
        poses[:, f] = Q
    return poses
