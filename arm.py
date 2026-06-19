"""
arm.py — physics core. The ground-truth world AND the (deliberately imperfect)
nominal model used by the physics-residual detector both live here, so there is
exactly one place where the dynamics are defined and audited.

2-link planar arm in a vertical plane. Thin-rod inertia. Joint angles measured
from horizontal, gravity acts downward. Convention checked by test_invariants.py
(energy conservation under no actuation / no damping).

PORTED VERBATIM from the validated wedge v3 tree (phase-close commit 72866ed).
Do not modify without re-running tests/test_invariants.py.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

G = 9.81


@dataclass(frozen=True)
class TwoLinkArm:
    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    b1: float = 0.10        # viscous joint damping
    b2: float = 0.10
    mu_c: float = 0.0       # Coulomb friction magnitude (unmodeled structural term; 0 = off)

    # --- derived inertial constants (Spong form) ---------------------------
    def _consts(self):
        lc1, lc2 = self.l1 / 2, self.l2 / 2
        I1 = self.m1 * self.l1**2 / 12.0
        I2 = self.m2 * self.l2**2 / 12.0
        a1 = self.m1*lc1**2 + I1 + self.m2*(self.l1**2 + lc2**2) + I2
        a2 = self.m2*self.l1*lc2
        a3 = self.m2*lc2**2 + I2
        return a1, a2, a3, lc1, lc2

    def M(self, q):
        a1, a2, a3, *_ = self._consts()
        c2 = np.cos(q[1])
        return np.array([[a1 + 2*a2*c2, a3 + a2*c2],
                         [a3 + a2*c2,   a3]])

    def Cqd(self, q, qd):
        _, a2, _, *_ = self._consts()
        s2 = np.sin(q[1])
        C = np.array([[-a2*s2*qd[1], -a2*s2*(qd[0]+qd[1])],
                      [ a2*s2*qd[0],  0.0]])
        return C @ qd

    def grav(self, q):
        _, _, _, lc1, lc2 = self._consts()
        g1 = (self.m1*lc1 + self.m2*self.l1)*G*np.cos(q[0]) + self.m2*lc2*G*np.cos(q[0]+q[1])
        g2 = self.m2*lc2*G*np.cos(q[0]+q[1])
        return np.array([g1, g2])

    @property
    def b(self):
        return np.array([self.b1, self.b2])

    def qddot(self, q, qd, tau):
        rhs = tau - self.Cqd(q, qd) - self.grav(q) - self.b*qd - self.mu_c*np.sign(qd)
        return np.linalg.solve(self.M(q), rhs)

    def step_rk4(self, q, qd, tau, dt):
        def f(state):
            return np.concatenate([state[2:], self.qddot(state[:2], state[2:], tau)])
        s = np.concatenate([q, qd])
        k1 = f(s); k2 = f(s + dt/2*k1); k3 = f(s + dt/2*k2); k4 = f(s + dt*k3)
        s2 = s + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        return s2[:2], s2[2:]

    def energy(self, q, qd):
        """Total mechanical energy T+V. V consistent with grav() (the gradient check
        is in test_invariants). Used as a physics audit channel and a test oracle."""
        _, _, _, lc1, lc2 = self._consts()
        T = 0.5 * qd @ self.M(q) @ qd
        V = G*((self.m1*lc1 + self.m2*self.l1)*np.sin(q[0]) + self.m2*lc2*np.sin(q[0]+q[1]))
        return T + V

    # --- forward kinematics (added for render.py; pure geometry, no dynamics) ---
    def joint_positions(self, q):
        """Cartesian (x, y) of base, elbow, and tip. Used only for rendering."""
        base = np.array([0.0, 0.0])
        elbow = base + np.array([self.l1*np.cos(q[0]), self.l1*np.sin(q[0])])
        tip = elbow + np.array([self.l2*np.cos(q[0]+q[1]), self.l2*np.sin(q[0]+q[1])])
        return base, elbow, tip


def torque_residual(nominal: TwoLinkArm, q, qd, qd_next, tau, dt):
    """Analytic 'physics-consistency alarm' for ONE observed transition, using the
    NOMINAL arm's assumed parameters. Estimates qddot by finite difference of
    observed velocity (i.e. from sensing, not from sim internals), then checks how
    badly the manipulator equation is violated. NOT an oracle: if nominal params
    differ from the true world (mass, damping, or an unmodeled term), r > 0.
    Returns scalar ||tau_predicted - tau_applied||.
    """
    qdd_obs = (qd_next - qd) / dt
    tau_pred = nominal.M(q) @ qdd_obs + nominal.Cqd(q, qd) + nominal.grav(q) + nominal.b*qd
    return float(np.linalg.norm(tau_pred - tau))
