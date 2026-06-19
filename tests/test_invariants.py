"""
test_invariants.py — physics self-audit. These are the repo's pre-commit guards:
if the dynamics, the energy bookkeeping, or the residual silently break, these
fail BEFORE any experiment number is trusted. Same spirit as catching a torque
plot sitting at the floating-point noise floor.

PORTED VERBATIM from the wedge v3 tree (phase-close commit 72866ed).

Run:  python tests/test_invariants.py     (plain)
  or:  pytest tests/                       (if installed)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from arm import TwoLinkArm, torque_residual


def test_energy_conserved_no_actuation():
    """No torque, no damping => mechanical energy conserved (up to RK4 drift)."""
    arm = TwoLinkArm(b1=0.0, b2=0.0)
    q, qd = np.array([0.3, -0.4]), np.array([0.5, -0.7])
    E0 = arm.energy(q, qd)
    dt = 0.005
    for _ in range(400):                       # 2 s
        q, qd = arm.step_rk4(q, qd, np.zeros(2), dt)
    rel_drift = abs(arm.energy(q, qd) - E0) / abs(E0)
    assert rel_drift < 1e-3, f"energy drift {rel_drift:.2e} too large (integrator broken?)"
    return rel_drift


def test_damping_dissipates():
    """Positive damping, no torque => energy monotonically decreases."""
    arm = TwoLinkArm(b1=0.3, b2=0.3)
    q, qd = np.array([0.2, 0.5]), np.array([0.8, -0.6])
    E = [arm.energy(q, qd)]
    for _ in range(300):
        q, qd = arm.step_rk4(q, qd, np.zeros(2), 0.01)
        E.append(arm.energy(q, qd))
    E = np.array(E)
    assert np.all(np.diff(E) <= 1e-9), "energy increased under pure damping (sign error?)"
    return E[0] - E[-1]


def test_gravity_is_potential_gradient():
    """grav(q) must equal dV/dq (finite-difference check) — guards the V used in energy()."""
    arm = TwoLinkArm()
    rng = np.random.default_rng(3)
    h = 1e-6
    worst = 0.0
    for _ in range(20):
        q = rng.uniform(-2, 2, size=2)
        # V from energy() at zero velocity is purely potential
        def V(qq): return arm.energy(qq, np.zeros(2))
        dV = np.array([(V(q + h*np.eye(2)[i]) - V(q - h*np.eye(2)[i])) / (2*h) for i in range(2)])
        worst = max(worst, np.max(np.abs(dV - arm.grav(q))))
    assert worst < 1e-4, f"grav(q) != dV/dq (max err {worst:.2e}) — convention mismatch"
    return worst


def test_residual_zero_when_model_correct():
    """If nominal params == true params, the residual must vanish (validates the detector)."""
    true_arm = TwoLinkArm(m1=1.3, m2=0.7, b1=0.15, b2=0.08)
    dt, q, qd = 0.01, np.array([0.4, -0.2]), np.array([0.6, 0.3])
    tau = np.array([2.0, -1.0])
    _, qd2 = true_arm.step_rk4(q, qd, tau, dt)
    # midpoint-free single-step finite-diff residual carries O(dt) error; allow small tol
    r = torque_residual(true_arm, q, qd, qd2, tau, dt)
    assert r < 0.2, f"residual {r:.3f} should be ~0 when nominal==true (off by integration error only)"
    return r


def _main():
    tests = [test_energy_conserved_no_actuation, test_damping_dissipates,
             test_gravity_is_potential_gradient, test_residual_zero_when_model_correct]
    print("physics self-audit:")
    ok = True
    for t in tests:
        try:
            val = t()
            print(f"  PASS  {t.__name__:42s} ({val:.2e})")
        except AssertionError as e:
            ok = False
            print(f"  FAIL  {t.__name__:42s} -> {e}")
    print("ALL PASS" if ok else "FAILURES PRESENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _main()
