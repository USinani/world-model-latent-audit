"""
test_numpy_swaps.py — guards on the pure-numpy replacements for the sklearn/scipy
calls the validated v3 harness used. If these drift, the parity gate and the
post-hoc correlations become untrustworthy, so they are checked BEFORE any number.

Run:  python tests/test_numpy_swaps.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from metrics_np import roc_auc, spearman
from numpy_mlp import NumpyMLP


def test_roc_auc_perfect_separation():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    auc = roc_auc(labels, scores)
    assert abs(auc - 1.0) < 1e-9, f"perfect separation should be AUC=1, got {auc}"
    return auc


def test_roc_auc_random_is_half():
    rng = np.random.default_rng(0)
    labels = np.r_[np.zeros(2000), np.ones(2000)]
    scores = rng.standard_normal(4000)            # independent of labels
    auc = roc_auc(labels, scores)
    assert abs(auc - 0.5) < 0.03, f"random scores should be ~0.5, got {auc}"
    return auc


def test_roc_auc_tie_aware():
    # all-tie scores must give exactly 0.5 (Mann-Whitney with full ties)
    labels = np.array([0, 1, 0, 1])
    scores = np.array([1.0, 1.0, 1.0, 1.0])
    auc = roc_auc(labels, scores)
    assert abs(auc - 0.5) < 1e-9, f"all ties should be 0.5, got {auc}"
    return auc


def test_spearman_monotone():
    x = np.linspace(0, 1, 50)
    y = np.exp(3 * x)                              # strictly increasing, nonlinear
    rho = spearman(x, y)
    assert abs(rho - 1.0) < 1e-9, f"monotone increasing should be rho=1, got {rho}"
    return rho


def test_spearman_sign():
    x = np.linspace(0, 1, 50)
    rho = spearman(x, -x)
    assert abs(rho + 1.0) < 1e-9, f"monotone decreasing should be rho=-1, got {rho}"
    return rho


def test_mlp_fits_linear_map():
    """The backbone must actually learn a simple map (guards the optimiser/init)."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((800, 3))
    W = np.array([[1.0, -2.0], [0.5, 0.0], [0.0, 3.0]])
    Y = X @ W + 0.01 * rng.standard_normal((800, 2))
    net = NumpyMLP((3, 32, 2), activation="tanh", max_iter=200, seed=0,
                   standardize_x=True, standardize_y=True).fit(X, Y)
    assert not net.training_failed, "training failed on a trivial linear map"
    pred = net.predict(X)
    ss_res = ((Y - pred) ** 2).sum()
    ss_tot = ((Y - Y.mean(0)) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    assert r2 > 0.95, f"MLP failed to fit linear map, R2={r2:.3f}"
    return r2


def test_mlp_linear_bottleneck_autoencoder():
    """encode_to/decode_from with a linear bottleneck must round-trip a low-rank set."""
    rng = np.random.default_rng(2)
    Z = rng.standard_normal((600, 2))
    A = rng.standard_normal((8, 2))
    X = Z @ A.T                                    # rank-2 data in 8-D
    sizes = (8, 16, 2, 16, 8)
    ae = NumpyMLP(sizes, activation="tanh", max_iter=400, seed=0,
                  standardize_x=True, standardize_y=False, linear_layers=(1,)).fit(X, X)
    assert not ae.training_failed
    rec = ae.decode_from(ae.encode_to(X, 2), 2)
    mse = float(np.mean((rec - X) ** 2))
    assert mse < 0.05 * np.var(X), f"AE round-trip too lossy on rank-2 data, mse={mse:.4f}"
    return mse


def _main():
    tests = [test_roc_auc_perfect_separation, test_roc_auc_random_is_half,
             test_roc_auc_tie_aware, test_spearman_monotone, test_spearman_sign,
             test_mlp_fits_linear_map, test_mlp_linear_bottleneck_autoencoder]
    print("numpy-swap self-audit:")
    ok = True
    for t in tests:
        try:
            val = t()
            print(f"  PASS  {t.__name__:42s} ({val:.3e})")
        except AssertionError as e:
            ok = False
            print(f"  FAIL  {t.__name__:42s} -> {e}")
    print("ALL PASS" if ok else "FAILURES PRESENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _main()
