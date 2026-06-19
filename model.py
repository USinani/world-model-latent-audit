"""
model.py — bootstrap ensemble of MLPs that predict delta-state from (state, action),
trained ONLY on in-distribution data. Provides the learned-uncertainty channel
(ensemble disagreement) and the mean prediction used to compute true error eps.
z-scoring uses TRAIN statistics so eps is dimensionless and balanced across dims.

PORTED from the wedge v3 tree (phase-close commit 72866ed). The ONLY change vs v3
is the learned member: `sklearn.neural_network.MLPRegressor` -> `NumpyMLP` (pure
numpy, same tanh/(96,96)/alpha=1e-5/max_iter=150 configuration). Interface and
z-scoring are byte-for-byte the same as v3. The M1 parity gate verifies this swap
did not change the failure-mode table.
"""
from __future__ import annotations
import numpy as np
from config import Config
from numpy_mlp import NumpyMLP


class EnsembleDynamics:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.members = []
        self.xm = self.xs = self.ym = self.ys = None

    @staticmethod
    def xy(data):
        X = data[:, 0:6]                    # q, qd, tau
        Y = data[:, 6:10] - data[:, 0:4]    # delta-state
        return X, Y

    def fit(self, train):
        X, Y = self.xy(train)
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-8
        self.ym, self.ys = Y.mean(0), Y.std(0) + 1e-8
        cov = np.cov(X, rowvar=False) + 1e-6*np.eye(X.shape[1])
        self.x_mu, self.x_covinv = X.mean(0), np.linalg.pinv(cov)   # for Mahalanobis input-distance
        rng = np.random.default_rng(self.cfg.seed + 1)
        in_dim = X.shape[1]
        out_dim = Y.shape[1]
        sizes = (in_dim, *tuple(self.cfg.hidden), out_dim)
        for k in range(self.cfg.k_ensemble):
            idx = rng.integers(0, len(X), len(X))        # bootstrap
            mlp = NumpyMLP(sizes, activation="tanh", l2=1e-5,
                           max_iter=self.cfg.max_iter, seed=k)
            mlp.fit(self._zx(X[idx]), self._zy(Y[idx]))
            self.members.append(mlp)
        return self

    def _zx(self, X): return (X - self.xm) / self.xs
    def _zy(self, Y): return (Y - self.ym) / self.ys

    def predict_all(self, X):
        """Returns (mean_z, all_z) predictions in z-space. all_z shape (K, N, 4)."""
        P = np.stack([m.predict(self._zx(X)) for m in self.members])
        return P.mean(0), P

    def true_error_z(self, data):
        """eps: actual model error vs ground truth, in z-space L2."""
        X, Y = self.xy(data)
        mean_z, _ = self.predict_all(X)
        return np.linalg.norm(mean_z - self._zy(Y), axis=1)

    def input_distance(self, data):
        """d: Mahalanobis distance of (q,qd,tau) to the training inputs (an
        input-space OOD baseline — by construction blind to consequence-only shifts)."""
        X, _ = self.xy(data)
        D = X - self.x_mu
        return np.sqrt(np.einsum("ij,jk,ik->i", D, self.x_covinv, D))
