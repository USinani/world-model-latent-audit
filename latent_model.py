"""
latent_model.py — the latent-phase learned components, both pure numpy and both
built on the single NumpyMLP backbone:

  LatentAE        : an MLP autoencoder (stacked frames -> z -> reconstruction).
                    `z` is the bottleneck activation. Trained on CLEAN ID frames
                    only. Provides encode/decode/recon_error. This is what makes
                    the privileged state unavailable: downstream detectors see z,
                    not (q, qd, tau).

  LatentEnsemble  : a bootstrap ensemble that predicts the latent transition
                    (z_t, a_t) -> z_{t+1} (as a delta in z), trained on CLEAN ID
                    transitions only. Provides the latent analogues of the
                    state-space detectors:
                      u_z = ensemble disagreement on (z_t, a_t)        [input-based]
                      d_z = Mahalanobis distance of (z_t, a_t)          [input-based]
                    and the mean prediction used by the consistency detector c_z
                    and by the privileged true-error eps (computed in the experiment).

The AE is an MLP, NOT a conv net: a 2-link arm is intrinsically ~2-D, so conv
buys nothing and only adds instability surface (per the build decision).
"""
from __future__ import annotations
import numpy as np
from config import Config
from numpy_mlp import NumpyMLP


class LatentAE:
    def __init__(self, cfg: Config, in_dim: int, seed: int = 0):
        self.cfg = cfg
        self.in_dim = in_dim
        enc = tuple(cfg.ae_hidden)
        dec = tuple(reversed(cfg.ae_hidden))
        # one MLP: D -> enc... -> L -> dec... -> D, trained to reconstruct input
        self.sizes = (in_dim, *enc, cfg.latent_dim, *dec, in_dim)
        self.n_enc_layers = len(enc) + 1            # weight layers up to & incl. bottleneck
        # the bottleneck weight layer (index n_enc_layers-1) is LINEAR so the codes
        # are linearly decodable -> the probe R^2 measures real state content, not a
        # saturating-tanh artefact.
        self.net = NumpyMLP(self.sizes, activation="tanh", l2=1e-6,
                            max_iter=cfg.ae_max_iter, batch_size=256, lr=1e-3,
                            seed=seed, standardize_x=True, standardize_y=False,
                            linear_layers=(self.n_enc_layers - 1,))

    def fit(self, frames):
        """frames: (N, D) clean ID observations in [0,1]. Target = input."""
        self.net.fit(frames, frames)
        return self

    @property
    def training_failed(self):
        return self.net.training_failed

    def encode(self, frames):
        return self.net.encode_to(frames, self.n_enc_layers)        # (N, latent_dim)

    def decode(self, z):
        return self.net.decode_from(z, self.n_enc_layers)           # (N, in_dim)

    def reconstruct(self, frames):
        return self.decode(self.encode(frames))

    def recon_error(self, frames):
        """Per-sample reconstruction MSE (pixel space)."""
        rec = self.reconstruct(frames)
        return np.mean((rec - frames) ** 2, axis=1)


class LatentEnsemble:
    """Bootstrap ensemble of (z_t, a_t) -> delta_z nets, trained on clean ID only."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.members = []
        self.xm = self.xs = self.ym = self.ys = None
        self.x_mu = self.x_covinv = None

    def fit(self, Z, A, Z_next):
        X = np.hstack([Z, A])                         # (N, L+2)
        Y = Z_next - Z                                # delta-z target
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-8
        self.ym, self.ys = Y.mean(0), Y.std(0) + 1e-8
        # Mahalanobis on the STANDARDISED inputs (numerically stable; tau and z
        # live on very different scales, so raw covariance is ill-conditioned).
        Xz = (X - self.xm) / self.xs
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            cov = np.cov(Xz, rowvar=False) + 1e-6 * np.eye(X.shape[1])
            self.x_covinv = np.linalg.pinv(cov)
        self.x_mu = self.xm
        rng = np.random.default_rng(self.cfg.seed + 7)
        sizes = (X.shape[1], *tuple(self.cfg.dyn_hidden), Y.shape[1])
        for k in range(self.cfg.k_ensemble):
            idx = rng.integers(0, len(X), len(X))
            mlp = NumpyMLP(sizes, activation="tanh", l2=1e-5,
                           max_iter=self.cfg.dyn_max_iter, seed=100 + k)
            mlp.fit(self._zx(X[idx]), self._zy(Y[idx]))
            self.members.append(mlp)
        return self

    def _zx(self, X): return (X - self.xm) / self.xs
    def _zy(self, Y): return (Y - self.ym) / self.ys
    def _unzy(self, Yz): return Yz * self.ys + self.ym

    @property
    def training_failed(self):
        return any(m.training_failed for m in self.members)

    def predict_all(self, Z, A):
        """Returns (mean_next, members_next) in RAW z-space.
        mean_next: (N, L); members_next: (K, N, L) — predicted z_{t+1}."""
        X = np.hstack([Z, A])
        Pz = np.stack([m.predict(self._zx(X)) for m in self.members])   # (K,N,L) delta in z-norm
        members_delta = np.stack([self._unzy(p) for p in Pz])            # raw delta
        members_next = Z[None, :, :] + members_delta                     # (K,N,L)
        return members_next.mean(0), members_next

    def disagreement(self, Z, A):
        """u_z: ensemble variance of predicted z_{t+1}, summed over latent dims."""
        _, members_next = self.predict_all(Z, A)
        return members_next.var(0).sum(1)

    def input_distance(self, Z, A):
        """d_z: Mahalanobis distance of (z_t, a_t) to the training latent-inputs,
        computed in the standardised input space used at fit time."""
        X = np.hstack([Z, A])
        D = (X - self.xm) / self.xs
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", D, self.x_covinv, D), 0.0))
