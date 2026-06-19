"""
numpy_mlp.py — self-contained, pure-numpy MLP regressor.

This is the single learned-component backbone for the whole module: the
dynamics ensemble member (model.py), the autoencoder encoder/decoder, and the
latent-dynamics nets (latent_model.py) are all instances of NumpyMLP. Keeping
ONE backbone means the stability story is audited in one place.

It ports the stabilisation recipe proven in World_Models
`src/world_models/models/mlp.py` (WORLD-MODELS-003B) and generalises it from the
hardwired 6->4 shape to an arbitrary (in_dim, hidden..., out_dim) stack:

  - He/Xavier-style init (Xavier for tanh, He for relu)
  - Adam optimiser (matches the convergence behaviour of the validated sklearn
    MLPRegressor baseline at max_iter=150 far better than plain SGD would)
  - L2 weight decay on weight matrices (not biases)
  - global gradient clipping (shared scale across all parameter tensors)
  - finite-loss guard + post-update weight-finiteness guard -> training_failed
  - np.errstate around forward/backward so the guard IS the overflow detector

No autograd framework. Gradients computed analytically.
"""
from __future__ import annotations
import numpy as np

_ERR = dict(over="ignore", invalid="ignore", divide="ignore")


def _act(z, kind):
    if kind == "tanh":
        return np.tanh(z)
    if kind == "relu":
        return np.maximum(0.0, z)
    raise ValueError(kind)


def _act_grad(z, a, kind):
    """Derivative of activation given pre-activation z and activation a=act(z)."""
    if kind == "tanh":
        return 1.0 - a * a
    if kind == "relu":
        return (z > 0).astype(z.dtype)
    raise ValueError(kind)


class NumpyMLP:
    def __init__(
        self,
        layer_sizes,                 # e.g. (6, 96, 96, 4)
        activation: str = "tanh",
        l2: float = 1e-5,            # weight decay (sklearn alpha analogue)
        max_iter: int = 150,         # epochs
        batch_size: int = 200,
        lr: float = 1e-3,            # Adam step size
        max_grad_norm: float = 5.0,
        seed: int = 0,
        standardize_x: bool = False,
        standardize_y: bool = False,
        linear_layers=(),
    ):
        self.layer_sizes = tuple(int(s) for s in layer_sizes)
        self.activation = activation
        # weight-layer indices whose output is LINEAR (no activation). The final
        # layer is always linear; extra entries here let e.g. an AE bottleneck be
        # linear so its codes are linearly decodable (needed for the probe R^2).
        self.linear_layers = set(int(i) for i in linear_layers)
        self.l2 = l2
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.standardize_x = standardize_x
        self.standardize_y = standardize_y

        self.W = None    # list of weight matrices
        self.bvec = None  # list of bias vectors
        self.training_failed = False
        self.xm = self.xs = self.ym = self.ys = None

    # -- init ---------------------------------------------------------------
    def _init(self, rng):
        self.W, self.bvec = [], []
        for nin, nout in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            if self.activation == "relu":
                scale = np.sqrt(2.0 / nin)
            else:  # xavier for tanh
                scale = np.sqrt(1.0 / nin)
            self.W.append(rng.normal(0.0, scale, size=(nin, nout)))
            self.bvec.append(np.zeros(nout))

    # -- forward ------------------------------------------------------------
    def _forward(self, X):
        """Returns (zs, acts). acts[0]=X; last layer is linear (regression head)."""
        acts = [X]
        zs = []
        h = X
        L = len(self.W)
        for i in range(L):
            z = h @ self.W[i] + self.bvec[i]
            zs.append(z)
            if i < L - 1 and i not in self.linear_layers:
                h = _act(z, self.activation)
            else:
                h = z  # linear output (final layer or marked-linear layer)
            acts.append(h)
        return zs, acts

    # -- standardisation helpers -------------------------------------------
    def _sx(self, X):
        return (X - self.xm) / self.xs if self.standardize_x else X

    def _sy(self, Y):
        return (Y - self.ym) / self.ys if self.standardize_y else Y

    def _unsy(self, Yn):
        return Yn * self.ys + self.ym if self.standardize_y else Yn

    # -- fit ----------------------------------------------------------------
    def fit(self, X, Y):
        self.training_failed = False
        rng = np.random.default_rng(self.seed)
        self._init(rng)
        _EPS = 1e-8

        if self.standardize_x:
            self.xm = X.mean(0); s = X.std(0); self.xs = np.where(s < _EPS, 1.0, s)
        if self.standardize_y:
            self.ym = Y.mean(0); s = Y.std(0); self.ys = np.where(s < _EPS, 1.0, s)

        Xn, Yn = self._sx(X), self._sy(Y)
        N = len(Xn)
        # Adam state
        mW = [np.zeros_like(w) for w in self.W]; vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.bvec]; vb = [np.zeros_like(b) for b in self.bvec]
        b1, b2, eps = 0.9, 0.999, 1e-8
        t = 0

        for _ in range(self.max_iter):
            perm = rng.permutation(N)
            Xs, Ys = Xn[perm], Yn[perm]
            for start in range(0, N, self.batch_size):
                if not self._finite():
                    self.training_failed = True
                    return self
                Xb = Xs[start:start + self.batch_size]
                Yb = Ys[start:start + self.batch_size]
                B = len(Xb)
                with np.errstate(**_ERR):
                    zs, acts = self._forward(Xb)
                    out = acts[-1]
                    loss = float(np.mean((out - Yb) ** 2))
                    if not np.isfinite(loss):
                        self.training_failed = True
                        return self
                    # backprop
                    grads_W = [None] * len(self.W)
                    grads_b = [None] * len(self.bvec)
                    delta = (out - Yb) * (2.0 / B)        # dL/dout
                    for i in reversed(range(len(self.W))):
                        grads_W[i] = acts[i].T @ delta + self.l2 * self.W[i]
                        grads_b[i] = delta.sum(0)
                        if i > 0:
                            da = delta @ self.W[i].T
                            if (i - 1) in self.linear_layers:
                                delta = da     # linear layer: activation gradient is 1
                            else:
                                delta = da * _act_grad(zs[i - 1], acts[i], self.activation)
                # global grad clip
                gn = np.sqrt(sum(float(np.sum(g * g)) for g in grads_W + grads_b))
                scale = self.max_grad_norm / gn if (gn > self.max_grad_norm and gn > 0) else 1.0
                t += 1
                with np.errstate(**_ERR):
                    for i in range(len(self.W)):
                        gW = grads_W[i] * scale
                        gb = grads_b[i] * scale
                        mW[i] = b1 * mW[i] + (1 - b1) * gW
                        vW[i] = b2 * vW[i] + (1 - b2) * (gW * gW)
                        mb[i] = b1 * mb[i] + (1 - b1) * gb
                        vb[i] = b2 * vb[i] + (1 - b2) * (gb * gb)
                        mWh = mW[i] / (1 - b1 ** t); vWh = vW[i] / (1 - b2 ** t)
                        mbh = mb[i] / (1 - b1 ** t); vbh = vb[i] / (1 - b2 ** t)
                        self.W[i] -= self.lr * mWh / (np.sqrt(vWh) + eps)
                        self.bvec[i] -= self.lr * mbh / (np.sqrt(vbh) + eps)
                if not self._finite():
                    self.training_failed = True
                    return self
        return self

    def _finite(self):
        if self.W is None:
            return False
        return all(np.all(np.isfinite(w)) for w in self.W) and \
            all(np.all(np.isfinite(b)) for b in self.bvec)

    # -- predict ------------------------------------------------------------
    def predict(self, X):
        if self.W is None:
            raise RuntimeError("call fit() before predict()")
        with np.errstate(**_ERR):
            _, acts = self._forward(self._sx(X))
        return self._unsy(acts[-1])

    # -- partial forward (for autoencoder use) ------------------------------
    def encode_to(self, X, n_enc_layers):
        """Run the first n_enc_layers weight layers and return the bottleneck
        activation (tanh/relu applied since the bottleneck is not the output layer)."""
        if self.W is None:
            raise RuntimeError("call fit() before encode_to()")
        L = len(self.W)
        with np.errstate(**_ERR):
            h = self._sx(X)
            for i in range(n_enc_layers):
                z = h @ self.W[i] + self.bvec[i]
                h = z if (i == L - 1 or i in self.linear_layers) else _act(z, self.activation)
        return h

    def decode_from(self, Z, n_enc_layers):
        """Continue from a bottleneck activation Z through the remaining layers."""
        if self.W is None:
            raise RuntimeError("call fit() before decode_from()")
        L = len(self.W)
        with np.errstate(**_ERR):
            h = Z
            for i in range(n_enc_layers, L):
                z = h @ self.W[i] + self.bvec[i]
                h = z if (i == L - 1 or i in self.linear_layers) else _act(z, self.activation)
        return self._unsy(h)
