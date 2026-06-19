"""
render.py — turn arm STATE into small grayscale FRAMES, so the latent phase has
to recover physics-auditability from pixels instead of privileged (q, qd, tau).

Two anti-laundering requirements from the charter are enforced here:

  1. STACKED WINDOW. A single frame of a 2-link arm is a 2-DOF manifold; a small
     AE would invert it straight back to (q1, q2) and c_z would become the analytic
     residual in disguise. So an observation is a stack of `frame_stack` frames
     ending at the current pose, the preceding poses reconstructed from qd by
     first-order back-extrapolation (q_{t-k} ~= q_t - k*stride*dt*qd). Velocity is
     therefore INFERABLE from frame differences but never handed over as a clean
     coordinate. `frame_stride` widens the temporal baseline so qd is visible.

  2. VISUAL NUISANCE. A faint FIXED background texture (same every frame -> carries
     no state, but adds reconstruction burden) plus per-pixel uniform JITTER
     (stochastic -> blocks pixel-perfect inversion). Together they keep the AE
     lossy, which is the regime in which the latent test is actually informative.

Rendering uses ONLY kinematics (q + link lengths). It is deliberately blind to
mass/damping, so a physics_shift world looks identical to nominal at the same pose
-- i.e. the shift is consequence-only in pixel space too, exactly as in state space.

Implementation is vectorised: poses are rendered in batches (chunked to bound
memory), not one Python call per frame.
"""
from __future__ import annotations
import numpy as np
from config import Config

L1, L2 = 1.0, 1.0          # link lengths used for drawing (masses irrelevant to pixels)
_GRID_CACHE: dict = {}
_TEXTURE_CACHE: dict = {}
_CHUNK = 2000              # poses rendered per vectorised batch


def _grid(cfg: Config):
    key = (cfg.img_size, cfg.fov)
    if key not in _GRID_CACHE:
        coords = np.linspace(-cfg.fov, cfg.fov, cfg.img_size)
        gx, gy = np.meshgrid(coords, -coords)         # flip y so +y is up
        _GRID_CACHE[key] = (gx, gy)
    return _GRID_CACHE[key]


def _conv2_same(img, k):
    H, W = img.shape
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    p = np.pad(img, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros_like(img)
    for i in range(kh):
        for j in range(kw):
            out += k[i, j] * p[i:i + H, j:j + W]
    return out


def _fixed_texture(cfg: Config):
    key = (cfg.img_size, cfg.nuisance_texture)
    if key not in _TEXTURE_CACHE:
        rng = np.random.default_rng(12345)            # fixed -> same texture always
        base = rng.standard_normal((cfg.img_size, cfg.img_size))
        k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], float); k /= k.sum()
        sm = base.copy()
        for _ in range(2):
            sm = _conv2_same(sm, k)
        sm = (sm - sm.min()) / (np.ptp(sm) + 1e-9)
        _TEXTURE_CACHE[key] = (cfg.nuisance_texture * sm).astype(float)
    return _TEXTURE_CACHE[key]


def _seg_batch(gx, gy, p0, p1, thick):
    """Vectorised exp-falloff intensity of segment p0->p1 for many poses.
    gx,gy: (n,n); p0,p1: (M,2). Returns (M,n,n)."""
    p0x = p0[:, 0][:, None, None]; p0y = p0[:, 1][:, None, None]
    vx = (p1[:, 0] - p0[:, 0])[:, None, None]
    vy = (p1[:, 1] - p0[:, 1])[:, None, None]
    wx = gx[None] - p0x; wy = gy[None] - p0y
    seg2 = vx * vx + vy * vy + 1e-12
    t = np.clip((wx * vx + wy * vy) / seg2, 0.0, 1.0)
    dx = wx - t * vx; dy = wy - t * vy
    dist = np.sqrt(dx * dx + dy * dy)
    return np.exp(-(dist / max(thick, 1e-6)) ** 2)


def _blob_batch(gx, gy, p, thick):
    px = p[:, 0][:, None, None]; py = p[:, 1][:, None, None]
    dj = np.sqrt((gx[None] - px) ** 2 + (gy[None] - py) ** 2)
    return np.exp(-(dj / (1.6 * thick)) ** 2)


def _render_poses(Q, cfg: Config):
    """Q: (M,2) joint angles -> (M, n, n) frames in [0,1] with fixed texture."""
    gx, gy = _grid(cfg)
    n = cfg.img_size
    thick = getattr(cfg, "link_width", 0.12)        # link half-width in world units
    tex = _fixed_texture(cfg)
    out = np.empty((len(Q), n, n), dtype=float)
    for s in range(0, len(Q), _CHUNK):
        q = Q[s:s + _CHUNK]
        base = np.zeros((len(q), 2))
        elbow = np.stack([L1 * np.cos(q[:, 0]), L1 * np.sin(q[:, 0])], axis=1)
        tip = elbow + np.stack([L2 * np.cos(q[:, 0] + q[:, 1]),
                                L2 * np.sin(q[:, 0] + q[:, 1])], axis=1)
        img = _seg_batch(gx, gy, base, elbow, thick)
        img = np.maximum(img, _seg_batch(gx, gy, elbow, tip, thick))
        for P in (base, elbow, tip):
            img = np.maximum(img, _blob_batch(gx, gy, P, thick))
        out[s:s + _CHUNK] = np.clip(img + tex[None], 0.0, 1.0)
    return out


def render_frame(q, cfg: Config):
    """Single grayscale frame in [0,1] of the arm at pose q (no jitter)."""
    return _render_poses(np.asarray(q, dtype=float)[None, :], cfg)[0]


def render_window(poses, cfg: Config, rng, jitter=None):
    """poses: (N, F, 2) EXPLICIT joint angles for each of F stacked frames (no
    back-extrapolation -- the caller supplies real simulated poses, so the window
    can carry genuine curvature/acceleration). Returns (N, F*H*W) observations.
    jitter=0 -> privileged clean render (no noise)."""
    poses = np.asarray(poses, dtype=float)
    N, F, _ = poses.shape
    n = cfg.img_size
    flat = poses.reshape(N * F, 2)
    frames = _render_poses(flat, cfg).reshape(N, F, n, n)
    j = cfg.nuisance_jitter if jitter is None else jitter
    if j > 0:
        r = rng if rng is not None else np.random.default_rng(0)
        frames = frames + r.uniform(-j, j, size=frames.shape)
    frames = np.clip(frames, 0.0, 1.0)
    return frames.reshape(N, F * n * n)


def render_batch(states, cfg: Config, rng, jitter=None):
    """states: (N, 4) array of [q1, q2, qd1, qd2]. Returns (N, F*H*W) observations.
    Each observation is a stacked window ending at the current pose; earlier frames
    are back-extrapolated from qd (CONSTANT-VELOCITY; carries position+velocity but
    no curvature). Retained for the state-space parity path; the latent experiment
    now uses render_window with real simulated poses. jitter=0 -> clean render."""
    states = np.asarray(states, dtype=float)
    N = len(states)
    F = cfg.frame_stack
    q = states[:, 0:2]; qd = states[:, 2:4]
    stride = getattr(cfg, "frame_stride", 1)
    # build all poses: for k in [F-1 .. 0], pose = q - k*stride*dt*qd  -> (N, F, 2)
    ks = np.arange(F - 1, -1, -1)[None, :, None]                  # (1, F, 1)
    poses = q[:, None, :] - ks * (stride * cfg.dt) * qd[:, None, :]  # (N, F, 2)
    return render_window(poses, cfg, rng, jitter=jitter)


def obs_dim(cfg: Config):
    return cfg.frame_stack * cfg.img_size ** 2
