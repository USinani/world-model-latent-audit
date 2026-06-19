"""
config.py — one dataclass that fully specifies a run. Serialized into every
results/ folder so any number can be traced back to the exact config + git hash
that produced it (verifiability). Change knobs here, not in the code.

Base fields (physics / data / learned model) are PORTED VERBATIM from the wedge
v3 tree (phase-close commit 72866ed) so the M1 state-space parity run is identical.
Render/latent fields are additive (new section) and do not affect the parity run.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field


@dataclass
class Config:
    # physics / nominal model
    dt: float = 0.01
    steps: int = 200
    tau_max: float = 6.0
    nominal_m: float = 1.0
    nominal_b: float = 0.10
    damp_mismatch: float = 1.5      # residual's assumed damping is wrong by this factor (not an oracle)

    # data
    n_train: int = 120
    n_test_id: int = 40
    n_test_ood: int = 40
    id_lo: float = 0.8
    id_hi: float = 1.2
    ood_lo: float = 1.6
    ood_hi: float = 3.0

    # learned model
    k_ensemble: int = 5
    hidden: tuple = (96, 96)
    max_iter: int = 150

    # misc
    seed: int = 0
    quick: bool = False             # smaller run for CI / smoke test

    # ---- latent-phase additions (do not affect the state-space parity run) ----
    # rendering
    img_size: int = 48              # rendered frame is img_size x img_size grayscale (LW-11: 24->48 fidelity)
    frame_stack: int = 3            # stacked window so qd is inferable, not handed over
    frame_stride: int = 8           # temporal baseline (in dt) between stacked frames
    link_width: float = 0.18        # link half-width (world units) -> arm prominence in pixels
    nuisance_jitter: float = 0.03   # per-pixel uniform jitter amplitude (mild visual nuisance)
    nuisance_texture: float = 0.08  # amplitude of a fixed background texture
    fov: float = 2.3                # half-extent (metres) mapped to the frame edges

    # latent model
    latent_dim: int = 12            # AE bottleneck width
    ae_hidden: tuple = (192, 96)    # encoder hidden sizes (decoder mirrors)
    dyn_hidden: tuple = (96, 96)    # latent-dynamics net hidden sizes
    ae_max_iter: int = 250
    dyn_max_iter: int = 200
    max_ae_frames: int = 6000       # cap AE training frames (numpy budget)
    max_dyn_samples: int = 6000     # cap latent-dynamics training transitions

    def apply_quick(self):
        if self.quick:
            self.n_train, self.n_test_id, self.n_test_ood = 60, 20, 20
            self.steps, self.k_ensemble, self.max_iter = 120, 4, 120
            self.ae_max_iter, self.dyn_max_iter = 120, 100
            self.max_ae_frames, self.max_dyn_samples = 2500, 2500
        return self

    def to_dict(self):
        return asdict(self)
