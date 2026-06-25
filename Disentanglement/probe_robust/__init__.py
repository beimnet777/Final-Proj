"""Probe-robust disentanglement modules.

Adversary-free, structural-and-information-theoretic mechanisms targeting
probe-architecture-agnostic disentanglement. Lives in a dedicated package so
the existing GRL / adversarial code paths in the main `Disentanglement/`
namespace remain untouched and the boundary stays clean for the dissertation
("structural" vs "adversarial" methods comparison).

Currently provides:

  losses.vicreg_invariance_loss   : per-frame L2 between frame-aligned pairs
                                    (perturbation pairs only — no bilinear hack)
  losses.vicreg_covariance_loss   : off-diagonal cov regularizer; decorrelates
                                    bucket dims to block the orthogonal-subspace
                                    escape diagnosed in dual_inv_v1_soft_nogrl
  club.CLUBSampled                : sampled CLUB MI upper bound (Cheng 2020);
                                    minimising it bounds the error of ANY
                                    downstream speaker probe by Fano's
                                    inequality (probe-architecture-agnostic)
"""
from . import losses, club  # noqa: F401
