"""VICReg-style structural losses for probe-robust disentanglement.

Three terms (Bardes/Ponce/LeCun 2022):

  invariance : per-frame L2 between paired representations
               — assumes pair alignment is IDENTITY (perturbation pairs only)
               — does NOT use bilinear time resample (the v1 design flaw)
  variance   : already provided by `Disentanglement.losses.variance_floor_loss`
               — re-exported here for convenience
  covariance : off-diagonal squared covariance over selected bucket dims
               — decorrelates dims; blocks the orthogonal-subspace escape

The covariance term is the key addition vs v1. With cosine slack at 0.978
(v1's converged value), z_L has ~21% of vector magnitude available in directions
orthogonal to the shared content direction; speaker info can sit there
unconstrained. Decorrelating dims shrinks the structured "speaker subspace"
that this escape route relied on.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# Re-export the existing variance floor so callers can import all three from
# this module (`from probe_robust.losses import variance_floor_loss, ...`).
from Disentanglement.losses import variance_floor_loss  # noqa: F401


def vicreg_invariance_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    lens_a: torch.Tensor,
    lens_b: torch.Tensor,
) -> torch.Tensor:
    """Per-frame L2 invariance between two frame-aligned views.

    Assumes z_a[b, t, :] corresponds to z_b[b, t, :] by *identity* alignment
    (the case for perturbation pairs, which preserve duration). Does NOT
    perform bilinear time resample — that hack is unjustified for cross-speaker
    pairs (SiamCTC 2025) and unnecessary for perturbation pairs.

    Returns a scalar averaged over valid (batch, time, dim) entries — invariant
    to per-utterance length so short utterances aren't down-weighted.
    """
    T = min(z_a.shape[1], z_b.shape[1])
    K = z_a.shape[-1]
    lengths = torch.minimum(lens_a, lens_b).clamp(max=T)
    fmask = (torch.arange(T, device=z_a.device).unsqueeze(0) < lengths.unsqueeze(1)
             ).float().unsqueeze(-1)                          # (B, T, 1)
    diff_sq = (z_a[:, :T] - z_b[:, :T]).pow(2)                # (B, T, K)
    n_valid = fmask.sum().clamp(min=1.0) * K
    return (diff_sq * fmask).sum() / n_valid


def vicreg_covariance_loss(
    z: torch.Tensor,
    lengths: torch.Tensor,
    mask_dim: torch.Tensor | None = None,
) -> torch.Tensor:
    """Off-diagonal squared covariance, averaged over selected dims.

    Computes `Cov(z)` over (batch × valid time) samples for the dims selected
    by `mask_dim` (e.g. routing soft assignment to z_L). The loss is the sum
    of squared off-diagonal covariance entries divided by the number of
    selected dims, matching VICReg's normalization.

    mask_dim semantics: (K,) bool, OR (B,T,K) bool (will be reduced to (K,)
    by `any` across batch+time — captures the union of bucket-assigned dims
    in this batch).
    """
    B, T, K = z.shape
    if mask_dim is not None:
        m = mask_dim.bool()
        if m.dim() > 1:
            m = m.reshape(-1, K).any(0)
        if m.sum() < 2:
            return z.new_zeros(())
        z = z[..., m]
    K_sel = z.shape[-1]
    if K_sel < 2:
        return z.new_zeros(())

    fmask = (torch.arange(T, device=z.device).unsqueeze(0) < lengths.unsqueeze(1)
             ).float().unsqueeze(-1)                          # (B, T, 1)
    n_valid = fmask.sum().clamp(min=1.0)
    # Centre across (batch, time) — broadcasting per-dim mean
    z_mean = (z * fmask).sum(dim=(0, 1), keepdim=True) / n_valid
    z_centered = (z - z_mean) * fmask                         # zero-out padding
    # Flatten valid frames into rows for (K_sel, K_sel) cov matrix
    flat = z_centered.reshape(B * T, K_sel)
    cov = flat.t().matmul(flat) / (n_valid - 1.0).clamp(min=1.0)
    # Sum squared off-diagonal = sum of all squared entries minus diagonal squared
    diag_sq = cov.diagonal().pow(2).sum()
    all_sq  = cov.pow(2).sum()
    return (all_sq - diag_sq) / float(K_sel)
