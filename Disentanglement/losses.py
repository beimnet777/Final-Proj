"""Loss functions for the SAE system."""

from __future__ import annotations

import torch


def recon_loss(
    h_t: torch.Tensor,
    h_hat: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Masked mean-squared error over valid frames.

    h_t, h_hat : (B, T, D)
    lengths    : (B,)  valid frame counts
    """
    B, T, _ = h_t.shape
    mask = (
        torch.arange(T, device=h_t.device).unsqueeze(0) < lengths.unsqueeze(1)
    ).float()                               # (B, T)
    mse = (h_t - h_hat).pow(2).mean(-1)    # (B, T)
    return (mse * mask).sum() / mask.sum().clamp(min=1)
