"""Task heads: PR (CTC), SID (CE), GRL adversarial speaker head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- GRL helpers

class _GRLFunction(torch.autograd.Function):
    """Gradient Reversal Layer: identity forward, -λ·grad backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return -ctx.lam * grad, None


def gradient_reversal(x: torch.Tensor, lam: float) -> torch.Tensor:
    return _GRLFunction.apply(x, lam)


# ---------------------------------------------------------------- Heads

class PRHead(nn.Module):
    """Frame-level phoneme CTC head operating on the L-routed latents.

    Input  : z_L_t  (B, T, K)   sparse, only L-group units non-zero
    Output : logits (B, T, vocab_size)  raw (pre-log-softmax); CTCLoss is in
             losses.py and applies log_softmax internally.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.K, cfg.K)  # optional bottleneck
        self.fc2 = nn.Linear(cfg.K, cfg.vocab_size)

    def forward(self, z_L_t: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(z_L_t))
        return self.fc2(x)   # (B, T, vocab_size)


class SIDHead(nn.Module):
    """Utterance-level speaker classification on the P-routed representation.

    Input  : z_P_bar  (B, 2K)   mean+std pool of z_P over time
    Output : logits   (B, num_speakers)
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.fc = nn.Linear(2 * cfg.K, cfg.num_speakers)

    def forward(self, z_P_bar: torch.Tensor) -> torch.Tensor:
        return self.fc(z_P_bar)  # (B, num_speakers)


class GRLHead(nn.Module):
    """Adversarial speaker head on z_L with Gradient Reversal.

    Trained to predict speaker identity from z_L.  The GRL ensures that
    gradients flowing back to the encoder/SAE are reversed, penalising
    speaker information retained in the linguistic subspace.

    Input  : z_L_t  (B, T, K)
    Output : logits (B, num_speakers)
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.K, cfg.num_speakers)

    def forward(
        self,
        z_L_t: torch.Tensor,
        out_lengths: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        # Mean-pool over valid frames
        B, T, K = z_L_t.shape
        mask = (
            torch.arange(T, device=z_L_t.device).unsqueeze(0)
            < out_lengths.unsqueeze(1)
        ).float().unsqueeze(-1)                                      # (B, T, 1)
        z_L_mean = (z_L_t * mask).sum(1) / out_lengths.float().unsqueeze(1).clamp(min=1)
        # (B, K)

        # Apply GRL then classify
        z_rev = gradient_reversal(z_L_mean, lam)
        return self.fc(z_rev)   # (B, num_speakers)
