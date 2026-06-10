"""Task heads for stage 2 disentanglement.

PRHead     : CTC phoneme recognition on z_L  (frame-level)
SIDHead    : Speaker classification on z_P_bar  (utterance-level mean pool)
GRLHead    : Adversarial speaker head on z_L with gradient reversal
PR_GRL_Head: Adversarial phoneme head on z_P with gradient reversal  (Exp 1 — dual GRL)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- gradient reversal

class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return -ctx.lam * grad, None


def gradient_reversal(x: torch.Tensor, lam: float) -> torch.Tensor:
    return _GRL.apply(x, lam)


# ---------------------------------------------------------------- PR head

def _head_input_dim(cfg) -> int:
    if getattr(cfg, "projection_disentanglement", False):
        return getattr(cfg, "projection_dim", cfg.K)
    return cfg.K


class PRHead(nn.Module):
    """Linear CTC head: z_L (B, T, input_dim) -> logits."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.fc = nn.Linear(_head_input_dim(cfg), cfg.vocab_size)

    def forward(self, z_L: torch.Tensor) -> torch.Tensor:
        return self.fc(z_L)


# ---------------------------------------------------------------- SID head

class SIDHead(nn.Module):
    """Speaker CE head: mean(z_P) (B, input_dim) -> 256 -> logits.

    Two-layer MLP matching probing head style — single linear was too weak
    for 247-way classification on sparse features.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_head_input_dim(cfg), 256),
            nn.ReLU(),
            nn.Linear(256, cfg.num_speakers),
        )

    def forward(self, z_P_bar: torch.Tensor) -> torch.Tensor:
        return self.net(z_P_bar)


# ---------------------------------------------------------------- PR-GRL head  (Exp 1 — dual GRL)

class PR_GRL_Head(nn.Module):
    """Adversarial phoneme head on z_P with gradient reversal.

    Frame-level linear CTC on z_P with reversed gradient.  When z_P encodes
    phonemes, the reversed gradient penalises the SAE for putting phoneme
    information into the paralinguistic bucket.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.fc = nn.Linear(_head_input_dim(cfg), cfg.vocab_size)

    def forward(self, z_P: torch.Tensor, lam: float) -> torch.Tensor:
        # z_P : (B, T, K) — frame-level paralinguistic features
        return self.fc(gradient_reversal(z_P, lam))


# ---------------------------------------------------------------- GRL head

class GRLHead(nn.Module):
    """Adversarial speaker head on z_L with gradient reversal.

    Mean-pools z_L over valid frames, applies GRL, then classifies speaker.
    GRL reverses gradients so the SAE is penalised for encoding speaker in L features.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.fc = nn.Linear(_head_input_dim(cfg), cfg.num_speakers)

    def forward(
        self,
        z_L: torch.Tensor,
        lengths: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """
        z_L     : (B, T, K)
        lengths : (B,)  valid frame counts
        lam     : GRL reversal strength
        """
        B, T, K = z_L.shape
        mask   = (torch.arange(T, device=z_L.device).unsqueeze(0) < lengths.unsqueeze(1)
                  ).float().unsqueeze(-1)                          # (B, T, 1)
        z_mean = (z_L * mask).sum(1) / lengths.float().unsqueeze(1).clamp(min=1)  # (B, K)
        return self.fc(gradient_reversal(z_mean, lam))
