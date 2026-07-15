"""MSP-local head overrides, defined here so the shared model/heads.py stays
untouched (legacy runs unaffected).

GELUSpeakerGRLHead: the speaker adversary, identical to the default GRLHead pooled
path but with GELU instead of ReLU after the projector.  Swapped onto the model in
msp.train after build_dis_model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.heads import gradient_reversal, gradient_reversal_norm, _head_input_dim


class GELUSpeakerGRLHead(nn.Module):
    """z_L → GRL(λ) → Linear(K→256) → GELU → masked mean-pool → Linear(256→S).

    Mirrors GRLHead's default branch exactly, swapping ReLU→GELU.  Signature matches
    GRLHead so it is a drop-in for model.grl_head: forward(z_L, lengths, lam)."""

    def __init__(self, cfg) -> None:
        super().__init__()
        P = 256
        self.projector = nn.Linear(_head_input_dim(cfg), P)
        self.fc = nn.Linear(P, cfg.num_speakers)
        self.grad_norm = bool(getattr(cfg, "grl_grad_norm", False))
        self.grad_norm_target = float(getattr(cfg, "grl_grad_norm_target", 1.0))

    def forward(self, z_L: torch.Tensor, lengths: torch.Tensor, lam: float) -> torch.Tensor:
        z = (gradient_reversal_norm(z_L, lam, self.grad_norm_target)
             if self.grad_norm else gradient_reversal(z_L, lam))
        z = F.gelu(self.projector(z))                              # (B, T, P) — GELU
        T = z.shape[1]
        fmask = (torch.arange(T, device=z.device).unsqueeze(0) < lengths.unsqueeze(1)
                 ).float().unsqueeze(-1)
        n = lengths.float().clamp(min=1).unsqueeze(1)
        z_mean = (z * fmask).sum(1) / n                            # masked mean over time
        return self.fc(z_mean)                                     # (B, num_speakers)


class GELUEmotionGRLHead(nn.Module):
    """z_L → GRL(λ) → Linear(K→256) → GELU → masked mean+std → Linear(512→E).

    This mirrors the shared ``Emotion_GRL_Head`` architecture but optionally uses
    per-frame GRL normalization.  Keeping the same ``projector``/``fc`` names
    preserves checkpoint compatibility with the shared head.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        P = 256
        self.projector = nn.Linear(_head_input_dim(cfg), P)
        self.fc = nn.Linear(2 * P, getattr(cfg, "emotion_num_classes", 4))
        self.grad_norm = bool(getattr(cfg, "grl_emotion_grad_norm", False))
        self.grad_norm_target = float(getattr(cfg, "grl_emotion_grad_norm_target", 1.0))

    def forward(self, z: torch.Tensor, lengths: torch.Tensor, lam: float) -> torch.Tensor:
        z = (gradient_reversal_norm(z, lam, self.grad_norm_target)
             if self.grad_norm else gradient_reversal(z, lam))
        z = F.gelu(self.projector(z))
        B, T, C = z.shape
        mask = (torch.arange(T, device=z.device).unsqueeze(0) < lengths.unsqueeze(1)
                ).float().unsqueeze(-1)
        n = lengths.float().clamp(min=1).unsqueeze(-1)
        mean = (z * mask).sum(1) / n
        var = (((z - mean.unsqueeze(1)) ** 2) * mask).sum(1) / n
        std = (var + 1e-5).sqrt()
        return self.fc(torch.cat([mean, std], dim=-1))
