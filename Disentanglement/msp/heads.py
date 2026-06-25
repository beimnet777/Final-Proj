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

from model.heads import gradient_reversal, _head_input_dim


class GELUSpeakerGRLHead(nn.Module):
    """z_L → GRL(λ) → Linear(K→256) → GELU → masked mean-pool → Linear(256→S).

    Mirrors GRLHead's default branch exactly, swapping ReLU→GELU.  Signature matches
    GRLHead so it is a drop-in for model.grl_head: forward(z_L, lengths, lam)."""

    def __init__(self, cfg) -> None:
        super().__init__()
        P = 256
        self.projector = nn.Linear(_head_input_dim(cfg), P)
        self.fc = nn.Linear(P, cfg.num_speakers)

    def forward(self, z_L: torch.Tensor, lengths: torch.Tensor, lam: float) -> torch.Tensor:
        z = gradient_reversal(z_L, lam)
        z = F.gelu(self.projector(z))                              # (B, T, P) — GELU
        T = z.shape[1]
        fmask = (torch.arange(T, device=z.device).unsqueeze(0) < lengths.unsqueeze(1)
                 ).float().unsqueeze(-1)
        n = lengths.float().clamp(min=1).unsqueeze(1)
        z_mean = (z * fmask).sum(1) / n                            # masked mean over time
        return self.fc(z_mean)                                     # (B, num_speakers)
