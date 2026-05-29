"""DISModel — single-stage, all losses active from step 1.

Gao et al. 2024 SAE encoder + routed (D, 4K) decoder.
Routing, task heads, and all losses are always active.
No staged training.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .spear_encoder import SpearWeightedEncoder
from .sae import SparseAutoencoder
from .routing import RoutingModule
from .heads import PRHead, SIDHead, GRLHead


def _mean_std_pool(z: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Utterance-level mean+std pool over valid frames → (B, 2K)."""
    B, T, K = z.shape
    mask  = (
        torch.arange(T, device=z.device).unsqueeze(0) < lengths.unsqueeze(1)
    ).float().unsqueeze(-1)
    count = lengths.float().clamp(min=1).view(B, 1, 1)
    mean  = (z * mask).sum(1) / count.squeeze(-1)
    diff  = z - mean.unsqueeze(1)
    var   = ((diff * diff) * mask).sum(1) / count.squeeze(-1)
    std   = (var + 1e-8).sqrt()
    return torch.cat([mean, std], dim=-1)


class DISModel(nn.Module):
    """Disentanglement model — single stage, all objectives active."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg      = cfg
        self.encoder  = SpearWeightedEncoder(cfg)
        self.sae      = SparseAutoencoder(cfg)
        self.routing  = RoutingModule(cfg)
        self.pr_head  = PRHead(cfg)
        self.sid_head = SIDHead(cfg)
        self.grl_head = GRLHead(cfg)

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        grl_lambda: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        audio         : (B, T_samples)
        audio_lengths : (B,)
        grl_lambda    : DANN reversal strength
        """
        # 1. SPEAR weighted layer mix
        h_t, out_lengths = self.encoder(audio, audio_lengths)    # (B, T, D)

        # 2. SAE encode — always exactly topk active features
        z_t, z_pre = self.sae.encode(h_t)                        # (B, T, K)

        # 3. Route z_t → L / P / U
        m_L, m_P, m_U = self.routing()
        z_L   = m_L * z_t
        z_P   = m_P * z_t
        z_U   = m_U * z_t
        z_P_bar = _mean_std_pool(z_P, out_lengths)               # (B, 2K)

        # 4. Decode from routed representations
        h_hat = self.sae.decode(z_L, z_P_bar, z_U)              # (B, T, D)

        return {
            "h_t":         h_t,
            "h_hat":       h_hat,
            "z_t":         z_t,
            "z_pre":       z_pre,
            "z_L":         z_L,
            "z_P":         z_P,
            "z_U":         z_U,
            "z_P_bar":     z_P_bar,
            "out_lengths": out_lengths,
            "pr_logits":   self.pr_head(z_L),
            "sid_logits":  self.sid_head(z_P_bar),
            "grl_logits":  self.grl_head(z_L, out_lengths, grl_lambda),
        }


def build_dis_model(cfg) -> DISModel:
    model = DISModel(cfg)
    model.to(cfg.device)
    return model
