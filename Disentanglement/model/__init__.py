"""DISModel: frozen SPEAR final-layer encoder + TopK SAE for reconstruction."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .spear_encoder import SpearEncoder
from .sae import SparseAutoencoder


class DISModel(nn.Module):
    """
    encoder : SpearEncoder        — frozen SPEAR, returns final-layer h_t
    sae     : SparseAutoencoder   — TopK SAE trained to reconstruct h_t
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.encoder = SpearEncoder(cfg)
        self.sae     = SparseAutoencoder(cfg)

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        audio         : (B, T_samples)  16 kHz waveform, zero-padded
        audio_lengths : (B,)            true sample counts

        Returns dict with keys: h_t, h_hat, z_t, z_pre, out_lengths
        """
        h_t, out_lengths = self.encoder(audio, audio_lengths)   # (B, T, D)
        z_t, z_pre       = self.sae.encode(h_t)                 # (B, T, K)
        h_hat            = self.sae.decode(z_t)                 # (B, T, D)
        return {
            "h_t":         h_t,
            "h_hat":       h_hat,
            "z_t":         z_t,
            "z_pre":       z_pre,
            "out_lengths": out_lengths,
        }


def build_dis_model(cfg) -> DISModel:
    model = DISModel(cfg)
    model.to(cfg.device)
    return model
