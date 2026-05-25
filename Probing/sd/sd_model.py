"""Spoof Detection probe model.

Architecture (from the paper):
    1. Weighted-sum of L encoder layers  →  (B, T, D)
    2. Linear(D, proj_dim=256)           →  (B, T, 256)   [frame projection]
    3. Masked mean pool                  →  (B, 256)       [utterance vector]
    4. Linear(256, mlp_hidden=128) → ReLU → Dropout
       → Linear(128, 1)                 →  (B, 1)          [raw logit]

BCE with logits loss is applied externally.
Sigmoid of the logit gives the bonafide score used for EER computation.

Also supports probe_type='final' (single layer, no learnable mix) for ablation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import FrozenEncoder
from sd_config import SDConfig


# ====================================================================
# Shared masked mean pool
# ====================================================================

def _masked_mean(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Exclude padded frames from the mean.  x:(B,T,D)  lengths:(B,) → (B,D)."""
    B, T, D = x.shape
    mask  = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask_f = mask.unsqueeze(-1).float()
    return (x * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)


# ====================================================================
# Probe heads
# ====================================================================

class WeightedSpoofProbe(nn.Module):
    """Learnable softmax layer mix → projection → pool → MLP."""

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        proj_dim: int   = 256,
        mlp_hidden: int = 128,
        dropout: float  = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layer_norm = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(num_layers)]
        )
        self.weights    = nn.Parameter(torch.zeros(num_layers))
        self.projection = nn.Linear(hidden_size, proj_dim)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    @property
    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.weights, dim=0).detach().cpu()

    def forward(
        self,
        hidden_states: List[torch.Tensor],
        frame_lengths: torch.Tensor,
    ) -> torch.Tensor:
        w = torch.softmax(self.weights, dim=0)
        x = sum(
            w[i] * self.layer_norm[i](hidden_states[i])
            for i in range(self.num_layers)
        )                                          # (B, T, D)
        x = self.projection(x)                     # (B, T, proj_dim)
        x = _masked_mean(x, frame_lengths)         # (B, proj_dim)
        return self.classifier(x)                  # (B, 1)


class FinalLayerSpoofProbe(nn.Module):
    """Single encoder layer → projection → pool → MLP."""

    def __init__(
        self,
        hidden_size: int,
        proj_dim: int   = 256,
        mlp_hidden: int = 128,
        dropout: float  = 0.1,
        layer_idx: int  = -1,
    ) -> None:
        super().__init__()
        self.layer_idx  = layer_idx
        self.projection = nn.Linear(hidden_size, proj_dim)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(
        self,
        hidden_states: List[torch.Tensor],
        frame_lengths: torch.Tensor,
    ) -> torch.Tensor:
        x = hidden_states[self.layer_idx]          # (B, T, D)
        x = self.projection(x)                     # (B, T, proj_dim)
        x = _masked_mean(x, frame_lengths)         # (B, proj_dim)
        return self.classifier(x)                  # (B, 1)


# ====================================================================
# Factory
# ====================================================================

def build_sd_model(cfg: SDConfig):
    """Construct frozen encoder + spoof detection probe.

    Returns (encoder, probe).
    cfg.encoder_layer_count is set as a side-effect.
    """
    encoder = FrozenEncoder(cfg.model_id, model_family=cfg.model_family)
    cfg.encoder_layer_count = encoder.num_layers

    if cfg.probe_type == "weighted":
        probe = WeightedSpoofProbe(
            num_layers  = encoder.num_layers,
            hidden_size = encoder.hidden_size,
            proj_dim    = cfg.proj_dim,
            mlp_hidden  = cfg.mlp_hidden,
            dropout     = cfg.probe_dropout,
        )
    elif cfg.probe_type == "final":
        probe = FinalLayerSpoofProbe(
            hidden_size = encoder.hidden_size,
            proj_dim    = cfg.proj_dim,
            mlp_hidden  = cfg.mlp_hidden,
            dropout     = cfg.probe_dropout,
            layer_idx   = cfg.layer_idx,
        )
    else:
        raise ValueError(f"Unknown probe_type: {cfg.probe_type!r}")

    n_probe = sum(p.numel() for p in probe.parameters())
    print(
        f"[build_sd_model] probe={cfg.probe_type}"
        f"  hidden={encoder.hidden_size}  layers={encoder.num_layers}"
        f"  proj_dim={cfg.proj_dim}  mlp_hidden={cfg.mlp_hidden}"
        f"  probe_params={n_probe:,}"
    )
    return encoder, probe
