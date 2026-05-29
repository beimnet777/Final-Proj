"""CTC probe heads for Phone Recognition.

Two variants matching the SUPERB spec:
    CTCFinalLayerProbe      — single encoder layer → 2-layer MLP → log-softmax
    CTCWeightedLayerProbe   — softmax mix of all layers → 2-layer MLP → log-softmax

The downstream model is a frame-wise 2-layer linear network optimised by CTC
loss, as described in the SUPERB paper.

Both output (B, T, vocab_size) log-softmax scores suitable for nn.CTCLoss.

Factory:
    build_pr_model(cfg) -> (encoder, probe)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import FrozenEncoder
from pr_config import PRConfig


# ====================================================================
# Probe heads
# ====================================================================

class CTCFinalLayerProbe(nn.Module):
    """Single encoder layer → projector → linear → log-softmax.

    Matches SUPERB: Linear(upstream_dim, proj_dim) before the CTC head.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        proj_dim: int = 256,
        layer_idx: int = -1,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.projector = nn.Linear(hidden_size, proj_dim)
        self.linear    = nn.Linear(proj_dim, vocab_size)

    def forward(
        self,
        hidden_states: List[torch.Tensor],
        frame_lengths: torch.Tensor,
    ) -> torch.Tensor:
        x = hidden_states[self.layer_idx]       # (B, T, D)
        x = self.projector(x)                   # (B, T, proj_dim)
        x = self.linear(x)                      # (B, T, V)
        return F.log_softmax(x, dim=-1)


class CTCWeightedLayerProbe(nn.Module):
    """Weighted sum of all encoder layers → projector → linear → log-softmax.

    Matches SUPERB: weighted sum first, then Linear(upstream_dim, proj_dim) before CTC head.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        vocab_size: int,
        proj_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layer_norm = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(num_layers)]
        )
        self.weights   = nn.Parameter(torch.zeros(num_layers))
        self.projector = nn.Linear(hidden_size, proj_dim)
        self.linear    = nn.Linear(proj_dim, vocab_size)

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
        )                                       # (B, T, D)
        x = self.projector(x)                   # (B, T, proj_dim)
        x = self.linear(x)                      # (B, T, V)
        return F.log_softmax(x, dim=-1)


# ====================================================================
# Factory
# ====================================================================

def build_pr_model(cfg: PRConfig):
    """Construct frozen encoder + CTC probe head.

    Returns (encoder, probe).
    cfg.encoder_layer_count is set as a side-effect.
    """
    encoder = FrozenEncoder(cfg.model_id, model_family=cfg.model_family)
    cfg.encoder_layer_count = encoder.num_layers

    if cfg.probe_type == "final":
        probe = CTCFinalLayerProbe(
            hidden_size=encoder.hidden_size,
            vocab_size=cfg.vocab_size,
            proj_dim=cfg.proj_dim,
            layer_idx=cfg.layer_idx,
        )
    elif cfg.probe_type == "weighted":
        probe = CTCWeightedLayerProbe(
            num_layers=encoder.num_layers,
            hidden_size=encoder.hidden_size,
            vocab_size=cfg.vocab_size,
            proj_dim=cfg.proj_dim,
        )
    else:
        raise ValueError(f"Unknown probe_type for PR: {cfg.probe_type!r}")

    n_probe = sum(p.numel() for p in probe.parameters())
    print(
        f"[build_pr_model] probe={cfg.probe_type}"
        f"  hidden={encoder.hidden_size}  layers={encoder.num_layers}"
        f"  vocab={cfg.vocab_size}  probe_params={n_probe:,}"
    )
    return encoder, probe
