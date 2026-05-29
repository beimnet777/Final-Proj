"""Sparse Autoencoder: Gao et al. 2024 encoder + routed (D, 4K) decoder.

Encoder (Gao et al. 2024):
    z = TopK( W_enc (x − b_pre) )     no ReLU, no enc_bias

Decoder (original routed design):
    x̂ = W_dec cat(z_L, z_P_bar, z_U) + b_pre    shape (D, 4K)

Decoder input layout:
    z_L_t       (B, T, K)   L-routed latents
    z_P_bar     (B, 2K)     utterance-level mean+std of z_P, broadcast over T
    z_U_t       (B, T, K)   U-routed latents
→   (B, T, 4K)

Aligned init (Gao et al.):
    enc_weight rows → unit vectors
    L-slot and U-slot decoder columns start equal to enc_weight rows
    P-slot initialised independently (utterance-level path)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _topk_straight_through(pre: torch.Tensor, k: int) -> torch.Tensor:
    """Keep top-k values per token; zero the rest. Straight-through backward.

    Forward : exactly k non-zero per (B, T) position
    Backward: identity on pre (gradient flows to all K positions)
    """
    topk_vals, topk_idx = pre.topk(k, dim=-1)
    z_sparse = torch.zeros_like(pre).scatter_(-1, topk_idx, topk_vals)
    return pre + (z_sparse - pre).detach()


class SparseAutoencoder(nn.Module):
    """Gao et al. 2024 encoder with original routed (D, 4K) decoder."""

    def __init__(self, cfg) -> None:
        super().__init__()
        D, K = cfg.D, cfg.K
        self.K    = K
        self.topk = cfg.topk

        # Pre-bias: absorbs data mean, shared encoder/decoder
        self.b_pre = nn.Parameter(torch.zeros(D))

        # Encoder weight (K, D) — no bias
        self.enc_weight = nn.Parameter(torch.empty(K, D))

        # Decoder weight (D, 4K) — input is cat(z_L, z_P_bar, z_U)
        self.dec_weight = nn.Parameter(torch.empty(D, 4 * K))

        self._init_weights()

    def _init_weights(self) -> None:
        K = self.K

        # Encoder rows → unit vectors
        nn.init.kaiming_uniform_(self.enc_weight, a=math.sqrt(5))
        with torch.no_grad():
            self.enc_weight.data = F.normalize(self.enc_weight.data, dim=1)

        # Aligned init: L-slot and U-slot columns = enc_weight rows
        with torch.no_grad():
            self.dec_weight.data[:, :K]     = self.enc_weight.data.T  # L-slot
            self.dec_weight.data[:, 3*K:]   = self.enc_weight.data.T  # U-slot

        # P-slot (utterance-level path) independently initialised
        bound = 1.0 / math.sqrt(2 * K)
        nn.init.uniform_(self.dec_weight.data[:, K:3*K], -bound, bound)

    # ---------------------------------------------------------------- encode

    def encode(self, h_t: torch.Tensor):
        """
        h_t   : (B, T, D)
        returns
            z_t   : (B, T, K)  TopK-sparse, straight-through gradient
            z_pre : (B, T, K)  pre-TopK activations (monitoring)
        """
        centred = h_t - self.b_pre
        z_pre   = F.linear(centred, self.enc_weight)       # no bias
        return _topk_straight_through(z_pre, self.topk), z_pre

    # ---------------------------------------------------------------- decode

    def decode(
        self,
        z_L_t: torch.Tensor,
        z_P_bar: torch.Tensor,
        z_U_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        z_L_t   : (B, T, K)
        z_P_bar : (B, 2K)
        z_U_t   : (B, T, K)
        returns ĥ_t : (B, T, D)
        """
        T = z_L_t.size(1)
        z_P_bc  = z_P_bar.unsqueeze(1).expand(-1, T, -1)          # (B, T, 2K)
        dec_in  = torch.cat([z_L_t, z_P_bc, z_U_t], dim=-1)       # (B, T, 4K)
        return F.linear(dec_in, self.dec_weight) + self.b_pre
