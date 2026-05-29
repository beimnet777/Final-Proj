"""Sparse Autoencoder: overcomplete encoder + decoder.

Follows Gao et al. 2024 (Scaling and evaluating sparse autoencoders):
    z  = TopK( W_enc (x − b_pre) )          no ReLU, no enc_bias
    x̂  = W_dec z + b_pre                     decoder takes full z_t (K-dim)

The decoder always reconstructs from the full TopK sparse features z_t.
Routing is NOT involved in reconstruction — it only carves z_t into
z_L / z_P / z_U for the downstream task heads (stage 2 only).

This means:
  - No pooling-gradient disadvantage for P features
  - Decoder shape (D, K) matches encoder perfectly
  - Aligned init is one-to-one: dec_weight = enc_weight.T
  - Stage 1 and stage 2 use the identical decoder
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _topk_straight_through(pre: torch.Tensor, k: int) -> torch.Tensor:
    """Keep top-k values per token; zero the rest.  Straight-through backward.

    Operates on raw pre-activations (no prior ReLU).
    Forward : sparse z  (exactly k non-zero per (B, T) position)
    Backward: identity on pre  (gradient flows to all K positions)
    """
    topk_vals, topk_idx = pre.topk(k, dim=-1)           # (B, T, k)
    z_sparse = torch.zeros_like(pre).scatter_(-1, topk_idx, topk_vals)
    return pre + (z_sparse - pre).detach()


class SparseAutoencoder(nn.Module):
    """Overcomplete SAE with TopK sparsity (Gao et al. 2024).

    Encoder: z = TopK( W_enc (x − b_pre) )   shape: (B, T, D) → (B, T, K)
    Decoder: x̂ = W_dec z + b_pre              shape: (B, T, K) → (B, T, D)
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        D, K = cfg.D, cfg.K
        self.K = K
        self.topk = cfg.topk

        # Pre-bias: absorbs data mean. Shape (D,).
        # Shared between encoder (subtracted) and decoder (added back).
        self.b_pre = nn.Parameter(torch.zeros(D))

        # Encoder weight (K, D) — no bias term.
        self.enc_weight = nn.Parameter(torch.empty(K, D))

        # Decoder weight (D, K) — takes full z_t, not routed features.
        self.dec_weight = nn.Parameter(torch.empty(D, K))

        self._init_weights()

    def _init_weights(self) -> None:
        # 1. Random encoder rows, normalised to unit vectors.
        nn.init.kaiming_uniform_(self.enc_weight, a=math.sqrt(5))
        with torch.no_grad():
            self.enc_weight.data = F.normalize(self.enc_weight.data, dim=1)

        # 2. Aligned init (Gao et al. 2024): decoder columns = encoder rows.
        #    enc_weight[k] is the read direction for feature k.
        #    dec_weight[:, k] starts as the same unit vector — write = read.
        #    Weights diverge freely during training; only aligned at init.
        with torch.no_grad():
            self.dec_weight.data = self.enc_weight.data.T   # (D, K)

    # ---------------------------------------------------------------- encode

    def encode(self, h_t: torch.Tensor):
        """
        h_t  : (B, T, D)
        returns
            z_t   : (B, T, K)  TopK-sparse, straight-through gradient
            z_pre : (B, T, K)  pre-TopK activations (monitoring)
        """
        centred = h_t - self.b_pre                         # (B, T, D)
        z_pre   = F.linear(centred, self.enc_weight)       # (B, T, K)  no bias
        return _topk_straight_through(z_pre, self.topk), z_pre

    # ---------------------------------------------------------------- decode

    def decode(self, z_t: torch.Tensor) -> torch.Tensor:
        """
        z_t  : (B, T, K)  TopK-sparse features (full, not routed)
        returns ĥ_t : (B, T, D)
        """
        return F.linear(z_t, self.dec_weight) + self.b_pre
