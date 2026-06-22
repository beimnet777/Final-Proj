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


def _blockwise_topk_straight_through(pre: torch.Tensor, spec) -> torch.Tensor:
    """Per-block TopK: keep top-k within each contiguous index block, separately.

    spec: list of (start, end, k).  Blocks must be contiguous and tile [0, K).
    Guarantees each factor block a fixed active budget so none can be starved by
    a global TopK competition (the failure mode of learned routing).
    """
    parts = [_topk_straight_through(pre[..., s:e], k) for s, e, k in spec]
    return torch.cat(parts, dim=-1)


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

        # Fixed-block per-block TopK (Option A): top-k chosen within each L/P/U
        # block separately, so each factor gets a guaranteed active budget.
        # block_spec set only for per-block TopK (Exp 1/2).  When fixed_blocks but
        # per_block_topk=False (Exp 3), block_spec stays None → global TopK, and the
        # per-block active counts emerge from the fixed-membership masks downstream.
        self.block_spec = None
        if getattr(cfg, 'fixed_blocks', False) and getattr(cfg, 'per_block_topk', True):
            kL, kP, kU = cfg.K_L, cfg.K_P, cfg.K_U
            assert kL + kP + kU == K, f"K_L+K_P+K_U ({kL+kP+kU}) must equal K ({K})"
            self.block_spec = [(0, kL, cfg.topk_L),
                               (kL, kL + kP, cfg.topk_P),
                               (kL + kP, K, cfg.topk_U)]

        # Pre-bias: absorbs data mean. Shape (D,).
        # Shared between encoder (subtracted) and decoder (added back).
        self.b_pre = nn.Parameter(torch.zeros(D))

        # Encoder weight (K, D) — no bias term.
        self.enc_weight = nn.Parameter(torch.empty(K, D))

        # Decoder weight (D, K) — takes full z_t, not routed features.
        self.dec_weight = nn.Parameter(torch.empty(D, K))

        self._init_weights()

        # Dead-latent revival (Gao AuxK).  steps_since_fired is a transient training
        # counter (not checkpointed); aux_k>0 turns the mechanism on.
        self.aux_k          = int(getattr(cfg, 'aux_k', 0))
        self.dead_threshold = int(getattr(cfg, 'dead_steps_threshold', 256))
        self.register_buffer('steps_since_fired', torch.zeros(K), persistent=False)

    # ---------------------------------------------------------------- dead latents / AuxK
    @torch.no_grad()
    def update_dead(self, z_t: torch.Tensor) -> None:
        """Increment the not-fired counter; reset latents that fired this batch."""
        fired = (z_t != 0).any(dim=tuple(range(z_t.dim() - 1)))   # (K,)
        self.steps_since_fired += 1
        self.steps_since_fired[fired] = 0

    def aux_reconstruct(self, z_pre: torch.Tensor):
        """AuxK: reconstruct the main recon's residual using the top-aux_k among
        currently-DEAD latents — giving them gradient so they revive.
        Returns ê (B,T,D) or None if there aren't aux_k dead latents yet."""
        if self.aux_k <= 0:
            return None
        dead = self.steps_since_fired > self.dead_threshold        # (K,)
        if int(dead.sum()) < self.aux_k:
            return None
        masked = z_pre.masked_fill(~dead, -1e30)                   # keep only dead latents
        vals, idx = masked.topk(self.aux_k, dim=-1)
        z_aux = torch.zeros_like(z_pre).scatter_(-1, idx, vals)
        return F.linear(z_aux, self.dec_weight)                    # residual recon — no b_pre

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Unit-norm the decoder columns (per Gao, applied after each step)."""
        self.dec_weight.data = F.normalize(self.dec_weight.data, dim=0)

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
        if self.block_spec is None:
            z_t = _topk_straight_through(z_pre, self.topk)
        else:
            z_t = _blockwise_topk_straight_through(z_pre, self.block_spec)
        return z_t, z_pre

    # ---------------------------------------------------------------- decode

    def decode(self, z_t: torch.Tensor) -> torch.Tensor:
        """
        z_t  : (B, T, K)  TopK-sparse features (full, not routed)
        returns ĥ_t : (B, T, D)
        """
        return F.linear(z_t, self.dec_weight) + self.b_pre
