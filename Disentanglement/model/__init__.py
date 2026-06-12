"""DISModel: frozen SPEAR encoder + TopK SAE + stage-2 routing and task heads.

Stage 1 : encoder → SAE encode → SAE decode(z_t) → recon loss only
Stage 2 : routing carves z_t → z_L / z_P / z_U
          decoder input: z_L (frame) + z_U (frame) + broadcast(mean_t(z_P))
          decoder shape stays (D, K) — stage-1 weights load directly

Experiment flags (all via cfg):
  no_routing=True       (D) — bypass routing, feed full z to all heads
  fixed_routing=True    (E) — freeze routing at init split
  n_routes=2            (F) — binary L/P, no U bucket
  grl_phoneme_weight>0  (1) — dual GRL: phoneme adversary on z_P
  ste_routing=True      (5) — straight-through estimator on routing mask mult:
                              forward = m × z_t (sparse), backward = m × z_pre (dense)
  projection_disentanglement=True — learned compressed views z_t → z_L/z_P
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .spear_encoder import SpearEncoder
from .sae import SparseAutoencoder
from .routing import RoutingModule
from .heads import PRHead, SIDHead, GRLHead, PR_GRL_Head


# ---------------------------------------------------------------- pooling

def _mean_pool(z: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Utterance-level mean pool over valid frames → (B, K)."""
    B, T, K = z.shape
    mask  = (torch.arange(T, device=z.device).unsqueeze(0) < lengths.unsqueeze(1)
             ).float().unsqueeze(-1)
    count = lengths.float().clamp(min=1).view(B, 1, 1)
    return (z * mask).sum(1) / count.squeeze(-1)


def _instance_norm_time(z: torch.Tensor, lengths: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Per-utterance, per-channel normalization over valid time frames (no affine).

    Removes each channel's per-utterance mean/scale — where speaker identity
    largely lives — while keeping frame-to-frame (content) variation.  Padding
    frames are masked out of the statistics and zeroed in the output.
    z : (B, T, C)  ->  (B, T, C)
    """
    B, T, C = z.shape
    mask = (torch.arange(T, device=z.device).unsqueeze(0) < lengths.unsqueeze(1)
            ).float().unsqueeze(-1)                              # (B, T, 1)
    n    = mask.sum(1).clamp(min=1.0)                            # (B, 1)
    mean = (z * mask).sum(1) / n                                # (B, C)
    var  = (((z - mean.unsqueeze(1)) ** 2) * mask).sum(1) / n   # (B, C)
    z_norm = (z - mean.unsqueeze(1)) / (var.unsqueeze(1) + eps).sqrt()
    return z_norm * mask                                        # zero padding


# ---------------------------------------------------------------- model

class ProjectionView(nn.Module):
    """True learned projection view: z_t (K) -> z_view (projection_dim)."""

    def __init__(self, K: int, dim: int) -> None:
        super().__init__()
        dim = max(1, min(dim, K))
        self.proj = nn.Linear(K, dim, bias=False)
        nn.init.kaiming_uniform_(self.proj.weight, a=5 ** 0.5)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)


class ProjectionUp(nn.Module):
    """Up-projection: a view (dim) -> SAE code space (K), for reconstructive projection."""

    def __init__(self, dim: int, K: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, K, bias=False)
        nn.init.kaiming_uniform_(self.proj.weight, a=5 ** 0.5)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)


class DISModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg        = cfg
        self.encoder    = SpearEncoder(cfg)
        self.sae        = SparseAutoencoder(cfg)
        self.routing    = RoutingModule(cfg)
        self.pr_head    = PRHead(cfg)
        self.sid_head   = SIDHead(cfg)
        self.grl_head   = GRLHead(cfg)
        self.pr_grl_head = PR_GRL_Head(cfg)   # Exp 1: phoneme GRL on z_P
        if getattr(cfg, 'projection_disentanglement', False):
            dim = getattr(cfg, 'projection_dim', 128)
            self.proj_L = ProjectionView(cfg.K, dim)
            self.proj_P = ProjectionView(cfg.K, dim)
            if getattr(cfg, 'projection_reconstruct', False):
                # Reconstruct h_t solely through the views: up-project each back
                # to code space, sum, then decode.  Optional penalized residual z_U.
                self.up_L = ProjectionUp(dim, cfg.K)
                self.up_P = ProjectionUp(dim, cfg.K)
                u_dim = int(getattr(cfg, 'projection_u_dim', 0))
                if u_dim > 0:
                    self.proj_U = ProjectionView(cfg.K, u_dim)
                    self.up_U   = ProjectionUp(u_dim, cfg.K)

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        stage: int = 1,
        grl_lambda: float = 0.0,
        grl_p_lambda: float | None = None,
    ) -> Dict[str, torch.Tensor]:
        # grl_p_lambda: separate reversal strength for the phoneme adversary on
        # z_P (canonical DANN folds per-adversary weights into the lambda).
        # None -> fall back to grl_lambda (legacy shared-lambda behaviour).
        if grl_p_lambda is None:
            grl_p_lambda = grl_lambda

        h_t, out_lengths = self.encoder(audio, audio_lengths)    # (B, T, D)
        z_t, z_pre       = self.sae.encode(h_t)                  # (B, T, K)

        if stage == 1:
            h_hat = self.sae.decode(z_t)
            return {"h_t": h_t, "h_hat": h_hat, "z_t": z_t,
                    "z_pre": z_pre, "out_lengths": out_lengths}

        no_routing = getattr(self.cfg, 'no_routing', False)
        n_routes   = getattr(self.cfg, 'n_routes', 3)
        ste        = getattr(self.cfg, 'ste_routing', False)
        grl_p      = getattr(self.cfg, 'grl_phoneme_weight', 0.0)
        projection = getattr(self.cfg, 'projection_disentanglement', False)

        # ---- Experiment D: bypass routing entirely ----
        if no_routing:
            h_hat    = self.sae.decode(z_t)
            z_t_pool = _mean_pool(z_t, out_lengths)
            return {
                "h_t": h_t, "h_hat": h_hat, "z_t": z_t,
                "z_pre": z_pre, "out_lengths": out_lengths,
                "z_L": z_t, "z_P": z_t, "z_P_bar": z_t_pool,
                "pr_logits":   self.pr_head(z_t),
                "sid_logits":  self.sid_head(z_t_pool),
                "grl_logits":  self.grl_head(z_t, out_lengths, grl_lambda),
            }

        # ---- Projection disentanglement: learned views instead of unit routing ----
        if projection:
            reconstruct = getattr(self.cfg, 'projection_reconstruct', False)
            z_L = self.proj_L(z_t)
            z_P = self.proj_P(z_t)
            if getattr(self.cfg, 'instance_norm_zL', False):
                # Strip per-utterance speaker statistics from z_L; reconstruction
                # must then source speaker from z_P (up_P).
                z_L = _instance_norm_time(z_L, out_lengths)
            z_P_bar = _mean_pool(z_P, out_lengths)
            z_U = None
            if reconstruct:
                # Reconstruct SOLELY through the views (no decode(z_t) path).
                z_hat = self.up_L(z_L) + self.up_P(z_P)
                if hasattr(self, 'proj_U'):
                    z_U = self.proj_U(z_t)
                    z_hat = z_hat + self.up_U(z_U)
                h_hat = self.sae.decode(z_hat)
            else:
                h_hat = self.sae.decode(z_t)
            out = {
                "h_t":         h_t,
                "h_hat":       h_hat,
                "z_t":         z_t,
                "z_pre":       z_pre,
                "out_lengths": out_lengths,
                "z_L":         z_L,
                "z_P":         z_P,
                "z_P_bar":     z_P_bar,
                "pr_logits":   self.pr_head(z_L),
                "sid_logits":  self.sid_head(z_P_bar),
                "grl_logits":  self.grl_head(z_L, out_lengths, grl_lambda),
            }
            if z_U is not None:
                out["z_U"] = z_U
            if grl_p > 0.0:
                out["pr_grl_logits"] = self.pr_grl_head(z_P, grl_p_lambda)
            return out

        # ---- Normal stage 2: get routing masks ----
        if n_routes == 2:
            m_L, m_P = self.routing(h_t)
            m_U = torch.zeros_like(m_L)
        else:
            m_L, m_P, m_U = self.routing(h_t)   # static (K,) or dynamic (B,1,K)

        # ---- Reconstruction: always from sparse z_t (decoder-consistent) ----
        z_L_sp = m_L * z_t
        z_P_sp = m_P * z_t
        z_U_sp = m_U * z_t
        z_P_bar_sp    = _mean_pool(z_P_sp, out_lengths)
        z_P_broadcast = z_P_bar_sp.unsqueeze(1).expand_as(z_t)
        z_recon       = z_L_sp + z_U_sp + z_P_broadcast
        h_hat         = self.sae.decode(z_recon)

        # ---- Exp 5: STE — task heads use dense z_pre in backward only ----
        # Forward: m × z_t (sparse, identical to standard).
        # Backward: gradient flows through m × z_pre (dense) → routing gets gradient.
        if ste:
            z_L = m_L * z_pre + (m_L * z_t - m_L * z_pre).detach()
            z_P = m_P * z_pre + (m_P * z_t - m_P * z_pre).detach()
            z_P_bar = _mean_pool(z_P, out_lengths)
        else:
            z_L, z_P, z_P_bar = z_L_sp, z_P_sp, z_P_bar_sp

        out = {
            "h_t":         h_t,
            "h_hat":       h_hat,
            "z_t":         z_t,
            "z_pre":       z_pre,
            "out_lengths": out_lengths,
            "z_L":         z_L,
            "z_P":         z_P,
            "z_P_bar":     z_P_bar,
            "m_L":         m_L,    # Exp 4: needed for ub_loss
            "m_P":         m_P,
            "routing_logits": self.routing.current_logits,   # route/spec loss (static or dynamic)
            "pr_logits":   self.pr_head(z_L),
            "sid_logits":  self.sid_head(z_P_bar),
            "grl_logits":  self.grl_head(z_L, out_lengths, grl_lambda),
        }

        # Exp 1: phoneme GRL on z_P (only when weight > 0)
        if grl_p > 0.0:
            out["pr_grl_logits"] = self.pr_grl_head(z_P, grl_p_lambda)

        return out


def build_dis_model(cfg) -> DISModel:
    model = DISModel(cfg)
    model.to(cfg.device)
    return model
