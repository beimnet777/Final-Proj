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
from .heads import PRHead, SIDHead, GRLHead, PR_GRL_Head, ProsodyHead, Prosody_GRL_Head


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
    """Learned projection view: z_t (K) -> z_view (dim).

    hidden=0 -> single linear map (legacy).  hidden>0 -> a 2-layer MLP
    (Linear-ReLU-Linear): a genuine NONLINEAR demixer that can compute a
    speaker-invariant content code even when speaker is linearly entangled
    in the frozen features.
    """

    def __init__(self, K: int, dim: int, hidden: int = 0) -> None:
        super().__init__()
        dim = max(1, min(dim, K))
        if hidden > 0:
            self.proj = nn.Sequential(nn.Linear(K, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        else:
            lin = nn.Linear(K, dim, bias=False)
            nn.init.kaiming_uniform_(lin.weight, a=5 ** 0.5)
            self.proj = lin

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)


class ProjectionUp(nn.Module):
    """Up-projection: a view (dim) -> SAE code space (K), for reconstructive projection."""

    def __init__(self, dim: int, K: int, hidden: int = 0) -> None:
        super().__init__()
        if hidden > 0:
            self.proj = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, K))
        else:
            lin = nn.Linear(dim, K, bias=False)
            nn.init.kaiming_uniform_(lin.weight, a=5 ** 0.5)
            self.proj = lin

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
        # Adversaries on the residual z_U: force it to carry NEITHER factor.
        if getattr(cfg, 'grl_u_weight', 0.0) > 0 or getattr(cfg, 'grl_phoneme_u_weight', 0.0) > 0:
            self.grl_head_u    = GRLHead(cfg)       # anti-speaker on z_U
            self.pr_grl_head_u = PR_GRL_Head(cfg)   # anti-phoneme on z_U
        # Prosody: per-frame F0/energy regression on z_P (gives z_P a frame-level
        # task so it survives emergent top-k), + optional anti-prosody adversaries.
        if getattr(cfg, 'prosody', False):
            self.prosody_head = ProsodyHead(cfg)
            if getattr(cfg, 'grl_prosody_weight', 0.0) > 0:
                self.prosody_grl_head = Prosody_GRL_Head(cfg)     # anti-prosody on z_L
            if getattr(cfg, 'grl_prosody_u_weight', 0.0) > 0:
                self.prosody_grl_head_u = Prosody_GRL_Head(cfg)   # anti-prosody on z_U
        # Option A: fixed L/P/U index blocks — constant masks, routing disabled.
        if getattr(cfg, 'fixed_blocks', False):
            kL, kP = cfg.K_L, cfg.K_P
            idx = torch.zeros(cfg.K, dtype=torch.long)
            idx[kL:kL + kP] = 1
            idx[kL + kP:]   = 2
            self.register_buffer('block_idx', idx)
            self.register_buffer('block_m_L', (idx == 0).float())
            self.register_buffer('block_m_P', (idx == 1).float())
            self.register_buffer('block_m_U', (idx == 2).float())
            for p in self.routing.parameters():   # routing is dead in this mode
                p.requires_grad_(False)
        _projection = getattr(cfg, 'projection_disentanglement', False)
        _proj_dim   = getattr(cfg, 'projection_dim', 128)
        # VIB on z_L: learned per-feature log-variance for the bottleneck.  In
        # projection mode the bottleneck lives on the projected view (dim), else
        # on the full sparse code (K).
        if getattr(cfg, 'vib_zL_weight', 0.0) > 0:
            self.vib_logvar = nn.Parameter(torch.zeros(_proj_dim if _projection else cfg.K))
        if _projection:
            dim    = _proj_dim
            hidden = int(getattr(cfg, 'projection_hidden', 0)) if getattr(cfg, 'projection_nonlinear', False) else 0
            self.proj_L = ProjectionView(cfg.K, dim, hidden)
            self.proj_P = ProjectionView(cfg.K, dim, hidden)
            if getattr(cfg, 'projection_reconstruct', False):
                # Reconstruct h_t solely through the views: up-project each back
                # to code space, sum, then decode.  Optional penalized residual z_U.
                self.up_L = ProjectionUp(dim, cfg.K, hidden)
                self.up_P = ProjectionUp(dim, cfg.K, hidden)
                u_dim = int(getattr(cfg, 'projection_u_dim', 0))
                if u_dim > 0:
                    self.proj_U = ProjectionView(cfg.K, u_dim, hidden)
                    self.up_U   = ProjectionUp(u_dim, cfg.K, hidden)

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        stage: int = 1,
        grl_lambda: float = 0.0,
        grl_p_lambda: float | None = None,
        grl_u_lambda: float = 0.0,
        grl_p_u_lambda: float = 0.0,
        grl_prosody_lambda: float = 0.0,
        grl_prosody_u_lambda: float = 0.0,
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
            # VIB on the z_L view: KL(N(mu,sigma) || N(0,1)) compresses z_L, dropping
            # anything PR/recon don't need (incl. separable speaker).  Reparameterized
            # sample is used for PR, GRL and reconstruction → a true info bottleneck.
            vib_kl = None
            if getattr(self.cfg, 'vib_zL_weight', 0.0) > 0 and hasattr(self, 'vib_logvar'):
                if getattr(self.cfg, 'vib_zL_layernorm', False):
                    # param-free per-frame LayerNorm: bounds ||z_L|| so KL (½·mu²) can't
                    # run away.  Compression then happens via the learned noise, not scale.
                    _m = z_L.mean(-1, keepdim=True)
                    _v = z_L.var(-1, keepdim=True, unbiased=False)
                    z_L = (z_L - _m) / (_v + 1e-5).sqrt()
                mu     = z_L
                logvar = self.vib_logvar.clamp(-8.0, 8.0)
                std    = torch.exp(0.5 * logvar)
                fmask  = (torch.arange(z_t.shape[1], device=z_t.device).unsqueeze(0)
                          < out_lengths.unsqueeze(1)).float()                  # (B, T)
                kl_bt  = (0.5 * (mu ** 2 + std ** 2 - logvar - 1.0)).sum(-1)    # (B, T)
                vib_kl = (kl_bt * fmask).sum() / fmask.sum().clamp(min=1)
                if self.training:
                    z_L = mu + std * torch.randn_like(mu)                       # reparameterized
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
            if vib_kl is not None:
                out["vib_kl"] = vib_kl
            if grl_p > 0.0:
                out["pr_grl_logits"] = self.pr_grl_head(z_P, grl_p_lambda)
            return out

        # ---- Option A: fixed-block supervised SAE (no routing) ----
        if getattr(self.cfg, 'fixed_blocks', False):
            m_L, m_P, m_U = self.block_m_L, self.block_m_P, self.block_m_U
            z_L_sp = m_L * z_t
            z_P_sp = m_P * z_t
            z_U_sp = m_U * z_t
            if getattr(self.cfg, 'instance_norm_zL', False):
                # Structural speaker removal from z_L: strip per-utterance per-channel
                # mean/scale over time (the dual of P's pooling).  Used for BOTH the
                # heads and reconstruction, so recon must source speaker from z_P.
                z_L_sp = _instance_norm_time(z_L_sp, out_lengths)
            vib_kl = None
            if getattr(self.cfg, 'vib_zL_weight', 0.0) > 0 and hasattr(self, 'vib_logvar'):
                # VIB: z_L ~ N(mu, sigma^2) on active features; KL to N(0,1) compresses
                # mu→0 for anything PR doesn't need (incl. separable speaker).
                mu      = z_L_sp
                logvar  = self.vib_logvar.clamp(-8.0, 8.0)
                std     = torch.exp(0.5 * logvar)
                active  = (mu != 0).float()
                fmask   = (torch.arange(z_t.shape[1], device=z_t.device).unsqueeze(0)
                           < out_lengths.unsqueeze(1)).float()             # (B, T)
                kl_bt   = (0.5 * (mu ** 2 + std ** 2 - logvar - 1.0) * active).sum(-1)
                vib_kl  = (kl_bt * fmask).sum() / fmask.sum().clamp(min=1)
                if self.training:
                    z_L_sp = mu + active * std * torch.randn_like(mu)      # reparameterized sample
            z_P_bar_sp    = _mean_pool(z_P_sp, out_lengths)   # kept ONLY as the SID-head readout
            z_recon       = z_L_sp + z_U_sp + z_P_sp          # per-frame z_P (no static assumption)
            h_hat         = self.sae.decode(z_recon)
            out = {
                "h_t":         h_t,
                "h_hat":       h_hat,
                "z_t":         z_t,
                "z_pre":       z_pre,
                "out_lengths": out_lengths,
                "z_L":         z_L_sp,
                "z_P":         z_P_sp,
                "z_U":         z_U_sp,
                "z_P_bar":     z_P_bar_sp,
                "m_L":         m_L,
                "m_P":         m_P,
                "pr_logits":   self.pr_head(z_L_sp),
                "sid_logits":  self.sid_head(z_P_bar_sp),
                "grl_logits":  self.grl_head(z_L_sp, out_lengths, grl_lambda),
            }
            if grl_p > 0.0:
                out["pr_grl_logits"] = self.pr_grl_head(z_P_sp, grl_p_lambda)
            if hasattr(self, 'grl_head_u'):
                out["grl_u_logits"]    = self.grl_head_u(z_U_sp, out_lengths, grl_u_lambda)
                out["pr_grl_u_logits"] = self.pr_grl_head_u(z_U_sp, grl_p_u_lambda)
            if hasattr(self, 'prosody_head'):
                out["prosody_pred"] = self.prosody_head(z_P_sp)        # per-frame [logF0, logE]
                if hasattr(self, 'prosody_grl_head'):
                    out["prosody_grl_pred"]   = self.prosody_grl_head(z_L_sp, grl_prosody_lambda)
                if hasattr(self, 'prosody_grl_head_u'):
                    out["prosody_grl_u_pred"] = self.prosody_grl_head_u(z_U_sp, grl_prosody_u_lambda)
            if vib_kl is not None:
                out["vib_kl"] = vib_kl
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
        z_P_bar_sp    = _mean_pool(z_P_sp, out_lengths)   # kept ONLY as the SID-head readout
        z_recon       = z_L_sp + z_U_sp + z_P_sp          # per-frame z_P (no static assumption)
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
            "z_U":         z_U_sp,
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
        # Adversaries on the residual z_U (anti-speaker + anti-phoneme)
        if hasattr(self, 'grl_head_u'):
            out["grl_u_logits"]    = self.grl_head_u(z_U_sp, out_lengths, grl_u_lambda)
            out["pr_grl_u_logits"] = self.pr_grl_head_u(z_U_sp, grl_p_u_lambda)
        # Prosody on z_P (+ optional anti-prosody adversaries on z_L / z_U)
        if hasattr(self, 'prosody_head'):
            out["prosody_pred"] = self.prosody_head(z_P)
            if hasattr(self, 'prosody_grl_head'):
                out["prosody_grl_pred"]   = self.prosody_grl_head(z_L, grl_prosody_lambda)
            if hasattr(self, 'prosody_grl_head_u'):
                out["prosody_grl_u_pred"] = self.prosody_grl_head_u(z_U_sp, grl_prosody_u_lambda)

        return out


def build_dis_model(cfg) -> DISModel:
    model = DISModel(cfg)
    model.to(cfg.device)
    return model
