"""Gumbel-softmax routing: assigns each of the K SAE features to L / P / U (or L / P).

Routing does NOT touch reconstruction — the decoder always uses full z_t.
It only carves z_t into masks for the downstream task heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoutingModule(nn.Module):
    """Per-feature soft assignment to n_routes groups (default 3: L, P, U).

    During training : soft Gumbel-softmax by default; optionally hard ST-Gumbel
                      when cfg.hard_gumbel_routing=True.
    During eval     : hard argmax (deterministic one-hot)

    cfg.n_routes    : 3 = L/P/U (default)  |  2 = binary L/P (experiment F)
    cfg.fixed_routing : if True, init to a fixed split and freeze logits (experiment E)
    cfg.fixed_routing_split : fraction of features assigned to L (default 0.7)
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.n_routes = getattr(cfg, 'n_routes', 3)
        self.K        = cfg.K
        # Static base partition (input-independent). In dynamic mode it acts as a
        # learned bias the per-utterance router deviates from.
        self.logits   = nn.Parameter(torch.zeros(cfg.K, self.n_routes))
        self.tau: float = cfg.gumbel_tau_start
        self.hard_gumbel_routing = getattr(cfg, 'hard_gumbel_routing', False)

        # Symmetry-breaking init: zeros sit at a symmetric saddle the optimizer
        # struggles to leave; a small random init gives Gumbel something to amplify.
        init_std = getattr(cfg, 'routing_init_std', 0.0)
        if init_std > 0:
            with torch.no_grad():
                self.logits.normal_(mean=0.0, std=init_std)

        # Dynamic (input-dependent) routing: a per-utterance router conditioned on
        # mean-pooled h_t produces a per-feature delta added to the static base, so
        # the L/P/U partition can adapt per utterance.  Default off → static.
        self.dynamic = bool(getattr(cfg, 'routing_dynamic', False))
        if self.dynamic:
            hidden = int(getattr(cfg, 'routing_dynamic_hidden', 256))
            self.router = nn.Sequential(
                nn.Linear(cfg.D, hidden), nn.ReLU(),
                nn.Linear(hidden, cfg.K * self.n_routes),
            )
            nn.init.normal_(self.router[-1].weight, std=1e-2)  # tiny delta → start ≈ static, but trainable
            nn.init.zeros_(self.router[-1].bias)
        self.current_logits = None   # last forward's effective logits (grad-carrying)

        if getattr(cfg, 'fixed_routing', False):
            self._init_fixed(cfg.K, getattr(cfg, 'fixed_routing_split', 0.7))
            self.logits.requires_grad_(False)

    def _init_fixed(self, K: int, split: float) -> None:
        n_L = int(K * split)
        with torch.no_grad():
            self.logits[:n_L, 0] = 10.0   # first split → L
            self.logits[n_L:, 1] = 10.0   # remainder  → P

    def _effective_logits(self, context):
        """(K, n_routes) if static; (B, K, n_routes) if dynamic (per utterance)."""
        if not self.dynamic:
            return self.logits
        ctx = context.mean(dim=1)                                   # (B, D)  utterance summary
        delta = self.router(ctx).view(-1, self.K, self.n_routes)   # (B, K, n_routes)
        return self.logits.unsqueeze(0) + delta                    # static base + dynamic delta

    def forward(self, context=None):
        """Returns (m_L, m_P[, m_U]).  Static → each (K,); dynamic → each (B,1,K)."""
        logits = self._effective_logits(context)
        if self.training:
            soft = F.gumbel_softmax(logits, tau=self.tau, hard=self.hard_gumbel_routing, dim=-1)
        else:
            idx  = logits.argmax(dim=-1)
            soft = F.one_hot(idx, num_classes=self.n_routes).float()
        self.current_logits = logits     # grad-carrying — used for route/spec loss

        if self.dynamic:
            # (B, K, n_routes) → per-route (B, 1, K) for broadcasting over time
            masks = [soft[:, :, r].unsqueeze(1) for r in range(self.n_routes)]
        else:
            masks = [soft[:, r] for r in range(self.n_routes)]     # each (K,)
        return tuple(masks[:2]) if self.n_routes == 2 else tuple(masks)

    def _diag_logits(self):
        src = self.current_logits if (self.dynamic and self.current_logits is not None) else self.logits
        return src.detach().reshape(-1, self.n_routes)   # (N, n_routes), N=K or B*K

    @property
    def hard_counts(self):
        """(n_L, n_P, n_U) — feature counts (per-utterance avg in dynamic mode)."""
        lg  = self._diag_logits()
        bf  = lg.shape[0] / self.K if self.dynamic else 1.0   # batch factor
        idx = lg.argmax(dim=-1)
        n_L = int((idx == 0).sum().item() / bf)
        n_P = int((idx == 1).sum().item() / bf)
        n_U = int((idx == 2).sum().item() / bf) if self.n_routes == 3 else 0
        return n_L, n_P, n_U

    @property
    def routing_entropy(self) -> float:
        """Legacy balance entropy of the mean soft-routing distribution.

        This measures global bucket balance, not whether individual units have
        specialised. Kept for backwards-compatible logs/analysis.
        """
        return self.routing_balance_entropy

    @property
    def routing_balance_entropy(self) -> float:
        """Entropy of mean soft-routing distribution (nats). Max = log(n_routes)."""
        p = F.softmax(self._diag_logits(), dim=-1).mean(dim=0)
        return -(p * p.log().clamp(min=-100)).sum().item()

    @property
    def routing_unit_entropy(self) -> float:
        """Mean per-unit routing entropy. Lower means units are more specialised."""
        p = F.softmax(self._diag_logits(), dim=-1)
        unit_H = -(p * p.log().clamp(min=-100)).sum(dim=-1)
        return unit_H.mean().item()

    @property
    def routing_diagnostics(self) -> dict[str, float]:
        """Detailed routing diagnostics for balance and specialisation."""
        logits = self._diag_logits()
        p = F.softmax(logits, dim=-1)
        p_mean = p.mean(dim=0)
        unit_H = -(p * p.log().clamp(min=-100)).sum(dim=-1)
        top2 = p.topk(k=min(2, self.n_routes), dim=-1).values
        margin = top2[:, 0] - top2[:, 1] if self.n_routes > 1 else top2[:, 0]

        stats = {
            "balance_entropy": -(p_mean * p_mean.log().clamp(min=-100)).sum().item(),
            "unit_entropy": unit_H.mean().item(),
            "unit_entropy_min": unit_H.min().item(),
            "unit_entropy_max": unit_H.max().item(),
            "specialized_frac_h_lt_0_5": (unit_H < 0.5).float().mean().item(),
            "specialized_frac_h_lt_0_8": (unit_H < 0.8).float().mean().item(),
            "top1_top2_margin": margin.mean().item(),
            "logit_std": logits.std().item(),
            "logit_range": (logits.max() - logits.min()).item(),
        }
        for i in range(self.n_routes):
            stats[f"mean_p_route_{i}"] = p_mean[i].item()
        return stats
