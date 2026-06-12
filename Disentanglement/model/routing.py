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
        self.logits   = nn.Parameter(torch.zeros(cfg.K, self.n_routes))
        self.tau: float = cfg.gumbel_tau_start
        self.hard_gumbel_routing = getattr(cfg, 'hard_gumbel_routing', False)

        # Symmetry-breaking init: zeros sit at a symmetric saddle the optimizer
        # struggles to leave; a small random init gives Gumbel something to amplify.
        init_std = getattr(cfg, 'routing_init_std', 0.0)
        if init_std > 0:
            with torch.no_grad():
                self.logits.normal_(mean=0.0, std=init_std)

        if getattr(cfg, 'fixed_routing', False):
            self._init_fixed(cfg.K, getattr(cfg, 'fixed_routing_split', 0.7))
            self.logits.requires_grad_(False)

    def _init_fixed(self, K: int, split: float) -> None:
        n_L = int(K * split)
        with torch.no_grad():
            self.logits[:n_L, 0] = 10.0   # first split → L
            self.logits[n_L:, 1] = 10.0   # remainder  → P

    def forward(self):
        """Returns (m_L, m_P) or (m_L, m_P, m_U) — each (K,)."""
        if self.training:
            soft = F.gumbel_softmax(
                self.logits,
                tau=self.tau,
                hard=self.hard_gumbel_routing,
                dim=-1,
            )
        else:
            idx  = self.logits.argmax(dim=-1)
            soft = F.one_hot(idx, num_classes=self.n_routes).float()

        if self.n_routes == 2:
            return soft[:, 0], soft[:, 1]
        return soft[:, 0], soft[:, 1], soft[:, 2]

    @property
    def hard_counts(self):
        """(n_L, n_P, n_U) — hard feature counts; n_U=0 when n_routes=2."""
        idx = self.logits.detach().argmax(dim=-1)
        n_L = (idx == 0).sum().item()
        n_P = (idx == 1).sum().item()
        n_U = (idx == 2).sum().item() if self.n_routes == 3 else 0
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
        p = F.softmax(self.logits.detach(), dim=-1).mean(dim=0)
        return -(p * p.log().clamp(min=-100)).sum().item()

    @property
    def routing_unit_entropy(self) -> float:
        """Mean per-unit routing entropy. Lower means units are more specialised."""
        p = F.softmax(self.logits.detach(), dim=-1)
        unit_H = -(p * p.log().clamp(min=-100)).sum(dim=-1)
        return unit_H.mean().item()

    @property
    def routing_diagnostics(self) -> dict[str, float]:
        """Detailed routing diagnostics for balance and specialisation."""
        logits = self.logits.detach()
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
