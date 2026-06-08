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

    During training : Gumbel-softmax (soft, differentiable, temperature τ annealed 1.0→0.1)
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
            soft = F.gumbel_softmax(self.logits, tau=self.tau, hard=False, dim=-1)
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
        """Entropy of mean soft-routing distribution (nats).  Max = log(n_routes)."""
        p = F.softmax(self.logits.detach(), dim=-1).mean(dim=0)
        return -(p * p.log().clamp(min=-100)).sum().item()
