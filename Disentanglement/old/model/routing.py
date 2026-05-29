"""Per-unit hard routing with ST-Gumbel-Softmax.

Each of the K latent units is assigned to exactly one of three groups:
    L — linguistic
    P — paralinguistic
    U — unused / residual

Group indices: 0=L, 1=P, 2=U.

Non-overlap is guaranteed by construction: the three masks are one-hot per unit
and their element-wise products are zero.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoutingModule(nn.Module):
    """Global routing logits over {L, P, U} for K latent units.

    Attributes
    ----------
    logits : Parameter (K, 3)
        Raw (un-normalised) routing logits.  Softmax over dim=1 gives the
        soft assignment probabilities used during backward.
    tau : float
        Current Gumbel temperature.  Set by the trainer each step via
        ``module.tau = value``; not a registered parameter.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        K = cfg.K
        # Small random init so the three groups start roughly balanced.
        self.logits = nn.Parameter(torch.randn(K, 3) * 0.01)
        self.tau: float = cfg.gumbel_tau_start

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return three binary masks, each shape (K,).

        Forward pass uses a hard one-hot assignment (argmax).
        Backward pass uses the soft Gumbel-Softmax relaxation
        (straight-through estimator).
        """
        # Hard assignment (no gradient)
        a_hard = F.one_hot(
            self.logits.argmax(dim=-1), num_classes=3
        ).float()                                       # (K, 3)

        # Soft relaxation (carries gradient through logits)
        a_soft = F.gumbel_softmax(self.logits, tau=self.tau, hard=False)  # (K, 3)

        # Straight-through: forward=hard, backward=soft
        a = a_soft + (a_hard - a_soft).detach()        # (K, 3)

        m_L = a[:, 0]   # (K,)
        m_P = a[:, 1]
        m_U = a[:, 2]
        return m_L, m_P, m_U

    @property
    def hard_counts(self) -> Tuple[int, int, int]:
        """Number of units hard-assigned to L, P, U (for logging)."""
        idx = self.logits.detach().argmax(dim=-1)       # (K,)
        return (
            (idx == 0).sum().item(),
            (idx == 1).sum().item(),
            (idx == 2).sum().item(),
        )

    @property
    def routing_entropy(self) -> float:
        """Mean per-unit entropy (nats) over K routing distributions.

        Correct specialisation metric: starts at log(3)≈1.099 when all units
        are uniform, drops toward 0 as individual units commit to one group.
        Previous implementation computed H(mean_k p_k) which is always ≈log(3)
        regardless of how specialised individual units are.
        """
        p = F.softmax(self.logits.detach(), dim=-1)          # (K, 3)
        unit_H = -(p * p.log().clamp(min=-100)).sum(-1)      # (K,)
        return unit_H.mean().item()
