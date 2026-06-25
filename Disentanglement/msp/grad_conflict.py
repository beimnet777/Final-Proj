"""PCGrad (Yu et al. 2020) gradient surgery on the shared SAE trunk.

The legacy logs showed cos(recon, grl) ~ -0.17 etc. — task gradients fighting on
the shared encoder weights.  GradNorm (already in the repo) only balances gradient
*magnitudes*; it does nothing about *direction* conflict.  PCGrad fixes direction:
when two task gradients conflict (negative cosine), one is projected off the
other's conflicting component.

Important design choice for THIS setup: we de-conflict only the COOPERATIVE tasks
(recon, pr, sid, prosody, emotion, inv).  The adversarial GRL gradients are left
untouched and simply added — their opposition to reconstruction is the mechanism
that strips speaker/prosody/emotion from z_L, so "fixing" that conflict would
weaken the disentanglement.

Operates on the trainable shared params (the SAE + routing); heads are task-private
and keep their ordinary gradients from total.backward().
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

import torch


class PCGrad:
    def __init__(self, shared_params, seed: int = 0) -> None:
        self.shared: List[torch.Tensor] = [p for p in shared_params if p.requires_grad]
        self.rng = random.Random(seed)

    # -- per-loss gradient on shared params, flattened (zeros if no path) --
    def _flat_grad(self, loss: torch.Tensor) -> torch.Tensor:
        if not loss.requires_grad:
            return torch.cat([p.reshape(-1) * 0 for p in self.shared])
        grads = torch.autograd.grad(loss, self.shared, retain_graph=True,
                                    allow_unused=True)
        return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                          for g, p in zip(grads, self.shared)])

    def project(self, named_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """PCGrad-combined flat gradient over the cooperative losses."""
        grads = [self._flat_grad(L) for L in named_losses.values()]
        pc = [g.clone() for g in grads]
        for i in range(len(pc)):
            order = list(range(len(grads)))
            self.rng.shuffle(order)
            for j in order:
                if i == j:
                    continue
                gj = grads[j]
                dot = torch.dot(pc[i], gj)
                if dot < 0:
                    pc[i] = pc[i] - (dot / (torch.dot(gj, gj) + 1e-12)) * gj
        return torch.stack(pc).sum(0)

    def flat_grad(self, loss: torch.Tensor) -> torch.Tensor:
        """Public single-loss flat grad (used for the adversary bundle)."""
        return self._flat_grad(loss)

    def write_(self, flat: torch.Tensor) -> None:
        """Assign a flat shared-gradient vector back onto each shared param's .grad."""
        off = 0
        for p in self.shared:
            n = p.numel()
            p.grad = flat[off:off + n].view_as(p).clone()
            off += n

    @torch.no_grad()
    def cos_table(self, grads: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Pairwise cosines between the supplied flat grads (for logging)."""
        names = list(grads)
        out = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = grads[names[i]], grads[names[j]]
                d = float(a.norm() * b.norm())
                out[f"{names[i]}|{names[j]}"] = float(torch.dot(a, b) / d) if d > 0 else 0.0
        return out
