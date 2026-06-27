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

Operates on the trainable SAE parameters. Routing and task-private heads keep their
ordinary gradients from total.backward().
"""
from __future__ import annotations

import random
from typing import Dict, List

import torch


def named_gradient_diagnostics(
    named_losses: Dict[str, torch.Tensor],
    params,
    reference: torch.Tensor | None = None,
) -> dict:
    """Read-only weighted-gradient diagnostics for an arbitrary parameter group.

    When `reference` is supplied, `push_cos[name]` is the signed first-order
    direction in which gradient descent on that loss moves the reference:
    positive increases it, negative decreases it.
    """
    shared = [p for p in params if p.requires_grad]

    def _flat_grad(loss: torch.Tensor) -> torch.Tensor:
        if not loss.requires_grad:
            return torch.cat([p.reshape(-1) * 0 for p in shared])
        grads = torch.autograd.grad(loss, shared, retain_graph=True, allow_unused=True)
        return torch.cat([
            (g if g is not None else torch.zeros_like(p)).reshape(-1)
            for g, p in zip(grads, shared)
        ])

    vectors = {}
    for name, loss in named_losses.items():
        vectors[name] = _flat_grad(loss)

    def _norm(x: torch.Tensor) -> float:
        return float(x.detach().float().norm().item())

    def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
        af, bf = a.detach().float(), b.detach().float()
        den = af.norm() * bf.norm()
        return float(torch.dot(af, bf).div(den.clamp(min=1e-12)).item())

    names = list(vectors)
    cosines = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            cosines[f"{a}|{b}"] = _cos(vectors[a], vectors[b])
    out = {
        "vectors": vectors,
        "norms": {name: _norm(g) for name, g in vectors.items()},
        "cosines": cosines,
    }
    if reference is not None:
        ref_grad = _flat_grad(reference)
        total_grad = torch.stack(list(vectors.values())).sum(0)
        # theta <- theta - eta*g_loss, so delta(reference) ~ -eta*g_ref.g_loss.
        out["reference_value"] = float(reference.detach())
        out["reference_grad_norm"] = _norm(ref_grad)
        out["push_cos"] = {name: -_cos(ref_grad, g) for name, g in vectors.items()}
        out["push_effect"] = {
            name: -float(torch.dot(ref_grad.detach().float(), g.detach().float()).item())
            for name, g in vectors.items()
        }
        out["total_push_cos"] = -_cos(ref_grad, total_grad)
        out["total_push_effect"] = -float(
            torch.dot(ref_grad.detach().float(), total_grad.detach().float()).item())
    return out


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

    def _project(self, grads: List[torch.Tensor]) -> List[torch.Tensor]:
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
        return pc

    def project(self, named_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """PCGrad-combined flat gradient over the cooperative losses."""
        grads = [self._flat_grad(loss) for loss in named_losses.values()]
        return torch.stack(self._project(grads)).sum(0)

    def project_with_diagnostics(
        self,
        named_losses: Dict[str, torch.Tensor],
        external_losses: Dict[str, torch.Tensor],
    ):
        """Project cooperative gradients and summarize what PCGrad actually sees.

        External losses are not projected. They are returned as one flat gradient
        for the normal adversarial addition, while their individual norms and
        cosines remain visible in the diagnostics.
        """
        coop = {name: self._flat_grad(loss) for name, loss in named_losses.items()}
        external = {name: self._flat_grad(loss) for name, loss in external_losses.items()}
        projected_parts = self._project(list(coop.values()))
        projected = torch.stack(projected_parts).sum(0)
        raw_sum = torch.stack(list(coop.values())).sum(0)
        if external:
            external_sum = torch.stack(list(external.values())).sum(0)
        else:
            external_sum = torch.zeros_like(projected)

        def _norm(x: torch.Tensor) -> float:
            return float(x.detach().float().norm().item())

        def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
            af, bf = a.detach().float(), b.detach().float()
            den = af.norm() * bf.norm()
            return float(torch.dot(af, bf).div(den.clamp(min=1e-12)).item())

        coop_names = list(coop)
        coop_cos = {}
        for i, a in enumerate(coop_names):
            for b in coop_names[i + 1:]:
                coop_cos[f"{a}|{b}"] = _cos(coop[a], coop[b])
        diagnostics = {
            "norms": {name: _norm(g) for name, g in {**coop, **external}.items()},
            "raw_coop_norm": _norm(raw_sum),
            "projected_coop_norm": _norm(projected),
            "external_norm": _norm(external_sum),
            "coop_cosines": coop_cos,
            "coop_conflicts": sum(c < 0.0 for c in coop_cos.values()),
            "external_vs_recon": (
                {name: _cos(coop["recon"], g) for name, g in external.items()}
                if "recon" in coop else {}
            ),
            "coop_vs_external": {
                name: _cos(g, external_sum) for name, g in coop.items()
            },
        }
        return projected, external_sum, diagnostics

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
