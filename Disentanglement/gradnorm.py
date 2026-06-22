"""GradNorm (Chen et al. 2018) for the disentanglement stage-2 losses.

Balances each managed task's gradient magnitude on a single shared parameter
(the SAE encoder weight) so no loss term dominates the shared backbone — the
automatic version of hand-tuning alpha/beta/grl weights to "gradient parity".

Usage (per training step, before total.backward()):
    w = ctrl.weights()                      # dict name -> float (use in `total`)
    ...build total with those weights...
    ctrl.update({name: L_i, ...})           # learns the weights for the NEXT step
    total.backward()                        # graph retained by update()'s autograd.grad
"""
from typing import Dict, List
import torch


class GradNormController:
    def __init__(self, task_names: List[str], shared_param: torch.Tensor,
                 alpha: float = 1.5, lr: float = 0.025, device: str = "cuda") -> None:
        self.names = list(task_names)
        self.n = len(self.names)
        self.shared = shared_param                    # one tensor (the shared layer W)
        self.alpha = float(alpha)
        self.w = torch.ones(self.n, device=device, requires_grad=True)
        self.opt = torch.optim.Adam([self.w], lr=lr)
        self.L0: torch.Tensor | None = None           # initial per-task losses

    @torch.no_grad()
    def weights(self) -> Dict[str, float]:
        """Current weights as plain floats — multiply into the model's total loss."""
        return {n: float(self.w[i]) for i, n in enumerate(self.names)}

    def update(self, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """One GradNorm step on the weights. `losses` must contain every task name
        (unweighted scalar loss tensors, still attached to the graph).

        Uses retain_graph=True so the caller can run total.backward() afterwards.
        """
        L = torch.stack([losses[n] for n in self.names])              # (T,)
        if self.L0 is None:
            self.L0 = L.detach().clamp(min=1e-8)

        # Per-task gradient norm on the shared parameter (detached → constant).
        gnorms = []
        for n in self.names:
            Ln = losses[n]
            # A task can be a constant (no grad_fn) early on — e.g. `aux` before any
            # dead latents exist, so aux_reconstruct returns a plain zeros tensor.
            # Treat its gradient norm as 0 (its weight just stays put until it activates).
            if not Ln.requires_grad:
                gnorms.append(torch.zeros((), device=self.w.device))
                continue
            g = torch.autograd.grad(Ln, self.shared, retain_graph=True,
                                    allow_unused=True)[0]
            gn = (g.detach().float().norm() if g is not None
                  else torch.zeros((), device=self.w.device))
            gnorms.append(gn)
        gnorm = torch.stack(gnorms)                                   # (T,) detached

        G = self.w * gnorm                                            # differentiable wrt w
        G_avg = G.detach().mean()
        loss_ratio = L.detach() / self.L0                            # relative training progress
        r = loss_ratio / loss_ratio.mean().clamp(min=1e-8)          # relative inverse rate
        target = (G_avg * r.pow(self.alpha)).detach()
        L_grad = (G - target).abs().sum()                           # GradNorm objective on w

        self.opt.zero_grad(set_to_none=True)
        L_grad.backward()                                            # grad flows only to self.w
        self.opt.step()
        with torch.no_grad():
            self.w.clamp_(min=1e-3)                                  # keep positive
            self.w.mul_(self.n / self.w.sum().clamp(min=1e-8))      # renormalize: sum == n_tasks
        return self.weights()
