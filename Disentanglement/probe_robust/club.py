"""CLUB: Contrastive Log-ratio Upper Bound of Mutual Information.

Used here to minimise I(z_L_pooled ; speaker_id) directly. By Fano's
inequality, low MI bounds the error of ANY downstream speaker classifier —
so this is a *probe-architecture-agnostic* speaker-removal mechanism, in
contrast to adversarial methods (GRL) which fight a single probe-head proxy
and are seed-fragile by their training dynamics.

Reference: Cheng, P., Hao, W., Dai, S., Liu, J., Gan, Z., Carin, L. (2020).
"CLUB: A Contrastive Log-ratio Upper Bound of Mutual Information." ICML 2020.
arXiv:2006.12013.

Speech-disentanglement precedent: Mun, S. et al. "Disentangled Speaker
Representation Learning via Mutual Information Minimization." arXiv:2208.08012.

Why CLUB is NOT a GAN-style adversary
-------------------------------------
The variational network q_phi(y|x) is a *density estimator*: trained
separately by cross-entropy to model p(speaker|z_L). The main model's
gradient comes from a *bound* that is a smooth function of q_phi's outputs
on both positive and shuffled-negative pairs. There is no minimax fixed
point to find. Even an imperfect q_phi provides a meaningful gradient
direction for I(z; y) reduction. This avoids the seed-fragile equilibrium
dynamics that make GRL-style adversaries unreliable (cf. statsgrl multi-seed
sweep, 0.006 / 0.378 / 0.418 across probe seeds).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _NormalizeCLUBGradient(torch.autograd.Function):
    """Identity forward with a magnitude-controlled, sign-preserving backward.

    CLUB is already minimised by the main model, so unlike a GRL this operation
    must *not* reverse its gradient.  Normalisation is per (batch, time) item
    over the feature dimension.  ``scale`` carries the objective weight and
    gradient-accumulation divisor; ``amp_scale`` compensates for GradScaler's
    later unscale so FP16 and BF16/FP32 produce the same requested magnitude.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        target: float,
        scale: float,
        amp_scale: float,
    ) -> torch.Tensor:
        ctx.target = float(target)
        ctx.scale = float(scale)
        ctx.amp_scale = float(amp_scale)
        return x.clone()

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        grad32 = grad.float()
        norm = grad32.norm(dim=-1, keepdim=True)
        # A truly zero CLUB gradient must remain zero; manufacturing a direction
        # here would turn label-collision/no-signal batches into arbitrary noise.
        unit = torch.where(
            norm > 1e-12,
            grad32 / norm.clamp_min(1e-12),
            torch.zeros_like(grad32),
        )
        magnitude = ctx.target * ctx.scale * ctx.amp_scale
        return (unit * magnitude).to(grad.dtype), None, None, None


def normalize_club_gradient(
    z_l: torch.Tensor,
    *,
    target: float,
    weight: float = 1.0,
    accumulation: int = 1,
    amp_scale: float = 1.0,
) -> torch.Tensor:
    """Return ``z_l`` unchanged while normalising only CLUB's backward branch.

    The per-microbatch gradient delivered to ``z_l`` has norm
    ``target * weight / accumulation`` for every item with a non-zero CLUB
    gradient. Summing all microbatches therefore preserves the configured
    effective-batch objective strength. ``amp_scale`` should be the current
    ``GradScaler`` scale in FP16 mode and 1 otherwise.
    """
    if target <= 0:
        raise ValueError("CLUB gradient-normalisation target must be positive")
    if accumulation < 1:
        raise ValueError("gradient accumulation must be at least one")
    return _NormalizeCLUBGradient.apply(
        z_l, float(target), float(weight) / int(accumulation), float(amp_scale)
    )


class CLUBSampled(nn.Module):
    """vCLUB-Sampled MI upper bound estimator.

    Architecture
    ------------
    q_phi : MLP classifier  Linear(in_dim, hidden) -> ReLU -> Linear(hidden, num_classes)
    optimizer : separate Adam for q_phi (does not touch main model params)

    Usage
    -----
        # Note: in_dim = 2*K_L for the mean+std pool below (x-vector tradition).
        club = CLUBSampled(2*K_L, num_speakers, lr=1e-3).to(device)
        # In training loop, AFTER main forward pass:
        #   stats-pool z_L over time: concat(mean, std) — closes the variance
        #   escape route that mean-only pool would leave open.
        fm = (arange(T) < lengths.unsqueeze(1)).float().unsqueeze(-1)
        n  = lengths.float().clamp(min=1).unsqueeze(1)
        mean = (z_L * fm).sum(1) / n
        var  = (((z_L - mean.unsqueeze(1)) ** 2) * fm).sum(1) / n
        std  = (var + 1e-5).sqrt()
        z_pool = cat([mean, std], dim=-1)                     # (B, 2*K_L)
        ce, acc = club.inner_step(z_pool.detach(), speaker_idx, k=3)
        mi_bound = club.mi_bound(z_pool, speaker_idx)         # gradient flows to z_L
        total_loss = total_loss + club_weight * mi_bound
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden: int = 512,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        # Input LayerNorm: z_pool is sparse (~5% active) high-dim (10240); without
        # normalization, default-init Linear(10240, .) produces pre-activations
        # of order 1e-3 -> logits ~0 -> softmax uniform -> CLUB bound stuck at 0.
        # Mid-layer LayerNorm: keeps hidden pre-softmax scale stable as q_phi
        # learns. GELU (not ReLU) so units do not die at near-zero init pre-acts.
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes),
        )
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        self.num_classes = num_classes
        self.last_diagnostics: dict[str, float] = {}

    def inner_step(self, z: torch.Tensor, y: torch.Tensor, k: int = 1) -> tuple[float, float]:
        """Update q_phi for k cross-entropy steps on (z, y).

        z must be DETACHED — q_phi is the density estimator, it is trained
        separately and its gradients must not flow into the main encoder.

        Returns (final CE loss, batch accuracy after last update). The accuracy
        is a real-time leakage diagnostic: as the main model minimises the
        CLUB bound, z_L should carry less speaker info, so q_phi's accuracy
        on the same batch should DROP over training. Rising q_phi_acc = main
        model not removing speaker; flat near chance = success.
        """
        last_loss = 0.0
        last_acc  = 0.0
        for _ in range(k):
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.classifier(z)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            self.optimizer.step()
            last_loss = float(loss.item())
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                last_acc = float((pred == y).float().mean().item())
        # Prevent stale q_phi gradients from being mistaken for gradients from
        # the subsequent main-model MI-minimisation pass.
        self.optimizer.zero_grad(set_to_none=True)
        return last_loss, last_acc

    def mi_bound(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """vCLUB-Sampled upper bound on I(z; y).

        Formula (Cheng 2020, Eq. 11 of arXiv:2006.12013):
            I_vCLUB-S = E_i[ log q(y_i | z_i) - log q(y_{σ(i)} | z_i) ]
        where σ is a random permutation (in-batch negative).

        Gradient flows through z back to the main encoder; the classifier's
        params get gradient too but are normally updated only by inner_step()
        (this method's gradient w.r.t. q_phi is incidental and small).
        """
        # q_phi is fitted only by inner_step. Freeze its parameters for this
        # bilevel/main-model pass while preserving the differentiable path to z.
        params = list(self.classifier.parameters())
        old = [p.requires_grad for p in params]
        try:
            for p in params: p.requires_grad_(False)
            logits = self.classifier(z)                         # (B, num_classes)
            log_probs = F.log_softmax(logits, dim=-1)
            log_q_pos = log_probs.gather(1, y.unsqueeze(1)).squeeze(1)
            perm = torch.randperm(y.shape[0], device=y.device)
            y_neg = y[perm]
            log_q_neg = log_probs.gather(1, y_neg.unsqueeze(1)).squeeze(1)
            bound = (log_q_pos - log_q_neg).mean()
            with torch.no_grad():
                probs = logits.softmax(dim=-1)
                entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1).mean()
                self.last_diagnostics = {
                    "positive_log_q": float(log_q_pos.mean()),
                    "negative_log_q": float(log_q_neg.mean()),
                    "negative_label_collision": float((y_neg == y).float().mean()),
                    "prediction_entropy": float(entropy),
                    "label_class_coverage": float(y.unique().numel() / max(1, self.num_classes)),
                }
            return bound
        finally:
            for p, enabled in zip(params, old): p.requires_grad_(enabled)
