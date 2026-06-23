"""Task heads for stage 2 disentanglement.

PRHead     : CTC phoneme recognition on z_L  (frame-level)
SIDHead    : Speaker classification on z_P_bar  (utterance-level mean pool)
GRLHead    : Adversarial speaker head on z_L with gradient reversal
PR_GRL_Head: Adversarial phoneme head on z_P with gradient reversal  (Exp 1 — dual GRL)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- gradient reversal

class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return -ctx.lam * grad, None


def gradient_reversal(x: torch.Tensor, lam: float) -> torch.Tensor:
    return _GRL.apply(x, lam)


class _GRLNorm(torch.autograd.Function):
    """Gradient reversal with PER-FRAME gradient NORMALIZATION.

    Backward L2-normalizes the incoming gradient over the feature dim (per (B,T)
    frame) to a fixed `target` magnitude, then reverses it.  This decouples the
    encoder-side removal push from the discriminator's confidence: a near-chance
    discriminator (tiny gradient) is boosted to full strength, and every frame
    receives an equal-magnitude push — directly countering the per-frame
    dilution that sinks the pooled speaker adversary.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float, target: float) -> torch.Tensor:
        ctx.lam = lam; ctx.target = target
        return x.clone()

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        n = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        g = grad / n * ctx.target                       # unit per-frame, scaled to target
        return -ctx.lam * g, None, None


def gradient_reversal_norm(x: torch.Tensor, lam: float, target: float) -> torch.Tensor:
    return _GRLNorm.apply(x, lam, target)


# ---------------------------------------------------------------- PR head

def _head_input_dim(cfg) -> int:
    if getattr(cfg, "projection_disentanglement", False):
        return getattr(cfg, "projection_dim", cfg.K)
    return cfg.K


class PRHead(nn.Module):
    """Linear CTC head: z_L (B, T, input_dim) -> logits."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.fc = nn.Linear(_head_input_dim(cfg), cfg.vocab_size)

    def forward(self, z_L: torch.Tensor) -> torch.Tensor:
        return self.fc(z_L)


# ---------------------------------------------------------------- SID head

class SIDHead(nn.Module):
    """Speaker CE head: mean(z_P) (B, input_dim) -> linear -> logits.

    Deliberately the WEAKEST head: a single linear layer on the mean-pooled z_P
    (no projector, no nonlinearity).  Per the task<=probe<=adversary principle,
    a weak task head forces z_P to make speaker linearly accessible rather than
    letting the head do the work.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.fc = nn.Linear(_head_input_dim(cfg), cfg.num_speakers)

    def forward(self, z_P_bar: torch.Tensor) -> torch.Tensor:
        return self.fc(z_P_bar)


# ---------------------------------------------------------------- PR-GRL head  (Exp 1 — dual GRL)

class PR_GRL_Head(nn.Module):
    """Adversarial phoneme head on z_P with gradient reversal.

    Mirrors the diagnostic PR probe style: frame projection followed by a
    frame classifier.  When z_P encodes phonemes, the reversed gradient
    penalises the model for putting phone information into the paralinguistic
    bucket.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.projector = nn.Linear(_head_input_dim(cfg), 256)
        self.fc = nn.Linear(256, cfg.vocab_size)
        # grad_norm=True: per-frame normalize the reversed gradient to a fixed
        # magnitude (same mechanism as the speaker GRLHead) — a CONSTANT per-frame
        # content-removal push on z_P, decoupled from the discriminator's confidence.
        self.grad_norm        = bool(getattr(cfg, "grl_p_grad_norm", False))
        self.grad_norm_target = float(getattr(cfg, "grl_p_grad_norm_target",
                                               getattr(cfg, "grl_grad_norm_target", 0.001)))

    def forward(self, z_P: torch.Tensor, lam: float) -> torch.Tensor:
        z_P = (gradient_reversal_norm(z_P, lam, self.grad_norm_target)
               if self.grad_norm else gradient_reversal(z_P, lam))
        # MLP (ReLU) — stronger than the linear PR probe it must beat.
        return self.fc(F.relu(self.projector(z_P)))


# ---------------------------------------------------------------- Prosody head

class ProsodyHead(nn.Module):
    """Per-frame prosody regressor on z_P: projection -> ReLU -> [log-F0, log-E].

    Prosody is a frame-level (suprasegmental) signal, so this predicts the F0 and
    energy contour at EVERY frame (no pooling) — the opposite of the pooled SID
    head.  This per-frame supervision is what gives z_P a frame-level reason to
    win the emergent top-k, so the paralinguistic block survives without a forced
    per-block budget.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.projector = nn.Linear(_head_input_dim(cfg), 256)
        self.head = nn.Linear(256, 2)                     # [log-F0, log-energy]

    def forward(self, z_P: torch.Tensor) -> torch.Tensor:
        return self.head(F.relu(self.projector(z_P)))     # (B, T, 2)


class Prosody_GRL_Head(nn.Module):
    """Adversarial prosody regressor with gradient reversal (for z_L / z_U).

    Mirrors PR_GRL_Head but regresses the prosody contour.  When the bucket
    encodes prosody, the reversed gradient penalises the model for leaving F0 /
    energy in the linguistic / residual blocks — pushing prosody into z_P.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.projector = nn.Linear(_head_input_dim(cfg), 256)
        self.head = nn.Linear(256, 2)

    def forward(self, z: torch.Tensor, lam: float) -> torch.Tensor:
        z = gradient_reversal(z, lam)
        return self.head(F.relu(self.projector(z)))       # (B, T, 2)


# ---------------------------------------------------------------- GRL head

class GRLHead(nn.Module):
    """Adversarial speaker head on z_L with gradient reversal.

    Mirrors the diagnostic SID probe style: frame projection, masked mean pool,
    then speaker classifier.  GRL reverses gradients so the model is penalised
    for encoding speaker in L features.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        P = 256
        self.projector = nn.Linear(_head_input_dim(cfg), P)
        # frame_level=True: predict speaker at every frame (wrong granularity —
        # speaker is utterance-level, so a frame classifier stalls at chance).
        # Default: pool over time, then classify.
        self.frame_level    = bool(getattr(cfg, "grl_frame_level", False))
        # attention_pool=True: ATTENTIVE STATISTICS pooling — a learned scorer
        # weights each frame, then we pool a weighted mean+std.  Lets the
        # discriminator concentrate on the most speaker-informative frames →
        # a much stronger adversary than a flat mean-pool.
        self.attention_pool = bool(getattr(cfg, "grl_attention_pool", False))
        # stats_pool=True: match the diagnostic SID-stats probe exactly after
        # the projector: ReLU -> masked mean+std -> classifier.  This tests
        # whether the training adversary can remove what the leakage probe sees.
        self.stats_pool     = bool(getattr(cfg, "grl_stats_pool", False))
        # dense_context=True: per-frame speaker prediction (like grl_p) but with a
        # temporal conv for local context, so each frame gets its OWN removal
        # gradient (dense) instead of one diluted pooled gradient over T frames.
        self.dense_context  = bool(getattr(cfg, "grl_dense_context", False))
        # grad_norm=True: per-frame normalize the reversed gradient to a fixed
        # magnitude (decouples removal strength from discriminator confidence).
        self.grad_norm        = bool(getattr(cfg, "grl_grad_norm", False))
        self.grad_norm_target = float(getattr(cfg, "grl_grad_norm_target", 1.0))
        if self.attention_pool:
            self.attn = nn.Sequential(nn.Linear(P, P), nn.Tanh(), nn.Linear(P, 1))
        if self.attention_pool or self.stats_pool:
            self.fc   = nn.Linear(2 * P, cfg.num_speakers)        # [weighted mean ; weighted std]
        else:
            self.fc   = nn.Linear(P, cfg.num_speakers)
        if self.dense_context:
            k = int(getattr(cfg, "grl_context_kernel", 31))
            self.context_conv = nn.Conv1d(P, P, kernel_size=k, padding=k // 2)

    def forward(
        self,
        z_L: torch.Tensor,
        lengths: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """
        z_L     : (B, T, K)
        lengths : (B,)  valid frame counts
        lam     : GRL reversal strength

        Returns (B, T, num_speakers) if frame_level, else (B, num_speakers).
        """
        z_L = (gradient_reversal_norm(z_L, lam, self.grad_norm_target)
               if self.grad_norm else gradient_reversal(z_L, lam))
        # MLP (ReLU) — stronger than the linear SID probe it must beat, so z_L
        # must remove speaker in a way that survives a real probe.
        z_proj = F.relu(self.projector(z_L))                      # (B, T, P)
        if self.dense_context:
            # local temporal context, then per-frame speaker logits (B, T, num_speakers)
            z_ctx = F.relu(self.context_conv(z_proj.transpose(1, 2))).transpose(1, 2)
            return self.fc(z_ctx)
        if self.frame_level:
            return self.fc(z_proj)                                # (B, T, num_speakers)
        B, T, P = z_proj.shape
        mask = (torch.arange(T, device=z_L.device).unsqueeze(0) < lengths.unsqueeze(1)
                ).unsqueeze(-1)                                    # (B, T, 1) bool
        if self.attention_pool:
            a = self.attn(z_proj).masked_fill(~mask, -1e9)        # (B, T, 1)
            a = torch.softmax(a, dim=1)                           # attention weights over time
            mean = (z_proj * a).sum(1)                            # (B, P)
            var  = (a * (z_proj - mean.unsqueeze(1)) ** 2).sum(1)
            std  = (var + 1e-5).sqrt()                            # (B, P)
            return self.fc(torch.cat([mean, std], dim=-1))
        fmask  = mask.float()
        if self.stats_pool:
            n    = lengths.float().clamp(min=1).unsqueeze(1)
            mean = (z_proj * fmask).sum(1) / n
            var  = (((z_proj - mean.unsqueeze(1)) ** 2) * fmask).sum(1) / n
            std  = (var + 1e-5).sqrt()
            return self.fc(torch.cat([mean, std], dim=-1))
        z_mean = (z_proj * fmask).sum(1) / lengths.float().unsqueeze(1).clamp(min=1)
        return self.fc(z_mean)
