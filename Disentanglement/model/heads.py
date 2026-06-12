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

    def forward(self, z_P: torch.Tensor, lam: float) -> torch.Tensor:
        z_P = gradient_reversal(z_P, lam)
        # MLP (ReLU) — stronger than the linear PR probe it must beat.
        return self.fc(F.relu(self.projector(z_P)))


# ---------------------------------------------------------------- GRL head

class GRLHead(nn.Module):
    """Adversarial speaker head on z_L with gradient reversal.

    Mirrors the diagnostic SID probe style: frame projection, masked mean pool,
    then speaker classifier.  GRL reverses gradients so the model is penalised
    for encoding speaker in L features.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.projector = nn.Linear(_head_input_dim(cfg), 256)
        self.fc = nn.Linear(256, cfg.num_speakers)
        # frame_level=True: predict speaker at every frame (dense gradient to z_L,
        # like the frame-level PR-GRL).  False: utterance mean-pool then classify
        # (gradient diluted ~1/T per frame).
        self.frame_level = bool(getattr(cfg, "grl_frame_level", False))

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
        z_L = gradient_reversal(z_L, lam)
        # MLP (ReLU before pooling) — stronger than the linear SID probe it must
        # beat, so z_L must remove speaker in a way that survives a real probe.
        if self.frame_level:
            return self.fc(F.relu(self.projector(z_L)))           # (B, T, num_speakers)
        B, T, K = z_L.shape
        mask   = (torch.arange(T, device=z_L.device).unsqueeze(0) < lengths.unsqueeze(1)
                  ).float().unsqueeze(-1)                          # (B, T, 1)
        z_proj = F.relu(self.projector(z_L))
        z_mean = (z_proj * mask).sum(1) / lengths.float().unsqueeze(1).clamp(min=1)
        return self.fc(z_mean)
