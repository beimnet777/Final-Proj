"""All loss functions for the disentanglement system.

All functions are pure (no state) and return a scalar tensor.

Total objective (stage 2):
    L = L_recon + α·L_PR + β·L_SID + δ·L_decorr + ρ·L_route + L_GRL
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- reconstruction

def recon_loss(
    h_t: torch.Tensor,
    h_hat: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error over valid frames.

    h_t, h_hat : (B, T, D)
    lengths     : (B,)  valid frame counts
    """
    B, T, _ = h_t.shape
    mask = (
        torch.arange(T, device=h_t.device).unsqueeze(0) < lengths.unsqueeze(1)
    ).float()                                           # (B, T)

    mse = (h_t - h_hat).pow(2).mean(-1)                # (B, T)
    return (mse * mask).sum() / mask.sum().clamp(min=1)


# ---------------------------------------------------------------- phoneme CTC

def ctc_pr_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
) -> torch.Tensor:
    """CTC loss for frame-level phoneme recognition.

    logits         : (B, T, vocab_size)  raw (pre-log-softmax)
    targets        : (B, P_max)          padded phone id sequences
    input_lengths  : (B,)                valid frame counts
    target_lengths : (B,)                valid phone sequence lengths
    """
    log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)   # (T, B, V)
    return F.ctc_loss(
        log_probs, targets, input_lengths, target_lengths,
        blank=0, reduction="mean", zero_infinity=True,
    )


# ---------------------------------------------------------------- speaker CE

def sid_ce_loss(
    logits: torch.Tensor,
    speaker_ids: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy speaker classification loss.

    logits      : (B, num_speakers)
    speaker_ids : (B,)
    """
    return F.cross_entropy(logits, speaker_ids)


# ---------------------------------------------------------------- decorrelation

def decorr_loss(
    z_t: torch.Tensor,
    lengths: torch.Tensor,
    max_frames: int = 1000,
) -> torch.Tensor:
    """Barlow Twins redundancy-reduction loss on the SAE latent z.

    L_BT = Σ_i (1-C_ii)² + Σ_{i≠j} C_ij²

    where C_ij = (Σ_b z_{b,i}·z_{b,j}) / (‖z_i‖·‖z_j‖)  (column-L2 normalised)

    In the single-view case (z^A = z^B = z) the diagonal satisfies C_ii = 1
    exactly, so the first term is always 0 and only the off-diagonal
    redundancy penalty is active.  The overall weight is controlled by cfg.delta
    in the training loop; λ=1 is kept here for a clean interface.

    z_t     : (B, T, K)
    lengths : (B,)  valid frame counts
    max_frames : subsample frames to bound memory for the (K×K) matrix.
                 For K=5120 that is ~100 MB fp32 — keep this ≤ 1000.
    """
    B, T, K = z_t.shape
    mask = (
        torch.arange(T, device=z_t.device).unsqueeze(0) < lengths.unsqueeze(1)
    )                                                     # (B, T)

    z_flat = z_t[mask]                                    # (N, K)  valid frames
    N = z_flat.size(0)

    if N > max_frames:
        idx = torch.randperm(N, device=z_flat.device)[:max_frames]
        z_flat = z_flat[idx]

    # L2-normalise each feature column across the batch dimension.
    # clamp avoids div-by-zero on always-zero columns (common with TopK sparse
    # vectors); the backward of x/clamp(x,min=eps) is numerically stable.
    col_norms = z_flat.norm(dim=0, keepdim=True).clamp(min=1e-6)  # (1, K)
    z_n = z_flat / col_norms                              # (N, K) — unit-L2 columns

    # Cross-correlation matrix  C_ij = Σ_b z_n_{b,i} · z_n_{b,j}
    C = z_n.T @ z_n                                       # (K, K)

    # Barlow Twins objective (diagonal = 1 trivially in single-view, so
    # on_diag = 0 but included for completeness / two-view extension)
    on_diag  = (C.diagonal() - 1.0).pow(2).sum()
    off_diag = C.pow(2).sum() - C.diagonal().pow(2).sum()
    return on_diag + off_diag


# ---------------------------------------------------------------- routing anti-collapse

def route_loss(routing_logits: torch.Tensor) -> torch.Tensor:
    """Negative entropy of the mean soft-routing distribution.

    routing_logits : (K, 3)

    Minimising this loss maximises the entropy of the average routing
    probability over all K units, preventing all units from collapsing to
    a single group.

    Loss range: [ -log(3), 0 ]  →  low (= good, uniform) … high (= bad, collapsed)
    """
    p_soft = F.softmax(routing_logits, dim=-1)          # (K, 3)
    p_mean = p_soft.mean(dim=0)                         # (3,)  average over units
    entropy = -(p_mean * p_mean.log().clamp(min=-100)).sum()
    return -entropy   # minimise → maximise entropy
