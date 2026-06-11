"""Loss functions for the disentanglement system."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def recon_loss(
    h_t: torch.Tensor,
    h_hat: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Masked MSE over valid frames.

    h_t, h_hat : (B, T, D)
    lengths    : (B,)
    """
    B, T, _ = h_t.shape
    mask = (torch.arange(T, device=h_t.device).unsqueeze(0) < lengths.unsqueeze(1)).float()
    mse  = (h_t - h_hat).pow(2).mean(-1)
    return (mse * mask).sum() / mask.sum().clamp(min=1)


def ctc_pr_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
) -> torch.Tensor:
    """CTC loss for frame-level phoneme recognition.

    logits         : (B, T, vocab_size)
    targets        : (B, P_max)  padded phone ids
    input_lengths  : (B,)
    target_lengths : (B,)
    """
    log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)   # (T, B, V)
    return F.ctc_loss(
        log_probs, targets, input_lengths, target_lengths,
        blank=0, reduction="mean", zero_infinity=True,
    )


def sid_ce_loss(
    logits: torch.Tensor,
    speaker_ids: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy speaker classification loss.

    logits      : (B, num_speakers)
    speaker_ids : (B,)
    """
    return F.cross_entropy(logits, speaker_ids)


def sid_ce_loss_frames(
    logits: torch.Tensor,
    speaker_ids: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Frame-level speaker CE (for the frame-level GRL adversary).

    logits      : (B, T, num_speakers)  — a speaker prediction per frame
    speaker_ids : (B,)                  — one label per utterance, broadcast to frames
    lengths     : (B,)                  — valid frame counts (padding masked out)
    """
    B, T, S = logits.shape
    mask = (torch.arange(T, device=logits.device).unsqueeze(0) < lengths.unsqueeze(1)).float()
    tgt  = speaker_ids.unsqueeze(1).expand(B, T).reshape(B * T)
    ce   = F.cross_entropy(logits.reshape(B * T, S), tgt, reduction="none").reshape(B, T)
    return (ce * mask).sum() / mask.sum().clamp(min=1)


def route_loss(routing_logits: torch.Tensor) -> torch.Tensor:
    """Negative entropy of mean soft-routing distribution — minimise to prevent collapse.

    routing_logits : (K, n_routes)
    """
    p_mean  = F.softmax(routing_logits, dim=-1).mean(dim=0)
    entropy = -(p_mean * p_mean.log().clamp(min=-100)).sum()
    return -entropy


def decor_loss(z_t: torch.Tensor, lengths: torch.Tensor,
               max_frames: int = 1000) -> torch.Tensor:
    """Full K×K Barlow-Twins-style off-diagonal correlation penalty at frame level.

    Each frame has exactly topk non-zero SAE features. We compute the K×K
    cross-correlation matrix across frames and penalise all off-diagonal entries,
    encouraging every pair of features to be uncorrelated.

    Uses random frame subsampling to bound the (N, K) matmul to O(max_frames × K²).

    z_t        : (B, T, K)  TopK-sparse SAE latent
    lengths    : (B,)       valid frame counts
    max_frames : cap on frames fed to the covariance matrix (memory guard)
    """
    B, T, K = z_t.shape
    mask   = (torch.arange(T, device=z_t.device).unsqueeze(0) < lengths.unsqueeze(1))

    z_flat = z_t.reshape(B * T, K)[mask.reshape(B * T)]   # (N_valid, K)
    N = z_flat.shape[0]

    if N > max_frames:
        idx    = torch.randperm(N, device=z_flat.device)[:max_frames]
        z_flat = z_flat[idx]
        N      = max_frames

    # Z-score each feature column across the sampled frames
    z_norm = (z_flat - z_flat.mean(0, keepdim=True)) / (z_flat.std(0, keepdim=True) + 1e-8)

    # Full K×K cross-correlation matrix
    corr = (z_norm.T @ z_norm) / N          # (K, K),  diagonal ≈ 1

    # Mean squared off-diagonal entries
    eye  = torch.eye(K, device=corr.device)
    return (corr * (1 - eye)).pow(2).mean()


def ub_loss(m_L: torch.Tensor, m_P: torch.Tensor) -> torch.Tensor:
    """Exp 4 — U-bucket information bottleneck.

    Penalises the total soft feature count claimed by L and P.  Minimising this
    forces the routing to assign only as many features to L and P as the task
    losses justify, with the remainder naturally falling to U.

    m_L, m_P : (K,)  Gumbel-softmax masks, values in [0, 1]
    Returns a scalar in [0, 2] (sum of mean soft assignments to L and P).
    """
    return m_L.mean() + m_P.mean()
