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
    This is the H(route) (marginal/balance) half of the MI objective: minimising
    -H pushes the GLOBAL bucket usage toward balanced (no route collapses).
    """
    p_mean  = F.softmax(routing_logits, dim=-1).reshape(-1, routing_logits.shape[-1]).mean(dim=0)
    entropy = -(p_mean * p_mean.log().clamp(min=-100)).sum()
    return -entropy


def routing_spec_loss(routing_logits: torch.Tensor) -> torch.Tensor:
    """Mean per-unit routing entropy — minimise to make each feature decisive.

    routing_logits : (K, n_routes)
    This is the H(route | feature) (conditional) half of the MI objective:
    minimising it pushes every feature off the uniform fence toward a single
    route.  Paired with route_loss it maximises MI(feature; route) =
    H(route) - H(route | feature): decisive units AND balanced buckets.
    """
    p = F.softmax(routing_logits, dim=-1)                       # (K, n_routes)
    unit_H = -(p * p.log().clamp(min=-100)).sum(dim=-1)         # (K,)
    return unit_H.mean()


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


def _masked_stats_pool(z: torch.Tensor, lengths: torch.Tensor, eps: float = 1e-5):
    """Per-utterance masked mean and std over valid frames.

    z       : (B, T, K)
    lengths : (B,)
    Returns (mean, std) each (B, K).
    """
    B, T, K = z.shape
    mask = (torch.arange(T, device=z.device).unsqueeze(0) < lengths.unsqueeze(1)
            ).float().unsqueeze(-1)
    n    = lengths.float().clamp(min=1).view(B, 1)
    mean = (z * mask).sum(1) / n
    var  = (((z - mean.unsqueeze(1)) ** 2) * mask).sum(1) / n
    std  = (var + eps).sqrt()
    return mean, std


def _bilinear_time_resample(z: torch.Tensor, lengths: torch.Tensor, target: int) -> torch.Tensor:
    """Resample each utterance's z (B, T, K) along the time axis to length `target`.

    Uses bilinear interp over only the valid prefix (per-utterance), so padding
    is excluded.  Returns (B, target, K).
    """
    B, T, K = z.shape
    out = z.new_zeros(B, target, K)
    for b in range(B):
        L = int(lengths[b].clamp(min=1).item())
        # (1, K, L) -> (1, K, target)
        zb = z[b, :L].transpose(0, 1).unsqueeze(0)                          # (1, K, L)
        rb = F.interpolate(zb.float(), size=target, mode="linear", align_corners=False)
        out[b] = rb.squeeze(0).transpose(0, 1).to(z.dtype)
    return out


def inv_L_frame_cosine_loss(
    z_L_a: torch.Tensor, lens_a: torch.Tensor,
    z_L_b: torch.Tensor, lens_b: torch.Tensor,
    target_frames: int = 200,
) -> torch.Tensor:
    """Frame-aligned cosine invariance for pair-alpha (same content, paralinguistic varies).

    Both inputs are resampled along time to `target_frames`, then 1 - cos
    per aligned position averaged over time and batch.
    """
    a = _bilinear_time_resample(z_L_a, lens_a, target_frames)              # (B, T*, K)
    b = _bilinear_time_resample(z_L_b, lens_b, target_frames)
    # Cosine per position; small eps to avoid div-by-zero when a frame is all zero.
    a_n = a / (a.float().norm(dim=-1, keepdim=True).clamp(min=1e-8))
    b_n = b / (b.float().norm(dim=-1, keepdim=True).clamp(min=1e-8))
    cos = (a_n * b_n).sum(dim=-1)                                          # (B, T*)
    return (1.0 - cos).mean()


def inv_P_stats_pool_loss(
    z_P_a: torch.Tensor, lens_a: torch.Tensor,
    z_P_b: torch.Tensor, lens_b: torch.Tensor,
) -> torch.Tensor:
    """Scale-normalised L2 between stats-pool(cat(mean,std)) for pair-beta.

    Matches the SID stats-pool probe family.  Same-speaker-same-session pairs
    differ in content; pooled paralinguistic statistics should match.
    """
    mu_a, sd_a = _masked_stats_pool(z_P_a, lens_a)
    mu_b, sd_b = _masked_stats_pool(z_P_b, lens_b)
    va = torch.cat([mu_a, sd_a], dim=-1)                                   # (B, 2K)
    vb = torch.cat([mu_b, sd_b], dim=-1)
    num = (va - vb).pow(2).sum(dim=-1)
    den = va.pow(2).sum(dim=-1).clamp(min=1e-8)
    return (num / den).mean()


def variance_floor_loss(z: torch.Tensor, lengths: torch.Tensor, gamma: float = 1.0,
                         weight: torch.Tensor | None = None) -> torch.Tensor:
    """VICReg-style per-dimension variance floor.

    Computes per-dim std across (B, T) valid frames, then penalises any dim
    with std < gamma.  Pushes z to maintain at least `gamma` std per dim,
    preventing collapse to a constant or low-rank subspace.

    z       : (B, T, K)  any view
    lengths : (B,)
    weight  : optional (K,) per-dim weight in [0, 1] — apply the floor more
              strongly to dims the routing has assigned to this view.  None =
              uniform.  Use the routing mask m_L (or m_P) to skip the dims
              the router pushed out of this view, otherwise hard routing
              makes ~half of dims mechanically std=0 and the loss flags it
              as collapse incorrectly.
    Returns scalar  weighted_mean_d max(0, gamma - std_d)^2.
    """
    B, T, K = z.shape
    mask = (torch.arange(T, device=z.device).unsqueeze(0) < lengths.unsqueeze(1)
            ).float().unsqueeze(-1)                                        # (B, T, 1)
    n_valid = mask.sum().clamp(min=1.0)
    z_flat = (z * mask).reshape(B * T, K)
    mean_d = z_flat.sum(0) / n_valid                                       # (K,)
    var_d  = ((z_flat - mean_d) ** 2 * mask.reshape(B * T, 1)).sum(0) / n_valid
    std_d  = (var_d + 1e-5).sqrt()
    term   = F.relu(gamma - std_d).pow(2)                                  # (K,)
    if weight is not None:
        w = weight.detach().to(term.dtype)
        if w.dim() > 1:
            w = w.reshape(-1, K).mean(0)                                   # collapse dynamic-routing batch dim
        return (w * term).sum() / w.sum().clamp(min=1e-6)
    return term.mean()


def bucket_diag(z: torch.Tensor, lengths: torch.Tensor, mask_dim: torch.Tensor | None,
                gamma: float = 1.0) -> dict:
    """Diagnostic: per-dim std stats restricted to bucket-assigned dims.

    Returns (p10_std, frac_below_gamma, utt_norm_mean, utt_norm_std) over the
    dims selected by `mask_dim` (bool, (K,)).  If mask_dim is None, uses all dims.
    `utt_norm` is the L2 norm of the per-utterance mean of z over the bucket dims —
    catches trivial-constant collapse (norm shrinks to ~0 across the batch).
    """
    B, T, K = z.shape
    fmask = (torch.arange(T, device=z.device).unsqueeze(0) < lengths.unsqueeze(1)
             ).float().unsqueeze(-1)                                       # (B, T, 1)
    n_valid = fmask.sum().clamp(min=1.0)
    z_flat = (z * fmask).reshape(B * T, K)
    mean_d = z_flat.sum(0) / n_valid
    var_d  = ((z_flat - mean_d) ** 2 * fmask.reshape(B * T, 1)).sum(0) / n_valid
    std_d  = (var_d + 1e-5).sqrt().float()
    if mask_dim is not None:
        m = mask_dim.bool()
        if m.dim() > 1:
            m = m.reshape(-1, K).any(0)
        if m.sum() > 0:
            std_d = std_d[m]
            z_sub = z[..., m]
        else:
            z_sub = z
    else:
        z_sub = z
    p10 = std_d.kthvalue(max(1, int(0.1 * std_d.numel())))[0].item() if std_d.numel() else float('nan')
    frac = float((std_d < gamma).float().mean().item()) if std_d.numel() else float('nan')
    # Per-utterance pooled L2 norm over the bucket dims
    nlen = lengths.float().clamp(min=1).view(B, 1)
    z_utt = (z_sub * fmask).sum(1) / nlen                                  # (B, K_sub)
    utt_norm = z_utt.float().norm(dim=-1)                                  # (B,)
    return {
        "p10_std":       p10,
        "frac_blw_g":    frac,
        "utt_norm_mean": float(utt_norm.mean().item()),
        "utt_norm_std":  float(utt_norm.std(unbiased=False).item()),
        "k_active":      int(std_d.numel()),
    }


def effective_rank(z: torch.Tensor, lengths: torch.Tensor, max_frames: int = 2000
                    ) -> float:
    """exp(H(p)) where p = softmax of normalised singular values.  Diagnostic only.

    Returns a Python float; do NOT use as a training loss (uses SVD).
    """
    B, T, K = z.shape
    mask = (torch.arange(T, device=z.device).unsqueeze(0) < lengths.unsqueeze(1)
            ).reshape(B * T)
    z_flat = z.reshape(B * T, K)[mask]
    if z_flat.shape[0] > max_frames:
        idx = torch.randperm(z_flat.shape[0], device=z.device)[:max_frames]
        z_flat = z_flat[idx]
    if z_flat.shape[0] < 2:
        return 0.0
    try:
        s = torch.linalg.svdvals(z_flat.float())
    except Exception:
        return 0.0
    p = s / s.sum().clamp(min=1e-12)
    H = -(p * p.clamp(min=1e-12).log()).sum()
    return float(torch.exp(H).item())


def ub_loss(m_L: torch.Tensor, m_P: torch.Tensor) -> torch.Tensor:
    """Exp 4 — U-bucket information bottleneck.

    Penalises the total soft feature count claimed by L and P.  Minimising this
    forces the routing to assign only as many features to L and P as the task
    losses justify, with the remainder naturally falling to U.

    m_L, m_P : (K,)  Gumbel-softmax masks, values in [0, 1]
    Returns a scalar in [0, 2] (sum of mean soft assignments to L and P).
    """
    return m_L.mean() + m_P.mean()
