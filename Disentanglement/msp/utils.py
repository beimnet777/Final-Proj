"""Self-contained helpers for the MSP pipeline.

These are copied (not imported) from the legacy train.py so this folder does not
depend on or perturb the existing training code.  Loss primitives (recon/CTC/SID)
are imported from the shared losses.py, which is a clean library module.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from losses import sid_ce_loss, sid_ce_loss_frames


# ---------------------------------------------------------------- CTC decode / PER
def greedy_ctc_decode(logits, lengths, blank_id: int = 0):
    preds = logits.argmax(dim=-1)
    out = []
    for i, n in enumerate(lengths.tolist()):
        ids, prev = [], -1
        for tok in preds[i, :n].tolist():
            if tok != prev:
                ids.append(tok); prev = tok
        out.append([t for t in ids if t != blank_id])
    return out


def edit_distance(a, b) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


@torch.no_grad()
def ctc_errors(logits, targets, input_lengths, target_lengths):
    """(edit-distance, ref-length) over a batch via greedy CTC decode."""
    preds = greedy_ctc_decode(logits, input_lengths)
    num = den = 0
    for i, pred_ids in enumerate(preds):
        ref = targets[i, :target_lengths[i]].tolist()
        num += edit_distance(pred_ids, ref)
        den += len(ref)
    return num, den


# ---------------------------------------------------------------- adversary readouts
def speaker_adv_loss(logits, speaker_ids, lengths) -> torch.Tensor:
    if isinstance(logits, (tuple, list)):
        return torch.stack([speaker_adv_loss(b, speaker_ids, lengths) for b in logits]).mean()
    if logits.dim() == 3:
        return sid_ce_loss_frames(logits, speaker_ids, lengths)
    return sid_ce_loss(logits, speaker_ids)


@torch.no_grad()
def speaker_correct(logits, speaker_ids, lengths):
    if isinstance(logits, (tuple, list)):
        cts = [speaker_correct(b, speaker_ids, lengths) for b in logits]
        return max(cts, key=lambda ct: ct[0] / max(ct[1], 1))
    if logits.dim() == 3:
        B, T, _ = logits.shape
        pred = logits.argmax(dim=-1)
        mask = (torch.arange(T, device=logits.device).unsqueeze(0) < lengths.unsqueeze(1))
        tgt = speaker_ids.unsqueeze(1).expand(B, T)
        return int(((pred == tgt) & mask).sum().item()), int(mask.sum().item())
    pred = logits.argmax(dim=-1)
    return int((pred == speaker_ids).sum().item()), int(speaker_ids.numel())


@torch.no_grad()
def class_correct(logits, labels):
    pred = logits.argmax(dim=-1)
    return int((pred == labels).sum().item()), int(labels.numel())


# ---------------------------------------------------------------- prosody / invariance
def _masked_mse(a, b, lengths) -> torch.Tensor:
    T = a.shape[1]
    mask = (torch.arange(T, device=a.device).unsqueeze(0) < lengths.unsqueeze(1)
            ).float().unsqueeze(-1)
    return (((a - b) ** 2) * mask).sum() / mask.sum().clamp(min=1) / a.shape[-1]


def invariance_loss(zL, zLp, lengths) -> torch.Tensor:
    """Scale-normalised per-frame fraction of z_L energy that changes under the
    speaker perturbation (0=invariant, 1=orthogonal).  Same form as legacy."""
    T = min(zL.shape[1], zLp.shape[1])
    zL, zLp = zL[:, :T], zLp[:, :T]
    diff = (zL - zLp).pow(2).sum(-1)
    den = 0.5 * (zL.pow(2).sum(-1) + zLp.pow(2).sum(-1)) + 1e-6
    r = diff / den
    mask = (torch.arange(T, device=zL.device).unsqueeze(0) < lengths.unsqueeze(1)).float()
    return (r * mask).sum() / mask.sum().clamp(min=1)


def _interp1d(x, T):
    L = x.shape[0]
    if L == T:
        return x
    if L < 2:
        return x.new_full((T,), float(x.mean()) if L else 0.0)
    return F.interpolate(x.view(1, 1, L), size=T, mode="linear", align_corners=True).view(T)


@torch.no_grad()
def prosody_targets_fast(audios, audio_lengths, out_lengths, sr: int = 16_000):
    """Per-frame (logF0 on voiced, voiced mask, log-RMS energy) aligned to the SAE
    frame grid.  F0 via torchaudio NCCF, computed on the fly in fp32."""
    import torchaudio.functional as AF
    B = audios.shape[0]
    Tmax = int(out_lengths.max().item()) if B else 0
    dev = audios.device
    f0o = torch.zeros(B, Tmax, device=dev)
    vo = torch.zeros(B, Tmax, device=dev)
    eo = torch.zeros(B, Tmax, device=dev)
    frame, hop = 400, 160
    ac_ctx = torch.autocast("cuda" if audios.is_cuda else "cpu", enabled=False)
    aud = audios.float()
    with ac_ctx:
        for i in range(B):
            n = int(audio_lengths[i].item())
            Ti = int(out_lengths[i].item())
            if Ti <= 0 or n < frame:
                continue
            w = aud[i, :n]
            loge = w.unfold(0, frame, hop).pow(2).mean(-1).clamp_min(1e-8).log()
            # Per-utterance mean-centering of log-energy: the raw contour sits ~-10
            # (recording gain / mic distance — pure nuisance), so an init head
            # predicting ~0 gives a huge MSE (the step-1 spike).  Subtracting the
            # mean removes that offset while KEEPING the dynamic range (emphasis /
            # intensity is prosodic).  We deliberately do NOT divide by std — it
            # would discard that range and amplify noise on quiet segments.  F0
            # stays raw so z_P also carries absolute pitch as a speaker cue.
            loge = loge - loge.mean()
            try:
                f0 = AF.detect_pitch_frequency(
                    w.unsqueeze(0), sr, frame_time=hop / sr,
                    win_length=30, freq_low=65, freq_high=400).squeeze(0)
            except Exception:
                f0 = torch.zeros_like(loge)
            voiced = ((f0 >= 65.0) & (f0 <= 400.0)).float()
            logf0 = torch.where(voiced.bool(), f0.clamp_min(1.0).log(), torch.zeros_like(f0))
            f0o[i, :Ti] = _interp1d(logf0, Ti)
            vo[i, :Ti] = (_interp1d(voiced, Ti) > 0.5).float()
            eo[i, :Ti] = _interp1d(loge, Ti)
    return f0o, vo, eo


def prosody_train_loss(pred, f0, voiced, energy, lengths) -> torch.Tensor:
    """Masked MSE: F0 on voiced frames + energy on all valid frames (pred (B,T,2))."""
    T = min(pred.shape[1], f0.shape[1])
    pred = pred[:, :T]
    valid = (torch.arange(T, device=pred.device).unsqueeze(0) < lengths.unsqueeze(1)).float()
    vmask = voiced[:, :T] * valid
    f0e = ((pred[..., 0] - f0[:, :T]) ** 2 * vmask).sum() / vmask.sum().clamp(min=1)
    ee = ((pred[..., 1] - energy[:, :T]) ** 2 * valid).sum() / valid.sum().clamp(min=1)
    return f0e + ee


# ---------------------------------------------------------------- emotion: weights + UAR
def emotion_class_weights(rows, n_classes: int, device) -> torch.Tensor:
    """Inverse-frequency CE weights from the train manifest rows (normalised to
    mean 1), so the neutral-heavy MSP distribution doesn't swamp rarer emotions."""
    counts = torch.zeros(n_classes)
    for r in rows:
        counts[int(r["emotion_idx"])] += 1
    counts = counts.clamp(min=1)
    w = counts.sum() / (n_classes * counts)
    return (w / w.mean()).to(device)


@torch.no_grad()
def uar_from_confusion(conf: torch.Tensor) -> float:
    """Unweighted Average Recall (macro recall) from a (C,C) confusion matrix
    rows=true, cols=pred.  The imbalance-robust SER metric."""
    per_class = conf.diag() / conf.sum(dim=1).clamp(min=1)
    present = conf.sum(dim=1) > 0
    return float(per_class[present].mean()) if present.any() else 0.0
