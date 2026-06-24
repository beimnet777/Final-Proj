#!/usr/bin/env python3
"""Post-hoc probing for disentanglement analysis (experiments A / B / C).

For each checkpoint, trains SUPERB-style probes on every source representation
and evaluates cross-task leakage.

Sources probed
--------------
  h_t   raw SPEAR features (D=1280)        — upper-bound baseline
  z_t   full SAE latent (K=5120)           — SAE baseline (no routing)
  z_L   linguistic route (K=5120)          — should encode phones, not speakers
  z_P   paralinguistic route (K=5120)      — should encode speakers, not phones

Tasks
-----
  PR    CTC phoneme recognition  → metric: PER
  SID   speaker classification   → metric: accuracy

Cross-leakage cells:
  z_L → SID  : speaker info leaking into linguistic bucket
  z_P → PR   : phoneme info leaking into speaker bucket

Usage
-----
  # Mode B — baselines only (no stage-2 checkpoint)
  python diag_probe/probe_runner.py --stage1_ckpt checkpoints/best.pt --run_name probe_B

  # Mode A — probe Run 2 checkpoint
  python diag_probe/probe_runner.py --stage1_ckpt checkpoints/best.pt \\
      --stage2_ckpt checkpoints/sid1_weakgrl/stage2_best.pt --run_name probe_A

  # Mode C — probe Run 3 checkpoint
  python diag_probe/probe_runner.py --stage1_ckpt checkpoints/best.pt \\
      --stage2_ckpt checkpoints/sid1_nogrl/stage2_best.pt --run_name probe_C
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

try:
    import jiwer
except ImportError:
    jiwer = None

try:
    import librosa
except ImportError:
    librosa = None

DIS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DIS_DIR.parent


def _prioritize_import_paths() -> None:
    """Keep Disentanglement imports ahead of Probing's top-level model.py."""
    dis_path = str(DIS_DIR)
    pr_path = str(REPO_ROOT / "Probing" / "pr")
    for path in (dis_path, pr_path):
        while path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, dis_path)
    sys.path.insert(1, pr_path)


_prioritize_import_paths()

from config import DISConfig
from pr_config import PRConfig

_pr_data = None


# ---------------------------------------------------------------- probe heads

class _SIDProbe(nn.Module):
    """SUPERB-style SID probe: frame projection -> masked mean -> linear."""

    def __init__(self, in_dim: int, num_speakers: int, proj_dim: int = 256) -> None:
        super().__init__()
        self.projector = nn.Linear(in_dim, proj_dim)
        self.linear = nn.Linear(proj_dim, num_speakers)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.projector(x)
        x = _mean_pool(x, lengths)
        return self.linear(x)


class _SIDProbeStats(nn.Module):
    """Pooling-robust SID probe: projection -> ReLU -> masked mean+std pool -> linear.

    The plain _SIDProbe (linear -> mean-pool -> linear) is structurally blind to
    instance-normalized features: a linear map commutes with the time-mean, and
    IN forces mean_t(z) = 0, so every utterance pools to the same constant.
    Here the ReLU before pooling breaks that commutation, and the std half of
    the x-vector-style statistics pooling reads second moments directly — so
    speaker info in higher-order/temporal structure remains visible.
    """

    def __init__(self, in_dim: int, num_speakers: int, proj_dim: int = 256) -> None:
        super().__init__()
        self.projector = nn.Linear(in_dim, proj_dim)
        self.linear = nn.Linear(2 * proj_dim, num_speakers)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.projector(x))                       # (B, T, P)
        B, T, _ = x.shape
        mask = (torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
                ).float().unsqueeze(-1)                          # (B, T, 1)
        n    = lengths.float().clamp(min=1).unsqueeze(-1)        # (B, 1)
        mean = (x * mask).sum(1) / n                             # (B, P)
        var  = (((x - mean.unsqueeze(1)) ** 2) * mask).sum(1) / n
        std  = (var + 1e-5).sqrt()                               # (B, P)
        return self.linear(torch.cat([mean, std], dim=-1))


class _SIDProbeMLP(nn.Module):
    """SUPERB-style SID with one ReLU non-linearity between projection and classifier.

    Layout: projector(linear) -> ReLU -> masked mean-pool -> linear.
    Compared to _SIDProbe: same pooling, adds a non-linearity (so the probe
    can read interactions a pure-linear head misses).
    Compared to _SIDProbeStats: no std pool, smaller (P-dim, not 2P-dim) head.
    """

    def __init__(self, in_dim: int, num_speakers: int, proj_dim: int = 256) -> None:
        super().__init__()
        self.projector = nn.Linear(in_dim, proj_dim)
        self.linear = nn.Linear(proj_dim, num_speakers)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.projector(x))
        x = _mean_pool(x, lengths)
        return self.linear(x)


class _PRProbe(nn.Module):
    """SUPERB-style PR CTC probe: frame projection -> linear -> log-softmax."""

    def __init__(self, in_dim: int, vocab_size: int, proj_dim: int = 256) -> None:
        super().__init__()
        self.projector = nn.Linear(in_dim, proj_dim)
        self.linear = nn.Linear(proj_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projector(x)
        x = self.linear(x)
        return F.log_softmax(x, dim=-1)


class _PRProbeMLP(nn.Module):
    """PR CTC probe with one ReLU non-linearity between projection and classifier."""

    def __init__(self, in_dim: int, vocab_size: int, proj_dim: int = 256) -> None:
        super().__init__()
        self.projector = nn.Linear(in_dim, proj_dim)
        self.linear = nn.Linear(proj_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.projector(x))
        x = self.linear(x)
        return F.log_softmax(x, dim=-1)


class _ProsodyProbe(nn.Module):
    """Per-frame prosody regressor: projection -> ReLU -> linear -> [logF0, logE].

    No pooling — prosody is a frame-level (suprasegmental) signal, so the probe
    predicts the F0 and energy contour at every frame.  Reported metric is the
    Pearson correlation between predicted and true contour (scale/offset-free),
    i.e. how much of the prosody contour is linearly recoverable from the bucket.
    """

    def __init__(self, in_dim: int, proj_dim: int = 256) -> None:
        super().__init__()
        self.projector = nn.Linear(in_dim, proj_dim)
        self.head = nn.Linear(proj_dim, 2)            # [log-F0, log-energy]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.projector(x)))   # (B, T, 2)


# ---------------------------------------------------------------- prosody targets

def _prosody_targets(audios, audio_lens, out_lengths, sr: int = 16_000):
    """Per-frame prosody targets aligned to the SAE frame count (out_lengths).

    Returns (f0, voiced, energy), each (B, Tmax):
      f0     : log-F0 (raw, NOT speaker-normalized — absolute pitch is shared
               signal that serves both SID and prosody), 0 where unvoiced.
      voiced : 1.0 on voiced frames (pyin found a pitch), else 0.0 — the F0
               loss/metric is masked to these.
      energy : log frame-RMS, defined on every valid frame.
    F0 via librosa.pyin, energy via frame RMS; both contours linearly resampled
    to each utterance's T = out_lengths[i]."""
    if librosa is None:
        raise RuntimeError("librosa is required for prosody probing (pip install librosa).")
    B = audios.shape[0]
    Tmax = int(out_lengths.max().item()) if B else 0
    f0o = torch.zeros(B, Tmax)
    vo  = torch.zeros(B, Tmax)
    eo  = torch.zeros(B, Tmax)
    a = audios.detach().cpu().numpy().astype(np.float64)
    hop, frame = 256, 1024
    for i in range(B):
        n = int(audio_lens[i].item())
        T = int(out_lengths[i].item())
        if T <= 0 or n < frame:
            continue
        wav = a[i, :n]
        try:
            f0, _vflag, _vprob = librosa.pyin(
                wav, fmin=65.0, fmax=400.0, sr=sr, frame_length=frame, hop_length=hop)
        except Exception:
            continue
        rms = librosa.feature.rms(y=wav, frame_length=frame, hop_length=hop)[0]
        voiced = (~np.isnan(f0)).astype(np.float64)
        logf0  = np.log(np.where(np.isnan(f0), 1.0, f0))
        loge   = np.log(rms + 1e-8)
        # linear-resample each contour to T frames
        xt = np.linspace(0.0, 1.0, T)
        f0_t = np.interp(xt, np.linspace(0.0, 1.0, len(logf0)), logf0)
        v_t  = (np.interp(xt, np.linspace(0.0, 1.0, len(voiced)), voiced) > 0.5).astype(np.float64)
        e_t  = np.interp(xt, np.linspace(0.0, 1.0, len(loge)), loge)
        f0o[i, :T] = torch.from_numpy(f0_t * v_t)       # zero logF0 on unvoiced frames
        vo[i, :T]  = torch.from_numpy(v_t)
        eo[i, :T]  = torch.from_numpy(e_t)
    return f0o, vo, eo


def _frame_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
    return (torch.arange(T, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)).float()


class _PearsonAcc:
    """Streaming Pearson-correlation accumulator over masked frames."""

    def __init__(self) -> None:
        self.n = self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0

    def update(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> None:
        m = mask.bool()
        x, y = x[m].double(), y[m].double()
        self.n   += x.numel()
        self.sx  += x.sum().item();  self.sy  += y.sum().item()
        self.sxx += (x * x).sum().item();  self.syy += (y * y).sum().item()
        self.sxy += (x * y).sum().item()

    def value(self) -> float:
        if self.n < 2:
            return float("nan")
        cov = self.sxy - self.sx * self.sy / self.n
        vx  = self.sxx - self.sx * self.sx / self.n
        vy  = self.syy - self.sy * self.sy / self.n
        denom = (vx * vy) ** 0.5
        return float(cov / denom) if denom > 1e-12 else float("nan")


# ---------------------------------------------------------------- helpers

def _mean_pool(z: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    B, T, _ = z.shape
    mask  = (torch.arange(T, device=z.device).unsqueeze(0) < lengths.unsqueeze(1)).float()
    return (z * mask.unsqueeze(-1)).sum(1) / lengths.float().clamp(min=1).unsqueeze(-1)


def _make_linear_schedule(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)
    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def _extract_representations(model, audios, audio_lengths, device, use_bf16, has_routing: bool):
    audios         = audios.to(device)
    audio_lengths  = audio_lengths.to(device)

    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cuda", enabled=False)
    with ctx:
        if has_routing:
            out = model(audios, audio_lengths, stage=2, grl_lambda=0.0)
        else:
            out = model(audios, audio_lengths, stage=1)

    return {
        "h_t":            out["h_t"].float(),
        "z_t":            out["z_t"].float(),
        "z_L":            out.get("z_L", out["z_t"]).float(),
        "z_P":            out.get("z_P", out["z_t"]).float(),
        "z_U":            out.get("z_U", out["z_t"]).float(),
        "out_lengths":    out["out_lengths"],
    }


def _safe_refs(refs: List[str]) -> List[str]:
    return [r if r else "SPN" for r in refs]


def _edit_distance(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _phone_error_rate(refs: List[str], hyps: List[str]) -> float:
    refs = _safe_refs(refs)
    if jiwer is not None:
        return float(jiwer.wer(refs, hyps))

    edits = total = 0
    for ref, hyp in zip(refs, hyps):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        edits += _edit_distance(hyp_tokens, ref_tokens)
        total += len(ref_tokens)
    return edits / max(total, 1)


def _phones_from_text(text: str, tokenizer) -> str:
    if _pr_data._LEXICON is None:
        raise RuntimeError("PR lexicon is not loaded. Call make_pr_dataloaders() first.")
    phones = _pr_data.text_to_phones(text, _pr_data._LEXICON)
    ids = tokenizer.encode(phones).tolist()
    return tokenizer.decode(ids)


# ---------------------------------------------------------------- probe training / eval

def _train_pr_probe(
    probe: nn.Module,
    src_key: str,
    train_dl,
    model,
    device,
    use_bf16: bool,
    has_routing: bool,
    steps: int,
    lr: float,
    warmup_steps: int,
    grad_clip: float,
    val_cache=None,
    tokenizer=None,
    val_every: int = 0,
    patience: int = 0,
):
    """Train the PR probe. If val_cache + val_every>0, evaluate PER on the cached
    dev set every val_every steps, keep the best-dev probe state, early-stop after
    `patience` evals without improvement, and restore the best-dev weights.
    Returns best dev PER (float) when validating, else None."""
    trainable = [p for p in probe.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=lr, weight_decay=1e-4)
    scheduler = _make_linear_schedule(opt, warmup_steps, steps)
    ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    probe.train()
    step  = 0
    model.eval()

    do_val = val_cache is not None and val_every > 0
    best_val: float = float("inf")
    best_state = None
    bad = 0
    stop = False

    while step < steps and not stop:
        for audios, audio_lens, targets, target_lens, _texts in train_dl:
            feats = _extract_representations(
                model, audios, audio_lens, device, use_bf16, has_routing
            )
            z     = feats[src_key]          # (B, T, dim)
            lens  = feats["out_lengths"]
            targets = targets.to(device, non_blocking=True)
            target_lens = target_lens.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            log_probs = probe(z)
            loss = ctc_loss(log_probs.permute(1, 0, 2), targets, lens, target_lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            opt.step()
            scheduler.step()
            step += 1

            if do_val and step % val_every == 0:
                per = _eval_pr_probe_cached(probe, val_cache, tokenizer, device)
                probe.train()
                if per < best_val - 1e-4:
                    best_val, bad = per, 0
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in probe.state_dict().items()}
                else:
                    bad += 1
                print(f"      [pr {src_key}] step {step}/{steps}  dev PER={per:.3f}  "
                      f"(best {best_val:.3f}, bad {bad}/{patience})", flush=True)
                if patience > 0 and bad >= patience:
                    stop = True
                    break
            if step >= steps:
                break

    if best_state is not None:
        probe.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return best_val if do_val else None


def _train_sid_probe(
    probe: nn.Module,
    src_key: str,
    train_dl,
    model,
    device,
    use_bf16: bool,
    has_routing: bool,
    steps: int,
    lr: float,
    warmup_steps: int,
    grad_clip: float,
    val_cache=None,
    val_every: int = 0,
    patience: int = 0,
):
    """Train the SID probe. If val_cache + val_every>0, evaluate accuracy on the
    cached dev set every val_every steps, keep the best-dev probe state, early-stop
    after `patience` evals without improvement, and restore the best-dev weights.
    Returns best dev accuracy (float) when validating, else None."""
    trainable = [p for p in probe.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=lr, weight_decay=1e-4)
    scheduler = _make_linear_schedule(opt, warmup_steps, steps)
    ce_loss = nn.CrossEntropyLoss()
    probe.train()
    step = 0
    model.eval()

    do_val = val_cache is not None and val_every > 0
    best_val: float = -1.0
    best_state = None
    bad = 0
    stop = False

    while step < steps and not stop:
        for audios, audio_lens, _targets, _target_lens, speaker_ids in train_dl:
            feats = _extract_representations(
                model, audios, audio_lens, device, use_bf16, has_routing
            )
            z = feats[src_key]
            lens = feats["out_lengths"]
            speaker_ids = speaker_ids.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            loss = ce_loss(probe(z, lens), speaker_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            opt.step()
            scheduler.step()
            step += 1

            if do_val and step % val_every == 0:
                acc = _eval_sid_probe_cached(probe, val_cache, device)
                probe.train()
                if acc > best_val + 1e-4:
                    best_val, bad = acc, 0
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in probe.state_dict().items()}
                else:
                    bad += 1
                print(f"      [sid {src_key}] step {step}/{steps}  dev acc={acc:.3f}  "
                      f"(best {best_val:.3f}, bad {bad}/{patience})", flush=True)
                if patience > 0 and bad >= patience:
                    stop = True
                    break
            if step >= steps:
                break

    if best_state is not None:
        probe.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return best_val if do_val else None


@torch.no_grad()
def _build_prosody_pool(dl, model, device, use_bf16, has_routing, max_examples: int):
    """One pass over `dl`: cache audio + per-frame prosody targets on CPU so the
    slow pyin F0 extraction runs ONCE (not every probe step).  The src feature
    (z) is re-extracted per step from the cached audio (cheap vs. caching big z).
    `out_lengths` is captured here so targets align to the SAE frame grid."""
    model.eval()
    pool, seen = [], 0
    for audios, audio_lens, _t, _tl, _last in dl:
        feats = _extract_representations(model, audios, audio_lens, device, use_bf16, has_routing)
        out_lengths = feats["out_lengths"].detach().cpu()
        f0, voiced, energy = _prosody_targets(audios.cpu(), audio_lens.cpu(), out_lengths)
        pool.append({
            "audios": audios.detach().cpu(), "audio_lens": audio_lens.detach().cpu(),
            "out_lengths": out_lengths, "f0": f0, "voiced": voiced, "energy": energy,
        })
        seen += audios.shape[0]
        if max_examples > 0 and seen >= max_examples:
            break
    return pool


def _prosody_loss(pred, f0, voiced, energy, lens):
    """Masked MSE: F0 on voiced frames, energy on all valid frames.

    Align pred (padded frame-dim) and targets (Tmax) to the common T."""
    T = min(pred.shape[1], f0.shape[1])
    pred = pred[:, :T]
    valid = _frame_mask(lens, T)                      # (B, T)
    vmask = voiced[:, :T] * valid
    f0_err = ((pred[..., 0] - f0[:, :T]) ** 2 * vmask).sum() / vmask.sum().clamp(min=1)
    e_err  = ((pred[..., 1] - energy[:, :T]) ** 2 * valid).sum() / valid.sum().clamp(min=1)
    return f0_err + e_err


def _train_prosody_probe(
    probe: nn.Module,
    src_key: str,
    pool,
    model,
    device,
    use_bf16: bool,
    has_routing: bool,
    steps: int,
    lr: float,
    warmup_steps: int,
    grad_clip: float,
    val_cache=None,
    val_every: int = 0,
    patience: int = 0,
):
    """Train the prosody probe from the cached audio+target pool.  Validates the
    mean(F0_corr, energy_corr) on the cached dev set; keeps best, early-stops.
    Returns best dev (f0_corr, energy_corr) when validating, else None."""
    trainable = [p for p in probe.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=lr, weight_decay=1e-4)
    scheduler = _make_linear_schedule(opt, warmup_steps, steps)
    probe.train()
    model.eval()
    step = 0
    do_val = val_cache is not None and val_every > 0
    best_val = -2.0
    best_pair = None
    best_state = None
    bad = 0
    stop = False

    while step < steps and not stop:
        for b in pool:
            feats = _extract_representations(model, b["audios"], b["audio_lens"],
                                             device, use_bf16, has_routing)
            z = feats[src_key]
            lens = feats["out_lengths"]
            T = z.shape[1]
            f0     = b["f0"][:, :T].to(device)
            voiced = b["voiced"][:, :T].to(device)
            energy = b["energy"][:, :T].to(device)

            opt.zero_grad(set_to_none=True)
            loss = _prosody_loss(probe(z), f0, voiced, energy, lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            opt.step()
            scheduler.step()
            step += 1

            if do_val and step % val_every == 0:
                f0c, ec = _eval_prosody_probe_cached(probe, val_cache, device)
                probe.train()
                score = np.nanmean([f0c, ec])
                if score > best_val + 1e-4:
                    best_val, best_pair, bad = score, (f0c, ec), 0
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in probe.state_dict().items()}
                else:
                    bad += 1
                print(f"      [pros {src_key}] step {step}/{steps}  dev F0r={f0c:.3f} Er={ec:.3f}  "
                      f"(best mean {best_val:.3f}, bad {bad}/{patience})", flush=True)
                if patience > 0 and bad >= patience:
                    stop = True
                    break
            if step >= steps:
                break

    if best_state is not None:
        probe.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return best_pair if do_val else None


@torch.no_grad()
def _eval_prosody_probe_cached(probe, cache, device):
    """Pearson correlation of predicted vs true contour over the cached dev/test
    set.  F0 correlation is over voiced frames; energy over all valid frames."""
    probe.eval()
    f0_acc, e_acc = _PearsonAcc(), _PearsonAcc()
    for e in cache:
        z = e["z"].to(device)
        lens = e["out_lengths"].to(device)
        T = min(z.shape[1], e["f0"].shape[1])     # align padded pred T and target Tmax
        pred = probe(z)[:, :T]
        valid  = _frame_mask(lens, T)
        voiced = e["voiced"][:, :T].to(device)
        f0     = e["f0"][:, :T].to(device)
        energy = e["energy"][:, :T].to(device)
        f0_acc.update(pred[..., 0], f0, voiced * valid)
        e_acc.update(pred[..., 1], energy, valid)
    return f0_acc.value(), e_acc.value()


@torch.no_grad()
def _eval_pr_probe(
    probe: nn.Module,
    src_key: str,
    val_dl,
    model,
    tokenizer,
    device,
    use_bf16: bool,
    has_routing: bool,
) -> float:
    probe.eval()
    model.eval()

    all_hyps: List[str] = []
    all_refs: List[str] = []
    for audios, audio_lens, _targets, _target_lens, texts in val_dl:
        feats = _extract_representations(
            model, audios, audio_lens, device, use_bf16, has_routing
        )
        log_probs = probe(feats[src_key])
        hyps = _greedy_pr_decode(log_probs.cpu(), feats["out_lengths"].cpu(), tokenizer)
        refs = [_phones_from_text(t, tokenizer) for t in texts]
        all_hyps.extend(hyps)
        all_refs.extend(refs)

    return _phone_error_rate(all_refs, all_hyps)


@torch.no_grad()
def _eval_sid_probe(
    probe: nn.Module,
    src_key: str,
    val_dl,
    model,
    device,
    use_bf16: bool,
    has_routing: bool,
) -> float:
    probe.eval()
    model.eval()
    correct = total = 0
    for audios, audio_lens, _targets, _target_lens, speaker_ids in val_dl:
        feats = _extract_representations(
            model, audios, audio_lens, device, use_bf16, has_routing
        )
        speaker_ids = speaker_ids.to(device, non_blocking=True)
        pred = probe(feats[src_key], feats["out_lengths"]).argmax(-1)
        correct += (pred == speaker_ids).sum().item()
        total += speaker_ids.size(0)
    return correct / max(total, 1)


# ---------------------------------------------------------------- cached eval (no encoder re-run)
@torch.no_grad()
def _cache_features(src_key, dl, model, device, use_bf16, has_routing, task):
    """Run the frozen model once over `dl`, caching the src feature + lengths +
    labels on CPU so repeated probe evals skip the (expensive) encoder forward.
    The model is frozen, so these features are invariant across probe steps."""
    model.eval()
    cache = []
    for audios, audio_lens, targets, target_lens, last in dl:
        feats = _extract_representations(
            model, audios, audio_lens, device, use_bf16, has_routing
        )
        entry = {
            "z": feats[src_key].detach().to("cpu"),
            "out_lengths": feats["out_lengths"].detach().to("cpu"),
        }
        if task == "sid":
            entry["speaker_ids"] = last.detach().to("cpu")
        elif task == "prosody":
            f0, voiced, energy = _prosody_targets(
                audios.cpu(), audio_lens.cpu(), feats["out_lengths"].detach().cpu())
            entry["f0"], entry["voiced"], entry["energy"] = f0, voiced, energy
        else:
            entry["texts"] = last
        cache.append(entry)
    return cache


@torch.no_grad()
def _eval_pr_probe_cached(probe, cache, tokenizer, device) -> float:
    probe.eval()
    all_hyps: List[str] = []
    all_refs: List[str] = []
    for e in cache:
        log_probs = probe(e["z"].to(device))
        hyps = _greedy_pr_decode(log_probs.cpu(), e["out_lengths"], tokenizer)
        refs = [_phones_from_text(t, tokenizer) for t in e["texts"]]
        all_hyps.extend(hyps)
        all_refs.extend(refs)
    return _phone_error_rate(all_refs, all_hyps)


@torch.no_grad()
def _eval_sid_probe_cached(probe, cache, device) -> float:
    probe.eval()
    correct = total = 0
    for e in cache:
        z = e["z"].to(device)
        lens = e["out_lengths"].to(device)
        sid = e["speaker_ids"].to(device)
        pred = probe(z, lens).argmax(-1)
        correct += (pred == sid).sum().item()
        total += sid.size(0)
    return correct / max(total, 1)


# ---------------------------------------------------------------- MDL / codelength probe
#
# Prequential MDL probe in the sense of Voita & Titov (2020, EMNLP).  The cached
# train features are partitioned into nested prefixes by the fractions below.
# For each prefix boundary t_i:
#   * the probe trained on examples [0, t_{i-1}) is evaluated on the slice
#     [t_{i-1}, t_i); its NLL (in nats) contributes to the running codelength,
#   * the probe is then trained on the prefix [0, t_i) for `steps_per_block`
#     optimisation steps (warm-started from the previous block).
# Block 0 is uncoded (no prior probe) and is paid for under a uniform prior
# (log K nats per example for SID; mean-target-length * log V for PR).
# Total codelength (nats) ≤ uniform-baseline (the "compression" the probe
# achieves over the prior).  Reported as kbits and bits/example, with a
# compression ratio (uniform - probe) / uniform.

_DEFAULT_MDL_FRACTIONS = (0.0, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0)


def _mdl_boundaries_by_examples(n_examples: int,
                                fractions=_DEFAULT_MDL_FRACTIONS) -> List[int]:
    bnd = [int(round(f * n_examples)) for f in fractions]
    bnd[0] = 0
    bnd[-1] = n_examples
    out = [bnd[0]]
    for b in bnd[1:]:
        if b > out[-1]:
            out.append(b)
    return out


def _cache_train_features(src_key, train_dl, model, device, use_bf16, has_routing,
                          task: str, max_examples: int):
    """Train-set version of `_cache_features`.  Caches features + labels on CPU
    so MDL passes do not re-run the (frozen) encoder."""
    model.eval()
    cache = []
    seen = 0
    for audios, audio_lens, targets, target_lens, last in train_dl:
        feats = _extract_representations(
            model, audios, audio_lens, device, use_bf16, has_routing
        )
        entry = {
            "z":           feats[src_key].detach().to("cpu"),
            "out_lengths": feats["out_lengths"].detach().to("cpu"),
            "B":           int(feats[src_key].shape[0]),
        }
        if task == "sid":
            entry["speaker_ids"] = last.detach().to("cpu")
        else:  # pr
            entry["targets"]     = targets.detach().to("cpu")
            entry["target_lens"] = target_lens.detach().to("cpu")
        cache.append(entry)
        seen += entry["B"]
        if max_examples > 0 and seen >= max_examples:
            break
    return cache, seen


def _example_index(cache) -> List[tuple]:
    """Flatten cache into [(batch_idx, within_batch_idx)] in cache order."""
    out: List[tuple] = []
    for bi, e in enumerate(cache):
        for ei in range(e["B"]):
            out.append((bi, ei))
    return out


def _train_probe_on_prefix(probe, cache, prefix_examples: int, task: str,
                           steps: int, lr: float, device, num_classes: int) -> None:
    """Train (or continue training) `probe` on the first `prefix_examples` items
    of the cached train set, drawing batches uniformly at random."""
    if prefix_examples <= 0 or steps <= 0:
        return
    # restrict to whole cached batches that fit inside the prefix
    cum = 0
    prefix_batches = 0
    for e in cache:
        if cum + e["B"] <= prefix_examples:
            cum += e["B"]; prefix_batches += 1
        else:
            break
    if prefix_batches == 0:
        prefix_batches = 1  # fall back to at least one batch
    trainable = [p for p in probe.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=lr, weight_decay=1e-4)
    probe.train()
    rng = np.random.default_rng(0)
    if task == "sid":
        loss_fn = nn.CrossEntropyLoss(reduction="mean")
        for _ in range(steps):
            bi = int(rng.integers(0, prefix_batches))
            e = cache[bi]
            z    = e["z"].to(device, non_blocking=True)
            lens = e["out_lengths"].to(device, non_blocking=True)
            sid  = e["speaker_ids"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(probe(z, lens), sid)
            loss.backward()
            opt.step()
    else:  # pr
        ctc = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        for _ in range(steps):
            bi = int(rng.integers(0, prefix_batches))
            e = cache[bi]
            z       = e["z"].to(device, non_blocking=True)
            lens    = e["out_lengths"].to(device, non_blocking=True)
            tgt     = e["targets"].to(device, non_blocking=True)
            tgt_len = e["target_lens"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            log_probs = probe(z)
            loss = ctc(log_probs.permute(1, 0, 2), tgt, lens, tgt_len)
            loss.backward()
            opt.step()


@torch.no_grad()
def _eval_probe_nll_on_slice(probe, cache, lo: int, hi: int, task: str, device):
    """Evaluate per-example NLL (in nats) on the example-range [lo, hi).
    Returns (sum_nll_nats, num_examples)."""
    probe.eval()
    sum_nll = 0.0
    n_examples = 0
    # Walk cache, find batches that overlap [lo, hi)
    cum = 0
    for e in cache:
        b_start, b_end = cum, cum + e["B"]
        cum = b_end
        if b_end <= lo or b_start >= hi:
            continue
        lo_i = max(0, lo - b_start)
        hi_i = min(e["B"], hi - b_start)
        if hi_i <= lo_i:
            continue
        z    = e["z"][lo_i:hi_i].to(device, non_blocking=True)
        lens = e["out_lengths"][lo_i:hi_i].to(device, non_blocking=True)
        if task == "sid":
            sid = e["speaker_ids"][lo_i:hi_i].to(device, non_blocking=True)
            logits = probe(z, lens)
            nll = F.cross_entropy(logits, sid, reduction="sum")
            sum_nll += float(nll.item())
            n_examples += sid.size(0)
        else:  # pr
            tgt     = e["targets"][lo_i:hi_i].to(device, non_blocking=True)
            tgt_len = e["target_lens"][lo_i:hi_i].to(device, non_blocking=True)
            log_probs = probe(z)  # (B, T, V)
            ctc = nn.CTCLoss(blank=0, reduction="sum", zero_infinity=True)
            nll = ctc(log_probs.permute(1, 0, 2), tgt, lens, tgt_len)
            sum_nll += float(nll.item())
            n_examples += tgt.size(0)
    return sum_nll, n_examples


def _uniform_baseline_nats(cache, lo: int, hi: int, task: str, num_classes: int) -> float:
    """Codelength under the uninformative prior over the slice [lo, hi).
    SID: log(num_speakers) per example.  PR: log(V) * mean target length per example."""
    if hi <= lo:
        return 0.0
    if task == "sid":
        return float(hi - lo) * float(np.log(max(num_classes, 1)))
    # PR: sum of target lengths in [lo, hi)
    total_phones = 0
    cum = 0
    for e in cache:
        b_start, b_end = cum, cum + e["B"]
        cum = b_end
        if b_end <= lo or b_start >= hi:
            continue
        lo_i = max(0, lo - b_start)
        hi_i = min(e["B"], hi - b_start)
        total_phones += int(e["target_lens"][lo_i:hi_i].sum().item())
    return float(total_phones) * float(np.log(max(num_classes, 1)))


def run_mdl_probe(src_key: str, task: str, in_dim: int, num_classes: int,
                  train_dl, model, device, use_bf16: bool, has_routing: bool,
                  lr: float, steps_per_block: int, max_train_examples: int,
                  sid_probe_arch: str = "stats",
                  pr_probe_arch: str = "linear") -> Dict[str, float]:
    """End-to-end prequential MDL probe.  Returns a dict with codelength /
    uniform / compression-ratio for one (src, task) pair."""
    if task == "sid":
        sid_map = {"stats": _SIDProbeStats, "mlp": _SIDProbeMLP, "linear": _SIDProbe}
        probe_cls = sid_map.get(sid_probe_arch, _SIDProbe)
        probe = probe_cls(in_dim, num_classes).to(device)
    elif task == "pr":
        pr_cls = _PRProbeMLP if pr_probe_arch == "mlp" else _PRProbe
        probe = pr_cls(in_dim, num_classes).to(device)
    else:
        raise ValueError(f"MDL probe supports sid|pr, got {task}")

    cache, n = _cache_train_features(
        src_key, train_dl, model, device, use_bf16, has_routing, task, max_train_examples
    )
    if n == 0:
        return {"codelength_nats": float("nan"), "uniform_nats": float("nan"),
                "compression": float("nan"), "n_examples": 0}

    bnd = _mdl_boundaries_by_examples(n)
    code_nats = 0.0
    unif_nats = 0.0
    # Block 0: [0, bnd[1]) — predicted with uniform prior.
    code_nats += _uniform_baseline_nats(cache, 0, bnd[1], task, num_classes)
    unif_nats += _uniform_baseline_nats(cache, 0, bnd[1], task, num_classes)
    # Train probe on prefix [0, bnd[1]).
    _train_probe_on_prefix(probe, cache, bnd[1], task, steps_per_block, lr, device, num_classes)

    for i in range(1, len(bnd) - 1):
        lo, hi = bnd[i], bnd[i + 1]
        # Evaluate the current probe on the next slice.
        sum_nll, _ = _eval_probe_nll_on_slice(probe, cache, lo, hi, task, device)
        code_nats += sum_nll
        unif_nats += _uniform_baseline_nats(cache, lo, hi, task, num_classes)
        # Now include this slice in the training prefix and keep training.
        _train_probe_on_prefix(probe, cache, hi, task, steps_per_block, lr, device, num_classes)
        print(f"      [mdl {src_key}/{task}] block {i}/{len(bnd)-2}  "
              f"slice=[{lo}, {hi})  block_nats={sum_nll:.1f}  "
              f"running_kbits={code_nats / np.log(2) / 1000:.2f}", flush=True)

    return {
        "codelength_nats":  code_nats,
        "uniform_nats":     unif_nats,
        "codelength_kbits": code_nats / float(np.log(2)) / 1000.0,
        "uniform_kbits":    unif_nats / float(np.log(2)) / 1000.0,
        "compression":      0.0 if unif_nats <= 0 else max(0.0, 1.0 - code_nats / unif_nats),
        "n_examples":       n,
        "n_blocks":         len(bnd) - 1,
    }


def _greedy_pr_decode(log_probs: torch.Tensor, lengths: torch.Tensor, tokenizer, blank_id: int = 0) -> List[str]:
    preds = log_probs.argmax(dim=-1)
    out: List[str] = []
    for row, n in zip(preds, lengths.tolist()):
        ids = row[:n].tolist()
        collapsed, prev = [], -1
        for i in ids:
            if i != prev:
                collapsed.append(i)
                prev = i
        keep = [i for i in collapsed if i != blank_id]
        out.append(tokenizer.decode(keep))
    return out


# ---------------------------------------------------------------- main

def _parse_args():
    cfg = DISConfig()
    p   = argparse.ArgumentParser()
    p.add_argument("--stage1_ckpt",  required=True)
    p.add_argument("--stage2_ckpt",  default=None,
                   help="If given, loads routing and enables z_L / z_P probes")
    p.add_argument("--run_name",     default="probe")
    p.add_argument("--probe_steps",  type=int, default=2000)
    p.add_argument("--topk",         type=int, default=0,
                   help="Override cfg.topk (e.g. 128 for K=10240 checkpoints)")
    p.add_argument("--librispeech_cache_dir", default=str(cfg.librispeech_cache_dir))
    p.add_argument("--lexicon_path",          default=str(cfg.lexicon_path))
    p.add_argument("--max_train_examples",    type=int, default=0)   # 0 = full set (all 251 speakers)
    p.add_argument("--max_val_examples",      type=int, default=500)
    p.add_argument("--pr_max_examples",       type=int, default=0,
                   help="Cap PR train/val/test examples. 0 = full SUPERB PR splits.")
    p.add_argument("--pr_probe_lr",           type=float, default=5e-4)
    p.add_argument("--sid_probe_lr",          type=float, default=1e-4)
    p.add_argument("--probe_warmup_steps",    type=int, default=500)
    p.add_argument("--probe_grad_clip",       type=float, default=1.0)
    return p.parse_args()


def main():
    args   = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global _pr_data
    import pr_data as _pr_data_module
    _prioritize_import_paths()  # pr_data prepends Probing/; restore Disentanglement first.
    from model import build_dis_model
    from train import _load_stage1_checkpoint
    from data.dataset import make_stage2_dataloaders
    _pr_data = _pr_data_module

    cfg                       = DISConfig()
    cfg.device                = str(device)
    cfg.librispeech_cache_dir = Path(args.librispeech_cache_dir)
    cfg.lexicon_path          = Path(args.lexicon_path)
    cfg.max_train_examples    = args.max_train_examples
    cfg.max_val_examples      = args.max_val_examples

    print(f"[probe] run={args.run_name}  device={device}")
    print(f"[probe] stage1_ckpt={args.stage1_ckpt}")
    print(f"[probe] stage2_ckpt={args.stage2_ckpt or '(none — baselines only)'}")
    print(
        f"[probe] probe_steps={args.probe_steps}  "
        f"sid_train_examples={args.max_train_examples}  pr_max_examples={args.pr_max_examples}"
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # PR uses the same SUPERB phone preparation as Probing/pr:
    # train-clean-100 -> train, dev-clean -> val, 74-token stress-marked phones.
    pr_cfg = PRConfig()
    pr_cfg.data_cache_dir = cfg.librispeech_cache_dir
    pr_cfg.librispeech_lexicon = cfg.lexicon_path
    pr_cfg.batch_size = cfg.batch_size
    pr_cfg.eval_batch_size = cfg.eval_batch_size
    pr_cfg.num_workers = cfg.num_workers
    pr_cfg.max_examples = args.pr_max_examples
    pr_tokenizer, pr_train_dl, pr_val_dl, _ = _pr_data.make_pr_dataloaders(pr_cfg)

    # SID keeps the current LibriSpeech speaker diagnostic split for now, but
    # uses the SUPERB SID probe head below.
    _, sid_train_dl, sid_val_dl, _sid_test_dl = make_stage2_dataloaders(cfg)

    # If a stage2 checkpoint is given, override num_speakers and infer K from checkpoint
    if args.stage2_ckpt:
        _tmp = torch.load(args.stage2_ckpt, map_location="cpu", weights_only=False)
        _state = _tmp["model_state"]
        cfg.num_speakers = _state["sid_head.fc.weight"].shape[0]
        ckpt_K = _state["sae.enc_weight"].shape[0]
        if ckpt_K != cfg.K:
            print(f"[probe] K overridden from checkpoint: {cfg.K} → {ckpt_K}")
            cfg.K = ckpt_K
        if "routing.logits" in _state:
            ckpt_routes = _state["routing.logits"].shape[1]
            if ckpt_routes != cfg.n_routes:
                print(f"[probe] n_routes overridden from checkpoint: {cfg.n_routes} → {ckpt_routes}")
                cfg.n_routes = ckpt_routes
        if "proj_L.proj.weight" in _state:
            cfg.projection_disentanglement = True
            ckpt_dim = _state["proj_L.proj.weight"].shape[0]
            if ckpt_dim != cfg.projection_dim:
                print(f"[probe] projection_dim overridden from checkpoint: {cfg.projection_dim} -> {ckpt_dim}")
                cfg.projection_dim = ckpt_dim
            print("[probe] projection_disentanglement enabled from checkpoint")
        del _tmp
        print(f"[probe] num_speakers overridden from checkpoint → {cfg.num_speakers}")

    if args.topk > 0:
        print(f"[probe] topk overridden: {cfg.topk} → {args.topk}")
        cfg.topk = args.topk

    # Build and load model (frozen for extraction)
    model = build_dis_model(cfg)
    _load_stage1_checkpoint(Path(args.stage1_ckpt), model, cfg)
    has_routing = False

    if args.stage2_ckpt:
        ckpt = torch.load(args.stage2_ckpt, map_location=device, weights_only=False)
        missing, _ = model.load_state_dict(ckpt["model_state"], strict=False)
        non_spear  = [k for k in missing if not k.startswith("encoder._spear.")]
        if non_spear:
            print(f"[probe] WARNING missing keys: {non_spear[:5]}")
        print(f"[probe] loaded stage2 weights from {args.stage2_ckpt}")
        has_routing = True

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    D, K, num_spk, pr_vocab_size = cfg.D, cfg.K, cfg.num_speakers, pr_cfg.vocab_size

    # Sources and their input dimensions
    sources: List[str] = ["h_t", "z_t"]
    view_dim = cfg.projection_dim if cfg.projection_disentanglement else K
    dims: Dict[str, int] = {"h_t": D, "z_t": K, "z_L": view_dim, "z_P": view_dim}
    if has_routing:
        sources += ["z_L", "z_P"]

    tasks = ["pr", "sid"]

    print(f"\n[probe] speakers={num_spk}  pr_vocab={pr_vocab_size}  D={D}  K={K}")
    print("[probe] PR data/head: Probing/pr SUPERB-style loader + CTC projector head")
    print("[probe] SID head: Probing/sid SUPERB-style projector + masked mean + linear")
    print(f"[probe] sources={sources}  tasks={tasks}\n")

    results: Dict[str, Dict[str, float]] = {}

    for src in sources:
        results[src] = {}
        in_dim = dims[src]
        for task in tasks:
            label = f"{src} → {task.upper()}"
            print(f"  training probe: {label} ...", flush=True)

            if task == "sid":
                probe = _SIDProbe(in_dim, num_spk).to(device)
                _train_sid_probe(
                    probe, src, sid_train_dl, model, device, use_bf16, has_routing,
                    steps=args.probe_steps, lr=args.sid_probe_lr,
                    warmup_steps=args.probe_warmup_steps,
                    grad_clip=args.probe_grad_clip,
                )
                score = _eval_sid_probe(
                    probe, src, sid_val_dl, model, device, use_bf16, has_routing
                )
            else:
                probe = _PRProbe(in_dim, pr_vocab_size).to(device)
                _train_pr_probe(
                    probe, src, pr_train_dl, model, device, use_bf16, has_routing,
                    steps=args.probe_steps, lr=args.pr_probe_lr,
                    warmup_steps=args.probe_warmup_steps,
                    grad_clip=args.probe_grad_clip,
                )
                score = _eval_pr_probe(
                    probe, src, pr_val_dl, model, pr_tokenizer, device, use_bf16,
                    has_routing,
                )
            results[src][task] = score
            metric = f"PER={score:.3f}" if task == "pr" else f"acc={score:.3f}"
            print(f"    {label:<22s}  {metric}", flush=True)

    # Results table
    print(f"\n{'='*60}")
    print(f"  PROBE RESULTS — {args.run_name}")
    print(f"{'='*60}")
    print(f"  {'Source':<8s}  {'PR (PER↓)':>12s}  {'SID (acc↑)':>12s}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}")
    for src in sources:
        per = results[src].get("pr", float('nan'))
        acc = results[src].get("sid", float('nan'))
        flag = ""
        if src == "z_L" and results[src].get("sid", 0) > results.get("z_t", {}).get("sid", 0) * 0.8:
            flag = "  ← leakage?"
        if src == "z_P" and results[src].get("pr", 1) < results.get("h_t", {}).get("pr", 1) * 1.5:
            flag = "  ← leakage?"
        print(f"  {src:<8s}  {per:>12.3f}  {acc:>12.3f}{flag}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
