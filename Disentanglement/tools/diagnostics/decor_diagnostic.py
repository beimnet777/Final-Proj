#!/usr/bin/env python3
"""Measure SAE feature decorrelation directly from stage-1 checkpoints.

The stage-1 training log never recorded the decorrelation loss value, so this
script recomputes it post-hoc. For each checkpoint it runs the frozen SAE over
the validation set, accumulates per-utterance mean-pooled z_t, and computes the
off-diagonal feature correlation — the exact quantity decor_loss penalises, but
measured globally over the whole val set rather than per training batch.

Reported per checkpoint:
  K_active      number of features active at least once on the val set
  offdiag_sq    mean squared off-diagonal correlation  (== decor_loss objective)
  offdiag_abs   mean |off-diagonal correlation|         (interpretable scale)
  frac|r|>0.3   fraction of feature pairs with |corr| > 0.3 (redundancy tail)
  frac|r|>0.5   fraction of feature pairs with |corr| > 0.5

Usage:
  python tools/diagnostics/decor_diagnostic.py \
      --ckpt baseline=checkpoints/best.pt \
      --ckpt decor_only=checkpoints/decor_only/stage1_best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

DIS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DIS_DIR))

from config import DISConfig
from model import build_dis_model
from train import _load_stage1_checkpoint
from data.dataset import make_stage2_dataloaders


def _mean_pool(z: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    B, T, _ = z.shape
    mask = (torch.arange(T, device=z.device).unsqueeze(0) < lengths.unsqueeze(1)).float()
    return (z * mask.unsqueeze(-1)).sum(1) / lengths.float().clamp(min=1).unsqueeze(-1)


@torch.no_grad()
def collect_pooled_z(model, val_dl, device, use_bf16) -> torch.Tensor:
    """Return (N_utts, K) matrix of per-utterance mean-pooled z_t."""
    model.eval()
    chunks = []
    for batch in val_dl:
        audios, audio_lengths = batch[0].to(device), batch[1].to(device)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cuda", enabled=False)
        with ctx:
            out = model(audios, audio_lengths, stage=1)
        z_pool = _mean_pool(out["z_t"].float(), out["out_lengths"])
        chunks.append(z_pool.cpu())
    return torch.cat(chunks, 0)


def decor_stats(z_pool: torch.Tensor) -> dict:
    """Off-diagonal correlation stats over features active on the val set."""
    N, K = z_pool.shape
    active = (z_pool.abs() > 0).any(0)           # (K,)
    z = z_pool[:, active]                          # (N, K_active)
    Ka = z.shape[1]

    z = (z - z.mean(0, keepdim=True)) / (z.std(0, keepdim=True) + 1e-8)
    corr = (z.T @ z) / N                           # (Ka, Ka)

    eye = torch.eye(Ka)
    off = corr * (1 - eye)                          # zero the diagonal
    n_off = Ka * (Ka - 1)

    return {
        "N_utts":       N,
        "K_active":     Ka,
        "offdiag_sq":   off.pow(2).sum().item() / n_off,
        "offdiag_abs":  off.abs().sum().item() / n_off,
        "frac_gt_0.3":  (off.abs() > 0.3).sum().item() / n_off,
        "frac_gt_0.5":  (off.abs() > 0.5).sum().item() / n_off,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", action="append", required=True,
                   help="name=path  (repeatable)")
    p.add_argument("--max_val_examples", type=int, default=500)
    args = p.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    cfg = DISConfig()
    cfg.device = str(device)
    cfg.max_val_examples = args.max_val_examples
    _, _, val_dl, _ = make_stage2_dataloaders(cfg)

    rows = []
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        print(f"\n[decor] {name}: loading {path}")
        model = build_dis_model(cfg)
        _load_stage1_checkpoint(Path(path), model, cfg)
        model.to(device)
        z_pool = collect_pooled_z(model, val_dl, device, use_bf16)
        stats = decor_stats(z_pool)
        stats["name"] = name
        rows.append(stats)
        print(f"[decor] {name}: K_active={stats['K_active']}  "
              f"offdiag_sq={stats['offdiag_sq']:.6f}  "
              f"offdiag_abs={stats['offdiag_abs']:.4f}")
        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*84}")
    print("  SAE FEATURE DECORRELATION  (lower offdiag = more decorrelated)")
    print(f"{'='*84}")
    print(f"  {'checkpoint':<16s}{'K_active':>9s}{'offdiag_sq':>13s}"
          f"{'offdiag_abs':>13s}{'frac|r|>.3':>12s}{'frac|r|>.5':>12s}")
    print(f"  {'-'*16}{'-'*9}{'-'*13}{'-'*13}{'-'*12}{'-'*12}")
    for r in rows:
        print(f"  {r['name']:<16s}{r['K_active']:>9d}{r['offdiag_sq']:>13.6f}"
              f"{r['offdiag_abs']:>13.4f}{r['frac_gt_0.3']:>12.4f}{r['frac_gt_0.5']:>12.4f}")
    print(f"{'='*84}\n")


if __name__ == "__main__":
    main()
