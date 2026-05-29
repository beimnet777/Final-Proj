#!/usr/bin/env python3
"""Compute per-bucket (L/P/U) active feature counts from the trained checkpoint.

Patches SPEAR with a fake model so this runs on CPU.
Runs N forward passes with synthetic audio to measure:
  - Active feature count per frame in z_dense overall
  - How those active features split across L / P / U buckets
  - Per-bucket density (active features / features-in-bucket)
  - Routing logit statistics (are logits truly near-zero?)
"""
import sys, math
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


class _FakeSpear(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MagicMock()
        self.config.hidden_size = 0
        self.config.num_hidden_layers = 0

    def forward(self, audio, lengths):
        B, T_audio = audio.shape
        T = max(int(T_audio) // 640, 1)
        D, L = 1280, 13
        hidden = [torch.randn(B, T, D) for _ in range(L)]
        return {"hidden_states": hidden}


def main():
    with patch("transformers.AutoModel.from_pretrained", return_value=_FakeSpear()):
        _run()


def _run():
    from config import DISConfig
    from model import build_dis_model
    from train import _load_checkpoint

    ckpt_path = Path("checkpoints/stage2_best.pt")
    cfg = DISConfig()
    cfg.device = "cpu"
    cfg.bf16 = False
    # Pre-load checkpoint to get num_speakers so model is built with correct head sizes
    ckpt_pre = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg.num_speakers = ckpt_pre.get("num_speakers", cfg.num_speakers)
    cfg.vocab_size   = ckpt_pre.get("vocab_size",   cfg.vocab_size)

    print("Building model and loading stage2_best checkpoint ...")
    model = build_dis_model(cfg)
    _load_checkpoint(ckpt_path, model, cfg)
    model.eval()

    K = cfg.K
    topk = cfg.topk
    print(f"K={K}  topk={topk}  TopK-budget={100*topk/K:.1f}%\n")

    # ----------------------------------------------------------------
    # 1. Routing logit statistics
    logits = model.routing.logits.detach()          # (K, 3)
    hard_idx = logits.argmax(dim=-1)                # (K,)
    n_L = (hard_idx == 0).sum().item()
    n_P = (hard_idx == 1).sum().item()
    n_U = (hard_idx == 2).sum().item()

    print(f"Hard routing counts: L={n_L} ({100*n_L/K:.1f}%)  "
          f"P={n_P} ({100*n_P/K:.1f}%)  U={n_U} ({100*n_U/K:.1f}%)")

    logit_range = logits.max().item() - logits.min().item()
    logit_std   = logits.std().item()
    print(f"Logit range: [{logits.min().item():.4f}, {logits.max().item():.4f}]  "
          f"std={logit_std:.4f}")

    # Mean per-unit entropy (the CORRECT specialisation metric)
    p_per_unit = torch.softmax(logits, dim=-1)                        # (K, 3)
    per_unit_H = -(p_per_unit * p_per_unit.log().clamp(-100)).sum(-1) # (K,)
    print(f"Mean per-unit entropy: {per_unit_H.mean().item():.4f} nats  "
          f"[max=log(3)={math.log(3):.4f}]")
    print(f"  fraction of units within 0.001 of log(3): "
          f"{((per_unit_H - math.log(3)).abs() < 0.001).float().mean().item()*100:.1f}%")
    print(f"  fraction of units with H < 0.5 nats (=~specialised): "
          f"{(per_unit_H < 0.5).float().mean().item()*100:.1f}%")
    print()

    # ----------------------------------------------------------------
    # 2. Forward passes to measure per-bucket density
    B_syn = 4
    T_audio = 19200  # 1.2 s → ~30 frames after 640× downsample

    all_density_total  = []
    all_density_L      = []
    all_density_P      = []
    all_density_U      = []
    all_active_total   = []
    all_active_L       = []
    all_active_P       = []
    all_active_U       = []

    # masks for routing (on CPU; boolean (K,))
    mask_L = (hard_idx == 0)
    mask_P = (hard_idx == 1)
    mask_U = (hard_idx == 2)

    torch.manual_seed(42)
    N_batches = 20
    print(f"Running {N_batches} forward passes (B={B_syn}, T_audio={T_audio}) ...")

    with torch.no_grad():
        for _ in range(N_batches):
            audio = torch.randn(B_syn, T_audio)
            audio_lens = torch.full((B_syn,), T_audio, dtype=torch.long)
            out = model(audio, audio_lens, stage=2, grl_lambda=0.0)
            z_dense = out["z_dense"]      # (B, T, K)
            lengths = out["out_lengths"]  # (B,)

            # Build frame-level mask (only valid frames, excluding padding)
            B, T, _ = z_dense.shape
            frame_mask = (
                torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)
            )  # (B, T) bool

            # z_dense over valid frames only
            z_valid = z_dense[frame_mask]   # (N_valid_frames, K)
            N_valid = z_valid.shape[0]

            active = (z_valid > 0).float()  # (N_valid, K)

            # Overall density (valid frames, all K)
            d_total = active.mean().item()
            # Per-frame average active count
            active_per_frame = active.sum(-1).mean().item()

            # Per-bucket
            d_L = active[:, mask_L].mean().item()
            d_P = active[:, mask_P].mean().item()
            d_U = active[:, mask_U].mean().item()

            act_L = active[:, mask_L].sum(-1).mean().item()  # avg active features in L per frame
            act_P = active[:, mask_P].sum(-1).mean().item()
            act_U = active[:, mask_U].sum(-1).mean().item()

            all_density_total.append(d_total)
            all_density_L.append(d_L)
            all_density_P.append(d_P)
            all_density_U.append(d_U)
            all_active_total.append(active_per_frame)
            all_active_L.append(act_L)
            all_active_P.append(act_P)
            all_active_U.append(act_U)

    dt  = np.mean(all_density_total)
    dL  = np.mean(all_density_L)
    dP  = np.mean(all_density_P)
    dU  = np.mean(all_density_U)
    atL = np.mean(all_active_L)
    atP = np.mean(all_active_P)
    atU = np.mean(all_active_U)
    at  = np.mean(all_active_total)

    print(f"\n{'='*60}")
    print("PER-BUCKET ACTIVE FEATURE ANALYSIS (valid frames only)")
    print(f"{'='*60}")
    print(f"Overall z_dense density (valid frames): {dt*100:.2f}%")
    print(f"Average active features per frame:      {at:.1f} / {K} total")
    print(f"  vs TopK budget:                       {topk} / {K}")
    print()
    print(f"{'Bucket':<8} {'#features':>10} {'% of K':>8} {'density':>10} {'active/frame':>14} {'% of active':>12}")
    print("-"*65)
    total_active = atL + atP + atU
    for name, n, d, a in [("L", n_L, dL, atL), ("P", n_P, dP, atP), ("U", n_U, dU, atU)]:
        print(f"  {name:<6} {n:>10} {100*n/K:>7.1f}%  {d*100:>9.2f}%  {a:>13.1f}  {100*a/total_active:>11.1f}%")

    print()
    print("Interpretation:")
    print(f"  Of the {at:.0f} active features/frame, CTC sees {atL:.1f} (z_L) out of {topk} topk budget")
    print(f"  SID sees {atP:.1f} active features per frame (z_P), pooled → z̄_P")
    print(f"  Padding-corrected density ≈ {dt*100:.2f}%  (train.py reported 0.6% over ALL frames incl. padding)")


if __name__ == "__main__":
    main()
