#!/usr/bin/env python3
"""Post-hoc probing for disentanglement analysis (experiments A / B / C).

For each checkpoint, trains lightweight linear probes on every source representation
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
  python probe_runner.py --stage1_ckpt checkpoints/best.pt --run_name probe_B

  # Mode A — probe Run 2 checkpoint
  python probe_runner.py --stage1_ckpt checkpoints/best.pt \\
      --stage2_ckpt checkpoints/sid1_weakgrl/stage2_best.pt --run_name probe_A

  # Mode C — probe Run 3 checkpoint
  python probe_runner.py --stage1_ckpt checkpoints/best.pt \\
      --stage2_ckpt checkpoints/sid1_nogrl/stage2_best.pt --run_name probe_C
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).parent))

from config import DISConfig
from model import build_dis_model
from train import (_load_stage1_checkpoint, _make_scheduler,
                   _greedy_ctc_decode, _edit_distance)
from data.dataset import make_stage2_dataloaders
from losses import ctc_pr_loss


# ---------------------------------------------------------------- probe heads

class _SIDProbe(nn.Module):
    def __init__(self, in_dim: int, num_speakers: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, num_speakers)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _PRProbe(nn.Module):
    def __init__(self, in_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, vocab_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ---------------------------------------------------------------- helpers

def _mean_pool(z: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    B, T, _ = z.shape
    mask  = (torch.arange(T, device=z.device).unsqueeze(0) < lengths.unsqueeze(1)).float()
    return (z * mask.unsqueeze(-1)).sum(1) / lengths.float().clamp(min=1).unsqueeze(-1)


@torch.no_grad()
def _extract(model, batch, device, use_bf16, has_routing: bool):
    audios, audio_lengths, targets, target_lengths, speaker_ids = batch
    audios         = audios.to(device)
    audio_lengths  = audio_lengths.to(device)
    targets        = targets.to(device)
    target_lengths = target_lengths.to(device)
    speaker_ids    = speaker_ids.to(device)

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
        "out_lengths":    out["out_lengths"],
        "targets":        targets,
        "target_lengths": target_lengths,
        "speaker_ids":    speaker_ids,
    }


# ---------------------------------------------------------------- probe training / eval

def _train_probe(
    probe: nn.Module,
    src_key: str,
    task: str,
    train_dl,
    model,
    device,
    use_bf16: bool,
    has_routing: bool,
    steps: int,
) -> None:
    opt   = AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    probe.train()
    step  = 0
    model.eval()

    while step < steps:
        for batch in train_dl:
            feats = _extract(model, batch, device, use_bf16, has_routing)
            z     = feats[src_key]          # (B, T, dim)
            lens  = feats["out_lengths"]

            opt.zero_grad(set_to_none=True)

            if task == "sid":
                z_pool = _mean_pool(z, lens)                        # (B, dim)
                loss   = nn.CrossEntropyLoss()(probe(z_pool), feats["speaker_ids"])
            else:  # pr
                logits = probe(z)                                    # (B, T, vocab)
                loss   = ctc_pr_loss(logits, feats["targets"], lens, feats["target_lengths"])

            loss.backward()
            opt.step()
            step += 1
            if step >= steps:
                break


@torch.no_grad()
def _eval_probe(
    probe: nn.Module,
    src_key: str,
    task: str,
    val_dl,
    model,
    device,
    use_bf16: bool,
    has_routing: bool,
) -> float:
    probe.eval()
    model.eval()

    if task == "sid":
        correct = total = 0
        for batch in val_dl:
            feats  = _extract(model, batch, device, use_bf16, has_routing)
            z_pool = _mean_pool(feats[src_key], feats["out_lengths"])
            pred   = probe(z_pool).argmax(-1)
            correct += (pred == feats["speaker_ids"]).sum().item()
            total   += feats["speaker_ids"].size(0)
        return correct / max(total, 1)
    else:  # pr → PER
        per_num = per_den = 0
        for batch in val_dl:
            feats  = _extract(model, batch, device, use_bf16, has_routing)
            logits = probe(feats[src_key])
            preds  = _greedy_ctc_decode(logits, feats["out_lengths"])
            for i, pred in enumerate(preds):
                ref      = feats["targets"][i, :feats["target_lengths"][i]].tolist()
                per_num += _edit_distance(pred, ref)
                per_den += len(ref)
        return per_num / max(per_den, 1)


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
    return p.parse_args()


def main():
    args   = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg                       = DISConfig()
    cfg.device                = str(device)
    cfg.librispeech_cache_dir = Path(args.librispeech_cache_dir)
    cfg.lexicon_path          = Path(args.lexicon_path)
    cfg.max_train_examples    = args.max_train_examples
    cfg.max_val_examples      = args.max_val_examples

    print(f"[probe] run={args.run_name}  device={device}")
    print(f"[probe] stage1_ckpt={args.stage1_ckpt}")
    print(f"[probe] stage2_ckpt={args.stage2_ckpt or '(none — baselines only)'}")
    print(f"[probe] probe_steps={args.probe_steps}  train_examples={args.max_train_examples}")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # Build dataloaders first — populates cfg.num_speakers and cfg.vocab_size
    _, train_dl, val_dl = make_stage2_dataloaders(cfg)

    # If a stage2 checkpoint is given, override num_speakers and infer K from checkpoint
    if args.stage2_ckpt:
        _tmp = torch.load(args.stage2_ckpt, map_location="cpu", weights_only=False)
        _state = _tmp["model_state"]
        cfg.num_speakers = _state["sid_head.net.2.weight"].shape[0]
        ckpt_K = _state["sae.enc_weight"].shape[0]
        if ckpt_K != cfg.K:
            print(f"[probe] K overridden from checkpoint: {cfg.K} → {ckpt_K}")
            cfg.K = ckpt_K
        if "routing.logits" in _state:
            ckpt_routes = _state["routing.logits"].shape[1]
            if ckpt_routes != cfg.n_routes:
                print(f"[probe] n_routes overridden from checkpoint: {cfg.n_routes} → {ckpt_routes}")
                cfg.n_routes = ckpt_routes
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
    D, K, num_spk, vocab_size = cfg.D, cfg.K, cfg.num_speakers, cfg.vocab_size

    # Sources and their input dimensions
    sources: List[str] = ["h_t", "z_t"]
    dims: Dict[str, int] = {"h_t": D, "z_t": K, "z_L": K, "z_P": K}
    if has_routing:
        sources += ["z_L", "z_P"]

    tasks = ["pr", "sid"]

    print(f"\n[probe] speakers={num_spk}  vocab={vocab_size}  D={D}  K={K}")
    print(f"[probe] sources={sources}  tasks={tasks}\n")

    results: Dict[str, Dict[str, float]] = {}

    for src in sources:
        results[src] = {}
        in_dim = dims[src]
        for task in tasks:
            label = f"{src} → {task.upper()}"
            print(f"  training probe: {label} ...", flush=True)

            probe = (_SIDProbe(in_dim, num_spk) if task == "sid"
                     else _PRProbe(in_dim, vocab_size)).to(device)

            _train_probe(probe, src, task, train_dl, model, device, use_bf16,
                         has_routing, steps=args.probe_steps)

            score = _eval_probe(probe, src, task, val_dl, model, device, use_bf16, has_routing)
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
