"""Standalone one-epoch fine-tune of a saved SID probe checkpoint.

Loads a probe checkpoint produced by sid_run.py, runs N more epochs with a
fresh small LR (no warmup), evaluates on the test set, and saves a summary.
Does NOT modify any existing sid_*.py files.

Usage
-----
    python sid_finetune.py \
        --probe final \
        --checkpoint sid/checkpoints/sid_probe_final_best.pt \
        --voxceleb1_root /path/to/VoxCeleb1 \
        --extra_epochs 1 \
        --lr 1e-5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).parent))
from sid_config import SIDConfig
from sid_data   import make_sid_dataloaders
from sid_model  import build_sid_model
from sid_train  import evaluate_sid


def parse_args() -> SIDConfig:
    cfg = SIDConfig()
    p = argparse.ArgumentParser(description="One-shot fine-tune of a saved SID probe.")
    p.add_argument("--probe",            choices=["final", "weighted"], required=True)
    p.add_argument("--checkpoint",       required=True,
                   help="Path to the .pt probe checkpoint to resume from.")
    p.add_argument("--voxceleb1_root",   required=True)
    p.add_argument("--extra_epochs",     type=int,   default=1)
    p.add_argument("--lr",               type=float, default=1e-5)
    p.add_argument("--batch_size",       type=int,   default=cfg.batch_size)
    p.add_argument("--eval_batch_size",  type=int,   default=cfg.eval_batch_size)
    p.add_argument("--model_id",         default=cfg.model_id)
    p.add_argument("--model_family",     default=cfg.model_family, choices=["spear", "hf"])
    p.add_argument("--checkpoint_dir",   default=None)
    p.add_argument("--runs_dir",         default=None)
    p.add_argument("--num_workers",      type=int,   default=cfg.num_workers)
    p.add_argument("--max_duration_s",   type=float, default=cfg.max_duration_s)
    args = p.parse_args()

    cfg.probe_type      = args.probe
    cfg.voxceleb1_root  = Path(args.voxceleb1_root)
    cfg.model_id        = args.model_id
    cfg.model_family    = args.model_family
    cfg.num_epochs      = args.extra_epochs
    cfg.batch_size      = args.batch_size
    cfg.eval_batch_size = args.eval_batch_size
    cfg.num_workers     = args.num_workers
    cfg.max_duration_s  = args.max_duration_s
    cfg.checkpoint_dir  = Path(args.checkpoint_dir) if args.checkpoint_dir else cfg.checkpoint_dir
    cfg.runs_dir        = Path(args.runs_dir)        if args.runs_dir        else cfg.runs_dir

    cfg._resume_path = Path(args.checkpoint)
    cfg._finetune_lr = args.lr
    return cfg


def main() -> None:
    cfg = parse_args()
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== SID fine-tune ({'final' if cfg.probe_type == 'final' else 'weighted'} probe)")
    print(f"=== resume from   : {cfg._resume_path}")
    print(f"=== extra epochs  : {cfg.num_epochs}  lr={cfg._finetune_lr}")

    # ------------------------------------------------------------------ data
    train_dl, val_dl, test_dl, _ = make_sid_dataloaders(cfg)

    # ----------------------------------------------------------------- model
    encoder, probe = build_sid_model(cfg)

    ckpt = torch.load(cfg._resume_path, map_location="cpu", weights_only=True)
    probe.load_state_dict(ckpt["probe_state"])
    prior_val_acc = ckpt.get("val_acc", float("nan"))
    print(f"[finetune] loaded probe  prior val_acc={prior_val_acc:.4f}")

    # ------------------------------------------------------------ train loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    probe.to(device)

    trainable  = [p for p in probe.parameters() if p.requires_grad]
    optimizer  = AdamW(trainable, lr=cfg._finetune_lr, weight_decay=cfg.weight_decay)
    ce_loss    = nn.CrossEntropyLoss()

    best_acc   = prior_val_acc
    best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
    step       = 0
    n_batches  = len(train_dl)

    for epoch in range(cfg.num_epochs):
        probe.train()
        encoder.eval()
        running_loss = running_count = 0

        for batch_idx, (audios, audio_lens, labels) in enumerate(train_dl):
            audios     = audios.to(device,     non_blocking=True)
            audio_lens = audio_lens.to(device, non_blocking=True)
            labels     = labels.to(device,     non_blocking=True)

            with torch.no_grad():
                hidden     = encoder(audios, audio_lens)
                frame_lens = encoder.output_lengths(audio_lens)

            logits = probe(hidden, frame_lens)
            loss   = ce_loss(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()

            running_loss  += loss.item() * audios.size(0)
            running_count += audios.size(0)
            step += 1

            log_every = max(1, min(cfg.log_every, n_batches // 2 or 1))
            if step % log_every == 0 or batch_idx == n_batches - 1:
                avg = running_loss / max(1, running_count)
                print(f"  epoch {epoch+1}/{cfg.num_epochs}  step {step:>6d}  loss {avg:.4f}  lr {cfg._finetune_lr:.2e}")
                running_loss = running_count = 0

        metrics = evaluate_sid(cfg, encoder, probe, val_dl, label="val", epoch=epoch+1)
        if metrics["acc"] > best_acc:
            best_acc   = metrics["acc"]
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
            ckpt_path  = cfg.checkpoint_dir / f"sid_probe_{cfg.probe_type}_finetune_best.pt"
            torch.save({"probe_state": best_state, "val_acc": best_acc, "epoch": epoch+1}, ckpt_path)
            print(f"  ✓ saved best probe (val_acc={best_acc:.4f}) → {ckpt_path}")
        else:
            print(f"  (no improvement: {metrics['acc']:.4f} ≤ best {best_acc:.4f})")

    # ---------------------------------------------------------- test eval
    probe.load_state_dict(best_state)
    print("\n=== Final test-set evaluation ===")
    test_metrics = evaluate_sid(cfg, encoder, probe, test_dl, label="test", epoch=cfg.num_epochs)
    print(f"[SID finetune] test acc : {test_metrics['acc']:.4f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = cfg.runs_dir / f"{ts}_sid_{cfg.probe_type}_finetune_summary.json"
    with summary_path.open("w") as f:
        json.dump({
            "probe_type":     cfg.probe_type,
            "model_id":       cfg.model_id,
            "resumed_from":   str(cfg._resume_path),
            "prior_val_acc":  prior_val_acc,
            "best_val_acc":   best_acc,
            "test_acc":       test_metrics["acc"],
            "extra_epochs":   cfg.num_epochs,
            "finetune_lr":    cfg._finetune_lr,
        }, f, indent=2)
    print(f"[run done] summary → {summary_path}")


if __name__ == "__main__":
    main()
