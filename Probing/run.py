from __future__ import annotations
import os
import warnings

# MPS fallback for ops not yet implemented on Apple Silicon (e.g. ctc_loss).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Suppress noisy deprecation warnings from SPEAR's internal code and torchaudio.
# These are in third-party libs we don't control and don't affect correctness.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

"""CLI entry point.

Builds dataloaders, loads SPEAR, attaches the chosen probe head, trains, then
reports CER/WER on test-clean. For the weighted probe it also prints the
learned softmax layer weights so you can see which SPEAR layers the probe
relied on.

Examples
--------
    python run.py --probe final    --spear_model_id <hf-id>
    python run.py --probe weighted --spear_model_id <hf-id>  --epochs 30
"""



import argparse
import random
from pathlib import Path

import numpy as np
import torch

from config import Config
from data import make_dataloaders
from logger import RunLogger
from model import build_model
from train import fit, evaluate


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser()
    p.add_argument("--probe", choices=["final", "weighted"], default=cfg.probe_type,
                   help="Probe head type: linear on a single layer (default: final), "
                        "or linear on a learnable softmax mixture of all layers.")
    p.add_argument("--epochs", type=int, default=cfg.num_epochs)
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--eval_batch_size", type=int, default=cfg.eval_batch_size,
                   help="Batch size for val/test loaders. Can be 2-4x train batch size.")
    p.add_argument("--lr", type=float, default=cfg.learning_rate)
    p.add_argument("--spear_model_id", default=cfg.spear_model_id,
                   help="HuggingFace model id or local path for SPEAR.")
    p.add_argument("--train_hours", type=float, default=cfg.train_hours)
    p.add_argument("--data_cache_dir", default=str(cfg.data_cache_dir))
    p.add_argument("--device", default=cfg.device)
    p.add_argument("--layer_idx", type=int, default=cfg.layer_idx,
                   help="For --probe final, which SPEAR layer to use (0-based, -1 for last).")
    p.add_argument("--warmup_steps", type=int, default=cfg.warmup_steps,
                   help="Number of warmup steps for the learning rate scheduler.")
    p.add_argument("--runs_dir", default="./runs",
                   help="Root directory for per-run logs (CSV/JSON for analysis).")
    args = p.parse_args()

    cfg.probe_type = args.probe
    cfg.num_epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.eval_batch_size = args.eval_batch_size
    cfg.learning_rate = args.lr
    cfg.spear_model_id = args.spear_model_id
    cfg.train_hours = args.train_hours
    cfg.data_cache_dir = Path(args.data_cache_dir)
    cfg.device = args.device
    cfg.layer_idx = args.layer_idx
    cfg.warmup_steps = args.warmup_steps
    cfg.runs_dir = Path(args.runs_dir)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    print(f"=== probe_type     : {cfg.probe_type}")
    print(f"=== spear_model_id : {cfg.spear_model_id}")
    print(f"=== train_hours    : {cfg.train_hours}")

    tokenizer, train_dl, val_dl, test_dl = make_dataloaders(cfg)
    encoder, probe = build_model(cfg)

    logger = RunLogger(root=cfg.runs_dir, probe_type=cfg.probe_type, cfg=cfg)

    fit(cfg, encoder, probe, tokenizer, train_dl, val_dl, logger=logger)

    print("\n=== Final test-set evaluation ===")
    metrics = evaluate(cfg, encoder, probe, tokenizer, test_dl,
                       label="test", epoch=cfg.num_epochs, logger=logger)
    print(f"test_cer {metrics['cer']:.4f}  test_wer {metrics['wer']:.4f}")

    layer_weights = None
    if cfg.probe_type == "weighted" and hasattr(probe, "layer_weights"):
        layer_weights = probe.layer_weights.tolist()
        print("Learned softmax weights over SPEAR layers (layer_idx: weight):")
        for i, w in enumerate(layer_weights):
            print(f"  layer {i:>2d}: {w:.4f}")

    logger.write_summary({
        "probe_type": cfg.probe_type,
        "spear_model_id": cfg.spear_model_id,
        "train_hours": cfg.train_hours,
        "test_cer": metrics["cer"],
        "test_wer": metrics["wer"],
        "layer_weights": layer_weights,
    })
    print(f"\n[run done] logs in {logger.dir}")


if __name__ == "__main__":
    main()
