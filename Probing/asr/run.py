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
    python run.py --probe final    --model_id <hf-id>
    python run.py --probe weighted --model_id <hf-id>  --epochs 30
"""



import argparse
import random
from pathlib import Path

import numpy as np
import torch

from config import Config
from data import make_dataloaders, make_cached_dataloaders
from logger import RunLogger
from model import build_model
from train import fit, evaluate


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser()
    p.add_argument("--probe", choices=["final", "weighted", "lstm", "weighted_lstm"], default=cfg.probe_type,
                   help="Probe head: final (linear on last layer), weighted (softmax mix + linear), "
                        "lstm (BLSTM on last layer), weighted_lstm (softmax mix + BLSTM).")
    p.add_argument("--epochs", type=int, default=cfg.num_epochs)
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--eval_batch_size", type=int, default=cfg.eval_batch_size,
                   help="Batch size for val/test loaders. Can be 2-4x train batch size.")
    p.add_argument("--lr", type=float, default=cfg.learning_rate)
    p.add_argument("--model_id", default=cfg.model_id,
                   help="HuggingFace model id or local path for the encoder.")
    p.add_argument("--model_family", default=cfg.model_family,
                   choices=["spear", "hf"],
                   help="Encoder family: 'spear' for SPEAR (custom API) or 'hf' for standard "
                        "HuggingFace speech encoders (wav2vec2, HuBERT, WavLM, …).")
    p.add_argument("--train_hours", type=float, default=cfg.train_hours)
    p.add_argument("--data_cache_dir", default=str(cfg.data_cache_dir))
    p.add_argument("--device", default=cfg.device)
    p.add_argument("--layer_idx", type=int, default=cfg.layer_idx,
                   help="For --probe final, which SPEAR layer to use (0-based, -1 for last).")
    p.add_argument("--warmup_steps", type=int, default=cfg.warmup_steps,
                   help="Number of warmup steps for the learning rate scheduler.")
    p.add_argument("--lstm_hidden", type=int, default=cfg.lstm_hidden,
                   help="Hidden units per direction for --probe lstm.")
    p.add_argument("--lstm_layers", type=int, default=cfg.lstm_layers,
                   help="Number of LSTM layers for --probe lstm.")
    p.add_argument("--runs_dir", default="./runs",
                   help="Root directory for per-run logs (CSV/JSON for analysis).")
    p.add_argument("--feature_cache_dir", default=None,
                   help="If set, load pre-extracted SPEAR features from this directory "
                        "instead of running the encoder. Created by cache_features.py.")
    args = p.parse_args()

    cfg.probe_type = args.probe
    cfg.num_epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.eval_batch_size = args.eval_batch_size
    cfg.learning_rate = args.lr
    cfg.model_id = args.model_id
    cfg.model_family = args.model_family
    cfg.train_hours = args.train_hours
    cfg.data_cache_dir = Path(args.data_cache_dir)
    cfg.device = args.device
    cfg.layer_idx = args.layer_idx
    cfg.warmup_steps = args.warmup_steps
    cfg.lstm_hidden = args.lstm_hidden
    cfg.lstm_layers = args.lstm_layers
    cfg.runs_dir = Path(args.runs_dir)
    cfg.feature_cache_dir = Path(args.feature_cache_dir) if args.feature_cache_dir else None
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
    print(f"=== model_id       : {cfg.model_id}")
    print(f"=== train_hours    : {cfg.train_hours}")

    if getattr(cfg, "feature_cache_dir", None):
        print(f"=== feature_cache  : {cfg.feature_cache_dir}  (cached mode — encoder not loaded)")
        tokenizer, train_dl, val_dl, test_dl = make_cached_dataloaders(cfg, cfg.feature_cache_dir)
        # Only the probe head is needed; encoder stays None.
        _encoder_unused, probe = build_model(cfg)
        encoder = None
    else:
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
        "model_id": cfg.model_id,
        "train_hours": cfg.train_hours,
        "test_cer": metrics["cer"],
        "test_wer": metrics["wer"],
        "layer_weights": layer_weights,
    })
    print(f"\n[run done] logs in {logger.dir}")


if __name__ == "__main__":
    main()
