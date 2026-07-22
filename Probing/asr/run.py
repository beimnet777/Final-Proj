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
    python run.py --probe fixed_weighted_lstm --model_id <hf-id>
"""



import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))
from config import Config
from data import make_dataloaders, make_cached_dataloaders
from logger import RunLogger
from model import build_model
from reproducibility import set_seed
from train import fit, evaluate


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser()
    p.add_argument("--probe", choices=["final", "weighted", "lstm", "weighted_lstm", "fixed_weighted_lstm"], default=cfg.probe_type,
                   help="Probe head: final (linear on last layer), weighted (softmax mix + linear), "
                        "lstm (BLSTM on last layer), weighted_lstm (softmax mix + BLSTM), "
                        "fixed_weighted_lstm (uniform layer average + BLSTM).")
    p.add_argument("--epochs", type=int, default=cfg.num_epochs)
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--eval_batch_size", type=int, default=cfg.eval_batch_size,
                   help="Batch size for val/test loaders. Can be 2-4x train batch size.")
    p.add_argument("--lr", type=float, default=cfg.learning_rate)
    p.add_argument("--model_id", default=cfg.model_id,
                   help="HuggingFace model id or local path for the encoder.")
    p.add_argument("--model_family", default=cfg.model_family,
                   choices=["spear", "hf", "disentanglement"],
                   help="Encoder family: 'spear' for SPEAR (custom API) or 'hf' for standard "
                        "HuggingFace speech encoders (wav2vec2, HuBERT, WavLM, …).")
    p.add_argument("--checkpoint_path", default=None,
                   help="Disentanglement checkpoint when --model_family=disentanglement.")
    p.add_argument("--representation_source", choices=["z_t", "z_L", "z_P"],
                   default=cfg.representation_source)
    p.add_argument("--train_hours", type=float, default=cfg.train_hours)
    p.add_argument("--data_cache_dir", default=str(cfg.data_cache_dir))
    p.add_argument("--local_data", action="store_true",
                   help="Read an extracted LibriSpeech tree instead of HF streaming.")
    p.add_argument("--librispeech_root", default=str(cfg.librispeech_root))
    p.add_argument("--max_examples", type=int, default=cfg.max_examples)
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
    p.add_argument("--checkpoint_dir", default=None,
                   help="Directory for the trained downstream probe checkpoint.")
    p.add_argument("--num_workers", type=int, default=cfg.num_workers)
    p.add_argument("--seed", type=int, default=cfg.seed,
                   help="Random seed for model init, shuffling, and augmentation.")
    args = p.parse_args()

    cfg.probe_type = args.probe
    cfg.num_epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.eval_batch_size = args.eval_batch_size
    cfg.learning_rate = args.lr
    cfg.model_id = args.model_id
    cfg.model_family = args.model_family
    cfg.checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    cfg.representation_source = args.representation_source
    cfg.train_hours = args.train_hours
    cfg.data_cache_dir = Path(args.data_cache_dir)
    cfg.local_data = args.local_data
    cfg.librispeech_root = Path(args.librispeech_root)
    cfg.max_examples = args.max_examples
    cfg.device = args.device
    cfg.layer_idx = args.layer_idx
    cfg.warmup_steps = args.warmup_steps
    cfg.lstm_hidden = args.lstm_hidden
    cfg.lstm_layers = args.lstm_layers
    cfg.runs_dir = Path(args.runs_dir)
    cfg.checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else cfg.checkpoint_dir
    cfg.num_workers = args.num_workers
    cfg.feature_cache_dir = Path(args.feature_cache_dir) if args.feature_cache_dir else None
    cfg.seed = args.seed
    return cfg


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    print(f"=== probe_type     : {cfg.probe_type}")
    print(f"=== model_id       : {cfg.model_id}")
    print(f"=== model_family   : {cfg.model_family}")
    if cfg.model_family == "disentanglement":
        print(f"=== source         : {cfg.representation_source}")
        print(f"=== encoder ckpt   : {cfg.checkpoint_path}")
    print(f"=== train_hours    : {cfg.train_hours}")
    print(f"=== seed           : {cfg.seed}")

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
    if hasattr(probe, "layer_weights"):
        layer_weights = probe.layer_weights.tolist()
        weight_label = "Layer weights over SPEAR layers"
        if cfg.probe_type == "fixed_weighted_lstm":
            weight_label += " (fixed uniform)"
        else:
            weight_label += " (learned softmax)"
        print(f"{weight_label}:")
        for i, w in enumerate(layer_weights):
            print(f"  layer {i:>2d}: {w:.4f}")

    logger.write_summary({
        "probe_type": cfg.probe_type,
        "model_id": cfg.model_id,
        "model_family": cfg.model_family,
        "encoder_checkpoint": str(cfg.checkpoint_path) if cfg.checkpoint_path else None,
        "representation_source": cfg.representation_source,
        "train_hours": cfg.train_hours,
        "seed": cfg.seed,
        "deterministic": True,
        "num_workers": cfg.num_workers,
        "test_cer": metrics["cer"],
        "test_wer": metrics["wer"],
        "layer_weights": layer_weights,
    })
    logger.close()
    print(f"\n[run done] logs in {logger.dir}")


if __name__ == "__main__":
    main()
