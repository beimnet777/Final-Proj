"""CLI entry point for the SAE reconstruction system.

Usage
-----
    python run.py

    # Override key hyperparameters
    python run.py --total_steps 6000 --K 5120 --topk 256

    # Smoke-test with tiny data
    python run.py --total_steps 20 --max_train_examples 50 --max_val_examples 20
"""

from __future__ import annotations

import argparse
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.*")

sys.path.insert(0, str(Path(__file__).parent))

from config import DISConfig
from train import run


def _parse_args() -> DISConfig:
    cfg = DISConfig()
    p = argparse.ArgumentParser(
        description="SAE reconstruction on SPEAR-Large final-layer features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- data
    p.add_argument("--librispeech_cache_dir", default=str(cfg.librispeech_cache_dir))
    p.add_argument("--max_train_examples",    type=int,   default=cfg.max_train_examples,
                   help="0 = full train-clean-100 (~28 k)")
    p.add_argument("--max_val_examples",      type=int,   default=cfg.max_val_examples)

    # ---- model
    p.add_argument("--spear_model_id", default=cfg.spear_model_id)
    p.add_argument("--K",    type=int,   default=cfg.K,    help="SAE latent size")
    p.add_argument("--topk", type=int,   default=cfg.topk, help="Active features per frame")

    # ---- training schedule
    p.add_argument("--total_steps",  type=int,   default=cfg.total_steps)
    p.add_argument("--batch_size",   type=int,   default=cfg.batch_size)
    p.add_argument("--lr",           type=float, default=cfg.lr)
    p.add_argument("--lr_min",       type=float, default=cfg.lr_min)
    p.add_argument("--warmup_steps", type=int,   default=cfg.warmup_steps)

    # ---- paths
    p.add_argument("--checkpoint_dir", default=str(cfg.checkpoint_dir))
    p.add_argument("--runs_dir",       default=str(cfg.runs_dir))
    p.add_argument("--log_dir",        default=str(cfg.log_dir))

    # ---- misc
    p.add_argument("--seed",        type=int, default=cfg.seed)
    p.add_argument("--num_workers", type=int, default=cfg.num_workers)
    p.add_argument("--no_bf16",     action="store_true")

    args = p.parse_args()

    cfg.librispeech_cache_dir = Path(args.librispeech_cache_dir)
    cfg.max_train_examples    = args.max_train_examples
    cfg.max_val_examples      = args.max_val_examples
    cfg.spear_model_id        = args.spear_model_id
    cfg.K                     = args.K
    cfg.topk                  = args.topk
    cfg.total_steps           = args.total_steps
    cfg.batch_size            = args.batch_size
    cfg.lr                    = args.lr
    cfg.lr_min                = args.lr_min
    cfg.warmup_steps          = args.warmup_steps
    cfg.checkpoint_dir        = Path(args.checkpoint_dir)
    cfg.runs_dir              = Path(args.runs_dir)
    cfg.log_dir               = Path(args.log_dir)
    cfg.seed                  = args.seed
    cfg.num_workers           = args.num_workers
    cfg.bf16                  = not args.no_bf16
    cfg.device                = "cuda" if torch.cuda.is_available() else "cpu"

    return cfg


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    cfg = _parse_args()
    _seed_all(cfg.seed)

    print(f"=== SAE Reconstruction")
    print(f"=== device   : {cfg.device}")
    print(f"=== bf16     : {cfg.bf16}")
    print(f"=== K={cfg.K}  topk={cfg.topk}  D={cfg.D}")
    print(f"=== steps    : {cfg.total_steps}  batch={cfg.batch_size}")
    print(f"=== lr       : {cfg.lr:.1e} → {cfg.lr_min:.1e}  warmup={cfg.warmup_steps}")
    print(f"=== train_ex : {cfg.max_train_examples or 'all'}")
    print(f"=== ckpt_dir : {cfg.checkpoint_dir}")

    best_ckpt = run(cfg)
    print(f"\n[done]  best checkpoint → {best_ckpt}")


if __name__ == "__main__":
    main()
