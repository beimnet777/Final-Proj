"""CLI entry point — single-stage disentanglement (old+Gao architecture)."""

from __future__ import annotations

import argparse, random, sys, warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.*")
sys.path.insert(0, str(Path(__file__).parent))

from config import DISConfig
from train import run


def _parse_args():
    cfg = DISConfig()
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data
    p.add_argument("--max_train_examples", type=int, default=cfg.max_train_examples)
    p.add_argument("--max_val_examples",   type=int, default=cfg.max_val_examples)

    # model
    p.add_argument("--topk",  type=int,   default=cfg.topk)

    # loss weights
    p.add_argument("--alpha", type=float, default=cfg.alpha)
    p.add_argument("--beta",  type=float, default=cfg.beta)
    p.add_argument("--delta", type=float, default=cfg.delta)
    p.add_argument("--rho",   type=float, default=cfg.rho)

    # training
    p.add_argument("--total_steps", type=int,   default=cfg.total_steps)
    p.add_argument("--batch_size",  type=int,   default=cfg.batch_size)
    p.add_argument("--lr_enc_dec",  type=float, default=cfg.lr_enc_dec)
    p.add_argument("--lr_routing",  type=float, default=cfg.lr_routing)

    # paths
    p.add_argument("--checkpoint_dir", default=str(cfg.checkpoint_dir))
    p.add_argument("--runs_dir",       default=str(cfg.runs_dir))
    p.add_argument("--log_dir",        default=str(cfg.log_dir))

    p.add_argument("--seed",    type=int, default=cfg.seed)
    p.add_argument("--no_bf16", action="store_true")

    args = p.parse_args()

    cfg.max_train_examples = args.max_train_examples
    cfg.max_val_examples   = args.max_val_examples
    cfg.topk               = args.topk
    cfg.alpha              = args.alpha
    cfg.beta               = args.beta
    cfg.delta              = args.delta
    cfg.rho                = args.rho
    cfg.total_steps        = args.total_steps
    cfg.batch_size         = args.batch_size
    cfg.lr_enc_dec         = args.lr_enc_dec
    cfg.lr_routing         = args.lr_routing
    cfg.checkpoint_dir     = Path(args.checkpoint_dir)
    cfg.runs_dir           = Path(args.runs_dir)
    cfg.log_dir            = Path(args.log_dir)
    cfg.seed               = args.seed
    cfg.bf16               = not args.no_bf16
    cfg.device             = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def main():
    cfg = _parse_args()
    random.seed(cfg.seed); np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)

    print(f"=== Old+Gao Disentanglement — single stage")
    print(f"=== device={cfg.device}  bf16={cfg.bf16}")
    print(f"=== K={cfg.K}  topk={cfg.topk}  D={cfg.D}")
    print(f"=== alpha={cfg.alpha}  beta={cfg.beta}  delta={cfg.delta}  rho={cfg.rho}")
    print(f"=== total_steps={cfg.total_steps}  train_examples={cfg.max_train_examples or 'all'}")
    print(f"=== checkpoint_dir={cfg.checkpoint_dir}")

    best_ckpt = run(cfg)
    print(f"\n[done]  best checkpoint → {best_ckpt}")


if __name__ == "__main__":
    main()
