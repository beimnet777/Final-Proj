"""CLI entry point for the disentanglement system.

Usage
-----
    # Stage 1 — SAE reconstruction
    python run.py --stage 1

    # Stage 2 — full disentanglement (calibration: few hundred steps, alpha=beta=grl=1)
    python run.py --stage 2 --stage1_ckpt checkpoints/stage1_best.pt \
                  --stage2_steps 500 --grad_log_every 50

    # Stage 2 — full training with calibrated weights
    python run.py --stage 2 --stage1_ckpt checkpoints/stage1_best.pt \
                  --stage2_steps 8000 --alpha 0.1 --beta 0.3 --grl_weight 0.2

    # Stage 2 from scratch — train SAE + routing + heads in one run
    python run.py --stage 2 --stage2_from_scratch --stage2_steps 8000 \
                  --alpha 0.02 --beta 0.01 --grl_weight 0.01

    # Smoke-test
    python run.py --stage 1 --total_steps 20 --max_train_examples 50 --max_val_examples 20
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
from train import run_stage1, run_stage2


def _parse_args():
    cfg = DISConfig()
    p   = argparse.ArgumentParser(
        description="Disentanglement system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--stage", type=int, choices=[1, 2], required=True)
    p.add_argument("--stage1_ckpt", default=None,
                   help="Path to stage-1 best checkpoint (required for --stage 2)")
    p.add_argument("--stage2_from_scratch", action="store_true",
                   help="For --stage 2, skip loading stage-1 SAE weights")

    # data
    p.add_argument("--librispeech_cache_dir", default=str(cfg.librispeech_cache_dir))
    p.add_argument("--lexicon_path",          default=str(cfg.lexicon_path))
    p.add_argument("--max_train_examples",    type=int,   default=cfg.max_train_examples)
    p.add_argument("--max_val_examples",      type=int,   default=cfg.max_val_examples)

    # model
    p.add_argument("--spear_model_id", default=cfg.spear_model_id)
    p.add_argument("--K",    type=int, default=cfg.K)
    p.add_argument("--topk", type=int, default=cfg.topk)

    # loss weights (stage 2)
    p.add_argument("--alpha",           type=float, default=cfg.alpha)
    p.add_argument("--beta",            type=float, default=cfg.beta)
    p.add_argument("--grl_weight",      type=float, default=cfg.grl_weight)
    p.add_argument("--grl_delay_steps", type=int,   default=cfg.grl_delay_steps)
    p.add_argument("--rho",             type=float, default=cfg.rho)

    # ablation flags (D / E / F)
    p.add_argument("--no_routing",          action="store_true", default=cfg.no_routing)
    p.add_argument("--fixed_routing",       action="store_true", default=cfg.fixed_routing)
    p.add_argument("--fixed_routing_split", type=float,          default=cfg.fixed_routing_split)
    p.add_argument("--n_routes",            type=int,            default=cfg.n_routes)
    p.add_argument("--pre_topk_routing",    action="store_true", default=cfg.pre_topk_routing)
    p.add_argument("--hard_gumbel_routing", action="store_true", default=cfg.hard_gumbel_routing,
                   help="Use one-hot straight-through Gumbel masks during training.")

    # experiment flags
    p.add_argument("--grl_phoneme_weight",  type=float, default=cfg.grl_phoneme_weight)
    p.add_argument("--decor_weight",        type=float, default=cfg.decor_weight)
    p.add_argument("--ub_weight",           type=float, default=cfg.ub_weight)
    p.add_argument("--ste_routing",         action="store_true", default=cfg.ste_routing)
    p.add_argument("--projection_disentanglement", action="store_true",
                   default=cfg.projection_disentanglement,
                   help="Use learned compressed z_t->z_L/z_P projections instead of routing masks.")
    p.add_argument("--projection_dim", type=int, default=cfg.projection_dim,
                   help="Output dimension for projection_disentanglement.")
    p.add_argument("--projection_reconstruct", action="store_true",
                   default=cfg.projection_reconstruct,
                   help="Reconstruct h_t solely through z_L/z_P (up-project + decode), no decode(z_t).")
    p.add_argument("--projection_u_dim", type=int, default=cfg.projection_u_dim,
                   help="Dim of residual view z_U for reconstructive projection (0 = 2-way, no z_U).")
    p.add_argument("--projection_u_l2", type=float, default=cfg.projection_u_l2,
                   help="L2 activity penalty on z_U (the residual bottleneck).")
    p.add_argument("--spear_layernorm", action="store_true", default=cfg.spear_layernorm,
                   help="LayerNorm each SPEAR layer before averaging (SUPERB-comparable h_t).")

    # schedule
    p.add_argument("--total_steps",   type=int,   default=cfg.total_steps)
    p.add_argument("--stage2_steps",  type=int,   default=cfg.stage2_steps)
    p.add_argument("--warmup_steps",  type=int,   default=cfg.warmup_steps)
    p.add_argument("--batch_size",    type=int,   default=cfg.batch_size)
    p.add_argument("--lr",            type=float, default=cfg.lr)
    p.add_argument("--lr_min",        type=float, default=cfg.lr_min)
    p.add_argument("--lr_routing",    type=float, default=cfg.lr_routing)
    p.add_argument("--lr_heads",      type=float, default=cfg.lr_heads)
    p.add_argument("--grad_log_every",type=int,   default=cfg.grad_log_every)

    # paths
    p.add_argument("--checkpoint_dir", default=str(cfg.checkpoint_dir))
    p.add_argument("--runs_dir",       default=str(cfg.runs_dir))
    p.add_argument("--log_dir",        default=str(cfg.log_dir))

    # misc
    p.add_argument("--seed",        type=int, default=cfg.seed)
    p.add_argument("--num_workers", type=int, default=cfg.num_workers)
    p.add_argument("--no_bf16",     action="store_true")

    args = p.parse_args()

    cfg.librispeech_cache_dir = Path(args.librispeech_cache_dir)
    cfg.lexicon_path          = Path(args.lexicon_path)
    cfg.max_train_examples    = args.max_train_examples
    cfg.max_val_examples      = args.max_val_examples
    cfg.spear_model_id        = args.spear_model_id
    cfg.K                     = args.K
    cfg.topk                  = args.topk
    cfg.alpha                 = args.alpha
    cfg.beta                  = args.beta
    cfg.grl_weight            = args.grl_weight
    cfg.grl_delay_steps       = args.grl_delay_steps
    cfg.rho                   = args.rho
    cfg.no_routing            = args.no_routing
    cfg.fixed_routing         = args.fixed_routing
    cfg.fixed_routing_split   = args.fixed_routing_split
    cfg.n_routes              = args.n_routes
    cfg.pre_topk_routing      = args.pre_topk_routing
    cfg.hard_gumbel_routing   = args.hard_gumbel_routing
    cfg.grl_phoneme_weight    = args.grl_phoneme_weight
    cfg.decor_weight          = args.decor_weight
    cfg.ub_weight             = args.ub_weight
    cfg.ste_routing           = args.ste_routing
    cfg.projection_disentanglement = args.projection_disentanglement
    cfg.projection_dim        = args.projection_dim
    cfg.projection_reconstruct = args.projection_reconstruct
    cfg.projection_u_dim      = args.projection_u_dim
    cfg.projection_u_l2       = args.projection_u_l2
    cfg.spear_layernorm       = args.spear_layernorm
    cfg.total_steps           = args.total_steps
    cfg.stage2_steps          = args.stage2_steps
    cfg.warmup_steps          = args.warmup_steps
    cfg.batch_size            = args.batch_size
    cfg.lr                    = args.lr
    cfg.lr_min                = args.lr_min
    cfg.lr_routing            = args.lr_routing
    cfg.lr_heads              = args.lr_heads
    cfg.grad_log_every        = args.grad_log_every
    cfg.checkpoint_dir        = Path(args.checkpoint_dir)
    cfg.runs_dir              = Path(args.runs_dir)
    cfg.log_dir               = Path(args.log_dir)
    cfg.seed                  = args.seed
    cfg.num_workers           = args.num_workers
    cfg.bf16                  = not args.no_bf16
    cfg.device                = "cuda" if torch.cuda.is_available() else "cpu"

    return cfg, args.stage, args.stage1_ckpt, args.stage2_from_scratch


def _seed_all(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main() -> None:
    cfg, stage, stage1_ckpt, stage2_from_scratch = _parse_args()
    _seed_all(cfg.seed)

    print(f"=== Disentanglement  stage={stage}")
    print(f"=== device={cfg.device}  bf16={cfg.bf16}")
    print(f"=== K={cfg.K}  topk={cfg.topk}  D={cfg.D}")
    if stage == 1:
        print(f"=== steps={cfg.total_steps}  lr={cfg.lr:.1e}→{cfg.lr_min:.1e}")
    else:
        print(f"=== steps={cfg.stage2_steps}  α={cfg.alpha}  β={cfg.beta}  grl={cfg.grl_weight}  ρ={cfg.rho}")

    if stage == 1:
        best = run_stage1(cfg)
    else:
        if stage1_ckpt is None and not stage2_from_scratch:
            raise ValueError("--stage1_ckpt required for stage 2")
        if cfg.stage2_steps == 0:
            raise ValueError("--stage2_steps required for stage 2")
        best = run_stage2(cfg, None if stage2_from_scratch else Path(stage1_ckpt))

    print(f"\n[done]  best checkpoint → {best}")


if __name__ == "__main__":
    main()
