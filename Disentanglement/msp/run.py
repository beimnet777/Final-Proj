#!/usr/bin/env python3
"""CLI entry point for the standalone MSP-Podcast disentanglement run.

Run from the Disentanglement/ directory:
    python -m msp.run --steps 12000 --run_name msp_v1
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import MSPConfig, to_dis_cfg
from . import train


def main() -> None:
    d = MSPConfig()
    p = argparse.ArgumentParser(description=__doc__)
    # data
    p.add_argument("--manifest", default=d.manifest)
    p.add_argument("--audio_root", default=d.audio_root)
    p.add_argument("--transcripts", default=d.transcripts)
    # schedule / scale
    p.add_argument("--steps", type=int, default=d.steps)
    p.add_argument("--warmup_steps", type=int, default=d.warmup_steps)
    p.add_argument("--batch_size", type=int, default=d.batch_size)
    p.add_argument("--eval_batch", type=int, default=d.eval_batch)
    p.add_argument("--num_workers", type=int, default=d.num_workers)
    p.add_argument("--soft_routing", action="store_true",
                   help="use soft Gumbel routing (default: hard ST-Gumbel)")
    p.add_argument("--seed", type=int, default=d.seed)
    # optimizer / routing
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--lr_min", type=float, default=d.lr_min)
    p.add_argument("--lr_heads", type=float, default=d.lr_heads)
    p.add_argument("--lr_disc", type=float, default=d.lr_disc)
    p.add_argument("--lr_routing", type=float, default=d.lr_routing)
    p.add_argument("--n_disc_steps", type=int, default=d.n_disc_steps)
    p.add_argument("--grad_clip", type=float, default=d.grad_clip)
    p.add_argument("--routing_init_std", type=float, default=d.routing_init_std)
    p.add_argument("--routing_spec_weight", type=float, default=d.routing_spec_weight)
    p.add_argument("--routing_tau", type=float, default=d.routing_tau)
    p.add_argument("--log_every", type=int, default=d.log_every)
    p.add_argument("--ckpt_every", type=int, default=d.ckpt_every)
    # gradient-conflict
    p.add_argument("--no_pcgrad", action="store_true", help="disable PCGrad surgery")
    p.add_argument("--pcgrad_tasks", default=d.pcgrad_tasks)
    p.add_argument("--grl_grad_norm", action="store_true",
                   help="normalize the z_L speaker-GRL gradient per frame")
    p.add_argument("--grl_grad_norm_target", type=float, default=d.grl_grad_norm_target)
    # task weights
    p.add_argument("--alpha", type=float, default=d.alpha)
    p.add_argument("--beta", type=float, default=d.beta)
    p.add_argument("--grl_weight", type=float, default=d.grl_weight)
    p.add_argument("--grl_phoneme_weight", type=float, default=d.grl_phoneme_weight)
    p.add_argument("--prosody_weight", type=float, default=d.prosody_weight)
    p.add_argument("--grl_prosody_weight", type=float, default=d.grl_prosody_weight)
    p.add_argument("--emotion_weight", type=float, default=d.emotion_weight)
    p.add_argument("--grl_emotion_weight", type=float, default=d.grl_emotion_weight)
    p.add_argument("--inv_weight", type=float, default=d.inv_weight)
    # misc
    p.add_argument("--run_name", default="msp_v1")
    p.add_argument("--checkpoint_dir", default=None)
    p.add_argument("--stage1_ckpt", default=None,
                   help="optional SAE init from a stage-1 checkpoint (default: from scratch)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny dry-run: 3 steps, eval every 3, to validate wiring")
    a = p.parse_args()

    m = MSPConfig(
        manifest=a.manifest, audio_root=a.audio_root, transcripts=a.transcripts,
        steps=a.steps, warmup_steps=a.warmup_steps, batch_size=a.batch_size,
        eval_batch=a.eval_batch, num_workers=a.num_workers, seed=a.seed,
        hard_routing=not a.soft_routing,
        lr=a.lr, lr_min=a.lr_min, lr_heads=a.lr_heads, lr_disc=a.lr_disc,
        lr_routing=a.lr_routing, n_disc_steps=a.n_disc_steps,
        grad_clip=a.grad_clip, routing_init_std=a.routing_init_std,
        routing_spec_weight=a.routing_spec_weight, routing_tau=a.routing_tau,
        log_every=a.log_every, ckpt_every=a.ckpt_every,
        pcgrad=not a.no_pcgrad, pcgrad_tasks=a.pcgrad_tasks,
        grl_grad_norm=a.grl_grad_norm,
        grl_grad_norm_target=a.grl_grad_norm_target,
        alpha=a.alpha, beta=a.beta, grl_weight=a.grl_weight,
        grl_phoneme_weight=a.grl_phoneme_weight, prosody_weight=a.prosody_weight,
        grl_prosody_weight=a.grl_prosody_weight, emotion_weight=a.emotion_weight,
        grl_emotion_weight=a.grl_emotion_weight, inv_weight=a.inv_weight,
    )
    cfg = to_dis_cfg(m)
    cfg.checkpoint_dir = Path(a.checkpoint_dir) if a.checkpoint_dir else \
        Path(__file__).resolve().parent / "checkpoints" / a.run_name
    if a.smoke:
        cfg.stage2_steps = 3
        cfg.warmup_steps = 1
        cfg.ckpt_every = 3
        cfg.log_every = 1
    print(f"=== MSP run '{a.run_name}'  pcgrad={cfg.pcgrad}  routing={'hard' if m.hard_routing else 'soft'} ===")
    train.run(cfg, stage1_ckpt=a.stage1_ckpt)


if __name__ == "__main__":
    main()
