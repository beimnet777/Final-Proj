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
    # gradient-conflict
    p.add_argument("--no_pcgrad", action="store_true", help="disable PCGrad surgery")
    p.add_argument("--pcgrad_tasks", default=d.pcgrad_tasks)
    p.add_argument("--gradnorm", action="store_true")
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
        pcgrad=not a.no_pcgrad, pcgrad_tasks=a.pcgrad_tasks, gradnorm=a.gradnorm,
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
