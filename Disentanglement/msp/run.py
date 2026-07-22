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
    p.add_argument("--lexicon_path", default=d.lexicon_path,
                   help="Pronunciation lexicon for transcript-derived PR targets.")
    p.add_argument("--spear_revision", default="",
                   help="optional immutable Hugging Face SPEAR commit")
    # schedule / scale
    p.add_argument("--steps", type=int, default=d.steps)
    p.add_argument("--warmup_steps", type=int, default=d.warmup_steps)
    p.add_argument("--dann_ramp_steps", type=int, default=d.dann_ramp_steps,
                   help="sigmoid DANN/GRL ramp length; reaches 1.0 by this step")
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
    p.add_argument("--grad_log_every", type=int, default=d.grad_log_every,
                   help="print detailed gradient diagnostics every N steps")
    p.add_argument("--ckpt_every", type=int, default=d.ckpt_every)
    p.add_argument("--freeze_learned_routing_on_resume", action="store_true",
                   default=d.freeze_learned_routing_on_resume,
                   help="after exact resume, freeze the learned static routing logits")
    p.add_argument("--freeze_route_topk_on_resume", action="store_true",
                   default=d.freeze_route_topk_on_resume,
                   help="after learned-route freeze, calibrate/enforce route-local TopK quotas")
    p.add_argument("--route_topk_calib_batches", type=int,
                   default=d.route_topk_calib_batches)
    p.add_argument("--fixed_blocks", action=argparse.BooleanOptionalAction,
                   default=d.fixed_blocks,
                   help="use fixed contiguous L/P blocks instead of learned routing")
    p.add_argument("--K_L", type=int, default=d.K_L)
    p.add_argument("--K_P", type=int, default=d.K_P)
    p.add_argument("--K_U", type=int, default=d.K_U)
    p.add_argument("--per_block_topk", action=argparse.BooleanOptionalAction,
                   default=d.per_block_topk,
                   help="enforce separate active-unit quotas inside each fixed block")
    p.add_argument("--topk_L", type=int, default=d.topk_L)
    p.add_argument("--topk_P", type=int, default=d.topk_P)
    p.add_argument("--topk_U", type=int, default=d.topk_U)
    # gradient-conflict
    p.add_argument("--no_pcgrad", action="store_true", help="disable PCGrad surgery")
    p.add_argument("--pcgrad_tasks", default=d.pcgrad_tasks)
    p.add_argument("--pcgrad_balance", choices=("none", "unit"),
                   default=d.pcgrad_balance,
                   help="balance cooperative SAE gradient norms before PCGrad")
    p.add_argument(
        "--adversary_balance", choices=("none", "unit_preserve_bundle"),
        default=d.adversary_balance,
        help=("balance factor-level adversarial SAE gradients, while preserving "
              "the original weighted adversary-bundle norm"),
    )
    p.add_argument("--separate_discriminator_optimizer",
                   action=argparse.BooleanOptionalAction,
                   default=d.separate_discriminator_optimizer,
                   help="update discriminator heads only with a detached-representation optimizer")
    p.add_argument("--separate_grad_clip", action=argparse.BooleanOptionalAction,
                   default=d.separate_grad_clip,
                   help="clip SAE, routing, positive-head, and discriminator groups separately")
    p.add_argument("--aux_k", type=int, default=d.aux_k,
                   help="number of dead SAE units used for residual reconstruction (0 disables)")
    p.add_argument("--aux_k_coef", type=float, default=d.aux_k_coef)
    p.add_argument("--dead_steps_threshold", type=int, default=d.dead_steps_threshold)
    p.add_argument("--valid_frame_dead_count", action=argparse.BooleanOptionalAction,
                   default=d.valid_frame_dead_count,
                   help="exclude padded frames when updating SAE dead-unit counters")
    p.add_argument("--grl_grad_norm", action=argparse.BooleanOptionalAction,
                   default=d.grl_grad_norm,
                   help="normalize the z_L speaker-GRL gradient per frame")
    p.add_argument("--grl_grad_norm_target", type=float, default=d.grl_grad_norm_target)
    p.add_argument("--grl_emotion_grad_norm", action=argparse.BooleanOptionalAction,
                   default=d.grl_emotion_grad_norm,
                   help="normalize the z_L emotion-GRL gradient per frame")
    p.add_argument("--grl_emotion_grad_norm_target", type=float,
                   default=d.grl_emotion_grad_norm_target)
    # task weights
    p.add_argument("--recon_weight", type=float, default=d.recon_weight)
    p.add_argument("--alpha", type=float, default=d.alpha)
    p.add_argument("--beta", type=float, default=d.beta)
    p.add_argument("--grl_weight", type=float, default=d.grl_weight)
    p.add_argument("--grl_phoneme_weight", type=float, default=d.grl_phoneme_weight)
    p.add_argument("--prosody_weight", type=float, default=d.prosody_weight)
    p.add_argument("--grl_prosody_weight", type=float, default=d.grl_prosody_weight)
    p.add_argument("--emotion_weight", type=float, default=d.emotion_weight)
    p.add_argument("--grl_emotion_weight", type=float, default=d.grl_emotion_weight)
    p.add_argument("--inv_weight", type=float, default=d.inv_weight)
    p.add_argument("--no_invariance", action="store_true",
                   help="disable perturbation generation and the invariance objective")
    # misc
    p.add_argument("--run_name", default="msp_v1")
    p.add_argument("--checkpoint_dir", default=None)
    p.add_argument("--stage1_ckpt", default=None,
                   help="optional SAE init from a stage-1 checkpoint (default: from scratch)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny dry-run: 3 steps, eval every 3, to validate wiring")
    p.add_argument("--resume", default="none")
    p.add_argument("--segment_steps", type=int, default=0)
    p.add_argument("--max_runtime_minutes", type=float, default=0.0)
    p.add_argument("--resume_every", type=int, default=0)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    p.add_argument("--dataset_fingerprint", default="")
    p.add_argument("--experiment_preset", default="")
    p.add_argument("--drive_mirror", default="")
    a = p.parse_args()

    m = MSPConfig(
        manifest=a.manifest, audio_root=a.audio_root, transcripts=a.transcripts,
        lexicon_path=a.lexicon_path,
        steps=a.steps, warmup_steps=a.warmup_steps, dann_ramp_steps=a.dann_ramp_steps,
        batch_size=a.batch_size,
        eval_batch=a.eval_batch, num_workers=a.num_workers, seed=a.seed,
        hard_routing=not a.soft_routing,
        lr=a.lr, lr_min=a.lr_min, lr_heads=a.lr_heads, lr_disc=a.lr_disc,
        lr_routing=a.lr_routing, n_disc_steps=a.n_disc_steps,
        grad_clip=a.grad_clip, routing_init_std=a.routing_init_std,
        routing_spec_weight=a.routing_spec_weight, routing_tau=a.routing_tau,
        log_every=a.log_every, grad_log_every=a.grad_log_every,
        ckpt_every=a.ckpt_every,
        freeze_learned_routing_on_resume=a.freeze_learned_routing_on_resume,
        freeze_route_topk_on_resume=a.freeze_route_topk_on_resume,
        route_topk_calib_batches=a.route_topk_calib_batches,
        fixed_blocks=a.fixed_blocks,
        K_L=a.K_L, K_P=a.K_P, K_U=a.K_U,
        per_block_topk=a.per_block_topk,
        topk_L=a.topk_L, topk_P=a.topk_P, topk_U=a.topk_U,
        pcgrad=not a.no_pcgrad, pcgrad_tasks=a.pcgrad_tasks,
        pcgrad_balance=a.pcgrad_balance,
        adversary_balance=a.adversary_balance,
        separate_discriminator_optimizer=a.separate_discriminator_optimizer,
        separate_grad_clip=a.separate_grad_clip,
        aux_k=a.aux_k, aux_k_coef=a.aux_k_coef,
        dead_steps_threshold=a.dead_steps_threshold,
        valid_frame_dead_count=a.valid_frame_dead_count,
        grl_grad_norm=a.grl_grad_norm,
        grl_grad_norm_target=a.grl_grad_norm_target,
        grl_emotion_grad_norm=a.grl_emotion_grad_norm,
        grl_emotion_grad_norm_target=a.grl_emotion_grad_norm_target,
        recon_weight=a.recon_weight,
        alpha=a.alpha, beta=a.beta, grl_weight=a.grl_weight,
        grl_phoneme_weight=a.grl_phoneme_weight, prosody_weight=a.prosody_weight,
        grl_prosody_weight=a.grl_prosody_weight, emotion_weight=a.emotion_weight,
        grl_emotion_weight=a.grl_emotion_weight, inv_weight=a.inv_weight,
    )
    cfg = to_dis_cfg(m)
    cfg.spear_revision = a.spear_revision
    if a.no_invariance:
        cfg.invariance = False
        cfg.inv_weight = 0.0
    cfg.checkpoint_dir = Path(a.checkpoint_dir) if a.checkpoint_dir else \
        Path(__file__).resolve().parent / "checkpoints" / a.run_name
    cfg.resume = a.resume
    cfg.segment_steps = a.segment_steps
    cfg.max_runtime_minutes = a.max_runtime_minutes
    cfg.resume_every = a.resume_every
    cfg.gradient_accumulation_steps = max(1, a.gradient_accumulation_steps)
    cfg.precision = a.precision
    cfg.dataset_fingerprint = a.dataset_fingerprint
    cfg.experiment_preset = a.experiment_preset
    cfg.drive_mirror = a.drive_mirror
    if not Path(cfg.lexicon_path).is_file():
        p.error(f"--lexicon_path does not exist: {cfg.lexicon_path}")
    if not cfg.spear_revision and cfg.resume not in {"none", ""}:
        _rp = cfg.checkpoint_dir / "latest-resume.pt" if cfg.resume == "auto" else Path(cfg.resume)
        if _rp.exists():
            import torch
            _meta = torch.load(_rp, map_location="cpu", weights_only=False)
            cfg.spear_revision = str(_meta.get("analysis_config", {}).get("spear_revision", ""))
    if a.smoke:
        cfg.stage2_steps = 3
        cfg.warmup_steps = 1
        cfg.dann_ramp_steps = 1
        cfg.ckpt_every = 3
        cfg.log_every = 1
        cfg.grad_log_every = 1
    if cfg.freeze_route_topk_on_resume and not cfg.freeze_learned_routing_on_resume:
        p.error("--freeze_route_topk_on_resume requires --freeze_learned_routing_on_resume")
    if cfg.route_topk_calib_batches <= 0:
        p.error("--route_topk_calib_batches must be positive")
    if cfg.fixed_blocks:
        if cfg.freeze_learned_routing_on_resume or cfg.freeze_route_topk_on_resume:
            p.error("fixed blocks cannot be combined with learned-route freeze options")
        block_sizes = (cfg.K_L, cfg.K_P, cfg.K_U)
        block_topks = (cfg.topk_L, cfg.topk_P, cfg.topk_U)
        if any(size < 0 for size in block_sizes):
            p.error(f"fixed block sizes must be non-negative, got {block_sizes}")
        if sum(block_sizes) != cfg.K:
            p.error(f"K_L+K_P+K_U must equal K={cfg.K}, got {sum(block_sizes)}")
        if any(k < 0 or k > size for k, size in zip(block_topks, block_sizes)):
            p.error(f"each fixed-block TopK must lie within its block: "
                    f"sizes={block_sizes}, topks={block_topks}")
        if cfg.per_block_topk and sum(block_topks) != cfg.topk:
            p.error(f"topk_L+topk_P+topk_U must equal topk={cfg.topk}, "
                    f"got {sum(block_topks)}")
    if cfg.aux_k < 0:
        p.error("--aux_k must be non-negative")
    if cfg.aux_k_coef < 0:
        p.error("--aux_k_coef must be non-negative")
    if cfg.dead_steps_threshold < 0:
        p.error("--dead_steps_threshold must be non-negative")
    if cfg.adversary_balance != "none" and not cfg.pcgrad:
        p.error("--adversary_balance requires PCGrad (remove --no_pcgrad)")
    routing_name = "fixed-block" if cfg.fixed_blocks else ("hard" if m.hard_routing else "soft")
    print(f"=== MSP run '{a.run_name}'  pcgrad={cfg.pcgrad}  routing={routing_name} ===")
    train.run(cfg, stage1_ckpt=a.stage1_ckpt)


if __name__ == "__main__":
    main()
