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
    p.add_argument("--local_data", action="store_true", default=cfg.local_data,
                   help="read raw flac from local disk instead of streaming from HF")
    p.add_argument("--librispeech_root", default=str(cfg.librispeech_root))
    p.add_argument("--train_split_dir", default=cfg.train_split_dir,
                   help="local training split dir (e.g. train-clean-100 or train-clean-360)")
    p.add_argument("--lexicon_path",          default=str(cfg.lexicon_path))
    p.add_argument("--max_train_examples",    type=int,   default=cfg.max_train_examples)
    p.add_argument("--max_val_examples",      type=int,   default=cfg.max_val_examples)
    p.add_argument("--max_test_examples",     type=int,   default=cfg.max_test_examples)

    # model
    p.add_argument("--spear_model_id", default=cfg.spear_model_id)
    p.add_argument("--K",    type=int, default=cfg.K)
    p.add_argument("--topk", type=int, default=cfg.topk)
    p.add_argument("--aux_k", type=int, default=cfg.aux_k,
                   help="AuxK dead-latent revival: # dead latents to model the residual (0=off, ~D/2 when scaling)")
    p.add_argument("--aux_k_coef", type=float, default=cfg.aux_k_coef)
    p.add_argument("--dead_steps_threshold", type=int, default=cfg.dead_steps_threshold)
    p.add_argument("--geom_median_bias", action="store_true", default=cfg.geom_median_bias)
    p.add_argument("--renorm_decoder", action="store_true", default=cfg.renorm_decoder)

    # loss weights (stage 2)
    p.add_argument("--alpha",           type=float, default=cfg.alpha)
    p.add_argument("--beta",            type=float, default=cfg.beta)
    p.add_argument("--gradnorm",        action="store_true", default=cfg.gradnorm,
                   help="learn the gradnorm_tasks weights online via GradNorm")
    p.add_argument("--gradnorm_alpha",  type=float, default=cfg.gradnorm_alpha)
    p.add_argument("--gradnorm_lr",     type=float, default=cfg.gradnorm_lr)
    p.add_argument("--gradnorm_tasks",  type=str,   default=cfg.gradnorm_tasks)
    p.add_argument("--gradnorm_every",  type=int,   default=cfg.gradnorm_every)
    p.add_argument("--grl_weight",      type=float, default=cfg.grl_weight)
    p.add_argument("--grl_delay_steps", type=int,   default=cfg.grl_delay_steps)
    p.add_argument("--grl_frame_level", action="store_true", default=cfg.grl_frame_level,
                   help="Speaker GRL predicts per-frame (dense gradient) instead of utterance mean-pool.")
    p.add_argument("--grl_attention_pool", action="store_true", default=cfg.grl_attention_pool,
                   help="Speaker GRL pools z_L with attentive statistics (weighted mean+std) instead of "
                        "flat mean — a stronger speaker discriminator.")
    p.add_argument("--grl_stats_pool", action="store_true", default=cfg.grl_stats_pool,
                   help="Speaker GRL uses diagnostic-style stats pooling: projector->ReLU->mean+std->linear.")
    p.add_argument("--grl_dense_context", action="store_true", default=cfg.grl_dense_context,
                   help="Speaker GRL predicts per-frame (dense) with a temporal conv for context — "
                        "gives z_L a dense per-frame removal gradient like grl_p.")
    p.add_argument("--grl_context_kernel", type=int, default=cfg.grl_context_kernel,
                   help="Temporal conv kernel (frames) for the dense-context speaker GRL.")
    p.add_argument("--grl_grad_norm", action="store_true", default=cfg.grl_grad_norm,
                   help="Per-frame normalize the reversed speaker gradient to a fixed magnitude "
                        "(decouples removal strength from discriminator confidence; counters dilution).")
    p.add_argument("--grl_grad_norm_target", type=float, default=cfg.grl_grad_norm_target,
                   help="Per-frame target L2 norm for grad-normalized GRL (effective push = grl_weight * this).")
    p.add_argument("--shuffle_grl_speaker_labels", action="store_true",
                   default=cfg.shuffle_grl_speaker_labels,
                   help="Negative control: train speaker adversaries with deterministic random "
                        "targets resampled each batch; the positive z_P SID head still uses true labels.")
    p.add_argument("--grl_p_grad_norm", action="store_true", default=cfg.grl_p_grad_norm,
                   help="Per-frame normalize the reversed PHONEME gradient on z_P (constant content-removal push).")
    p.add_argument("--grl_p_grad_norm_target", type=float, default=cfg.grl_p_grad_norm_target,
                   help="Per-frame target L2 norm for grad-normalized phoneme GRL on z_P.")
    p.add_argument("--invariance", action="store_true", default=cfg.invariance,
                   help="Enforce z_L invariance to a speaker perturbation (pitch+formant) — "
                        "dense per-frame speaker removal; train loader yields perturbed pairs.")
    p.add_argument("--inv_weight",      type=float, default=cfg.inv_weight)
    p.add_argument("--inv_ramp_end",    type=int,   default=cfg.inv_ramp_end,
                   help="Ramp inv_weight 0->full over this many steps (lets z_L form content first).")
    p.add_argument("--inv_f0_low",      type=float, default=cfg.inv_f0_low)
    p.add_argument("--inv_f0_high",     type=float, default=cfg.inv_f0_high)
    p.add_argument("--inv_formant_low", type=float, default=cfg.inv_formant_low)
    p.add_argument("--inv_formant_high",type=float, default=cfg.inv_formant_high)
    p.add_argument("--dann_full_discriminator", action="store_true", default=cfg.dann_full_discriminator,
                   help="Canonical DANN: adversary heads train at full strength; grl weights only scale "
                        "the reversed (encoder-side) gradient via lambda.")
    p.add_argument("--rho",             type=float, default=cfg.rho)

    # ablation flags (D / E / F)
    p.add_argument("--fixed_blocks", action="store_true", default=cfg.fixed_blocks,
                   help="Option A: fixed L/P/U index blocks + per-block TopK (no routing)")
    p.add_argument("--K_L",    type=int, default=cfg.K_L)
    p.add_argument("--K_P",    type=int, default=cfg.K_P)
    p.add_argument("--K_U",    type=int, default=cfg.K_U)
    p.add_argument("--per_block_topk", action=argparse.BooleanOptionalAction, default=cfg.per_block_topk,
                   help="True: per-block TopK (forced allocation); --no-per_block_topk: global TopK (emergent)")
    p.add_argument("--topk_L", type=int, default=cfg.topk_L)
    p.add_argument("--topk_P", type=int, default=cfg.topk_P)
    p.add_argument("--topk_U", type=int, default=cfg.topk_U)
    p.add_argument("--no_routing",          action="store_true", default=cfg.no_routing)
    p.add_argument("--fixed_routing",       action="store_true", default=cfg.fixed_routing)
    p.add_argument("--fixed_routing_split", type=float,          default=cfg.fixed_routing_split)
    p.add_argument("--n_routes",            type=int,            default=cfg.n_routes)
    p.add_argument("--pre_topk_routing",    action="store_true", default=cfg.pre_topk_routing)
    p.add_argument("--hard_gumbel_routing", action=argparse.BooleanOptionalAction, default=cfg.hard_gumbel_routing,
                   help="Routing mode: --hard_gumbel_routing (one-hot STE) / --no-hard_gumbel_routing (soft fractional).")
    p.add_argument("--gumbel_tau_start", type=float, default=cfg.gumbel_tau_start)
    p.add_argument("--gumbel_tau_end",   type=float, default=cfg.gumbel_tau_end,
                   help="Final Gumbel temperature (hold high, e.g. 0.5, to keep soft routing soft).")
    p.add_argument("--routing_init_std", type=float, default=cfg.routing_init_std,
                   help="Std of random routing-logit init (0 = zero init / symmetric saddle).")
    p.add_argument("--routing_spec_weight", type=float, default=cfg.routing_spec_weight,
                   help="Weight on per-unit specialization loss (minimise routing entropy Hu).")
    p.add_argument("--routing_dynamic", action="store_true", default=cfg.routing_dynamic,
                   help="Input-dependent (per-utterance) routing vs the static partition.")
    p.add_argument("--routing_dynamic_hidden", type=int, default=cfg.routing_dynamic_hidden)

    # experiment flags
    p.add_argument("--grl_phoneme_weight",  type=float, default=cfg.grl_phoneme_weight)
    p.add_argument("--grl_u_weight",         type=float, default=cfg.grl_u_weight,
                   help="speaker adversary on z_U (anti-speaker → push speaker to z_P)")
    p.add_argument("--grl_phoneme_u_weight", type=float, default=cfg.grl_phoneme_u_weight,
                   help="phoneme adversary on z_U (anti-phoneme → push phonemes to z_L)")
    p.add_argument("--prosody", action=argparse.BooleanOptionalAction, default=cfg.prosody,
                   help="enable the prosody factor: per-frame log-F0+log-E regression on z_P")
    p.add_argument("--prosody_weight",       type=float, default=cfg.prosody_weight,
                   help="weight on the z_P prosody regression task")
    p.add_argument("--grl_prosody_weight",   type=float, default=cfg.grl_prosody_weight,
                   help="anti-prosody adversary on z_L (push F0/energy → z_P)")
    p.add_argument("--grl_prosody_u_weight", type=float, default=cfg.grl_prosody_u_weight,
                   help="anti-prosody adversary on z_U (push F0/energy → z_P)")
    p.add_argument("--decor_weight",        type=float, default=cfg.decor_weight)
    p.add_argument("--ub_weight",           type=float, default=cfg.ub_weight)
    p.add_argument("--ub_ramp_start",       type=int,   default=cfg.ub_ramp_start)
    p.add_argument("--ub_ramp_end",         type=int,   default=cfg.ub_ramp_end,
                   help="Ramp ub_weight 0->full between ub_ramp_start and ub_ramp_end (0=constant).")
    p.add_argument("--ste_routing",         action="store_true", default=cfg.ste_routing)
    p.add_argument("--projection_disentanglement", action="store_true",
                   default=cfg.projection_disentanglement,
                   help="Use learned compressed z_t->z_L/z_P projections instead of routing masks.")
    p.add_argument("--projection_dim", type=int, default=cfg.projection_dim,
                   help="Output dimension for projection_disentanglement.")
    p.add_argument("--projection_nonlinear", action="store_true", default=cfg.projection_nonlinear,
                   help="Make the projection views 2-layer MLPs (nonlinear demixer) instead of linear.")
    p.add_argument("--projection_hidden", type=int, default=cfg.projection_hidden,
                   help="Hidden width of the nonlinear projection MLP.")
    p.add_argument("--projection_reconstruct", action="store_true",
                   default=cfg.projection_reconstruct,
                   help="Reconstruct h_t solely through z_L/z_P (up-project + decode), no decode(z_t).")
    p.add_argument("--projection_u_dim", type=int, default=cfg.projection_u_dim,
                   help="Dim of residual view z_U for reconstructive projection (0 = 2-way, no z_U).")
    p.add_argument("--projection_u_l2", type=float, default=cfg.projection_u_l2,
                   help="L2 activity penalty on z_U (the residual bottleneck).")
    p.add_argument("--spear_layernorm", action="store_true", default=cfg.spear_layernorm,
                   help="LayerNorm each SPEAR layer before averaging (SUPERB-comparable h_t).")
    p.add_argument("--vib_zL_weight", type=float, default=cfg.vib_zL_weight,
                   help="VIB KL penalty on z_L (information bottleneck; 0=off)")
    p.add_argument("--vib_zL_layernorm", action="store_true", default=cfg.vib_zL_layernorm,
                   help="Param-free LayerNorm on z_L before VIB — bounds magnitude so the KL can't diverge.")
    p.add_argument("--vib_zL_ramp_end", type=int, default=cfg.vib_zL_ramp_end,
                   help="ramp VIB weight 0→full by this step (0=constant)")
    p.add_argument("--instance_norm_zL", action="store_true", default=cfg.instance_norm_zL,
                   help="Instance-normalize z_L over time (strip per-utterance speaker stats).")

    # schedule
    p.add_argument("--total_steps",   type=int,   default=cfg.total_steps)
    p.add_argument("--stage2_steps",  type=int,   default=cfg.stage2_steps)
    p.add_argument("--stage2_schedule_steps", type=int, default=cfg.stage2_schedule_steps,
                   help="Stage-2 LR/DANN schedule horizon (0 = use --stage2_steps).")
    p.add_argument("--warmup_steps",  type=int,   default=cfg.warmup_steps)
    p.add_argument("--batch_size",    type=int,   default=cfg.batch_size)
    p.add_argument("--lr",            type=float, default=cfg.lr)
    p.add_argument("--lr_min",        type=float, default=cfg.lr_min)
    p.add_argument("--lr_routing",    type=float, default=cfg.lr_routing)
    p.add_argument("--lr_heads",      type=float, default=cfg.lr_heads)
    p.add_argument("--lr_disc",       type=float, default=cfg.lr_disc,
                   help="separate lr for adversary discriminators (0 = use lr_heads)")
    p.add_argument("--n_disc_steps",  type=int,   default=cfg.n_disc_steps,
                   help="discriminator updates per encoder update (GAN n_critic)")
    p.add_argument("--grad_log_every",type=int,   default=cfg.grad_log_every)
    p.add_argument("--grad_clip",     type=float, default=cfg.grad_clip,
                   help="Global gradient clipping max norm for stage1/stage2.")

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
    cfg.local_data            = args.local_data
    cfg.librispeech_root      = Path(args.librispeech_root)
    cfg.train_split_dir       = args.train_split_dir
    cfg.lexicon_path          = Path(args.lexicon_path)
    cfg.max_train_examples    = args.max_train_examples
    cfg.max_val_examples      = args.max_val_examples
    cfg.max_test_examples     = args.max_test_examples
    cfg.spear_model_id        = args.spear_model_id
    cfg.K                     = args.K
    cfg.topk                  = args.topk
    cfg.aux_k                 = args.aux_k
    cfg.aux_k_coef            = args.aux_k_coef
    cfg.dead_steps_threshold  = args.dead_steps_threshold
    cfg.geom_median_bias      = args.geom_median_bias
    cfg.renorm_decoder        = args.renorm_decoder
    cfg.alpha                 = args.alpha
    cfg.beta                  = args.beta
    cfg.gradnorm              = args.gradnorm
    cfg.gradnorm_alpha        = args.gradnorm_alpha
    cfg.gradnorm_lr           = args.gradnorm_lr
    cfg.gradnorm_tasks        = args.gradnorm_tasks
    cfg.gradnorm_every        = args.gradnorm_every
    cfg.grl_weight            = args.grl_weight
    cfg.grl_delay_steps       = args.grl_delay_steps
    cfg.grl_frame_level       = args.grl_frame_level
    cfg.grl_attention_pool    = args.grl_attention_pool
    cfg.grl_stats_pool        = args.grl_stats_pool
    cfg.grl_dense_context     = args.grl_dense_context
    cfg.grl_context_kernel    = args.grl_context_kernel
    cfg.grl_grad_norm         = args.grl_grad_norm
    cfg.grl_grad_norm_target  = args.grl_grad_norm_target
    cfg.shuffle_grl_speaker_labels = args.shuffle_grl_speaker_labels
    cfg.grl_p_grad_norm       = args.grl_p_grad_norm
    cfg.grl_p_grad_norm_target = args.grl_p_grad_norm_target
    cfg.invariance            = bool(args.invariance)
    cfg.inv_weight            = args.inv_weight
    cfg.inv_ramp_end          = args.inv_ramp_end
    cfg.inv_f0_low            = args.inv_f0_low
    cfg.inv_f0_high           = args.inv_f0_high
    cfg.inv_formant_low       = args.inv_formant_low
    cfg.inv_formant_high      = args.inv_formant_high
    cfg.dann_full_discriminator = args.dann_full_discriminator
    cfg.rho                   = args.rho
    cfg.fixed_blocks          = args.fixed_blocks
    cfg.K_L                   = args.K_L
    cfg.K_P                   = args.K_P
    cfg.K_U                   = args.K_U
    cfg.per_block_topk        = args.per_block_topk
    cfg.topk_L                = args.topk_L
    cfg.topk_P                = args.topk_P
    cfg.topk_U                = args.topk_U
    cfg.no_routing            = args.no_routing
    cfg.fixed_routing         = args.fixed_routing
    cfg.fixed_routing_split   = args.fixed_routing_split
    cfg.n_routes              = args.n_routes
    cfg.pre_topk_routing      = args.pre_topk_routing
    cfg.hard_gumbel_routing   = args.hard_gumbel_routing
    cfg.gumbel_tau_start      = args.gumbel_tau_start
    cfg.gumbel_tau_end        = args.gumbel_tau_end
    cfg.routing_init_std      = args.routing_init_std
    cfg.routing_spec_weight   = args.routing_spec_weight
    cfg.routing_dynamic       = args.routing_dynamic
    cfg.routing_dynamic_hidden = args.routing_dynamic_hidden
    cfg.grl_phoneme_weight    = args.grl_phoneme_weight
    cfg.grl_u_weight          = args.grl_u_weight
    cfg.grl_phoneme_u_weight  = args.grl_phoneme_u_weight
    cfg.prosody               = bool(args.prosody)
    cfg.prosody_weight        = args.prosody_weight
    cfg.grl_prosody_weight    = args.grl_prosody_weight
    cfg.grl_prosody_u_weight  = args.grl_prosody_u_weight
    cfg.decor_weight          = args.decor_weight
    cfg.ub_weight             = args.ub_weight
    cfg.ub_ramp_start         = args.ub_ramp_start
    cfg.ub_ramp_end           = args.ub_ramp_end
    cfg.ste_routing           = args.ste_routing
    cfg.projection_disentanglement = args.projection_disentanglement
    cfg.projection_dim        = args.projection_dim
    cfg.projection_nonlinear  = args.projection_nonlinear
    cfg.projection_hidden     = args.projection_hidden
    cfg.projection_reconstruct = args.projection_reconstruct
    cfg.projection_u_dim      = args.projection_u_dim
    cfg.projection_u_l2       = args.projection_u_l2
    cfg.spear_layernorm       = args.spear_layernorm
    cfg.instance_norm_zL      = args.instance_norm_zL
    cfg.vib_zL_weight         = args.vib_zL_weight
    cfg.vib_zL_ramp_end       = args.vib_zL_ramp_end
    cfg.vib_zL_layernorm      = args.vib_zL_layernorm
    cfg.total_steps           = args.total_steps
    cfg.stage2_steps          = args.stage2_steps
    cfg.stage2_schedule_steps = args.stage2_schedule_steps
    cfg.warmup_steps          = args.warmup_steps
    cfg.batch_size            = args.batch_size
    cfg.lr                    = args.lr
    cfg.lr_min                = args.lr_min
    cfg.lr_routing            = args.lr_routing
    cfg.lr_heads              = args.lr_heads
    cfg.lr_disc               = args.lr_disc
    cfg.n_disc_steps          = args.n_disc_steps
    cfg.grad_log_every        = args.grad_log_every
    cfg.grad_clip             = args.grad_clip
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
