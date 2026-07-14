"""Config for the standalone MSP-Podcast disentanglement run.

Reuses the shared DISConfig (so build_dis_model gets every field it expects) but
applies MSP-appropriate defaults here, in this folder — the legacy config.py is
left untouched.  Differences from the old Libri+IEMOCAP setup, all motivated by the
failure analysis:
  * one dataset: emotion trains EVERY batch (no IEMOCAP-every-8, no _cap_loss).
  * full-strength anti-emotion GRL on z_L after a DANN sigmoid ramp.
  * class-weighted emotion CE + UAR reporting (neutral-heavy corpus).
  * optional PCGrad over the cooperative tasks to defuse gradient conflict on the
    shared SAE trunk (adversaries are left alone — their conflict is the point).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from config import DISConfig

_DIS = Path(__file__).resolve().parent.parent          # Disentanglement/
_MSP = Path("/rds/project/rds-xyBFuSj0hm0/dataset/MSP-Podcast-2.0")


@dataclass
class MSPConfig:
    """MSP-specific knobs (paths + the gradient-conflict controls).  Training
    hyper-params live on the DISConfig built by `to_dis_cfg`."""
    manifest:    str = str(_DIS / "data" / "msp_subset")
    audio_root:  str = str(_DIS / "data" / "msp_audio")
    transcripts: str = str(_MSP / "Transcripts.zip")

    # gradient-conflict handling on the shared SAE trunk
    pcgrad:        bool = True
    # cooperative tasks PCGrad de-conflicts (adversaries excluded on purpose)
    pcgrad_tasks:  str  = "recon,pr,sid,prosody,emotion"

    # Optional per-frame normalization of the reversed speaker gradient entering
    # z_L. Experiments enable it and select its target in their Slurm script.
    grl_grad_norm:        bool  = True
    grl_grad_norm_target: float = 0.0002

    # task weights (full-strength emotion + its GRL — the key fix)
    recon_weight: float = 1.0
    alpha: float = 0.8                    # PR / CTC
    beta:  float = 0.6                    # SID
    grl_weight:          float = 1.0      # anti-speaker on z_L
    grl_phoneme_weight:  float = 0.15     # anti-phoneme on z_P
    prosody_weight:      float = 0.5
    grl_prosody_weight:  float = 0.5      # anti-prosody on z_L
    emotion_weight:      float = 0.5
    grl_emotion_weight:  float = 0.5      # anti-emotion on z_L  (was 0.2 + ramp)
    # Optional perturbation-invariance on z_L. Disabled by default for the initial
    # MSP search so the labeled factors remain cleanly interpretable.
    inv_weight:          float = 0.0

    steps:        int = 12000
    warmup_steps: int = 500
    # DANN/GRL sigmoid ramp.  Reaches 1.0 by this step and then stays there.
    # Keep separate from lr warmup so MSP can sweep adversary onset without
    # changing the optimizer schedule.  Default preserves the original 500-step
    # adversary warmup, but with the canonical DANN sigmoid shape.
    dann_ramp_steps: int = 500
    batch_size:   int = 16
    eval_batch:   int = 32
    num_workers:  int = 8
    hard_routing: bool = True
    seed:         int = 42

    # Optimizer and routing controls. The production Slurm script passes every
    # one explicitly so its submitted command is the experiment source of truth.
    lr:                  float = 1e-4
    lr_min:              float = 1e-5
    lr_heads:            float = 1e-4
    lr_disc:             float = 1e-3
    lr_routing:          float = 1e-3
    n_disc_steps:        int   = 3
    grad_clip:           float = 1.0
    routing_init_std:    float = 0.5
    routing_spec_weight: float = 0.01
    routing_tau:         float = 1.0
    log_every:           int   = 100
    # Detailed gradient diagnostics are useful, but printing them every train
    # log makes the MSP logs hard to read. Keep the compact trace frequent and
    # the gradient trace sparser.
    grad_log_every:      int   = 1000
    ckpt_every:          int   = 1000

    # Learned-routing continuation controls. These mirror the Libri
    # learned→freeze experiments: run a learn segment, exact-resume, freeze the
    # learned static route assignment, optionally calibrate route-local TopK
    # quotas from the learned active split, then continue training.
    freeze_learned_routing_on_resume: bool = False
    freeze_route_topk_on_resume:      bool = False
    route_topk_calib_batches:         int  = 20


def to_dis_cfg(m: MSPConfig) -> DISConfig:
    """Materialise a DISConfig the shared model/optimiser understand, with the MSP
    paths and choices stamped on."""
    c = DISConfig()
    # ---- routing / disentanglement structure (binary L/P, no z_U) ----
    c.n_routes = 2
    c.hard_gumbel_routing = m.hard_routing
    # Gumbel temperature anneal: sharpen the routing over training.  Hard ST-Gumbel
    # cools to 0.1; soft routing stops at 0.5 (matches the legacy schedule).
    c.gumbel_tau_start = m.routing_tau
    c.gumbel_tau_end = 0.1 if m.hard_routing else 0.5
    c.routing_init_std = m.routing_init_std
    c.routing_spec_weight = m.routing_spec_weight
    c.spear_layernorm = True
    c.dann_full_discriminator = True
    c.n_disc_steps = m.n_disc_steps
    c.rho = 0.0
    # ---- tasks on (build_dis_model reads these to create the heads) ----
    c.prosody = True
    c.emotion = True
    c.emotion_num_classes = 4
    c.invariance = m.inv_weight > 0
    c.recon_weight = m.recon_weight
    c.alpha, c.beta = m.alpha, m.beta
    c.grl_weight = m.grl_weight
    c.grl_phoneme_weight = m.grl_phoneme_weight
    c.prosody_weight = m.prosody_weight
    c.grl_prosody_weight = m.grl_prosody_weight
    c.grl_prosody_u_weight = 0.0
    c.emotion_weight = m.emotion_weight
    c.grl_emotion_weight = m.grl_emotion_weight
    c.inv_weight = m.inv_weight
    c.inv_ramp_end = 0
    # ---- optimisation ----
    c.lr = m.lr
    c.lr_min = m.lr_min
    c.lr_heads = m.lr_heads
    c.lr_disc = m.lr_disc
    c.lr_routing = m.lr_routing
    c.grad_clip = m.grad_clip
    c.warmup_steps = m.warmup_steps
    c.dann_ramp_steps = m.dann_ramp_steps
    c.stage2_steps = m.steps
    c.stage2_schedule_steps = m.steps
    c.batch_size = m.batch_size
    c.eval_batch_size = m.eval_batch
    c.num_workers = m.num_workers
    c.seed = m.seed
    c.log_every = m.log_every
    c.grad_log_every = m.grad_log_every
    c.ckpt_every = m.ckpt_every
    c.freeze_learned_routing_on_resume = m.freeze_learned_routing_on_resume
    c.freeze_route_topk_on_resume = m.freeze_route_topk_on_resume
    c.route_topk_calib_batches = m.route_topk_calib_batches
    # ---- MSP data paths (read by msp.data.make_msp_dataloaders) ----
    c.msp_manifest = m.manifest
    c.msp_audio_root = m.audio_root
    c.msp_transcripts = m.transcripts
    # ---- gradient-conflict controls (read by msp.train) ----
    c.pcgrad = m.pcgrad
    c.pcgrad_tasks = tuple(t.strip() for t in m.pcgrad_tasks.split(",") if t.strip())
    c.grl_grad_norm = m.grl_grad_norm
    c.grl_grad_norm_target = m.grl_grad_norm_target
    return c
