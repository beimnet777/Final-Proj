"""Config for the standalone MSP-Podcast disentanglement run.

Reuses the shared DISConfig (so build_dis_model gets every field it expects) but
applies MSP-appropriate defaults here, in this folder — the legacy config.py is
left untouched.  Differences from the old Libri+IEMOCAP setup, all motivated by the
failure analysis:
  * one dataset: emotion trains EVERY batch (no IEMOCAP-every-8, no _cap_loss).
  * full-strength anti-emotion GRL on z_L (was 0.2 and ramped — too weak).
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
    pcgrad_tasks:  str  = "recon,pr,sid,prosody,emotion,inv"
    gradnorm:      bool = False           # magnitude balancing (optional, off by default)

    # task weights (full-strength emotion + its GRL — the key fix)
    alpha: float = 0.8                    # PR / CTC
    beta:  float = 0.6                    # SID
    grl_weight:          float = 1.0      # anti-speaker on z_L
    grl_phoneme_weight:  float = 0.5      # anti-phoneme on z_P
    prosody_weight:      float = 0.5
    grl_prosody_weight:  float = 0.5      # anti-prosody on z_L
    emotion_weight:      float = 0.5
    grl_emotion_weight:  float = 0.5      # anti-emotion on z_L  (was 0.2 + ramp)
    # MSP has 700 real speakers + a GELU speaker-GRL, so perturbation-invariance is
    # now a stabiliser, not the main speaker-stripper — was 4.0 (Libri-tuned).
    inv_weight:          float = 1.0

    steps:        int = 12000
    warmup_steps: int = 500
    batch_size:   int = 16
    eval_batch:   int = 32
    num_workers:  int = 8
    hard_routing: bool = True
    seed:         int = 42


def to_dis_cfg(m: MSPConfig) -> DISConfig:
    """Materialise a DISConfig the shared model/optimiser understand, with the MSP
    paths and choices stamped on."""
    c = DISConfig()
    # ---- routing / disentanglement structure (binary L/P, no z_U) ----
    c.n_routes = 2
    c.hard_gumbel_routing = m.hard_routing
    # Gumbel temperature anneal: sharpen the routing over training.  Hard ST-Gumbel
    # cools to 0.1; soft routing stops at 0.5 (matches the legacy schedule).
    c.gumbel_tau_start = 1.0
    c.gumbel_tau_end = 0.1 if m.hard_routing else 0.5
    c.routing_init_std = 0.5
    c.routing_spec_weight = 0.01
    c.spear_layernorm = True
    c.dann_full_discriminator = True
    c.n_disc_steps = 3
    c.rho = 0.0
    # ---- tasks on (build_dis_model reads these to create the heads) ----
    c.prosody = True
    c.emotion = True
    c.emotion_num_classes = 4
    c.invariance = True
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
    c.lr = 1e-4
    c.lr_min = 1e-6
    c.lr_heads = 1e-4
    c.lr_disc = 1e-3
    c.lr_routing = 1e-3
    c.grad_clip = 1.0
    c.warmup_steps = m.warmup_steps
    c.stage2_steps = m.steps
    c.stage2_schedule_steps = m.steps
    c.batch_size = m.batch_size
    c.eval_batch_size = m.eval_batch
    c.num_workers = m.num_workers
    c.seed = m.seed
    # ---- MSP data paths (read by msp.data.make_msp_dataloaders) ----
    c.msp_manifest = m.manifest
    c.msp_audio_root = m.audio_root
    c.msp_transcripts = m.transcripts
    # ---- gradient-conflict controls (read by msp.train) ----
    c.pcgrad = m.pcgrad
    c.pcgrad_tasks = tuple(t.strip() for t in m.pcgrad_tasks.split(",") if t.strip())
    c.gradnorm = m.gradnorm
    return c
