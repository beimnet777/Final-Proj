"""DISConfig — hyperparameters for the SAE disentanglement system."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_DIS_DIR = Path(__file__).parent


@dataclass
class DISConfig:
    # ---------------------------------------------------------------- SPEAR
    spear_model_id: str = "marcoyang/spear-xlarge-speech-audio"
    D: int = 1280       # SPEAR-Large hidden size

    # ---------------------------------------------------------------- SAE
    K: int = 5120       # latent size  (4 × D)
    topk: int = 256     # active features per frame  (5% of K)

    # ---------------------------------------------------------------- Routing / Gumbel
    gumbel_tau_start: float = 1.0
    gumbel_tau_end:   float = 0.1
    hard_gumbel_routing: bool = False  # If True, stage-2 training uses one-hot ST-Gumbel masks.

    # ---------------------------------------------------------------- Loss weights  (stage 2)
    alpha:      float = 1.0     # PR (CTC) weight          — calibrated from grad norms
    beta:       float = 1.0     # SID (CE) weight           — calibrated from grad norms
    grl_weight:          float = 1.0    # adversarial speaker weight — calibrated from grad norms
    grl_delay_steps:     int   = 0     # steps before GRL is switched on (0 = no delay)
    rho:                 float = 0.001 # routing anti-collapse weight

    # ---------------------------------------------------------------- Ablation flags (D / E / F)
    no_routing:          bool  = False  # D: bypass routing, feed full z to all heads
    fixed_routing:       bool  = False  # E: freeze routing at init split (not learned)
    fixed_routing_split: float = 0.7   # E: fraction of K features assigned to L
    n_routes:            int   = 3     # F: 3 = L/P/U (default), 2 = binary L/P only

    # ---------------------------------------------------------------- Pre-TopK routing (deprecated — use ste_routing)
    pre_topk_routing:    bool  = False

    # ---------------------------------------------------------------- Experiment flags
    # Exp 1 — Dual GRL: phoneme adversary on z_P
    grl_phoneme_weight:  float = 0.0   # weight for phoneme-GRL CTC loss on z_P (0 = disabled)

    # Exp 2 — TopK decorrelation in stage 1
    decor_weight:        float = 0.0   # weight for off-diagonal correlation penalty on active features

    # Exp 4 — U-bucket information bottleneck
    ub_weight:           float = 0.0   # weight for (m_L + m_P).mean() bottleneck — forces U alive

    # Exp 5 — Straight-through estimator on routing mask multiplication
    # Forward: z_L = m_L × z_t (sparse, unchanged).  Backward: gradient flows through m_L × z_pre.
    ste_routing:         bool  = False

    # ---------------------------------------------------------------- Optimizer
    lr:          float = 1e-4   # SAE lr (stage 1);  also base lr for SAE in stage 2
    lr_min:      float = 1e-6   # cosine decay floor
    lr_routing:  float = 5e-6   # routing logits  (stage 2) — slow to prevent gradient-driven collapse
    lr_heads:    float = 1e-4   # task heads      (stage 2)
    weight_decay: float = 1e-4
    grad_clip:    float = 1.0

    # ---------------------------------------------------------------- Data
    sample_rate: int = 16_000
    librispeech_cache_dir: Path = _DIS_DIR.parent / "Probing" / "data"
    lexicon_path: Path = _DIS_DIR.parent / "Probing" / "data" / "librispeech-lexicon.txt"
    max_train_examples: int = 0     # 0 = full train-clean-100 (~28 k)
    max_val_examples:   int = 500
    num_workers: int = 0

    # ---------------------------------------------------------------- Training
    batch_size:      int = 16
    eval_batch_size: int = 32
    warmup_steps:    int = 500
    total_steps:     int = 6_000    # stage 1
    stage2_steps:    int = 0        # filled at launch (TBD)
    log_every:       int = 100
    grad_log_every:  int = 500
    ckpt_every:      int = 1_000

    # ---------------------------------------------------------------- Runtime (filled by data loader)
    vocab_size:   int = 41      # CTC: blank + 39 ARPAbet + SPN
    num_speakers: int = 0       # filled after dataset build

    # ---------------------------------------------------------------- Paths
    checkpoint_dir: Path = _DIS_DIR / "checkpoints"
    runs_dir:       Path = _DIS_DIR / "runs"
    log_dir:        Path = _DIS_DIR / "logs"

    # ---------------------------------------------------------------- Misc
    seed:   int  = 42
    device: str  = "cuda"
    bf16:   bool = True
