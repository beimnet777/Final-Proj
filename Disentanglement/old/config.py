"""DISConfig — all hyperparameters for the Disentanglement v1 system."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_DIS_DIR  = Path(__file__).parent
_PROJ_DIR = _DIS_DIR.parent.parent   # old/ -> Disentanglement/ -> Final-Proj/


@dataclass
class DISConfig:
    # ---------------------------------------------------------------- SPEAR
    spear_model_id: str = "marcoyang/spear-xlarge-speech-audio"
    D: int = 1280       # SPEAR-Large hidden size
    L: int = 13         # number of transformer layers

    # ---------------------------------------------------------------- SAE
    K: int = 5120       # latent size = 4 * D
    topk: int = 256     # 5% of K

    # ---------------------------------------------------------------- Routing / Gumbel
    gumbel_tau_start: float = 1.0
    gumbel_tau_end: float = 0.1

    # ---------------------------------------------------------------- Loss weights
    alpha: float = 1.0      # PR (CTC) weight
    beta: float = 1.0       # SID weight
    delta: float = 1e-5      # decorrelation weight (small — balanced with recon)
    rho: float = 0.0001      # routing anti-collapse weight
    # recon weight is fixed at 1.0; GRL uses its own DANN ramp (no tuned weight)

    # ---------------------------------------------------------------- Optimizer
    lr_enc_dec: float = 1e-4    # SAE enc/dec, heads, layer-weighted sum
    lr_routing: float = 3e-4   # routing logits
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # ---------------------------------------------------------------- Data
    sample_rate: int = 16_000
    # These paths mirror the shared Probing/data/ directory so LibriSpeech is not
    # downloaded again.
    librispeech_cache_dir: Path = _PROJ_DIR / "Probing" / "data"
    lexicon_path: Path = _PROJ_DIR / "Probing" / "data" / "librispeech-lexicon.txt"
    max_train_examples: int = 3000   # ~10–15 h; set 0 for full train-clean-100
    max_val_examples: int = 500
    max_duration_s: float = 5.0      # truncate long utterances to avoid OOM
    num_workers: int = 0

    # ---------------------------------------------------------------- Training
    batch_size: int = 16
    eval_batch_size: int = 32
    warmup_steps: int = 500
    total_steps: int = 50_000   # single stage — no stage separation
    stage1_steps: int = 10_000  # kept for scheduler compatibility
    stage2_steps: int = 40_000  # kept for scheduler compatibility
    log_every: int = 100
    probe_every: int = 1_000    # steps between in-loop fast-probe snapshots
    ckpt_every: int = 2_000

    # ---------------------------------------------------------------- Vocab (filled at runtime)
    vocab_size: int = 41        # CTC: blank + 39 ARPAbet phones + SPN
    num_speakers: int = 0       # filled after dataset build

    # ---------------------------------------------------------------- Decorrelation
    decorr_max_frames: int = 1000   # cap frames fed to covariance matrix

    # ---------------------------------------------------------------- Paths
    checkpoint_dir: Path = _DIS_DIR / "checkpoints"
    runs_dir: Path = _DIS_DIR / "runs"
    log_dir: Path = _DIS_DIR / "logs"

    # ---------------------------------------------------------------- Misc
    seed: int = 42
    device: str = "cuda"
    bf16: bool = True
