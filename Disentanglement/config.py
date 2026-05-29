"""DISConfig — hyperparameters for the SAE reconstruction system."""

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

    # ---------------------------------------------------------------- Optimizer
    lr: float = 1e-4
    lr_min: float = 1e-6        # cosine decay floor
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # ---------------------------------------------------------------- Data
    sample_rate: int = 16_000
    librispeech_cache_dir: Path = _DIS_DIR.parent / "Probing" / "data"
    lexicon_path: Path = _DIS_DIR.parent / "Probing" / "data" / "librispeech-lexicon.txt"
    max_train_examples: int = 0     # 0 = full train-clean-100 (~28 k)
    max_val_examples: int = 500
    num_workers: int = 0

    # ---------------------------------------------------------------- Training
    batch_size: int = 16
    eval_batch_size: int = 32
    warmup_steps: int = 500
    total_steps: int = 6_000
    log_every: int = 100
    grad_log_every: int = 500
    ckpt_every: int = 1_000

    # ---------------------------------------------------------------- Paths
    checkpoint_dir: Path = _DIS_DIR / "checkpoints"
    runs_dir: Path = _DIS_DIR / "runs"
    log_dir: Path = _DIS_DIR / "logs"

    # ---------------------------------------------------------------- Misc
    seed: int = 42
    device: str = "cuda"
    bf16: bool = True
