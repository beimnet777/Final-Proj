"""Configuration for the IEMOCAP Emotion Recognition task.

Kept separate from the ASR config so ER hyperparameters don't pollute that
namespace. This file is the single source of truth for the ER pipeline.

Emotion mapping (following SUPERB):
    neutral  → 0
    happy    → 1   (excited utterances are merged into this class)
    sad      → 2
    angry    → 3
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# All default paths are anchored to the er/ directory so the folder is
# self-contained regardless of where the script is called from.
_ER_DIR = Path(__file__).parent


# IEMOCAP emotion codes → integer class index.
# 'exc' (excited) is merged into 'hap' (happy) as is standard in the
# SUPERB / IEMOCAP literature.
EMOTION_MAP: dict[str, int] = {
    "neu": 0,
    "hap": 1,
    "exc": 1,
    "sad": 2,
    "ang": 3,
}

# Human-readable label for each index (used in reports).
EMOTION_NAMES: list[str] = ["neutral", "happy", "sad", "angry"]


@dataclass
class ERConfig:
    # ----------------------------------------------------------------- Data
    # Absolute path to the IEMOCAP_full_release directory obtained from USC.
    iemocap_root: Path = Path("/path/to/IEMOCAP_full_release")
    # Which session to hold out as the test set in this cross-validation fold.
    # Valid values: 1, 2, 3, 4, 5.
    test_fold: int = 1
    num_classes: int = 4
    sample_rate: int = 16_000

    # ------------------------------------------------------------ Encoder
    # Any HuggingFace model id or local path.
    model_id: str = "marcoyang/spear-xlarge-speech-audio"
    # 'spear' for SPEAR-XLarge (custom Zipformer API);
    # 'hf' for standard HF speech encoders (wav2vec2, HuBERT, WavLM, …).
    model_family: Literal["spear", "hf"] = "spear"
    # Populated at runtime once the encoder is loaded.
    encoder_layer_count: int = 0

    # ---------------------------------------------------------------- Probe
    # 'final'    — linear on a single encoder layer (selected by layer_idx)
    # 'weighted' — learnable softmax mix of all layers, then linear
    probe_type: Literal["final", "weighted"] = "weighted"
    # For probe_type='final': which layer to use (0-based, -1 = last).
    layer_idx: int = -1
    probe_dropout: float = 0.1

    # ------------------------------------------------------------- Training
    batch_size: int = 8
    eval_batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 20
    grad_clip: float = 1.0
    warmup_steps: int = 100

    # ----------------------------------------------------------------- Misc
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda"
    checkpoint_dir: Path = _ER_DIR / "checkpoints"
    runs_dir: Path = _ER_DIR / "runs"
    log_dir: Path = _ER_DIR / "logs"
    log_every: int = 20          # steps between train-loss log lines
