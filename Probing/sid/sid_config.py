"""Configuration for the VoxCeleb1 Speaker Identification task.

Task (SUPERB SID):
    Closed-set speaker classification over 1,251 speakers.
    Downstream model: mean-pool over encoder frames → linear classifier.
    Metric: Accuracy (ACC).

VoxCeleb1 directory layout expected on disk
-------------------------------------------
VoxCeleb1/
    dev/
        wav/
            {speaker_id}/           e.g. id00001/
                {video_id}/         e.g. 1zcIwhmdeo4/
                    {utt_id}.wav    e.g. 00001.wav
    test/
        wav/
            {speaker_id}/
                {video_id}/
                    {utt_id}.wav

The 1,251 speaker IDs are used directly as class labels (mapped to 0-based
indices sorted alphabetically so the mapping is deterministic across runs).

Download
--------
Request access and download from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# All default output paths are anchored to this sid/ directory.
_SID_DIR = Path(__file__).parent


@dataclass
class SIDConfig:
    # ----------------------------------------------------------------- Data
    # Absolute path to the VoxCeleb1 root (must contain dev/ sub-directory).
    voxceleb1_root: Path = Path("/path/to/VoxCeleb1")
    # SUPERB split manifest — defines exact train/val/test utterance assignments.
    # Download once: already bundled at sid/veri_test_class.txt
    meta_data: Path = _SID_DIR / "veri_test_class.txt"
    num_classes: int = 1251
    sample_rate: int = 16_000
    # Random-crop cap applied to TRAINING utterances only (SUPERB: 128,000 = 8 s).
    # Val and test are always evaluated on full utterances (no cap).
    train_max_duration_s: float = 8.0
    # Cap each split to this many examples (0 = no cap; for smoke tests).
    max_examples: int = 0

    # ------------------------------------------------------------ Encoder
    model_id: str = "marcoyang/spear-xlarge-speech-audio"
    # 'spear' for SPEAR-XLarge; 'hf' for standard HF speech encoders.
    model_family: Literal["spear", "hf"] = "spear"
    # Populated at runtime once the encoder is loaded.
    encoder_layer_count: int = 0

    # ---------------------------------------------------------------- Probe
    # 'final'    — linear on a single encoder layer (layer_idx)
    # 'weighted' — learnable softmax mix of all layers, then linear
    probe_type: Literal["final", "weighted"] = "weighted"
    layer_idx: int = -1
    proj_dim: int = 256        # frame-level projection dim before mean-pool (SUPERB: 256)
    probe_dropout: float = 0.1

    # ------------------------------------------------------------- Training
    batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 10
    grad_clip: float = 1.0
    warmup_steps: int = 500

    # ----------------------------------------------------------------- Misc
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda"
    checkpoint_dir: Path = _SID_DIR / "checkpoints"
    runs_dir: Path = _SID_DIR / "runs"
    log_dir: Path = _SID_DIR / "logs"
    log_every: int = 50
