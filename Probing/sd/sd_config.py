"""Configuration for the Spoof Detection (SD) probing task.

Architecture (from the paper):
    frozen encoder
      → weighted-sum of all layers
      → Linear(hidden, proj_dim=256)     per-frame projection
      → masked mean pool                 utterance-level vector
      → Linear(256, 128) → ReLU → Dropout → Linear(128, 1)
      → BCE with logits

Training data : ASVspoof 2019 LA train
Validation    : ASVspoof 2019 LA dev  (per-epoch early stopping)
Test sets     : ASV19 LA eval, ASV21 LA, ASV21 DF, ITW, DFEval2024,
                FamousFigures, ASVSpoofLD — evaluated ONCE after training.
                Any dataset whose root is None or missing is silently skipped.
Metric        : Equal Error Rate (EER) — lower is better.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

_SD_DIR = Path(__file__).parent


@dataclass
class SDConfig:
    # ── Training / validation data (ASVspoof 2019 LA) ──────────────────────
    # Expected layout:
    #   asv19_la_root/
    #     ASVspoof2019_LA_cm_protocols/
    #       ASVspoof2019.LA.cm.train.trn.txt
    #       ASVspoof2019.LA.cm.dev.trl.txt
    #       ASVspoof2019.LA.cm.eval.trl.txt
    #     ASVspoof2019_LA_train/flac/
    #     ASVspoof2019_LA_dev/flac/
    #     ASVspoof2019_LA_eval/flac/
    asv19_la_root: Path = Path("/path/to/ASVspoof2019/LA")

    # ── Test datasets (all optional — skipped if None or path not found) ───
    # ASVspoof 2019 eval (audio in asv19_la_root, protocol auto-located)
    # — no separate path needed, derived from asv19_la_root above.

    # ASVspoof 2021 LA eval
    #   asv21_la_root/flac/*.flac
    asv21_la_root: Optional[Path] = None
    asv21_la_keys: Optional[Path] = None   # path to the keys .txt file

    # ASVspoof 2021 DF eval
    #   asv21_df_root/flac/*.flac  (may be nested)
    asv21_df_root: Optional[Path] = None
    asv21_df_keys: Optional[Path] = None

    # In-The-Wild
    #   Either CSV-based (itw_root/meta.csv + audio/) or
    #   directory-based  (itw_root/{bonafide,spoof}/*.flac)
    itw_root: Optional[Path] = None

    # DFEval 2024
    dfeval24_root: Optional[Path] = None
    dfeval24_keys: Optional[Path] = None

    # Famous Figures
    famous_figures_root: Optional[Path] = None
    famous_figures_keys: Optional[Path] = None

    # ASVSpoofLD
    asvspoofld_root: Optional[Path] = None
    asvspoofld_keys: Optional[Path] = None

    # ── Encoder ────────────────────────────────────────────────────────────
    model_id: str            = "marcoyang/spear-xlarge-speech-audio"
    model_family: Literal["spear", "hf"] = "spear"
    encoder_layer_count: int = 0

    # ── Architecture ───────────────────────────────────────────────────────
    probe_type: Literal["final", "weighted"] = "weighted"
    layer_idx: int           = -1    # for probe_type='final' only
    proj_dim: int            = 256   # frame-level projection before pooling
    mlp_hidden: int          = 128   # MLP hidden layer
    probe_dropout: float     = 0.1
    sample_rate: int         = 16_000

    # ── Training ───────────────────────────────────────────────────────────
    batch_size: int          = 8
    eval_batch_size: int     = 16
    learning_rate: float     = 1e-4
    weight_decay: float      = 1e-4
    num_epochs: int          = 10
    grad_clip: float         = 1.0
    warmup_steps: int        = 500

    # ── Misc ───────────────────────────────────────────────────────────────
    num_workers: int         = 0
    seed: int                = 42
    device: str              = "cuda"
    checkpoint_dir: Path     = _SD_DIR / "checkpoints"
    runs_dir: Path           = _SD_DIR / "runs"
    log_dir: Path            = _SD_DIR / "logs"
    log_every: int           = 50
