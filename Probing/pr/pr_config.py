"""Configuration for the Phone Recognition (PR) probing task.

Data   : LibriSpeech train-clean-100 (full ~100h), validated on dev-clean,
         tested on test-clean.  Follows the SUPERB benchmark exactly.
Labels : Full CMU ARPAbet phone sequences WITH stress digits (71 phones),
         matching SUPERB's vocab/phoneme.txt exactly.
Loss   : CTC
Metric : Phone Error Rate (PER)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

_PR_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# SUPERB 71-phone vocab (s3prl/downstream/ctc/vocab/phoneme.txt, in order).
# Stress digits are kept (AH0, AH1, AH2 are three distinct tokens).
#
# Tokenizer index layout — matches SUPERB's CharacterTextEncoder:
#   0   <pad>  — CTC blank
#   1   <eos>  — appended to every encoded sequence
#   2   <unk>  — unknown phone
#   3   SIL
#   4   SPN
#   5   AA0  …  73  ZH
#
# vocab_size = 74  (3 special + 71 phones)
# ---------------------------------------------------------------------------
SUPERB_PHONES = [
    "SIL", "SPN",
    "AA0", "AA1", "AA2",
    "AE0", "AE1", "AE2",
    "AH0", "AH1", "AH2",
    "AO0", "AO1", "AO2",
    "AW0", "AW1", "AW2",
    "AY0", "AY1", "AY2",
    "B", "CH", "D", "DH",
    "EH0", "EH1", "EH2",
    "ER0", "ER1", "ER2",
    "EY0", "EY1", "EY2",
    "F", "G", "HH",
    "IH0", "IH1", "IH2",
    "IY0", "IY1", "IY2",
    "JH", "K", "L", "M", "N", "NG",
    "OW0", "OW1", "OW2",
    "OY0", "OY1", "OY2",
    "P", "R", "S", "SH", "T", "TH",
    "UH0", "UH1", "UH2",
    "UW0", "UW1", "UW2",
    "V", "W", "Y", "Z", "ZH",
]  # 71 entries → indices 3..73


@dataclass
class PRConfig:
    # ---------------------------------------------------------------- Data
    # Shared with the ASR probing pipeline — both read from Probing/data/.
    data_cache_dir: Path   = _PR_DIR.parent / "data"
    sample_rate: int       = 16_000
    # Optional raw LibriSpeech tree. Streaming remains the default for existing
    # probing/HPC jobs; Colab diagnostic probes set these explicitly.
    local_data: bool = False
    librispeech_root: Path = _PR_DIR.parent / "data" / "LibriSpeech"
    # Path to the official LibriSpeech lexicon file.
    # Download once:  wget -q https://www.openslr.org/resources/11/librispeech-lexicon.txt \
    #                      -O <data_cache_dir>/librispeech-lexicon.txt
    librispeech_lexicon: Path = _PR_DIR.parent / "data" / "librispeech-lexicon.txt"

    # --------------------------------------------------------------- Vocab
    # Populated at build time; kept here for serialisation.
    vocab_size: int        = 74   # 3 special (<pad>/<eos>/<unk>) + 71 SUPERB phones

    # ----------------------------------------------------- Encoder
    model_id: str          = "marcoyang/spear-xlarge-speech-audio"
    model_family: Literal["spear", "hf"] = "spear"
    encoder_layer_count: int = 0

    # ---------------------------------------------------------------- Probe
    probe_type: Literal["final", "weighted", "fixed_weighted"] = "weighted"
    layer_idx: int         = -1
    proj_dim: int          = 256   # frame-level projection dim before CTC head (SUPERB: 256)
    probe_dropout: float   = 0.1

    # ------------------------------------------------------------- Training
    batch_size: int        = 8
    eval_batch_size: int   = 16
    learning_rate: float   = 5e-4
    weight_decay: float    = 1e-4
    num_epochs: int        = 10
    grad_clip: float       = 1.0
    warmup_steps: int      = 500

    # ----------------------------------------------------------------- Misc
    num_workers: int       = 0
    seed: int              = 42
    device: str            = "cuda"
    checkpoint_dir: Path   = _PR_DIR / "checkpoints"
    runs_dir: Path         = _PR_DIR / "runs"
    log_dir: Path          = _PR_DIR / "logs"
    log_every: int         = 50
    # Cap dataset size for smoke tests (0 = no cap).
    max_examples: int      = 0
