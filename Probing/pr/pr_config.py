"""Configuration for the Phone Recognition (PR) probing task.

Data   : LibriSpeech train-clean-100 (full ~100h), validated on dev-clean,
         tested on test-clean.  Follows the SUPERB benchmark exactly.
Labels : ARPAbet phone sequences from the official LibriSpeech lexicon
         (librispeech-lexicon.txt from openslr.org/11); g2p_en used for OOV.
Loss   : CTC
Metric : Phone Error Rate (PER)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

_PR_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Standard 39-phone ARPAbet set (TIMIT-style, stress digits stripped).
# Blank (CTC) lives at index 0; phones at 1..39; SPN at 40.
# ---------------------------------------------------------------------------
ARPABET_39 = [
    "aa", "ae", "ah", "ao", "aw", "ay",
    "b",  "ch", "d",  "dh",
    "eh", "er", "ey",
    "f",  "g",  "hh",
    "ih", "iy", "jh",
    "k",  "l",  "m",  "n",  "ng",
    "ow", "oy", "p",  "r",
    "s",  "sh", "t",  "th",
    "uh", "uw", "v",  "w",  "y",  "z",  "zh",
]
# Special tokens (appended after the 39 phones)
SPN_TOKEN   = "spn"   # spoken noise / OOV placeholder

# Mapping from CMU dict stress-marked phones → ARPAbet 39 base phones.
# Vowels carry stress digit 0/1/2; we strip it. A few CMU phones need
# explicit remapping to the TIMIT-39 set.
_CMU_REMAP = {
    "ax":  "ah",   # reduced vowel → ah
    "axr": "er",
    "ix":  "ih",
    "ux":  "uw",
    "el":  "l",    # syllabic l
    "em":  "m",    # syllabic m
    "en":  "n",    # syllabic n
    "nx":  "ng",
}


@dataclass
class PRConfig:
    # ---------------------------------------------------------------- Data
    # Shared with the ASR probing pipeline — both read from Probing/data/.
    data_cache_dir: Path   = _PR_DIR.parent / "data"
    sample_rate: int       = 16_000
    # Path to the official LibriSpeech lexicon file.
    # Download once:  wget -q https://www.openslr.org/resources/11/librispeech-lexicon.txt \
    #                      -O <data_cache_dir>/librispeech-lexicon.txt
    librispeech_lexicon: Path = _PR_DIR.parent / "data" / "librispeech-lexicon.txt"

    # --------------------------------------------------------------- Vocab
    # Populated at build time; kept here for serialisation.
    vocab_size: int        = 41   # blank + 39 phones + SPN

    # ----------------------------------------------------- Encoder
    model_id: str          = "marcoyang/spear-xlarge-speech-audio"
    model_family: Literal["spear", "hf"] = "spear"
    encoder_layer_count: int = 0

    # ---------------------------------------------------------------- Probe
    probe_type: Literal["final", "weighted"] = "weighted"
    layer_idx: int         = -1
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
