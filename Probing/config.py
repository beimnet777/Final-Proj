"""Single source of truth for paths, hyperparameters, and probe options.

Everything that another file needs to know about *how* this run is shaped lives
here. run.py will add a thin CLI on top to override fields like `probe_type`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class Config:
    # ---------------------------------------------------------------- Data
    # LibriSpeech is fetched via `datasets` and cached under this directory.
    # We carve a ~10h slice out of train-clean-100 (see data.py).
    data_cache_dir: Path = Path("./data")
    train_hours: float = 10.0            # target size of the training slice
    val_split: float = 0.05              # fraction of the slice held out for validation
    sample_rate: int = 16_000            # SPEAR and LibriSpeech both expect 16 kHz mono

    # --------------------------------------------------------------- Vocab
    # Characters used in LibriSpeech transcripts after lowercasing:
    # 26 letters + space + apostrophe = 28 symbols. CTC needs an extra blank
    # symbol, which we place at index 0 by convention. Hence vocab_size = 29.
    vocab: str = " 'abcdefghijklmnopqrstuvwxyz"

    # ------------------------------------------------------- SPEAR encoder
    # Anything HF AutoModel can load.
    spear_model_id: str = "marcoyang/spear-xlarge-speech-audio"
    # Number of transformer layers SPEAR exposes via hidden_states.
    # Populated at runtime in model.py once the encoder is loaded.
    encoder_layer_count: int = 0

    # ---------------------------------------------------------------- Probe
    # "single"    -> linear classifier on the last SPEAR layer only.
    # "weighted" -> learnable softmax mixture across all SPEAR layers,
    #               then a linear classifier on the mixed representation.
    probe_type: Literal["single", "weighted"] = "weighted"
    layer_idx: int = -1
    probe_dropout: float = 0.1

    # ------------------------------------------------------------- Training
    batch_size: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    num_epochs: int = 20
    grad_clip: float = 1.0
    warmup_steps: int = 500

    # ----------------------------------------------------------------- Eval
    eval_every_epochs: int = 1
    # Eval has no gradients, so memory headroom is ~3-4x larger than train.
    # Use a bigger batch to cut the number of SPEAR forward passes during
    # validation/test. Bump higher on CUDA (e.g. 32-64).
    eval_batch_size: int = 16

    # ----------------------------------------------------------------- Misc
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda"                 # train.py will fall back to "cpu" if no GPU
    checkpoint_dir: Path = Path("./checkpoints")
    log_every: int = 50                  # steps between train-loss log lines

    # --------------------------------------------------------- Derived view
    @property
    def vocab_size(self) -> int:
        # +1 for the CTC blank symbol at index 0.
        return len(self.vocab) + 1

    @property
    def blank_id(self) -> int:
        return 0


if __name__ == "__main__":
    # Smoke test: instantiate, print, and sanity-check the derived fields.
    cfg = Config()
    print(cfg)
    print(f"vocab_size = {cfg.vocab_size}  (28 chars + 1 CTC blank)")
    print(f"blank_id   = {cfg.blank_id}")
    assert cfg.vocab_size == 29, "expected 28 chars + 1 blank"
    assert cfg.blank_id == 0
    print("config.py OK")
