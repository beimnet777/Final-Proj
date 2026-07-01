# Colab disentanglement experiments

Use `notebooks/Disentanglement_Colab.ipynb` for MSP or LibriSpeech stage-2
experiments. The notebook is an orchestration layer; model and loss code remains
in the normal trainers.

LibriSpeech does not need to be present on your computer. Before training or a
smoke test, the notebook downloads the complete corpus used by this project:
`train-clean-100`, `dev-clean`, and `test-clean`. The original OpenSLR archives
are cached under `MyDrive/FinalProjColab/assets/openslr/`, so later sessions only
copy and extract them. MSP is licensed and must still be prepared from an
authorized copy.

## Prepare data once

MSP is licensed data. Run this only on a machine where the corpus is available,
then place the resulting private archive in `MyDrive/FinalProjColab/assets/`:

```bash
python -m Disentanglement.colab_bundle prepare-msp \
  --manifest Disentanglement/data/msp_subset/manifest.csv \
  --audio-root /path/to/MSP \
  --transcripts /path/to/Transcripts.zip \
  --profile pilot --output msp_pilot.tar.gz
```

LibriSpeech uses the same archive contract:

```bash
python -m Disentanglement.colab_bundle prepare-librispeech \
  --librispeech-root /path/to/LibriSpeech-parent \
  --lexicon Probing/data/librispeech-lexicon.txt \
  --profile pilot --output librispeech_pilot.tar.gz
```

Use `--profile full` for final runs. Archives contain a manifest with per-file
SHA-256 hashes and are extracted to `/content`, not trained from Drive.

## Command-line equivalent

```bash
python -m Disentanglement.experiment_runner \
  --experiment libri_grl_stats_gelu \
  --data_root /content/data --profile pilot --phase train \
  --effective_batch_size 16 --microbatch_size auto \
  --resume auto --segment_steps 250 --max_runtime_minutes 600 \
  --output_dir /content/run
```

Available LibriSpeech presets are `libri_grl_stats_gelu`,
`libri_club_hybrid`, and experimental `libri_club_pure`. CLUB is an
architecture-agnostic MI-minimization objective, not proof of universal probe
failure; use `--phase probe` on a completed checkpoint.

The dependency cell also downloads NLTK's `cmudict` and
`averaged_perceptron_tagger_eng` data. These are runtime data used by `g2p_en`
for the small number of transcript words absent from the supplied lexicon;
installing the Python package alone does not install them.

## Fine-grained hyperparameters

The notebook's `OVERRIDES` dictionary changes selected values after loading the
named preset. For example:

```python
OVERRIDES = {
    "stage2_steps": 10_000,  # use "steps" for MSP presets
    "warmup_steps": 500,
    "lr": 1e-4,
    "lr_heads": 1e-4,
    "lr_disc": 1e-3,
    "lr_routing": 1e-3,
    "alpha": 0.8,
    "beta": 0.6,
    "grl_weight": 1.0,
    "grl_phoneme_weight": 0.15,
    "grad_clip": 1.0,
    "n_disc_steps": 3,
    "ckpt_every": 1_000,
}
```

The equivalent CLI form is repeatable: `--set lr=0.0001 --set alpha=0.8`.
Unknown or misspelled keys fail before training. `SEGMENT_STEPS` remains separate:
it limits only the current Colab session, while `stage2_steps`/`steps` is the
experiment's total optimizer-step target.

Give each distinct override set a distinct `RUN_TAG` (for example `lr2e4` or
`club_w03`). The tag creates a separate local/Drive checkpoint directory and
prevents `resume=auto` from mixing optimizers from different configurations.

The launcher validates overrides against the selected trainer's actual CLI, not
a shortened notebook list. The dry-run cell prints every available key and the
fully resolved command. The training cell streams stdout/stderr live, writes a
timestamped `segment_*.log`, mirrors it to Drive, and prints the final 80 lines
when a subprocess fails.

Speaker CLUB can optionally use a sign-preserving, per-frame normalized gradient
on its private `z_L` branch:

```python
OVERRIDES = {
    "club_grad_norm": True,
    "club_grad_norm_target": 0.005,
}
```

This is not a GRL: the CLUB minimization direction is preserved. The delivered
per-frame magnitude is `club_weight * club_grad_norm_target`; accumulation and
FP16 scaling are compensated internally. Phoneme GRL-P is unchanged.

Every completed segment writes `latest-resume.pt`. `best.pt` and `final.pt` are
compact inference/analysis checkpoints without frozen SPEAR or optimizer state.
Exact resume rejects changed datasets and architecture-critical settings.
