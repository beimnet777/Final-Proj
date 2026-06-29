# Colab disentanglement experiments

Use `notebooks/Disentanglement_Colab.ipynb` for MSP or LibriSpeech stage-2
experiments. The notebook is an orchestration layer; model and loss code remains
in the normal trainers.

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

Every completed segment writes `latest-resume.pt`. `best.pt` and `final.pt` are
compact inference/analysis checkpoints without frozen SPEAR or optimizer state.
Exact resume rejects changed datasets and architecture-critical settings.
