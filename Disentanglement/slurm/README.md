# Disentanglement Slurm Scripts

Use scripts under `active/` for new submissions. Scripts under `archive/` are
kept for provenance and should not be submitted without re-checking paths,
arguments, and log destinations.

## Active scripts

- `active/training/stage1.sh`: current stage-1 SAE training entrypoint.
- `active/training/stage2.sh`: current generic stage-2 training entrypoint.
- `active/probing/probe_best5_diag.sh`: seeded, cheap diagnostic best-5 probe.
  This is not an official SUPERB evaluation.
- `active/diagnostics/decor_diag.sh`: post-hoc decorrelation diagnostic.

## Archive

- `archive/probing_legacy/`: older probe scripts, including the misnamed
  `probe_best5_superb.sh`.
- `archive/stage1_legacy/`: older stage-1 run/sweep scripts.
- `archive/stage2_legacy/`: older stage-2 run/sweep/launcher scripts,
  including the previous one-stage weak-GRL train-and-old-probe script.

Archived scripts are historical evidence, not trusted current entrypoints.
