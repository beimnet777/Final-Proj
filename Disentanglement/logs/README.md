# Disentanglement Logs

This folder separates training logs, diagnostic probe logs, analysis figures,
and archived failed/incomplete runs.

## Training Logs

- `train/stage1/`: stage-1 SAE training logs.
- `train/stage2/main/`: main stage-2 training logs.
- `train/stage2/sweep/`: early SID/GRL sweep logs.
- `train/stage2/beta_sweep/`: beta sweep logs.
- `train/stage2/experiments/`: individual stage-2 experiment logs.
- `train/stage2/one_stage/`: one-stage weak-GRL train-and-probe logs.
- `train/stage2/archive_failed_or_incomplete/`: logs kept for provenance, not
  clean result runs.

## Probe Logs

- `probes/diagnostic_historical/`: older diagnostic probe results.
- `probes/diagnostic_historical/successful/`: older completed diagnostic probe
  runs that were previously grouped as successful.
- `probes/archive_failed_or_incomplete/`: failed, incomplete, or stale probe
  runs kept for debugging history.
- `probes/diag_best5_old_lr_mislabeled_superb/`: previous best-5 probe run that
  was misnamed as SUPERB and used the underpowered SID probe setup. Treat these
  as diagnostic-only and not comparable to official SUPERB.
- `probes/diag_best5_seeded/`: target location for the new seeded diagnostic
  best-5 probes.

## Figures

- `../figures/`: curated report-level figures.
- `analysis_figures/`: run-specific generated analysis images tied to logs.

Use exact folder names when reporting results so diagnostic, legacy, and
official-style evaluations are not mixed.
