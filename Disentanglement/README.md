# Speech Disentanglement — project guide

> Orientation for humans **and** for AI coding agents. Read this first, then
> open [`EXPERIMENTS.md`](EXPERIMENTS.md) for the list of every run and its result.

## 1. What this project does

Stage 1 trains a **sparse autoencoder (SAE)** on frozen speech-encoder features.
Stage 2 splits the SAE latent into buckets and uses adversaries to **disentangle**
them:

| bucket | meaning | should contain | should NOT contain |
|---|---|---|---|
| `z_L` | linguistic / content | phonetic content | speaker identity |
| `z_P` | paralinguistic / speaker | speaker identity | content |
| `z_U` | residual / everything-else | leftovers | — (some runs drop it → "2-way") |

A run "works" when `z_L` keeps content but sheds speaker, and `z_P` keeps speaker
but sheds content.

## 2. Metrics & vocabulary (read before interpreting any number)

| term | what it measures | good direction |
|---|---|---|
| **PR / PER** | phoneme error rate — SUPERB 74-phone CTC readout | `z_L`: **low** (content present) · `z_P`: **high** (content removed) |
| **SID** | speaker-ID accuracy — in-house closed-set LibriSpeech-100, **251 spk, chance = 0.004** | `z_L`: **low** (speaker removed) · `z_P`: **high** (speaker present) |

- **Stats probe** is the trustworthy evaluator: `Linear → ReLU → mean+std pool → Linear`.
  The plain linear+mean-pool probe is foolable by mean-removal — the *IN-liar*
  artifact (instance-norm gave `z_L` SID 0.002 linear but 0.74 stats). **Quote
  stats-probe numbers.**
- **Build heads vs adversaries.** In the per-bucket val line, `z_L PR` and
  `z_P SID` come from the *build* heads (the readouts we keep); `z_L SID` and
  `z_P PR` come from the *adversary* heads (the leakage we push down).

**Two unrelated "grad norm" knobs (naming collision — do not confuse):**
- `--grl_grad_norm` / `--grl_p_grad_norm` — per-frame normalize the *reversed
  adversary gradient* to a constant target (e.g. 0.001). This is the mechanism
  the winning `dense_gradnorm` run uses.
- `--gradnorm` (Chen et al. 2018) — a *loss-weight balancer* across tasks
  (`recon,pr,sid,aux`). It tunes loss weights, not gradients into the latent.
  Lives in [`gradnorm.py`](gradnorm.py).

## 3. Directory layout

| path | what |
|---|---|
| `run.py` | entry point: parse args → build cfg → train |
| `train.py` | training loop (stage 1 & 2), eval, in-loop probe |
| `config.py` | `Config` defaults / dataclass |
| `gradnorm.py` | Chen-et-al GradNorm loss-weight controller |
| `losses.py` | loss functions |
| `model/` | SAE, build + adversary heads, routing |
| `data/` | dataset, collate, perturbation |
| `diag_probe/` | standalone post-hoc probing (`probe_runner.py`, `run.py`, grad diagnostics) |
| `slurm/active/training/` | **live** SLURM job scripts |
| `logs/train/stage1/` | stage-1 SAE training `.out` logs |
| `logs/train/stage2/<topic>/` | stage-2 training `.out` logs, grouped by topic |
| `logs/diag/<probe>/` | diagnostic probe `.out` logs |
| `checkpoints/<run>/stage2_best.pt` | best checkpoint per run (per-step pruned; only `*_best.pt` kept) |
| `checkpoints/best.pt`, `checkpoints/ln_sae/stage1_best.pt` | base stage-1 SAEs |
| `runs/<run>/` | TensorBoard event files |
| `analysis/` | figures, CSVs, reports, **`build_index.py`** |
| **`EXPERIMENTS.md`**, **`experiments.json`** | the experiment index (auto-generated) |
| `README.md` | this file |

**Legacy (pre-reorg, superseded — do not add to):** `logs/probes/`,
`logs/figures/`, `logs/analysis_figures/`. Current probes go in `logs/diag/`.

## 4. The experiment index ← single source of truth

[`EXPERIMENTS.md`](EXPERIMENTS.md) (human) and `experiments.json` (machine) list
**every run**: date, run/job id, what it tested (the log banner), key config,
final result, and whether the checkpoint exists. Re-runs of the same experiment
are collapsed into one row.

**Regenerate after any new job finishes:**
```bash
python analysis/build_index.py
```
The generator parses the `.out` logs only — it never hand-edits, so the index is
always reproducible from the logs.

## 5. Anatomy of a stage-2 `.out` log

```
=== <run>: <one-line description> ===  <date>      ← banner (parsed into the index)
[stage 2] frozen=… trainable=… device=…            ← param counts
[stage 2] α=… β=… grl=… ρ=… grl_p=… …              ← hyperparameters (parsed: "key config")
step N/M  loss=… recon=… pr=… sid=… …  gn=[…]       ← gn=[…] only when --gradnorm is on
[val] step=N recon=… pr=… | z_L PR=… SID=… | z_P PR=… SID=… | z_U PR=… SID=…
best checkpoint (disent=… PER=… sid=…) → …/checkpoints/<run>/stage2_best.pt
DIAGNOSTIC PROBE RESULTS - <tag>                    ← if the job probes after training
  z_L   <PER>   <SID>                               ← stats-probe PR / SID per bucket
Finished …
```

## 6. Naming convention (keep this so the index stays clean)

One name threads through everything: **SLURM `--job-name` = `RUN_NAME` =
`checkpoints/<run>/` = the log `<topic>` folder.** The banner
`=== ${RUN_NAME}: <desc> ===  $(date)` is exactly what `build_index.py` reads, so
keep that format in every job script.

## 7. Add a new experiment

1. Copy a script in `slurm/active/training/`, set `RUN_NAME`, keep the
   `echo "=== ${RUN_NAME}: <desc> ===  $(date)"` banner.
2. Submit (always confirm before `sbatch`).
3. When it finishes: `python analysis/build_index.py` to refresh the index.

## 8. Where the analysis lives

- Latest narrative writeup: `analysis/weekly_report_2026-06-12_to_06-18.md`.
- Report figures + generators: `analysis/figures/`, `analysis/make_figures.py`,
  `analysis/make_scaling_figures.py`.
