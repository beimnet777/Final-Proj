# MSP-Podcast disentanglement (standalone)

Self-contained pipeline that trains content (PR) + speaker (SID) + prosody +
emotion disentanglement on **one** dataset (MSP-Podcast 2.0), so every batch
carries all labels. It does not modify the legacy trainer or CLI. It reuses
`DISConfig`, the model, loss primitives, and shared runtime helpers as read-only
libraries.

## Why this exists (vs the old Libri + IEMOCAP-every-8 setup)
The old emotion path failed: emotion CE was ~2–5% of a clamped IEMOCAP aux bundle
(dominated by prosody), fired only every 8 steps, and the anti-emotion GRL was 0.2
and ramped — so z_L kept *more* emotion than z_P. Fixes baked in here:
- **one dataset, per-batch emotion** — no `emotion_every`, no `_cap_loss_by_scaling`.
- **full-strength anti-emotion GRL** on z_L, with a canonical DANN sigmoid ramp
  controlled by `--dann_ramp_steps` (default reaches 1.0 by 500 steps).
- **per-frame normalized speaker-GRL gradient** on z_L (initial-search target 0.0002).
- **class-weighted emotion CE + UAR** reporting (MSP is neutral-heavy).
- **PCGrad** gradient surgery over the cooperative tasks on the shared SAE trunk,
  to defuse the `cos(recon, grl)<0` conflicts. Adversaries are **excluded** — their
  opposition to reconstruction is the disentanglement mechanism, not a bug.
- Perturbation invariance is available but disabled by default. The first MSP
  search uses only labeled factors: content, speaker, prosody, and emotion.

## Data prep (one-time)
```bash
cd Disentanglement
# 1) build the speaker/emotion-balanced subset manifest (metadata only)
python msp/build_subset.py --n_speakers 700 --cap_per_speaker 160 \
    --min_utts 25 --min_emotions 3 --out data/msp_subset
# 2) extract just the subset WAVs (GNU tar mishandles this PAX archive; use ours)
python msp/extract_audio.py --members data/msp_subset/members.txt --dest data/msp_audio
```
- `data/msp_subset/` (manifest.csv, members.txt, speakers.csv) is small and tracked.
- `data/msp_audio/` is gitignored (large).
- PR targets come from the **human transcript** via the shared `text_to_phones()`
  (SUPERB ARPABET), **not** the ForceAligned phones tier (that mixes ARPABET/IPA).

## Train
```bash
cd Disentanglement
python -m msp.run --run_name msp_v1 --steps 12000            # production defaults
python -m msp.run --run_name msp_smoke --smoke               # 3-step wiring check
python -m msp.run --no_pcgrad --run_name msp_nopcgrad        # ablate the surgery
python -m msp.run --no-grl_grad_norm --run_name msp_raw_grl  # ablate normalization
```
Or `sbatch msp/slurm/train_msp.sh`.

## Files
| file | role |
|---|---|
| `config.py` | `MSPConfig` + `to_dis_cfg` (DISConfig with MSP defaults) |
| `data.py` | manifest → audio + transcript-phones + speaker + emotion + A/V/D |
| `build_subset.py` | speaker/emotion-balanced subset selector → manifest |
| `extract_audio.py` | resume-friendly streaming WAV extractor |
| `grad_conflict.py` | PCGrad over the cooperative tasks (shared SAE trunk) |
| `utils.py` | copied prosody/CTC/adversary helpers + optional invariance + UAR |
| `train.py` | the standalone multi-task loop + eval (PER / SID / emotion UAR) |
| `probe.py` | fresh frozen MSP probes on z_t / z_L / z_P |
| `run.py` | CLI |

## Eval readout
`[val]` prints `PER` (z_L content), `SID` (z_P speaker), `z_P emo UAR/acc`, and
leakage: `z_L SID`, `z_P PR-PER`, `z_L emo UAR`. The win condition vs the old run:
**z_P emo UAR ≫ z_L emo UAR** (emotion in the paralinguistic bucket, not content).
These are jointly trained head proxies. At completion, the last checkpoint
`final.pt` is evaluated once on `test`; metrics are stored in `metrics.jsonl` and
the checkpoint's `auxiliary.test` field. `best.pt` is retained only as a
validation-selected diagnostic because its mixed proxy score is not a reliable
basis for choosing the test-evaluated checkpoint.
The manifest's arousal/valence/dominance values are carried in each batch as
metadata but are not currently optimization targets.

For independent MSP-native leakage measurements, train fresh frozen probes:
```bash
python -m msp.probe \
    --checkpoint msp/checkpoints/msp_v1/final.pt \
    --manifest data/msp_subset --audio_root data/msp_audio \
    --transcripts /path/to/MSP-Podcast-2.0/Transcripts.zip \
    --output msp/checkpoints/msp_v1/probe_results.json
```
Each PR, SID, and emotion probe is selected on validation independently; the
test split is evaluated only after probe training.
