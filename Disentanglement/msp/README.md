# MSP-Podcast disentanglement (standalone)

Self-contained pipeline that trains content (PR) + speaker (SID) + prosody +
emotion disentanglement on **one** dataset (MSP-Podcast 2.0), so every batch
carries all labels. It **does not modify or import** the legacy `train.py` /
`run.py` / `config.py`; it only reuses the model and loss primitives as read-only
libraries (`from model import build_dis_model`, `from losses import ...`).

## Why this exists (vs the old Libri + IEMOCAP-every-8 setup)
The old emotion path failed: emotion CE was ~2–5% of a clamped IEMOCAP aux bundle
(dominated by prosody), fired only every 8 steps, and the anti-emotion GRL was 0.2
and ramped — so z_L kept *more* emotion than z_P. Fixes baked in here:
- **one dataset, per-batch emotion** — no `emotion_every`, no `_cap_loss_by_scaling`.
- **full-strength anti-emotion GRL** on z_L (0.5, no ramp).
- **class-weighted emotion CE + UAR** reporting (MSP is neutral-heavy).
- **PCGrad** gradient surgery over the cooperative tasks on the shared SAE trunk,
  to defuse the `cos(recon, grl)<0` conflicts. Adversaries are **excluded** — their
  opposition to reconstruction is the disentanglement mechanism, not a bug.

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
python -m msp.run --run_name msp_v1 --steps 12000            # hard routing, PCGrad on
python -m msp.run --run_name msp_smoke --smoke               # 3-step wiring check
python -m msp.run --no_pcgrad --run_name msp_nopcgrad        # ablate the surgery
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
| `utils.py` | copied prosody/invariance/CTC/adversary helpers + UAR |
| `train.py` | the standalone multi-task loop + eval (PER / SID / emotion UAR) |
| `run.py` | CLI |

## Eval readout
`[val]` prints `PER` (z_L content), `SID` (z_P speaker), `z_P emo UAR/acc`, and
leakage: `z_L SID`, `z_P PR-PER`, `z_L emo UAR`. The win condition vs the old run:
**z_P emo UAR ≫ z_L emo UAR** (emotion in the paralinguistic bucket, not content).
These are in-training head proxies — run the existing `diag_probe/` for the
authoritative leakage signal.
