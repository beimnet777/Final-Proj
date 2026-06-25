# probe_robust — structural & information-theoretic disentanglement

A dedicated branch for probe-architecture-agnostic, adversary-free
disentanglement mechanisms. Kept separate from the existing GRL/adversarial
code paths in the main `Disentanglement/` namespace so the boundary stays
clean for the dissertation's "structural vs adversarial" comparison.

## Why this branch exists

`dual_inv_v1_soft_nogrl` (perturbation + ARCTIC + cosine-per-frame invariance
+ variance floor) achieved `inv_L = 0.022` in training (cosine ≈ 0.978) but
the held-out diag probe found `z_L → SID ≈ 1.000` on LibriSpeech AND
`z_L → SID ≈ 0.997` on ARCTIC matched-distribution. The mechanism failed
*in-distribution*, refuting the initial "training/probe distribution mismatch"
hypothesis.

Diagnosis (see *Experiment Analyses/v1 Soft NoGRL Probe Failure*):
cosine slack at 0.978 leaves `sqrt(1 - 0.978²) ≈ 0.21` — ~21% of vector
magnitude orthogonal to the shared direction. Speaker info hides there. The
cosine-per-frame loss does not constrain magnitude, temporal statistics, or
orthogonal subspaces. Adding a single matched-probe loss (e.g. stats-pool
invariance) would defeat one probe but tailor to it. We want a fix that
holds across probe architectures.

## Mechanisms

| module | what it constrains | probe-robustness |
|---|---|---|
| `losses.vicreg_invariance_loss` | per-frame L2 between frame-aligned pairs | matches L2 magnitude AND direction; literature-canonical (Bardes/Ponce/LeCun 2022) |
| `losses.vicreg_covariance_loss` | off-diagonal squared cov over bucket dims | decorrelates dims → shrinks orthogonal-subspace escape; structural, basis-agnostic |
| `losses.variance_floor_loss` (re-export) | per-dim std ≥ γ | blocks trivial collapse |
| `club.CLUBSampled` | `I(z_L_pooled ; speaker_id)` upper-bounded | **Fano's inequality bounds error of ANY downstream probe** — theoretical probe-agnostic guarantee |

## What this branch deliberately rejects

- **Matched-arch GRL** — seed-fragile (statsgrl 0.006/0.378/0.418 across
  three probe seeds with patience=0)
- **Stats-pool invariance loss** — probe-arch tailored; defeats one probe
  by construction while leaving others potentially open
- **Bilinear-200-frame time resample** — unjustified time-warp assumption
  for cross-speaker pairs (SiamCTC 2025); unnecessary for perturbation pairs

## Current scope

v1.2: perturbation-only pair α + VICReg-full (L2 inv + var + cov) + CLUB
on `(z_L_mean_pool, speaker_id)` for the 251 LibriSpeech training speakers.

Single training run, single moderately-strong probe (stats-pool, patience=0,
ARCTIC matched + LibriSpeech) at end-of-run. Multi-seed × multi-arch
verification only if v1.2 hits the joint gate (`z_L SID < 0.10` AND
`z_L PR ≤ 0.10` on both probes).
