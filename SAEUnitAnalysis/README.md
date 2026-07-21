# SAEUnitAnalysis

Standalone unit-level analysis for the sparse speech representations trained in
`Disentanglement/`.  The package never trains or edits the analyzed checkpoint.

## Run

From the repository root:

```bash
python -m SAEUnitAnalysis \
  --checkpoint /path/to/model.pt \
  --data /path/to/analysis_bundle \
  --analysis health,selectivity
```

The only required inputs are `--checkpoint`, `--data`, and `--analysis`.
Analyses may be comma-separated:

```text
health,atlas,selectivity,factor_metrics,clustering,similarity,geometry,causal,swap
```

For the current dissertation analysis, keep the main run focused on
`health,selectivity,factor_metrics,swap`. This answers the phone-vs-speaker unit question,
adds controlled full-space geometry and PCA/UMAP route views, and runs the
feature-level L/P swap intervention without enabling the broader similarity,
decoder-geometry, or prosody diagnostics.  The
default factor scope is also narrow: only phone identity and `speaker_id` are
scored.  Unit ranking uses `--score-splits train,validation` by default, so the
test split is not used to choose interesting units. Use `--factor-scope broad`
or `--score-splits all` only for deliberate side diagnostics.

Use `--profile quick` for a deterministic wiring check. Quick mode selects up
to eight speakers with three utterances each per split; it is still not a
scientific substitute for the full run. Full mode extracts all declared
train/validation/test utterances. Results default to
`SAEUnitAnalysis/results/<checkpoint>/<dataset fingerprint>/`.

For a reproducible 5,000-utterance run that preserves the complete validation
and held-out test splits, use:

```bash
python -m SAEUnitAnalysis \
  --checkpoint /path/to/model.pt \
  --data /path/to/analysis_bundle \
  --analysis health,selectivity,factor_metrics,swap \
  --profile full \
  --split-limits train=3000,validation=1000,test=1000
```

Split caps are sampled deterministically and speaker-balanced; they are part of
the feature-cache fingerprint and run manifest.

For the rolling dead-unit diagnostic alone, request `--analysis deadness`. It
requires at least two windows of `dead_steps_threshold` batches (8,192
utterances for batch size 16 and threshold 256), replays ten shuffled orders,
and avoids writing another large feature cache. This is a frozen-checkpoint,
valid-frame analogue of the trainer counter, not a reconstruction of its
unsaved historical state.

For a one-off large full-suite run, add `--no-persist-cache` to reuse any
compatible smaller cache during extraction without retaining the expanded
multi-gigabyte feature NPZ afterward. The report tables and figures are still
written normally.

The unit atlas is table-only by default: it writes the main report and CSV
tables, but does not generate thousands of low-information per-unit HTML pages,
per-example spectrogram PNGs, activation-trace PNG/CSVs, or audio controls.
If you need rich per-unit pages for manual inspection, opt in explicitly:

```bash
python -m SAEUnitAnalysis \
  --checkpoint /path/to/model.pt \
  --data /path/to/analysis_bundle \
  --analysis atlas \
  --atlas-assets traces,spectrograms,audio
```

## Phone/Speaker unit scores

The focused selectivity run writes:

```text
tables/unit_phone_speaker_scores.csv
tables/phone_units_ranked.csv
tables/speaker_units_ranked.csv
tables/entangled_units_ranked.csv
```

For each phone, a scalable AUROC is computed from the binary frame-level Top-K
selection indicator:

```text
AUROC_active      = 0.5 + 0.5 * (TPR - FPR)
directional_AUROC = 2 * (AUROC_active - 0.5)
positive_AUROC    = max(0, directional_AUROC)
```

`TPR` is the fraction of frames of that phone where the unit is selected by
Top-K, and `FPR` is the fraction of other frames where it is selected. Speaker
association instead uses the positive point-biserial correlation between the
speaker label and the unit's mean activation amplitude over an utterance. The
old “fired at least once” value remains a coverage diagnostic because it
saturates on long utterances. Speakers are scored within each dataset split so
LibriSpeech split differences cannot masquerade as speaker identity.

The default composite scores are:

```text
PhoneScore   = 0.5 * max_phone_positive_AUROC + 0.5 * mean_phone_positive_AUROC
SpeakerScore = max_speaker_positive_mean_activation_r
D            = (PhoneScore - SpeakerScore) / (PhoneScore + SpeakerScore + eps)
M            = PhoneScore + SpeakerScore
```

Headline categorical selections use these positive practical effect sizes
(plus the phone AUPRC-gain requirement), not frame-wise q-values. Frames from
one utterance are correlated, so the exported frame z/q columns are diagnostic
only and must not be interpreted as independent-sample significance tests.

High-phone and high-speaker thresholds default to the 90th percentile across
units and can be changed with `--threshold-percentile`. These scores are useful
for ranking the “highly associated” units used by the focused route summaries;
all lower-ranked per-level associations remain in `unit_factor_scores.csv`.
They are not yet the controlled ΔR² scores;
phone-controlled SID and speaker-controlled phone scores should be treated as
the next methodological layer.

The selected phone-unit matrix assigns unique units on train+validation by
maximizing `P(active|target phone) - max_other P(active|other phone)`, keeps
only positive selection margins, and evaluates the same assignments on held-out
test frames. The primary atlas shows all 39 assignments grouped by phonetic
family: filled circles are held-out test specificity margins, diamonds are the
original train+validation margins, and connecting lines show generalization
shifts. This value is an activity-specificity margin, not correlation. The
complete 39-phone raw coverage heatmap is retained as a deprecated diagnostic.

Route-vector figures use centered PCA. L and P display the same held-out
observations and labels. Labels are selected on the probe-training partition
with a route-neutral full-SAE cosine margin. The report gives frozen linear-
probe balanced accuracy, stratified nearest-centroid accuracy, and cosine
margin on the untouched evaluation partition in the original route space. The
expected signature is higher phone accuracy in L and higher speaker accuracy
in P; use these metrics before interpreting the two-dimensional picture.

The same observations are also shown with deterministic cosine UMAP
(`n_neighbors=30`, `min_dist=0.1`, seed 42). UMAP is supplementary: it is useful
for local neighbourhoods but can visually exaggerate gaps. PCA is the linear
global view, while quantitative claims should rely on the held-out metrics and
the full-space geometry table.

The classifier-free geometry analysis trains no prediction model. For every
anchor it compares cosine similarity to a same-label and a different-label
partner using identical pairs in L and P. Phone pairs cross speakers and
utterances; speaker pairs cross transcripts/content and utterances. The report
shows the paired same-minus-different cosine gap with cluster-bootstrap 95%
intervals. Disentanglement predicts a larger phone gap in L and a larger
speaker gap in P.

## Speech-adapted MIG, DCI and SAP

The optional `factor_metrics` analysis reuses the extracted sparse cache; it
does not rerun SPEAR or edit the checkpoint. It forms contiguous phone runs in
the held-out test split and represents each run by its mean SAE vector, retaining
the strongest `Top-K` mean activations. Observed speaker-phone cells are capped
approximately equally before scoring so common speakers and phones do not
dominate the natural, incomplete factorial design.

The evaluated object is the predefined subspace partition `z=(z_L,z_P)`, not
independence among units inside either route. Every primary score uses the
complete observed `z_L` or `z_P` vector on identical observations. Grouped
Route-MIG is the predictive lower bound

```text
RouteMIG(route, factor) = I(factor; held-out route prediction) / H(factor).
```

Grouped SAP is held-out balanced accuracy from a linear SVM trained on the
complete route. DCI informativeness is held-out balanced accuracy from an
ExtraTrees predictor. The chance-corrected DCI evidence matrix has two rows
(`L`, `P`) and two columns (phone, speaker); standard DCI entropy equations are
applied to this group matrix. The main contrasts are phone `L-P` and speaker
`P-L`, so positive values support the intended routing.

All metrics use ten repeated utterance-grouped fitting splits. Their intervals
measure estimator split stability rather than population-level uncertainty.
Shuffled-label controls are fitted independently. An additional capacity
control retains the same number of most-active observed units in `L` and `P`,
testing whether the crossover survives unequal route widths.

The earlier coordinate-gap MIG/SAP computation is retained as
`deprecated_unit_compactness_metrics.csv` for provenance. It measured
concentration into individual units and is not used to claim L/P subspace
disentanglement.
Label-shuffled controls are exported beside the observed scores:

```text
tables/speech_factor_metrics.csv
tables/route_dci_evidence.csv
tables/speech_factor_metric_repeats.csv
tables/deprecated_unit_compactness_metrics.csv
plots/route_factor_information.png
plots/route_factor_information_matrix.png
plots/route_factor_contrasts.png
speech_factor_metrics.json
```

To create the cross-checkpoint forest and learned-training trajectory after
upgrading result folders, run:

```bash
python -m SAEUnitAnalysis.compare_factor_metrics
```

These remain labelled post-hoc metrics. Controlled geometry and latent swapping
provide the classifier-free and intervention evidence.

The `swap` analysis is a feature-space intervention. Its main condition combines
recipient L with donor P, reconstructs frozen SPEAR features, and measures
recipient-phone retention plus donor/recipient speaker evidence. Donor-L /
recipient-P is the complementary intervention. Identity-P, same-speaker,
zero/mean-P and time-shuffled P provide controls. A P subset and a disjoint
non-P subset are paired one-to-one by unit activity and decoder-column norm.
This remains capacity-matched when learned routing assigns more than half of
the dictionary to P. The earlier shuffled-route
mask is retained as a diagnostic because it can overlap true P units; it is not
a clean negative control. Continuous P interpolation is evaluated at
`alpha=0,.25,.5,.75,1`.

The analysis writes `tables/swap_pairs.csv`. Supply that file to subsequent
checkpoints to force exactly the same intervention design:

```bash
python -m SAEUnitAnalysis \
  --checkpoint /path/to/another.pt \
  --data /path/to/analysis_bundle \
  --analysis swap \
  --split-limits train=3000,validation=1000,test=1000 \
  --swap-pair-manifest /path/to/reference/tables/swap_pairs.csv
```

Primary uncertainty is computed from paired mode-minus-baseline effects with a
two-way cluster bootstrap over recipient and donor speakers. Independent phone
and speaker feature evaluators are fit on unswapped SAE reconstructions only,
with disjoint fitting/evaluation
utterances; swapped examples are never used for fitting. Baseline reconstruction
scores must be inspected before interpreting any transfer.

After the fixed, post-GP and ramp-5k reports have been recomputed with the same
pair manifest, create the registered cross-checkpoint comparison with:

```bash
python -m SAEUnitAnalysis.compare_swap_protocols
```

This writes a double-dissociation profile and an equal-unit-count P-subset versus
non-P control figure to `results/swap_protocol_comparison_5k/`.

## Waveform conversion

Waveform validation is deliberately separate from `swap`: the SAE checkpoint is
never trained or edited. A single bridge is trained on ordinary unswapped pairs
from the bundle's training split:

```text
original SPEAR h -> predicted log-mel -> frozen vocoder -> waveform
```

The fixed, post-GP and ramp-5k Libri checkpoints share the same SPEAR revision,
layer-normalisation setting and 1280-dimensional feature domain, so one bridge
can be reused. Train it on CUDA for the scientific run:

```bash
python -m SAEUnitAnalysis.train_audio_bridge \
  --checkpoint /path/to/reference-final.pt \
  --data data/sae_analysis/librispeech_bundle_12k_mfa \
  --output-dir SAEUnitAnalysis/audio_models/spear_bigvgan_bridge \
  --device cuda --epochs 8 --batch-size 4
```

For an MPS wiring test, bound the data and model size:

```bash
python -m SAEUnitAnalysis.train_audio_bridge \
  --checkpoint /path/to/reference-final.pt \
  --data data/sae_analysis/librispeech_bundle_12k_mfa \
  --output-dir SAEUnitAnalysis/audio_models/smoke \
  --device mps --epochs 1 --batch-size 1 \
  --max-train-utterances 8 --max-validation-utterances 4 \
  --hidden-dim 64 --residual-layers 2
```

Render a bounded evidence set after re-running the upgraded feature swap:

```bash
git clone https://github.com/NVIDIA/BigVGAN.git /path/to/BigVGAN

python -m SAEUnitAnalysis.render_audio_swaps \
  --result-dir /path/to/SAEUnitAnalysis/result \
  --bridge SAEUnitAnalysis/audio_models/spear_bigvgan_bridge/best.pt \
  --device cuda --vocoder bigvgan --bigvgan-repo /path/to/BigVGAN \
  --max-pairs 24 \
  --interpolation-pairs 5 --grid-size 5
```

This saves recipient/donor references and the three reconstruction gates before
the route interventions. Only the requested demonstration pairs are persisted;
no spectrogram images or full-dataset waveform dump is produced. `griffinlim`
is available for a dependency-light diagnostic smoke test but is explicitly
marked unsuitable for audio-quality claims.

Finally, run independent ASR and open-set speaker evaluation. Enrollment uses
original utterances that exclude both members of each intervention pair:

```bash
python -m SAEUnitAnalysis.audio_evaluation \
  --audio-dir /path/to/result/audio_conversion \
  --data data/sae_analysis/librispeech_bundle_12k_mfa \
  --device cuda
```

The renderer creates an audio HTML report and links it from the main unit report.
The evaluator writes per-pair WER/CER, donor and recipient speaker cosine scores,
paired two-way-clustered contrasts, and a separate evaluation report. Objective
scores do not replace a blinded naturalness and speaker-similarity listening
study.

Once all reconstruction gates pass, prepare (but do not execute) a blinded
listening-study pack:

```bash
python -m SAEUnitAnalysis.prepare_listening_study \
  --audio-dir /path/to/result/audio_conversion \
  --max-pairs 24
```

The pack copies clips to neutral filenames, randomizes trials, separates the
private condition key, and provides naturalness and donor-versus-recipient ABX
forms. Human recruitment still requires the appropriate ethics approval and
consent; the command does not collect or invent participant data.

### Direct SPEAR-conditioned HiFi-GAN

The direct path is a separate experiment and does not replace or overwrite the
mel bridge above:

```text
original or SAE-reconstructed SPEAR h (50 Hz, 1280-D)
    -> HiFi-GAN V1 generator (320x)
    -> 16 kHz waveform
```

The total upsampling is 320 samples per SPEAR frame, matching the
frozen encoder exactly. The full generator is compatible with the public
kNN-VC WavLM HiFi-GAN: 235/236 parameter tensors are warm-started, while only
the input projection changes from 1024-D WavLM to 1280-D SPEAR. Training begins
with a short mel-only alignment phase, then uses the standard HiFi-GAN
least-squares adversarial, feature-matching, and `45 * log-mel L1` losses.

First cache frozen SPEAR features once. The cache contains train and validation
only, is stored as one float16 binary array plus an index, and is reusable by
all SAE checkpoints sharing the same SPEAR domain:

```bash
python -m SAEUnitAnalysis.cache_spear_audio_features \
  --checkpoint /path/to/reference-final.pt \
  --data data/sae_analysis/librispeech_bundle_12k_mfa \
  --output-dir SAEUnitAnalysis/audio_models/spear_direct_cache \
  --device cuda --batch-size 4 \
  --max-validation-utterances 1000
```

Then train the full direct vocoder:

```bash
python -m SAEUnitAnalysis.train_direct_hifigan \
  --cache SAEUnitAnalysis/audio_models/spear_direct_cache \
  --data-root data/sae_analysis/librispeech_bundle_12k_mfa \
  --output-dir SAEUnitAnalysis/audio_models/spear_direct_hifigan \
  --device cuda --model-size full --max-steps 250000 \
  --batch-size 16 --segment-frames 24
```

`last.pt` contains optimizers and resumes training; `best.pt` is the
validation-mel-selected inference checkpoint. Only three numbered recovery
checkpoints are retained. Validation previews are placed under `samples/`; no
spectrogram images or full waveform dump is produced.

Once the original-SPEAR and SAE-baseline reconstruction gates are intelligible,
render the same registered route interventions without the mel bridge:

```bash
python -m SAEUnitAnalysis.render_audio_swaps \
  --result-dir /path/to/SAEUnitAnalysis/result \
  --direct-hifigan SAEUnitAnalysis/audio_models/spear_direct_hifigan/best.pt \
  --device cuda --max-pairs 24 --interpolation-pairs 5 --grid-size 5
```

The CSD3 Slurm job performs both phases. It uses the existing
`Probing/data/LibriSpeech` tree and creates a symlink-only audio bundle under
the ignored `SAEUnitAnalysis/audio_models/` directory; the local MFA bundle is
not required for vocoder training. It validates the checkpoint, all three
LibriSpeech splits, registered test-pair audio, Python imports and CUDA before
extraction. It downloads and checksum-verifies the published 63 MiB warm start
once, then resumes from `last.pt` when present. After reaching `MAX_STEPS`, it
automatically renders ten held-out, speaker-diverse registered pairs. Each
report row contains recipient and donor references, original-SPEAR
reconstruction, SAE reconstruction, recipient-L + donor-P swapping, and the
complementary L swap:

```bash
sbatch SAEUnitAnalysis/slurm/train_direct_hifigan_blackwell.sh
```

The final listening page is written to:

```text
SAEUnitAnalysis/audio_models/spear_direct_hifigan_knnvc_init/
  final_demo_10_pairs/report/index.html
```

A bounded local wiring test uses the explicitly reduced model; its sound is not
an audio-quality result:

```bash
python -m SAEUnitAnalysis.train_direct_hifigan \
  --cache SAEUnitAnalysis/audio_models/spear_direct_cache_smoke \
  --data-root data/sae_analysis/librispeech_bundle_12k_mfa \
  --output-dir SAEUnitAnalysis/audio_models/spear_direct_hifigan_smoke \
  --device mps --model-size smoke --max-steps 2 --batch-size 1 \
  --segment-frames 4 --adversarial-start-step 1 \
  --validation-interval 1 --checkpoint-interval 2 \
  --validation-batches 1 --num-workers 0 --pretrained-generator none
```

## Analysis bundle version 1

An analysis bundle is deliberately independent of MSP-Podcast's original archive
layout. Paths in the manifest are relative to the bundle root.

```text
my_bundle/
  dataset.yaml
  utterances.csv
  alignments.csv                 # parquet is also supported
  audio/
    example.wav
```

Minimal `dataset.yaml` (JSON is valid YAML and works without PyYAML):

```yaml
schema_version: 1
sample_rate: 16000
manifest: utterances.csv
alignments: alignments.csv
splits:
  train: train
  validation: val
  test: test
factors:
  - {name: phone, family: linguistic, level: frame, type: categorical, source: alignment}
  - {name: speaker_id, family: paralinguistic, level: utterance, type: categorical, source: speaker_id}
```

### Recommended dissertation bundles

Use two complementary bundles rather than forcing one dataset to answer every
question:

1. **LibriSpeech in-domain bundle** — the primary evidence for the
   disentanglement experiments. Use the same domain as training/probing, with
   `speaker_id`, independent phone alignments, and transcripts. This is the
   bundle to cite for “does `z_L` keep phones while removing speaker
   information?”.
2. **TIMIT phonetic-validation bundle** — a cleaner phonetic sanity check. TIMIT
   has human phone boundaries, so it is useful for inspecting whether individual
   SAE units are phone/manner/place/boundary units. It should be treated as
   out-of-domain validation, not as a replacement for the LibriSpeech
   disentanglement result.

TIMIT-style manifest columns can be:

```csv
utterance_id,audio_path,split,transcript,speaker_id,sex,dialect_region
dr1-fcjf0-sa1,audio/dr1/fcjf0/sa1.wav,test,she had your dark suit in greasy wash water,fcjf0,F,DR1
```

TIMIT-style factors:

```yaml
factors:
  - {name: phone, family: linguistic, level: frame, type: categorical, source: alignment}
  - {name: speaker_id, family: paralinguistic, level: utterance, type: categorical, source: speaker_id}
```

If `factors` is omitted, the bundle loader auto-detects only the core
phone/speaker factors. Extra factors can be declared manually and scored with
`--factor-scope broad`, but they are intentionally not part of the main
phone-vs-speaker result.

Required manifest columns:

```csv
utterance_id,audio_path,split,transcript,speaker_id,emotion
u001,audio/u001.wav,train,example words,spk01,neutral
```

Required alignment columns:

```csv
utterance_id,start_sec,end_sec,phone
u001,0.00,0.12,EH
```

Phone timings must come from an aligner independent of the checkpoint's CTC
head. Missing factors are allowed for descriptive analyses; `causal`, `swap`,
and `all` require train/validation/test splits, phone alignments, and speakers.

For a first pass, run:

```bash
python -m SAEUnitAnalysis \
  --checkpoint /path/to/stage2.pt \
  --data /path/to/librispeech_analysis_bundle \
  --analysis health,atlas,selectivity
```

This first pass keeps the atlas lean. Add `--atlas-assets traces` only if you
want per-unit activation trace plots; add `spectrograms` or `audio` only for a
small manual audit run.

Use `clustering`, `similarity`, `geometry`, `causal`, and `swap` only as
secondary checks after the phone/speaker tables look sane.

If TIMIT or independent LibriSpeech phone alignments are not available yet, do
not fabricate frame-level phone labels.  Build an in-domain LibriSpeech bundle
without alignments and run the descriptive analyses:

```bash
python -m SAEUnitAnalysis.build_librispeech_bundle \
  --librispeech-root /scratch/$USER/data/LibriSpeech \
  --output /scratch/$USER/data/sae_analysis/librispeech_bundle \
  --max-train 2000 \
  --max-validation 500 \
  --max-test 500

python -m SAEUnitAnalysis \
  --checkpoint /path/to/stage2.pt \
  --data /scratch/$USER/data/sae_analysis/librispeech_bundle \
  --analysis health,atlas,selectivity
```

Without `alignments.csv`, phone scores are skipped, but speaker selectivity,
unit health, route summaries, top examples, and the HTML report still run.
`causal`, `swap`, and `all` still require independent phone alignments.

### LibriSpeech phone alignments with MFA

For in-domain phone interpretability on LibriSpeech, first build a Libri bundle,
then prepare a Montreal Forced Aligner corpus from that bundle:

```bash
python -m SAEUnitAnalysis.build_librispeech_bundle \
  --librispeech-root /scratch/$USER/data/LibriSpeech \
  --output /scratch/$USER/data/sae_analysis/librispeech_bundle_more \
  --max-train 10000 \
  --max-validation 1000 \
  --max-test 1000

python -m SAEUnitAnalysis.prepare_librispeech_mfa_corpus \
  --bundle /scratch/$USER/data/sae_analysis/librispeech_bundle_more \
  --output /scratch/$USER/data/sae_analysis/mfa_librispeech_corpus
```

Run MFA outside this package:

```bash
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
mfa align \
  /scratch/$USER/data/sae_analysis/mfa_librispeech_corpus \
  english_us_arpa \
  english_us_arpa \
  /scratch/$USER/data/sae_analysis/mfa_librispeech_aligned
```

Then import the MFA TextGrid phone intervals back into an SAE bundle:

```bash
python -m SAEUnitAnalysis.import_mfa_alignments \
  --bundle /scratch/$USER/data/sae_analysis/librispeech_bundle_more \
  --mfa-output /scratch/$USER/data/sae_analysis/mfa_librispeech_aligned \
  --utterance-map /scratch/$USER/data/sae_analysis/mfa_librispeech_corpus/mfa_utterance_map.csv \
  --output /scratch/$USER/data/sae_analysis/librispeech_bundle_more_mfa
```

By default, the importer strips ARPABET stress digits (`AH0 -> AH`) and skips
silence/noise intervals. Use `--preserve-stress` or `--keep-silence` if those
distinctions are part of the question.

For a TIMIT phonetic-validation bundle, use:

```bash
python -m SAEUnitAnalysis.build_timit_bundle \
  --timit-root /scratch/$USER/data/TIMIT \
  --output /scratch/$USER/data/sae_analysis/timit_bundle
```

## Old checkpoints

The loader accepts `model`, `model_state`, and raw state dictionaries. It reads
configuration in this order:

1. `analysis_config` embedded in the checkpoint;
2. `<checkpoint>.analysis.yaml`, `<stem>.analysis.yaml`, or a directory-level
   `analysis_config.yaml`;
3. the repository experiment index;
4. tensor shapes and reconstruction calibration.

The tool fails if two extraction configurations remain within 2% reconstruction
MSE. A sidecar for an old checkpoint can be as small as:

```yaml
topk: 256
spear_layernorm: true
hard_gumbel_routing: true
```

Fixed per-block Top-K checkpoints should also provide:

```yaml
block_topk: [160, 64, 32]
```

If `per_block_topk` is false, analysis preserves global Top-K even when fixed
route membership buffers are present. Learned quota-frozen checkpoints read
their persistent `sae.route_topk_idx` and `sae.route_topk_quotas` buffers, so
post-hoc extraction matches the representation used after freezing.

## Interpretation

Direct cosine similarity between `z_L` and `z_P` is intentionally not reported:
hard masks make it zero by construction. Causal metrics use separately trained
evaluators on original SPEAR features; checkpoint task/adversary heads are not
used as primary evidence. Because LibriSpeech speakers are disjoint across
official splits, the speaker evaluator uses a stratified utterance holdout
within test speakers and interventions run only on reserved utterances.
Swapping reconstructs SPEAR features only and does not synthesize waveform
audio.

## Tests

```bash
python -m unittest discover -s SAEUnitAnalysis/tests -v
```
