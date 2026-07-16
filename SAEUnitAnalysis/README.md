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
health,atlas,selectivity,clustering,similarity,geometry,causal,swap
```

For the current dissertation analysis, keep the main run focused on
`health,selectivity,swap`. This answers the phone-vs-speaker unit question,
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
  --analysis health,selectivity,swap \
  --profile full \
  --split-limits train=3000,validation=1000,test=1000
```

Split caps are sampled deterministically and speaker-balanced; they are part of
the feature-cache fingerprint and run manifest.

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
only positive margins, and displays their activity on held-out test frames.

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

The `swap` analysis is currently a feature-space intervention, not an audio
generation or listening experiment. Its main condition combines recipient L
with donor P, reconstructs frozen SPEAR features, and measures recipient-phone
retention plus donor/recipient speaker evidence. Donor-L/recipient-P is the
main complementary control. The exported shuffled-route mask is only a
diagnostic because it can overlap true P units, particularly when learned
routing assigns roughly half the SAE capacity to P; it is not a clean negative
control. Independent phone and speaker evaluators are
fit on unswapped SAE reconstructions only, with disjoint fitting/evaluation
utterances; swapped examples are never used for fitting. Baseline reconstruction
scores must be inspected before interpreting any transfer. Waveform synthesis is
a later, separate validation step.

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
