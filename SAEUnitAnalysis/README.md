# SAEUnitAnalysis

Standalone unit-level analysis for the sparse speech representations trained in
`Disentanglement/`.  The package never trains or edits the analyzed checkpoint.

## Run

From the repository root:

```bash
python -m SAEUnitAnalysis \
  --checkpoint /path/to/model.pt \
  --data /path/to/analysis_bundle \
  --analysis all
```

The only required inputs are `--checkpoint`, `--data`, and `--analysis`.
Analyses may be comma-separated:

```text
health,atlas,selectivity,clustering,similarity,geometry,causal,swap
```

Use `--profile quick` for a deterministic wiring check. Full mode evaluates the
complete declared test split. Results default to
`SAEUnitAnalysis/results/<checkpoint>/<dataset fingerprint>/`.

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
  - {name: emotion, family: paralinguistic, level: utterance, type: categorical, source: emotion}
  - {name: f0, family: paralinguistic, level: frame, type: continuous, source: "computed:f0"}
```

### Recommended dissertation bundles

Use two complementary bundles rather than forcing one dataset to answer every
question:

1. **LibriSpeech in-domain bundle** — the primary evidence for the
   disentanglement experiments. Use the same domain as training/probing, with
   `speaker_id`, independent phone alignments, transcripts, and computed
   acoustic factors (`f0`, `energy`, `voicing`). This is the bundle to cite for
   “does `z_L` keep phones while removing speaker information?”.
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
  - {name: sex, family: paralinguistic, level: utterance, type: categorical, source: sex}
  - {name: dialect_region, family: paralinguistic, level: utterance, type: categorical, source: dialect_region}
  - {name: energy, family: paralinguistic, level: frame, type: continuous, source: "computed:energy"}
  - {name: voicing, family: paralinguistic, level: frame, type: continuous, source: "computed:voicing"}
```

If `factors` is omitted, the bundle loader auto-detects common columns including
`speaker_id`, `sex`, `gender`, `dialect_region`, `age`, `emotion`, and computed
`f0/energy/voicing`.

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
  --analysis health,atlas,selectivity,clustering,similarity,geometry
```

Then rerun the same analysis on the TIMIT bundle. Use `causal` and `swap` only
after the descriptive tables look sane, because those train small external
evaluators and take longer.

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
  --analysis health,atlas,selectivity,clustering,similarity,geometry
```

Without `alignments.csv`, phone/manner/place/boundary scores are skipped, but
speaker/acoustic selectivity, unit health, route summaries, geometry, similarity
plots, top examples, and the HTML report still run.  `causal`, `swap`, and
`all` still require independent phone alignments.

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

## Interpretation

Direct cosine similarity between `z_L` and `z_P` is intentionally not reported:
hard masks make it zero by construction. Causal metrics use separately trained
evaluators on original SPEAR features; checkpoint task/adversary heads are not
used as primary evidence. Swapping reconstructs SPEAR features only and does not
synthesize waveform audio.

## Tests

```bash
python -m unittest discover -s SAEUnitAnalysis/tests -v
```
