# SAE unit analysis report: fixed `gn=1.5e-4`, `240L/16P`

This is a post-hoc unit analysis. The checkpoint is frozen; no training happens here. We run LibriSpeech utterances through the trained SAE, collect which units fire, and test whether individual units are associated with the available labels.

## Setup

| Item | Value |
|---|---:|
| Checkpoint | `libri_advfb_prrecover_gn00015_240L16P_aux64_12k_s42/stage2_step12000.pt` |
| SAE dictionary | `K=5120` |
| Route allocation | `4096 L / 1024 P / 0 U` |
| Active split per frame | `240 L / 16 P / 0 U` |
| Analysis data | 3000 LibriSpeech utterances |
| Available labels | `speaker_id`, `chapter_id`, `energy`, `voicing`, `speaking_rate` |
| Missing label | phone alignments |

The extraction is correct for this checkpoint: fixed blocks, `topk=256`, and `topk_L/topk_P/topk_U = 240/16/0`.

## How the metrics are computed

**Observed active unit:** a unit that fires at least once in this Libri analysis set. This is analysis-set coverage, not the same as training deadness. The training dead threshold was 256 inactive steps, while this analysis covers only 188 batches, so it cannot reproduce the training dead-unit rate.

**Speaker-selective unit:** for each unit, the analysis pools activation over each utterance and tests whether that unit is unusually associated with `speaker_id`. The score is an enrichment-style z score with FDR correction. A unit is flagged when it passes the significance/score threshold.

**Phone/content-selective unit:** not measured in this report. The current Libri bundle has no phone alignments, so `phone_selective = 0` means “not testable here,” not “no phone units exist.”

**Route violation:** under the speaker/content rule used here, an L unit is a violation if it is speaker-selective. Since phone labels are absent and there is no U block, the meaningful violation in this report is speaker leakage into L.

## Main results

| Measurement | Value | Interpretation |
|---|---:|---|
| Observed active units | 1617 / 5120 | Units that fired at least once in this analysis set. |
| Observed active L units | 593 / 4096 | Many L units were not observed in this subset; this is coverage, not training deadness. |
| Observed active P units | 1024 / 1024 | All P units fired at least once. |
| Speaker-selective L units | 7 / 4096 = 0.17% | Very small single-unit speaker leakage into L. |
| Speaker-selective active L units | 7 / 593 = 1.18% | Leakage remains small among observed active L units. |
| Speaker-selective P units | 0 / 1024 = 0.00% | No individual P unit is flagged as speaker-selective by this unit-level test. |
| Route violations | 7 / 5120 = 0.14% | Very low route-violation rate. |

The seven speaker-leaky L units are:

`3498, 921, 405, 817, 1939, 3921, 466`

## Necessary figures

### Route activity

![Route activity](../../../SAEUnitAnalysis/results/sae_units_gn00015_librispeech_full_240L16P_fixed/plots/route_activity.png)

This figure shows the firing-frequency distribution of observed active units by route. It is useful for checking that the corrected `240L/16P` extraction is being used and for understanding analysis-set coverage. P is fully observed in this bundle, while many L units do not fire in this Libri subset.

### Speaker association vs. firing frequency

![Speaker association vs firing frequency](../../../SAEUnitAnalysis/results/sae_units_gn00015_librispeech_full_240L16P_fixed/plots/frequency_vs_speaker_selectivity.png)

This figure is the speaker-only version of the frequency/selectivity plot. The x-axis is how often a unit fires; the y-axis is the unit's speaker-association score. The circled units are the seven L units flagged as speaker-selective.

This plot measures single-unit speaker association. It does not measure whether speaker identity is decodable from the whole P representation. In this analysis, P has no individually speaker-selective units, so P speaker information, if present, is distributed rather than localized to single units.

### Top leaky units

![Top leaky units](../../../SAEUnitAnalysis/results/sae_units_gn00015_librispeech_full_240L16P_fixed/plots/top_leaky_units.png)

This figure lists the actual route-violating units. All violations are `speaker_in_L`, and there are only seven. These are the best units to inspect or ablate later.

## Concise interpretation

The corrected `240L/16P` SAE analysis supports low single-unit speaker leakage into L: only 7 of 4096 L units are speaker-selective. This report cannot make unit-level claims about phone/content selectivity because the Libri bundle has no phone alignments.
