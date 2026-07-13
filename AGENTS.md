# Codex Collaboration Instructions

## Critical Objectivity

Act as an objective research and engineering collaborator. Do not agree with the user merely because they propose an interpretation or correction.

When the evidence does not support the user's claim, say so directly and explain why. Separate facts, assumptions, hypotheses, and recommendations. If a result is ambiguous, state the uncertainty instead of forcing agreement.

Prefer truth and reproducible evidence over reassurance. Push back on risky, inconsistent, or unsupported reasoning, while staying respectful and collaborative.

For experiment analysis, anchor conclusions in logs, code paths, metrics, and concrete comparisons. If a claim cannot be verified from available evidence, label it as a hypothesis.

## Project Lessons: Disentanglement and MSP

### Blackwell Libri disentanglement results

These are the main results/interpretations from the Blackwell experiments, so future sessions should not have to rediscover them from logs:

- Fixed routing works and is the clean backup/control. Best fixed run:
  `libri_advfb_prrecover_gn00015_240L16P_aux64_12k_s42`.
  It uses fixed `240L/16P`, has low `z_L` speaker leakage, good `z_L` PR/ASR, and strong `z_P` speaker retention.
- Naive learned routing / “learn forever” is a useful negative control, not the flagship. Best learning-only folder:
  `libri_advlearn_hard_gn00015_gp02_aux64_12k_s42`.
  It keeps content in `z_L`, but `z_L` still leaks speaker heavily.
- The best healthy learned-routing result is:
  `libri_advlearn_hardqfreeze4000_gn0002_dann6800_gp02_aux64_20k_s42`.
  Interpretation: learned hard routing + freeze at 4k + quota freeze + longer post-freeze adversarial training can recover content/speaker disentanglement without collapse.
- Immediate adversary reset after freeze can clean `z_L` speaker but tends to cause high dead-unit/collapse behavior; do not treat low SID alone as success. Delayed speaker reset is cleaner but less healthy:
  `libri_advlearn_hardqfreeze4000_spkresetdelay4500_dann6800_gn00015_gp02_aux64_16k_s42`.
- The working story is: fixed routing proves feasibility; naive learned routing creates a moving target; quota-freezing learned routes stabilizes that target; sufficient post-freeze adversarial training gives the learned-routing result.

### MSP folder notes

`Disentanglement/msp/` is a standalone MSP-Podcast pipeline. It intentionally avoids modifying the legacy Libri/IEMOCAP trainer while reusing shared model/config/loss utilities.

Key design lessons:

- MSP exists to avoid the old Libri + IEMOCAP-every-8 failure mode. The old setup trained emotion too rarely, with weak/ramped anti-emotion pressure, and the emotion loss was dominated by prosody. MSP uses one dataset where each batch has transcript/content, speaker, prosody, and emotion labels.
- PR/content targets in MSP are generated from human transcripts with the shared `text_to_phones()` / SUPERB-style ARPABET path. Do not use the MSP force-aligned phone tier as the default target, because it mixes phone conventions.
- Emotion must be read with UAR, not just accuracy, because MSP is neutral-heavy. Accuracy can look okay while minority emotions are poor.
- PCGrad is applied only to cooperative tasks on the shared SAE trunk. For the
  initial MSP search this means reconstruction, PR, SID, prosody, and emotion;
  perturbation invariance is disabled by default. Adversarial losses are excluded
  on purpose because their conflict with reconstruction/content is the
  disentanglement mechanism.
- The MSP win condition is not “emotion high everywhere.” Desired pattern:
  `z_L` keeps content/PR, `z_P` keeps speaker and emotion/prosody, and leakage probes stay low: low `z_L SID`, high `z_P PR-PER`, and lower `z_L emotion UAR` than `z_P emotion UAR`.
- Current synced MSP logs are early/incomplete local evidence, not final results:
  `msp_31159712.out` reaches about step 800; `msp_31161572.out` reaches about step 1700 and has only an early validation at step 1000. Do not make final claims from these logs alone.
- MSP now uses the canonical DANN sigmoid ramp for GRL strength, separate from LR
  warmup (`--dann_ramp_steps`). The submitted MSP Slurm script intentionally uses
  a slower `DANN_RAMP_STEPS=6000`: the initial logs show active P capacity dropping
  very early when the adversary reaches full strength by step 500, so future MSP
  searches should treat adversary onset/routing starvation as a first-class issue.
- Current MSP search plan: do an initial hard-routing-only sweep before trying
  learned/freeze or soft-routing variants. The first sweep script is
  `Disentanglement/msp/slurm/sweep_msp_initial_hard.sh` with cases for slower
  DANN ramp, lower routing specialization, and P-support weighting. These initial
  runs disable perturbation invariance so the labeled factors are easier to
  interpret.
- For MSP final reporting, prefer `final.pt` for the final test-evaluated checkpoint. Treat `best.pt` as a validation-selected diagnostic unless the selection criterion is explicitly justified. Use `msp.probe` for independent frozen probes.
