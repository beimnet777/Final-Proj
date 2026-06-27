# Week 6 Disentanglement Report

## Evidence Scope

This report summarizes the Week 6 experiment thread using the logs and implementation files in `Disentanglement/`. The main chronological focus is June 22 onward. The SID+PR section starts with the Job2 dense GradNorm run from June 17 because it is the anchor run that the later replications, checkpoint-selection checks, and diagnostics were built around.

Important metric conventions:

| Metric | Meaning | Better direction |
|---|---|---|
| PR / PER | Phoneme recognition error rate | Lower |
| SID acc | Speaker-identification accuracy | Depends on bucket: high for `z_P`, low for `z_L` |
| `z_L` | Intended linguistic/content representation | Should support PR, suppress speaker |
| `z_P` | Intended paralinguistic representation | Should support SID, prosody, emotion, suppress phones |
| `z_U` | Residual/unused bucket | Removed in binary learned-routing runs |
| Val bucket readout | In-training heads evaluated during validation | Proxy only |
| Diagnostic probe | Fresh held-out probe trained after checkpointing | Main evidence for leakage |

## Head And Loss Glossary

| Name | What it does | Implementation structure | Used for |
|---|---|---|---|
| PR head | Predicts phones from `z_L` | `z_L` per frame -> `Linear(K, vocab=74)` -> CTC loss | Keeps phone/content information in `z_L` |
| SID head | Predicts speaker from `z_P` | masked mean over valid frames of `z_P` -> `Linear(K, num_speakers=251)` -> CE loss | Keeps speaker information in `z_P` |
| GRL | Gradient-reversal layer | forward pass is identity; backward multiplies the gradient by `-lambda` | Turns a classifier into an adversary against the encoder/routing |
| Speaker GRL head | Predicts speaker from `z_L`, with reversed gradient | default pooled form: `z_L` -> `Linear(K,256)` -> ReLU -> masked mean or mean+std -> speaker classifier. Dense form: `Linear(K,256)` -> ReLU -> `Conv1d(kernel=31)` -> ReLU -> per-frame speaker classifier | Pushes speaker information out of `z_L` |
| `grl_p` / PR-GRL head | Predicts phones from `z_P`, with reversed gradient | `z_P` -> GRL -> `Linear(K,256)` -> ReLU -> `Linear(256,74)` -> CTC loss | Pushes phone information out of `z_P` |
| Prosody head | Predicts frame-level F0 and energy from `z_P` | `z_P` -> `Linear(K,256)` -> ReLU -> `Linear(256,2)` for `[log-F0, log-energy]` | Gives `z_P` a frame-level paralinguistic task |
| Prosody GRL | Predicts F0/energy from `z_L` with reversed gradient | `z_L` -> GRL -> `Linear(K,256)` -> ReLU -> `Linear(256,2)` | Pushes prosody out of `z_L` |
| Emotion head | Predicts utterance emotion from `z_P` | `z_P` -> `Linear(K,256)` -> ReLU -> masked mean+std pooling -> `Linear(512,4)` | Gives `z_P` an utterance-level emotion task |
| Emotion GRL | Predicts emotion from `z_L` with reversed gradient | `z_L` -> GRL -> `Linear(K,256)` -> ReLU -> masked mean+std pooling -> `Linear(512,4)` | Pushes emotion out of `z_L` |
| Learned routing | Assigns SAE features to buckets | trainable logits of shape `K x routes`; soft Gumbel or hard straight-through Gumbel; masks are applied to task heads, not reconstruction | Lets the model learn which SAE features belong to `z_L` or `z_P` |

Source implementation: `Disentanglement/model/heads.py`, `Disentanglement/model/routing.py`, `Disentanglement/train.py`.

# 1. SID + PR Experiments

## 1.1 Job2 Dense GradNorm Main Run

Purpose: test whether a dense per-frame speaker adversary can remove speaker from `z_L` without destroying phone recognition, while `grl_p` removes phone content from `z_P`.

| Field | Value |
|---|---|
| Log | `logs/train/stage2/grl_dense/dense_gradnorm_30661246.out` |
| Date | Wed Jun 17, 2026 |
| Run name | `job2_dense_gradnorm` |
| Data | LibriSpeech train-clean-100 local split |
| Train / val / test | 27,539 / 500 / 500 utterances |
| Speakers / phone vocab | 251 speakers / 74 phones |
| SAE size | `K=5120`, `topk=256`, `D=1280` |
| Buckets | fixed blocks: `K_L=3072`, `K_P=1024`, `K_U=1024` |
| Per-block top-k | `topk_L=160`, `topk_P=64`, `topk_U=32` |
| Main weights | `alpha=0.8` PR, `beta=0.6` SID |
| Speaker removal | dense speaker GRL on `z_L`, `grl_weight=1.0`, kernel 31 |
| Speaker GRL gradient | per-frame GradNorm target `0.001` |
| Phone removal | `grl_p` on `z_P`, `grl_phoneme_weight=0.5` |
| Discriminator setting | DANN full discriminator, `lr_disc=1e-3`, `n_disc_steps=3` |
| Steps | 12,000 |

Architecture wiring:

| Path | Head | Training pressure |
|---|---|---|
| `z_L -> PR` | linear CTC PR head | make `z_L` phonetic |
| `z_P -> SID` | mean-pool + linear SID head | make `z_P` speaker-discriminative |
| `z_L -> SID` | dense speaker GRL: GRL -> linear -> ReLU -> local conv -> ReLU -> speaker logits | remove speaker from `z_L` |
| `z_P -> PR` | phone GRL: GRL -> linear -> ReLU -> linear -> CTC logits | remove phones from `z_P` |

Final validation and held-out probe:

| Signal | `z_L -> PR` PER | `z_L -> SID` acc | `z_P -> PR` PER | `z_P -> SID` acc |
|---|---:|---:|---:|---:|
| Validation proxy at step 12,000 | 0.066 | not shown in old bucket format | not shown in old bucket format | 0.538 |
| Held-out diagnostic probe | 0.067 | 0.010 | 0.534 | 0.972 |

Interpretation supported by evidence: the held-out probe says `z_L` keeps phone information and nearly removes speaker information. `z_P` keeps speaker information and loses much of the phone information. This is the strongest completed fixed-block GRL anchor run.

## 1.2 Job2 Seed Replications And Checkpoint-Selection Issue

Purpose: test whether the Job2 dense GradNorm result survives training-seed changes.

All three replications used the same architecture as Job2. Only the training seed changed; the diagnostic probe seed stayed fixed at 42.

| Train seed | Started | Final validation at step 12,000 | Selected `stage2_best.pt` | Probe on selected checkpoint | Evidence status |
|---:|---|---|---|---|---|
| 7 | Jun 22 15:03 | `z_L PR=0.090`, proxy `z_L SID=0.008`, `z_P PR=1.000`, `z_P SID=0.998` | step 5,000 | `z_L PR=0.071`, `z_L SID=0.704`, `z_P PR=0.805`, `z_P SID=0.998` | selected checkpoint leaks speaker under probe |
| 21 | Jun 22 19:00 | `z_L PR=0.061`, proxy `z_L SID=0.001`, `z_P PR=1.000`, `z_P SID=1.000` | step 10,000 | `z_L PR=0.061`, `z_L SID=0.452`, `z_P PR=0.906`, `z_P SID=0.998` | selected checkpoint leaks speaker under probe |
| 84 | Jun 22 23:35 | `z_L PR=0.093`, proxy `z_L SID=0.006`, `z_P PR=1.000`, `z_P SID=0.998` | step 8,000 | `z_L PR=0.074`, `z_L SID=0.006`, `z_P PR=0.903`, `z_P SID=0.994` | selected checkpoint is clean for `z_L` SID |

Objective conclusion: the replication did not cleanly prove that Job2 fails under seed changes. It proved that checkpoint selection is a confound. The bad-looking seed 7 and seed 21 probes were run on `stage2_best.pt`, and the chosen checkpoints were not necessarily the final step. The logs also show that the in-training proxy `z_L SID` could be low while the held-out probe still recovered speaker. Therefore, the correct statement is:

| Claim | Supported? | Why |
|---|---|---|
| Some selected checkpoints leaked speaker under diagnostic probes | Yes | seed 7 and seed 21 selected checkpoints had `z_L SID` 0.704 and 0.452 |
| The entire method failed for those seeds | Not proven | final checkpoints were not fully re-probed for all seeds in the available logs |
| `stage2_best.pt` selection can be misleading | Yes | selected steps differed from final step and proxy leakage disagreed with probes |

## 1.3 Diagnostic Probe-Seed Sweep On Job2

Purpose: test whether the diagnostic probe itself is stable across probe seeds.

Checkpoint: `job2_dense_gradnorm/stage2_best.pt`, step 12,000.

| Probe seed | `z_L -> PR` PER | `z_L -> SID` acc | `z_P -> PR` PER | `z_P -> SID` acc |
|---:|---:|---:|---:|---:|
| 7 | 0.067 | 0.382 | 0.529 | 0.966 |
| 42 | 0.067 | 0.126 | 0.531 | 0.970 |
| 123 | 0.068 | 0.258 | 0.529 | 0.968 |

MDL summary:

| Probe seed | `z_L PR` compression | `z_L SID` compression | `z_P PR` compression | `z_P SID` compression |
|---:|---:|---:|---:|---:|
| 7 | 89.5% | 0.0% | 41.2% | 21.8% |
| 42 | 81.0% | 0.0% | 40.9% | 23.2% |
| 123 | 89.4% | 0.0% | 41.0% | 20.7% |

Interpretation: phone probing and `z_P` SID were stable, but `z_L` SID accuracy was probe-seed sensitive. The MDL result still showed no useful compression for `z_L SID`, so the high `z_L SID` accuracies should be treated cautiously rather than as a clean contradiction.

## 1.4 Invariance Structure

Purpose: remove speaker from `z_L` using input perturbation consistency instead of a `z_L` speaker GRL, while keeping `grl_p` active to remove phones from `z_P`.

| Field | Value |
|---|---|
| Log | `logs/train/stage2/invariance/inv_only_nr_30923230.out` |
| Started | Mon Jun 22, 2026 |
| Run name | `invariance_only_w4_noramp` |
| Routing | fixed blocks, same `K_L/K_P/K_U` as Job2 |
| Invariance | on, weight 4.0, no ramp |
| Perturbation | F0 scale `[0.7, 1.5]`, formant scale `[0.8, 1.45]` |
| Speaker GRL on `z_L` | off, `grl_weight=0.0` |
| Phone GRL on `z_P` | on, `grl_phoneme_weight=0.5` |
| Main tasks | `z_L -> PR`, `z_P -> SID` |

Implementation: for each training item, the loader supplies the original audio and a speaker-perturbed copy. The model encodes both. The invariance loss makes `z_L(original)` and `z_L(perturbed)` similar frame by frame. This gives a dense speaker-removal signal without needing a speaker classifier on `z_L`.

Results:

| Signal | `z_L -> PR` PER | `z_L -> SID` acc | `z_P -> PR` PER | `z_P -> SID` acc |
|---|---:|---:|---:|---:|
| Final validation proxy | 0.065 | 0.964 | 1.000 | 1.000 |
| Held-out diagnostic probe, seed 42 | 0.066 | 0.010 | 0.864 | 1.000 |
| No-early-stop `z_L SID` probe, seed 42 | n/a | 0.010 | n/a | n/a |
| `z_L SID` probe seed 7 | n/a | 0.002 | n/a | n/a |
| `z_L SID` probe seed 123 | n/a | 0.002 | n/a | n/a |

Objective caveat: the final validation proxy and the held-out probe disagree for `z_L SID`. The report should treat the held-out diagnostic probes as the leakage evidence, while using the validation proxy only as a training-time signal. Based on held-out probes, this invariance-only structure is very strong for removing speaker from `z_L`.

## 1.5 Learned Routing

Purpose: replace fixed blocks with learned binary `z_L/z_P` routing and test whether invariance-only or stats/robust speaker GRL still works when the model chooses the feature allocation.

Architecture:

| Component | Description |
|---|---|
| Routing | trainable per-feature logits assign each of 5,120 SAE features to `z_L` or `z_P` |
| Soft mode | Gumbel-softmax during training; deterministic soft masks at evaluation |
| Hard mode | straight-through Gumbel during training; argmax masks at evaluation |
| Reconstruction | not routed; decoder still sees the full SAE representation |
| Probes | final checkpoint only, not `stage2_best.pt`, because proxy-selected checkpoints had already been shown to mislead |

Results:

| Run | Routing | Training method | Step 12,000 val bucket readout | Held-out `z_L -> SID` probes | Held-out `z_P -> PR` probes |
|---|---|---|---|---|---|
| `lr_invariance_only_w4_soft_seed42` | soft | invariance only + `grl_p=0.5` | `z_L PR=0.064`, `z_L SID=0.976`, `z_P PR=1.000`, `z_P SID=0.996` | linear 0.964, MLP 0.002, stats 0.002 | linear 0.944, MLP 0.931 |
| `lr_invariance_only_w4_hard_seed42` | hard | invariance only + `grl_p=0.5` | `z_L PR=0.067`, `z_L SID=0.988`, `z_P PR=1.000`, `z_P SID=0.140` | linear 0.966, MLP 0.002, stats 0.002 | partial in available log |
| `lr_statsgrl_gp02_soft_seed42` | soft | stats-pool `z_L` GRL + phone GRL | `z_L PR=0.107`, `z_L SID=0.014`, `z_P PR=1.000`, `z_P SID=0.934` | seed 42: linear 0.988, MLP 0.002, stats 0.002; seed 7: linear 0.990, MLP 0.004, stats 0.002 | seed 42: linear 0.848, MLP 0.876; seed 7 linear 0.850 |
| `lr_statsgrl_gp02_hard_seed42` | hard | stats-pool `z_L` GRL + phone GRL | `z_L PR=0.085`, `z_L SID=0.004`, `z_P PR=1.000`, `z_P SID=0.944` | seed 42: linear 0.986, MLP 0.002, stats 0.002; seed 7: linear 0.992, MLP 0.004, stats 0.002 | seed 42: linear 0.980, MLP 0.980; seed 7: linear 0.980, MLP 0.979 |

Interpretation: learned routing did not produce clean speaker removal. It worked for MLP/stat-pool SID probes but failed badly under the linear `z_L SID` probe. This means speaker information is still linearly recoverable from `z_L`, even when nonlinear/statistical probes report chance. The safest wording is that learned routing changed the geometry of the leak, not that it removed speaker.

# 2. Multitask Prosody + Emotion Runs

Purpose: expand the `z_P` target beyond speaker identity by adding prosody and emotion, while preserving the original PR/SID disentanglement goal.

## 2.1 Detailed Implementation

| Field | Value |
|---|---|
| Logs | `logs/train/stage2/inv_dense_prosody_emotion_8to1/invpem_*.out` |
| Started | Thu Jun 25, 2026 |
| Run family | `inv_dense_prosody_emotion_8to1_*_seed42` |
| Routing | learned binary `z_L/z_P`, no `z_U` |
| Routing variants | soft and hard |
| Main data | LibriSpeech train-clean-100 |
| Auxiliary data | IEMOCAP fold 5 |
| Libri batch | 16 |
| IEMOCAP batch | 8 |
| Update cadence | Libri every step; IEMOCAP every 8 Libri steps |
| IEMOCAP classes | neutral, happy_excited, sad, angry |
| IEMOCAP split | train 3,432; val 858; test 1,241 |
| Stage2 steps planned | 12,000 |
| Available status | partial/truncated runs only, no final probes |

Wiring:

| Bucket | Positive tasks | Adversarial/removal pressure |
|---|---|---|
| `z_L` | PR head for phones | invariance weight 4.0, dense speaker GRL weight 1.0, prosody GRL weight 0.5, emotion GRL weight 0.2 |
| `z_P` | SID, prosody regression, emotion classification | phone GRL weight 0.5 |

Loss/head details:

| Task | Data source | Head | Target |
|---|---|---|---|
| PR | Libri | `z_L -> Linear(K,74)` with CTC | phone sequence |
| SID | Libri | mean(`z_P`) -> linear | speaker ID |
| Speaker GRL | Libri | dense `z_L` GRL with conv kernel 31 | speaker ID, reversed gradient |
| Phone GRL | Libri | `z_P` -> GRL -> linear -> ReLU -> linear -> CTC | phones, reversed gradient |
| Prosody | Libri and IEMOCAP audio | `z_P` -> linear -> ReLU -> linear 2-dim | per-frame log-F0 and log-energy |
| Prosody GRL | Libri and IEMOCAP audio | same regressor on `z_L` with GRL | F0/energy, reversed gradient |
| Emotion | IEMOCAP | `z_P` -> linear -> ReLU -> masked mean+std -> linear 4-way | emotion class |
| Emotion GRL | IEMOCAP | same classifier on `z_L` with GRL | emotion class, reversed gradient |

## 2.2 Run Status And Partial Results

The `.out` files do not show an explicit Slurm cancellation line, but the runs stop far before 12,000 steps and do not reach the scripted final-checkpoint probes. Therefore these should be reported as partial/truncated runs, not completed experiments.

| Log | Routing | Variant | Last observed step | Best available validation | IEMOCAP validation | Main observed issue |
|---|---|---|---:|---|---|---|
| `invpem_31034976_0.out` | soft | initial | 2,800 | step 2,000: `z_L PR=0.143`, `z_L SID=0.007`, `z_P PR=0.138`, `z_P SID=0.586` | `z_P emotion=0.536`, `z_L emotion=0.641` | phone leakage into `z_P` remained high; routing moved heavily toward `z_P` |
| `invpem_31034976_1.out` | hard | initial | 2,800 | step 2,000: `z_L PR=0.155`, `z_L SID=0.010`, `z_P PR=0.198`, `z_P SID=0.678` | `z_P emotion=0.580`, `z_L emotion=0.444` | hard routing also left strong phone leakage in `z_P` |
| `invpem_31043736_1.out` | hard | norm-target rerun | 2,500 | step 2,000: `z_L PR=0.190`, `z_L SID=0.000`, `z_P PR=0.936`, `z_P SID=0.322` | `z_P emotion=0.591`, `z_L emotion=0.507` | phone removal improved, but `z_P` speaker quality weakened and PR/recon degraded |
| `invpem_31045826_0.out` | soft | norm-target rerun | 2,100 | step 2,000: `z_L PR=0.171`, `z_L SID=0.007`, `z_P PR=0.520`, `z_P SID=0.402` | `z_P emotion=0.537`, `z_L emotion=0.548` | no clean emotion separation; phone removal only moderate |
| `invpem_31045826_1.out` | hard | norm-target rerun | 1,800 | step 1,000 only: `z_L PR=0.371`, `z_L SID=0.011`, `z_P PR=0.292`, `z_P SID=0.144` | `z_P emotion=0.466`, `z_L emotion=0.523` | early degradation; no later validation available |

## 2.3 Evidence For Objective Competition

| Evidence type | What the logs show | Interpretation |
|---|---|---|
| Route allocation | many runs drift from near-balanced routing toward more `z_P` features, e.g. soft initial reaches `L/P=1674/3446` by step 2,800 | multitask pressure pulls capacity into `z_P` |
| Phone leakage | early soft/hard initial runs have low `z_P PR` PER, e.g. 0.138 and 0.198 at step 2,000 | `z_P` still contains phone/content information |
| Strong adversarial losses | `grlPr` and `grl_p` losses oscillate while PR/SID are still learning | phone-removal pressure and useful task pressure compete |
| Emotion leakage | `z_L emotion` is often comparable to or higher than `z_P emotion` | emotion is not isolated into `z_P` |
| Gradient logs | speaker GRL and phone GRL gradients are often much larger than reconstruction gradients | auxiliary/adversarial tasks can dominate optimization |
| Aux clipping | `emoAux` repeatedly shows clipping scales below 1, e.g. `38.224x0.13`, `17.868x0.28` | emotion/prosody auxiliary contribution was large enough to need capping |

Current conclusion: the multitask setup is promising as a formulation, but the available runs are partial and show objective competition. It should not be presented as a successful disentanglement result yet.

## 2.4 Figures Needed For Multitask Section

| Figure | Purpose | Data to plot |
|---|---|---|
| Multitask validation curves | show partial training trajectory | `z_L PR`, `z_L SID`, `z_P PR`, `z_P SID`, IEMOCAP `z_P emotion`, `z_L emotion` over step |
| Loss stack over time | show which objective dominates | `recon`, `pr`, `sid`, `grl`, `grl_p`, `pros`, `grlPr`, `emo`, `grlE`, `inv` |
| Routing trajectory | show capacity migration | hard route counts `L/P`, active counts, entropy `H`, unit entropy `Hu`, specialization fraction |
| Gradient-ratio panel | show objective competition | per-task gradient norm ratios against reconstruction |
| Emotion/prosody auxiliary clipping | explain instability | raw aux loss, clipped aux loss, clip scale |

# 3. Pure Invariance / No-GRL: Dual Invariance V1

Purpose: test whether paired-data invariance alone, without any GRL adversary, can form `z_L` and `z_P`.

## 3.1 Detailed Implementation

| Field | Value |
|---|---|
| Log | `logs/train/stage2/dual_inv_v1/dual_inv_v1_soft_nogrl_31006673.out` |
| Started | Wed Jun 24, 2026 |
| Run name | `dual_inv_v1_soft_nogrl` |
| Routing | soft learned binary `z_L/z_P`, `n_routes=2`, no `z_U` |
| Gumbel tau | fixed at 1.0, no anneal |
| GRL | off: `grl_weight=0.0`, `grl_phoneme_weight=0.0` |
| Main data | LibriSpeech train-clean-100 |
| Libri split | train 27,539; val 500; test 500 |
| Speaker / phone vocab | 251 speakers / 74 phones |
| SAE size | `K=5120`, `topk=256`, `D=1280` |
| Steps | 12,000 |
| Routing regularization | `rho=0.001`, `routing_spec_weight=0.01` |

Dual-invariance losses:

| Loss | Pair source | Bucket | Implementation meaning |
|---|---|---|---|
| `inv_L` | pair-alpha: 60% CMU ARCTIC, 40% perturbed Libri | `z_L` | frame-aligned cosine consistency after interpolation to 200 frames |
| `inv_P` | pair-beta: within-chapter LibriSpeech | `z_P` | scale-normalized L2 on stats-pooled `z_P` |
| variance floor | main batch | `z_L` and `z_P` | prevents bucket collapse by keeping variance above a floor |

The design intention was: `z_L` should be stable across speaker or speaker-like changes, and `z_P` should be stable across same-chapter utterance pairs that share paralinguistic context better than arbitrary utterances.

## 3.2 Results

| Signal | Value |
|---|---:|
| Final validation recon | 0.0964 |
| Final validation PR loss | 0.2211 |
| Final validation `z_L PR` PER | 0.053 |
| Final validation proxy `z_L SID` acc | 0.002 |
| Final validation `z_P SID` acc | 1.000 |
| Final route count | `L/P=2457/2663` |
| Final active count | `active L/P=108/148` |
| Final unit entropy | `Hu=0.289` |
| Final specialization fraction `H<0.5` | 0.77 |

Held-out diagnostics:

| Probe | Result |
|---|---:|
| `z_L -> PR` | PER 0.054 |
| `z_L -> SID` | acc 1.000 |
| `z_P -> PR` | started but not completed in available output |
| ARCTIC matched SID probes | partial/truncated; not enough for final claim |

Objective conclusion: this run preserved strong phone information in `z_L`, but it did not remove speaker information. The validation proxy said `z_L SID=0.002`, while the held-out diagnostic probe recovered speaker perfectly from `z_L`. That is a direct proxy/probe contradiction, and the held-out probe should be treated as the stronger evidence.

## 3.3 Figures Needed For Dual Invariance V1

| Figure | Purpose | Data to plot |
|---|---|---|
| Dual-invariance loss curves | show whether invariance objectives converged | `inv_L`, `inv_P`, variance floor over step |
| Route allocation | show learned bucket balance and specialization | `L/P`, active `L/P`, `H`, `Hu`, specialization fraction |
| Proxy vs probe bar chart | show the central failure mode | validation `z_L SID=0.002` vs held-out `z_L SID=1.000` |
| Pair-source mix curve | show pair-alpha composition over training | logged `mix[arc/pert]` |
| Libri vs ARCTIC SID probe panel | check distribution-specific leakage | Libri held-out SID and ARCTIC partial probe trajectories |

# 4. Chronological Map

| Date | Experiment group | Run/log | Role in report |
|---|---|---|---|
| Jun 17 | SID+PR anchor | `dense_gradnorm_30661246.out` | main Job2 dense GradNorm result |
| Jun 22 | SID+PR seed replication | `dense_gn_seed_30921877_0.out` | seed 7 replication, selected checkpoint leak |
| Jun 22 | Invariance structure | `inv_only_nr_30923230.out` | invariance-only fixed-block control |
| Jun 22 | SID+PR seed replication | `dense_gn_seed_30921877_1.out` | seed 21 replication, selected checkpoint leak |
| Jun 22-23 | SID+PR seed replication | `dense_gn_seed_30921877_2.out` | seed 84 replication, clean selected checkpoint |
| Jun 23 | Diagnostic sweep | `probe_seed_sweep_30956200_*.out` | probe-seed sensitivity check for Job2 |
| Jun 24 | Pure invariance/no-GRL | `dual_inv_v1_soft_nogrl_31006673.out` | dual-invariance v1 result |
| Jun 25 | Learned routing | `lr_invstat_31020134_*.out` | learned routing and probe-architecture mismatch |
| Jun 25 | Multitask | `invpem_31034976_*.out`, `31043736_1.out`, `31045826_*.out` | partial prosody/emotion multitask pilots |

# 5. Evidence Files

| Type | Files |
|---|---|
| Head implementation | `Disentanglement/model/heads.py` |
| Routing implementation | `Disentanglement/model/routing.py` |
| Training implementation | `Disentanglement/train.py`, `Disentanglement/config.py` |
| Job2 main | `Disentanglement/logs/train/stage2/grl_dense/dense_gradnorm_30661246.out` |
| Job2 replications | `Disentanglement/logs/train/stage2/grl_dense/dense_gn_seed_30921877_*.out` |
| Job2 probe-seed sweep | `Disentanglement/logs/diag/probe_seed_sweep/probe_seed_sweep_30956200_*.out` |
| Invariance-only | `Disentanglement/logs/train/stage2/invariance/inv_only_nr_30923230.out` |
| Invariance probe sweep | `Disentanglement/logs/diag/invariance_noearly/probe_inv_noes_30996224.out`, `Disentanglement/logs/diag/invariance_zL_sid_seed_sweep/inv_zL_sid_ss_31000897_*.out` |
| Learned routing | `Disentanglement/logs/train/stage2/learned_routing_inv_statsgrl/lr_invstat_31020134_*.out` |
| Multitask prosody/emotion | `Disentanglement/logs/train/stage2/inv_dense_prosody_emotion_8to1/invpem_*.out` |
| Dual invariance v1 | `Disentanglement/logs/train/stage2/dual_inv_v1/dual_inv_v1_soft_nogrl_31006673.out` |

