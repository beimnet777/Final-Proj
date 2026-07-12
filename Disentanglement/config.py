"""DISConfig — hyperparameters for the SAE disentanglement system."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_DIS_DIR = Path(__file__).parent


@dataclass
class DISConfig:
    # ---------------------------------------------------------------- SPEAR
    spear_model_id: str = "marcoyang/spear-xlarge-speech-audio"
    spear_revision: str = ""  # optional immutable HF commit; resolved commit is checkpointed
    D: int = 1280       # SPEAR-Large hidden size
    spear_layernorm: bool = False  # LayerNorm each SPEAR layer (no affine) before averaging → SUPERB-comparable h_t

    # ---------------------------------------------------------------- SAE
    K: int = 5120       # latent size  (4 × D)
    topk: int = 256     # active features per frame  (5% of K)

    # Dead-latent revival (Gao et al. 2024 "AuxK") — needed when scaling K up, where
    # most latents otherwise die (never selected → never updated → dead forever).
    aux_k:                int   = 0        # # dead latents the aux loss reconstructs the residual with (0 = off; ~D/2 when on)
    aux_k_coef:           float = 0.03125  # 1/32 — weight on the aux loss
    dead_steps_threshold: int   = 256      # a latent is "dead" if it hasn't fired in this many steps
    geom_median_bias:     bool  = False    # init b_pre to the geometric median of a data sample
    renorm_decoder:       bool  = False    # unit-norm decoder columns after every step (not just at init)

    # ---------------------------------------------------------------- Routing / Gumbel
    gumbel_tau_start: float = 1.0
    gumbel_tau_end:   float = 0.1
    # Routing mode switch: False = soft Gumbel-softmax (fractional masks),
    # True = hard straight-through one-hot (matches the argmax used at eval).
    hard_gumbel_routing: bool = True
    # Break the zero-init symmetry: std>0 inits logits ~ N(0, std) so the routing
    # has something to amplify (0 = legacy zero init = symmetric saddle).
    routing_init_std:    float = 0.5
    # Per-unit specialization loss = mean per-unit routing entropy (Hu), minimized.
    # With route_loss (rho, balance) this maximizes MI(feature; route): each
    # feature decisive AND buckets globally balanced. Key tuning knob (sweep up
    # if units stay on the fence). 0 = off.
    routing_spec_weight: float = 0.01
    # Input-dependent routing: per-utterance router (conditioned on mean h_t)
    # produces a per-feature delta added to the static base, so the L/P/U partition
    # adapts per utterance. False = static (one fixed partition for all inputs).
    routing_dynamic:        bool = False
    routing_dynamic_hidden: int  = 256   # hidden width of the dynamic router MLP
    # Continuation-only mode: restore a learned-routing resume checkpoint, then
    # keep its learned partition deterministic and fixed while SAE/head training
    # continues. Unlike fixed_routing, this does not create a predetermined split.
    freeze_learned_routing_on_resume: bool = False
    # Optional continuation companion: after freezing learned route membership,
    # estimate the route-local active budgets produced by the learned model under
    # global TopK, then enforce those budgets for the remaining training.  This
    # keeps the split learned (not preset 240/16), while preventing the active
    # L/P count from drifting after the membership freeze.
    freeze_route_topk_on_resume: bool = False
    route_topk_calib_batches: int = 20
    # Continuation intervention for learned-route freeze experiments: reset the
    # adversarial discriminator heads after restoring a resume checkpoint.  This
    # tests whether a stale/co-adapted adversary is under-detecting leakage after
    # the route target stops moving.
    reset_adversaries_on_resume: bool = False
    # Narrower alternative to reset_adversaries_on_resume: comma-separated
    # adversary module names/aliases to reset after resume, e.g. "grl_head" or
    # "speaker".  Empty = off.  This lets learned-freeze experiments reset only
    # the z_L speaker adversary without disturbing the z_P phoneme adversary.
    reset_adversary_heads_on_resume: str = ""

    # ---------------------------------------------------------------- GradNorm (automatic loss balancing)
    # When on, the listed task weights are LEARNED online (Chen et al. 2018): each
    # task's gradient magnitude on the shared SAE encoder is balanced (× relative
    # training rate), replacing the fixed weights below for those tasks.
    gradnorm:        bool  = False
    gradnorm_alpha:  float = 1.5     # asymmetry: >0 up-weights slow-training tasks (0 = pure magnitude balance)
    gradnorm_lr:     float = 0.025   # lr for the weight optimizer
    gradnorm_tasks:  str   = "recon,pr,sid"   # which loss terms GradNorm manages (comma-sep)
    gradnorm_every:  int   = 1       # update the weights every N steps (amortize cost)

    # ---------------------------------------------------------------- Loss weights  (stage 2)
    alpha:      float = 1.0     # PR (CTC) weight          — calibrated from grad norms (or GradNorm)
    beta:       float = 1.0     # SID (CE) weight           — calibrated from grad norms (or GradNorm)
    grl_weight:          float = 1.0    # adversarial speaker weight — calibrated from grad norms
    grl_delay_steps:     int   = 0     # steps before GRL is switched on (0 = no delay)
    grl_frame_level:     bool  = False  # speaker adversary predicts per-frame (dense gradient) vs utterance mean-pool
    grl_attention_pool:  bool  = False  # speaker adversary pools z_L with attentive statistics (weighted mean+std) instead of flat mean → stronger discriminator
    # Single nonlinear speaker adversary: projector -> GELU ->
    # masked mean+std pooling -> speaker classifier.  This is stronger than
    # flat mean-pooling without adding attention as a second confound.
    grl_stats_pool:      bool  = False
    # Standalone signed-linear statistics adversary: projector -> masked
    # mean+std -> speaker classifier, with no activation and no companion branch.
    grl_linear_stats:    bool  = False
    # Standalone pure-linear adversary matching the diagnostic linear SID probe:
    # projector -> masked signed mean -> speaker classifier.  Keeping this
    # separate from linear_stats prevents a shared classifier from relying on
    # variance while leaving linearly decodable signed-mean leakage untouched.
    grl_linear_mean:     bool  = False
    # Dense speaker adversary: per-frame speaker prediction (like grl_p's per-frame
    # phoneme head) but with a temporal conv so each frame has local context — gives
    # z_L a DENSE per-frame removal gradient instead of one diluted pooled gradient.
    grl_dense_context:   bool  = False
    grl_context_kernel:  int   = 31    # conv kernel (frames) for the dense head's temporal context
    # Robust speaker adversary: train multiple z_L speaker readouts at once
    # (linear mean, nonlinear stats, and optionally dense-context).  This avoids
    # declaring speaker "removed" only because one adversary architecture is blind.
    grl_robust_sid:      bool  = False
    grl_robust_activation: str = "gelu"  # activation for nonlinear robust branches: relu|gelu
    # Per-frame gradient normalization for the PHONEME adversary on z_P (same mechanism
    # as the speaker grad-norm): a constant per-frame content-removal push on z_P.
    grl_p_grad_norm:        bool  = False
    grl_p_grad_norm_target: float = 0.001
    # grad-normalized GRL: per-frame L2-normalize the reversed gradient to a fixed
    # magnitude, so every frame gets an equal removal push regardless of the
    # discriminator's confidence (counters per-frame dilution).
    grl_grad_norm:        bool  = False
    grl_grad_norm_target: float = 1.0  # per-frame target L2 norm of the (unit) reversed gradient; magnitude = grl_weight * this
    # Optional schedule for the speaker GRL grad-norm target on z_L.  A negative
    # final target disables the schedule.  When enabled, the effective target is
    # constant at grl_grad_norm_target until decay_start, linearly decays to
    # grl_grad_norm_final_target by decay_end, then stays there.
    grl_grad_norm_decay_start: int = 0
    grl_grad_norm_decay_end: int = 0
    grl_grad_norm_final_target: float = -1.0
    # Optional task-level cap on the gradient that each GRL objective delivers
    # to the shared representation parameters.  Unlike the final global clip,
    # this bounds adversarial dominance before task gradients are combined.
    adversarial_task_grad_cap: bool = False
    grl_shared_grad_cap_ratio: float = 2.0
    grl_p_shared_grad_cap_ratio: float = 1.0
    # Negative control: train speaker adversaries against deterministic random
    # targets resampled each batch, while the positive z_P SID task keeps true
    # labels. A fixed class permutation would only rename speakers.
    shuffle_grl_speaker_labels: bool = False

    # ---------------------------------------------------------------- Invariance (speaker removal from z_L)
    # Enforce z_L(x) ~= z_L(perturb(x)) where perturb changes the SPEAKER (pitch+
    # formant) but keeps CONTENT and TIMING -> z_L becomes speaker-invariant.  This
    # is a DENSE per-frame removal signal (no per-frame speaker classifier needed),
    # which is why it works where the adversary can't.  0 = off.
    invariance:        bool  = False
    inv_weight:        float = 0.0
    # Linear ramp 0 -> inv_weight over this many steps (0 = constant).  Lets z_L
    # form CONTENT (recon+PR) before invariance strips speaker — without it, a
    # strong normalized weight can collapse z_L to a trivial (constant) invariant.
    inv_ramp_end:      int   = 0
    inv_f0_low:        float = 0.7     # F0 (pitch) random scale range
    inv_f0_high:       float = 1.5
    inv_formant_low:   float = 0.8     # formant (vocal-tract) random warp range (widened for ~half attenuation)
    inv_formant_high:  float = 1.45
    # Canonical DANN: discriminator heads train at FULL strength; grl_weight /
    # grl_phoneme_weight act only as reversal strengths (folded into lambda).
    # Default False preserves the legacy behaviour, where the weight also scaled
    # the discriminator's own learning signal (starving it at small weights).
    dann_full_discriminator: bool = False
    # Optional DANN ramp length. 0 preserves the default schedule that ramps over
    # the full optimizer schedule. Positive values ramp 0->1 over this many
    # optimizer steps and then hold at 1. Useful for learned-routing freeze runs
    # where the adversary should become effective before route freeze.
    dann_ramp_steps: int = 0
    rho:                 float = 0.001 # routing anti-collapse weight

    # ---------------------------------------------------------------- Option A: fixed-block supervised SAE
    # Disable learned routing entirely: partition K into fixed L/P/U index blocks
    # and apply per-block TopK, so disentanglement is shaped into the dictionary by
    # supervision (PR on z_L, SID on z_P, GRL adversaries) rather than sorted by a
    # Gumbel router.  Run as `--stage 2` from scratch (no stage1_ckpt) at full lr.
    fixed_blocks:        bool  = False
    K_L:                 int   = 3072   # phonetic block size   (K_L+K_P+K_U must == K)
    K_P:                 int   = 1024   # speaker block size     (speaker is low-dim)
    K_U:                 int   = 1024   # residual block size
    # per_block_topk=True  : fix the ACTIVE allocation too (top-k within each block) →
    #                        post-activation partition (Exp 1/2).
    # per_block_topk=False : fix only MEMBERSHIP; a single global top-k decides which
    #                        features fire → the per-block active counts EMERGE (Exp 3).
    per_block_topk:      bool  = True
    topk_L:              int   = 160    # per-block active budget (topk_L+topk_P+topk_U ≈ topk)
    topk_P:              int   = 64
    topk_U:              int   = 32

    # ---------------------------------------------------------------- Prosody (paralinguistic factor)
    # Master switch for the prosody factor.  OFF by default so any experiment runs
    # WITHOUT prosody unless explicitly enabled.  When on, it gates the prosody
    # probe (per-frame log-F0 + log-energy regression off the paralinguistic
    # bucket z_P) and — once added — the prosody training head / loss / adversaries.
    # Prosody lives in z_P alongside speaker (no separate block); F0 is raw
    # (not speaker-normalized) so z_P can serve both SID and prosody.
    prosody:        bool  = False
    prosody_weight: float = 0.0    # per-frame z_P -> [log-F0, log-E] regression task weight
    # Anti-prosody adversaries: push F0/energy OUT of the linguistic / residual
    # blocks so prosody concentrates in z_P (0 = off).
    grl_prosody_weight:   float = 0.0   # on z_L
    grl_prosody_u_weight: float = 0.0   # on z_U

    # ---------------------------------------------------------------- Emotion (IEMOCAP auxiliary factor)
    # Emotion is trained from IEMOCAP as a sparse auxiliary task, not mixed into the
    # Libri CTC/SID labels.  Every `emotion_every` Libri steps, one IEMOCAP batch
    # teaches z_P to classify the standard 4-way emotion target while optional
    # emotion-GRL pushes that label out of z_L.
    emotion:              bool  = False
    emotion_weight:       float = 0.0    # utterance-level z_P -> emotion CE
    grl_emotion_weight:   float = 0.0    # utterance-level anti-emotion adversary on z_L
    emotion_every:        int   = 8      # 8 Libri : 1 IEMOCAP update cadence
    emotion_grl_ramp_end: int   = 2000   # warm up z_L anti-emotion pressure
    emotion_aux_loss_clip: float = 5.0   # cap auxiliary contribution by scaling, preserving gradients
    emotion_num_classes:  int   = 4
    iemocap_root: Path = _DIS_DIR.parent / "Probing" / "data" / "IEMOCAP_full_release"
    iemocap_fold: int = 5
    iemocap_batch_size: int = 8
    iemocap_eval_batch_size: int = 16
    iemocap_val_fraction: float = 0.20

    # Adversaries on z_U (the residual): push BOTH factors out of U so it can't hoard
    # the phoneme/speaker info.  Unlike the z_L speaker-adv these have NO ceiling
    # (no task on U keeps the factors), so speaker→z_P and phonemes→z_L.
    grl_u_weight:         float = 0.0    # speaker adversary on z_U (anti-speaker)
    grl_phoneme_u_weight: float = 0.0    # phoneme adversary on z_U (anti-phoneme)

    # ---------------------------------------------------------------- Ablation flags (D / E / F)
    no_routing:          bool  = False  # D: bypass routing, feed full z to all heads
    fixed_routing:       bool  = False  # E: freeze routing at init split (not learned)
    fixed_routing_split: float = 0.7   # E: fraction of K features assigned to L
    n_routes:            int   = 3     # F: 3 = L/P/U (default), 2 = binary L/P only

    # ---------------------------------------------------------------- Pre-TopK routing (deprecated — use ste_routing)
    pre_topk_routing:    bool  = False

    # ---------------------------------------------------------------- Experiment flags
    # Exp 1 — Dual GRL: phoneme adversary on z_P
    grl_phoneme_weight:  float = 0.0   # weight for phoneme-GRL CTC loss on z_P (0 = disabled)

    # Exp 2 — TopK decorrelation in stage 1
    decor_weight:        float = 0.0   # weight for off-diagonal correlation penalty on active features

    # Exp 4 — U-bucket information bottleneck
    ub_weight:           float = 0.0   # weight for (m_L + m_P).mean() bottleneck — forces U alive
    # Delayed linear ramp for the IB: ub_weight is 0 until ub_ramp_start, then
    # ramps to full by ub_ramp_end (lets features specialize first, then prune).
    # ub_ramp_end=0 → no ramp (constant ub_weight).
    ub_ramp_start:       int   = 0
    ub_ramp_end:         int   = 0

    # Exp 5 — Straight-through estimator on routing mask multiplication
    # Forward: z_L = m_L × z_t (sparse, unchanged).  Backward: gradient flows through m_L × z_pre.
    ste_routing:         bool  = False

    # Projection disentanglement — learned compressed views z_t -> z_L and z_t -> z_P.
    projection_disentanglement: bool = False
    projection_dim: int = 128
    projection_nonlinear: bool = False    # views = 2-layer MLP (nonlinear demixer) instead of a single linear map
    projection_hidden:    int  = 512      # hidden width of the nonlinear projection MLP

    # Reconstructive projection — reconstruct h_t SOLELY through z_L/z_P (and an
    # optional penalized residual z_U), instead of decode(z_t).  Forces the views
    # to be a complete factorization of the signal (separate experiment family).
    projection_reconstruct: bool = False
    projection_u_dim:       int   = 0     # >0 adds residual view z_U of this dim (0 = 2-way, no z_U)
    projection_u_l2:        float = 0.0   # L2 activity penalty on z_U — the residual bottleneck
    instance_norm_zL:       bool  = False # instance-normalize z_L over time (strip per-utterance speaker stats)

    # Variational Information Bottleneck on z_L (linguistic block): add learned
    # per-feature noise + KL penalty so z_L keeps only what PR needs and sheds the
    # rest (incl. separable speaker).  Attacks the cause (excess capacity) rather
    # than fighting speaker adversarially.  0 = off.
    vib_zL_weight:          float = 0.0
    vib_zL_ramp_end:        int   = 0      # linear ramp 0→weight by this step (0 = constant)
    # Param-free LayerNorm on z_L (over feature dim, per frame) BEFORE the VIB, so the
    # projected magnitude can't run away (mu²→∞) — fixes the KL divergence in projection
    # mode.  Normalizes scale only (per-frame), does NOT strip per-utterance speaker.
    vib_zL_layernorm:       bool  = False

    # ---------------------------------------------------------------- Optimizer
    lr:          float = 1e-4   # SAE lr (stage 1);  also base lr for SAE in stage 2
    lr_min:      float = 1e-6   # cosine decay floor
    lr_routing:  float = 1e-3   # routing logits (stage 2) — raised from 5e-6 (which froze the
                                # logits); collapse is guarded by route_loss, not a tiny lr
    lr_heads:    float = 1e-4   # task heads      (stage 2) — pr_head + prosody/emotion
    # SID head gets its own lr (separate from pr_head) so it can be tuned when
    # CLUB-phn / GRL-phn are pulling phoneme info OUT of z_P at the same time
    # SID needs to keep speaker info IN z_P.  0 = fall back to lr_heads.
    lr_sid_head: float = 0.0
    # Adversary discriminators (grl_head, pr_grl_head) get their own, higher lr so
    # they can track the moving encoder instead of stalling at chance.  0 = fall
    # back to lr_heads (legacy: discriminators shared the task-head group).
    lr_disc:     float = 0.0
    # Discriminator updates per encoder update (GAN n_critic).  >1 takes extra
    # cheap gradient steps on the adversary heads (reusing this batch's detached
    # z_L/z_P — no extra encoder forward) so they stay ahead of the encoder.
    n_disc_steps: int  = 1
    weight_decay: float = 1e-4
    grad_clip:    float = 1.0

    # ---------------------------------------------------------------- Data
    sample_rate: int = 16_000
    librispeech_cache_dir: Path = _DIS_DIR.parent / "Probing" / "data"
    lexicon_path: Path = _DIS_DIR.parent / "Probing" / "data" / "librispeech-lexicon.txt"
    # Read raw flac from local disk instead of streaming from the HF CDN (which is
    # flaky and uncached).  librispeech_root must contain the split dirs
    # (train-clean-100, train-clean-360, dev-clean, test-clean).
    local_data:       bool = False
    librispeech_root: Path = _DIS_DIR.parent / "Probing" / "data" / "LibriSpeech"
    train_split_dir:  str  = "train-clean-100"   # set to "train-clean-360" for the scaled run
    max_train_examples: int = 0     # 0 = full train-clean-100 (~28 k)
    max_val_examples:   int = 500
    max_test_examples:  int = 500   # stage-2 closed-set SID test split (same speakers)
    num_workers: int = 0

    # ---------------------------------------------------------------- Training
    batch_size:      int = 16
    eval_batch_size: int = 32
    warmup_steps:    int = 500
    total_steps:     int = 6_000    # stage 1
    stage2_steps:    int = 0        # filled at launch (TBD)
    # Optional Stage-2 LR/DANN horizon. 0 means use stage2_steps.
    stage2_schedule_steps: int = 0
    log_every:       int = 100
    grad_log_every:  int = 500
    ckpt_every:      int = 1_000

    # ---------------------------------------------------------------- Runtime (filled by data loader)
    vocab_size:   int = 74      # SUPERB CTC: <pad>(blank)/<eos>/<unk> + 71 phones
    num_speakers: int = 0       # filled after dataset build

    # ---------------------------------------------------------------- Paths
    checkpoint_dir: Path = _DIS_DIR / "checkpoints"
    runs_dir:       Path = _DIS_DIR / "runs"
    log_dir:        Path = _DIS_DIR / "logs"

    # ---------------------------------------------------------------- Dual-invariance (v1)
    # Master switch.  When ON, training adds:
    #   L_inv_L: frame-aligned cosine between z_L of pair-alpha utterances
    #            (same content, paralinguistic varies — natural pairs from
    #            CMU ARCTIC or on-the-fly speaker-perturbation of LibriSpeech)
    #   L_inv_P: scale-normalised L2 between stats-pool(z_P) of pair-beta
    #            utterances (different content, same speaker+session — within
    #            LibriSpeech chapter)
    #   L_var  : VICReg-style per-dim variance floor on z_L and z_P
    # Recommended companion settings: n_routes=2 (drop z_U), grl_weight=0,
    # grl_phoneme_weight=0, hard_gumbel_routing chosen via --hard_gumbel_routing.
    dual_invariance:        bool  = False
    # Per-loss weights
    inv_L_weight:           float = 1.0
    inv_P_weight:           float = 1.0
    inv_var_weight:         float = 0.1
    inv_var_gamma:          float = 1.0
    # Pair sampling weights (relative; normalised internally)
    pair_alpha_arctic_w:    float = 0.6
    pair_alpha_pert_w:      float = 0.4    # synthetic-perturbation LibriSpeech
    pair_beta_libri_w:      float = 1.0
    # Pairs per step (each pair = 2 utterances forwarded)
    pairs_alpha_per_step:   int   = 8
    pairs_beta_per_step:    int   = 8
    # Frame-aligned interp target length for L_inv_L
    inv_L_interp_frames:    int   = 200
    # Corpus paths (under <repo>/Probing/data by default)
    arctic_root: Path = _DIS_DIR.parent / "Probing" / "data" / "CMU_ARCTIC"
    vctk_root:   Path = _DIS_DIR.parent / "Probing" / "data" / "VCTK"
    esd_root:    Path = _DIS_DIR.parent / "Probing" / "data" / "ESD"
    # Gumbel-tau annealing schedule for hard routing (linear from start to end
    # over [0, tau_anneal_steps]; 0 means no annealing — hold at start).
    gumbel_tau_anneal_steps: int = 0

    # ---------------------------------------------------------------- Probe-robust (VICReg-full + CLUB)
    # When `vicreg_full=True`, pair-α L_inv switches from cosine-per-frame-after-
    # bilinear-resample to **per-frame L2 on frame-aligned pairs only** (so use
    # with pair_alpha_pert_w=1.0, pair_alpha_arctic_w=0.0 — ARCTIC pairs are NOT
    # frame-aligned and bilinear resample is the v1 design flaw being removed).
    # Variance floor is unchanged; covariance regulariser is added on z_L and z_P
    # bucket dims (decorrelates → blocks orthogonal-subspace escape diagnosed in
    # dual_inv_v1_soft_nogrl).
    vicreg_full:        bool  = False
    vicreg_cov_weight:  float = 0.2
    # CLUB MI-min: minimises I(stats_pool(z_L); speaker_id) where stats_pool is
    # mean+std over time (x-vector canonical speaker representation). Choosing
    # mean+std (not mean-only) closes the "hide in temporal variance" escape
    # route that pure mean pooling would leave open. Probe-architecture-agnostic
    # by Fano's inequality w.r.t. classifiers reading the same stats vector.
    # Adversary-free (variational density estimator, not GAN minimax).
    # num_classes = cfg.num_speakers (set at runtime).
    club_enabled:       bool  = False
    club_weight:        float = 0.3
    club_lr:            float = 1e-3
    club_inner_steps:   int   = 3
    club_hidden:        int   = 512
    # Sign-preserving per-frame normalisation of only the speaker-CLUB gradient
    # entering z_L. The objective weight is applied after normalisation, as for
    # normalized GRL, but there is no reversal because CLUB is directly minimised.
    club_grad_norm:        bool  = False
    club_grad_norm_target: float = 0.005
    # Learned projection prepended to q_phi so the classifier does not have to
    # absorb a 2*K=10240-d sparse input directly. 0 disables (backward-compat).
    # Matches VQMIVC / Mun 2022 practice of feeding CLUB a small projection
    # rather than the raw high-dim latent.
    club_projection_dim: int = 0
    # CLUB warmup: hold the loss weight at 0 for this many optimiser steps
    # while still training q_phi (with `club_pretrain_inner_steps` inner steps
    # per boundary) so q_phi approximates p(y|z) before the encoder is asked
    # to descend its bound. Matches Cheng 2020 Algorithm 1.
    club_warmup_steps: int = 0
    club_pretrain_inner_steps: int = 20
    # Rejection-sample the shuffled negative labels so no negative collides
    # with the true label. Removes the ~6% floor observed when using
    # y.roll(1) or torch.randperm(y) at batch 16 / 251 classes.
    club_no_collision_negatives: bool = False
    # Expensive, opt-in diagnostics for one-shot CLUB/VICReg calibration runs.
    # Logs raw/delivered CLUB gradients, every objective's shared/routing
    # gradient, q_phi before/after updates, clipping, and representation scale.
    club_full_diagnostics: bool = False
    club_diagnostics_every: int = 100
    # Phoneme CLUB on z_P (frame-level, pr_head argmax as pseudo-labels).
    # Warmup gates the loss until pr_head's argmax stabilises — before that
    # the pseudo-labels are random and CLUB would chase noise.
    club_phoneme_enabled:       bool  = False
    club_phoneme_weight:        float = 0.3
    club_phoneme_lr:            float = 1e-3
    club_phoneme_inner_steps:   int   = 3
    club_phoneme_hidden:        int   = 512
    club_phoneme_warmup_steps:  int   = 1000

    # ---------------------------------------------------------------- Misc
    seed:   int  = 42
    device: str  = "cuda"
    bf16:   bool = True
