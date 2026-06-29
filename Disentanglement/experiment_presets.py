"""Canonical Colab/cluster experiment presets."""
from __future__ import annotations

from copy import deepcopy

MSP_BASELINE = {
    "steps": 12000, "warmup_steps": 500, "alpha": 0.8, "beta": 0.6,
    "grl_weight": 1.0, "grl_phoneme_weight": 0.15,
    "prosody_weight": 0.5, "grl_prosody_weight": 0.5,
    "emotion_weight": 0.5, "grl_emotion_weight": 0.5, "inv_weight": 1.0,
    "lr": 1e-4, "lr_min": 1e-6, "lr_heads": 1e-4, "lr_disc": 1e-3,
    "lr_routing": 1e-3, "n_disc_steps": 3, "routing_init_std": 0.5,
    "routing_spec_weight": 0.01, "routing_tau": 1.0, "pcgrad": True,
    "hard_routing": True, "grl_grad_norm": True,
    "grl_grad_norm_target": 0.0005,
}

LIBRI_COMMON = {
    "stage": 2, "stage2_from_scratch": True, "local_data": True,
    "train_split_dir": "train-clean-100", "spear_layernorm": True,
    "K": 5120, "topk": 256, "alpha": 0.8, "beta": 0.6,
    "stage2_steps": 10000, "warmup_steps": 500, "lr": 1e-4,
    "lr_min": 1e-6, "lr_heads": 1e-4, "lr_sid_head": 5e-4,
    "lr_routing": 1e-3, "grad_clip": 1.0, "n_routes": 2,
    "routing_init_std": 0.5, "log_every": 100, "grad_log_every": 500,
    "ckpt_every": 1000,
}


def _merged(base, **changes):
    out = deepcopy(base); out.update(changes); return out


PRESETS = {
    "msp_baseline": _merged(MSP_BASELINE),
    "msp_no_pcgrad": _merged(MSP_BASELINE, pcgrad=False),
    "msp_no_invariance": _merged(MSP_BASELINE, inv_weight=0.0, invariance=False),
    "msp_soft_routing": _merged(MSP_BASELINE, hard_routing=False),
    "msp_no_cross_adversaries": _merged(
        MSP_BASELINE, grl_phoneme_weight=0.0, grl_prosody_weight=0.0,
        grl_emotion_weight=0.0),
    "msp_no_adversaries": _merged(
        MSP_BASELINE, grl_weight=0.0, grl_phoneme_weight=0.0,
        grl_prosody_weight=0.0, grl_emotion_weight=0.0),
    "libri_grl_stats_gelu": _merged(
        LIBRI_COMMON, hard_gumbel_routing=True, grl_stats_pool=True,
        grl_robust_sid=False, grl_grad_norm=True, grl_grad_norm_target=0.00025,
        grl_weight=1.0, grl_phoneme_weight=0.15, dann_full_discriminator=True,
        lr_disc=1e-3, n_disc_steps=3, rho=0.0, routing_spec_weight=0.0),
    "libri_club_hybrid": _merged(
        LIBRI_COMMON, hard_gumbel_routing=False, gumbel_tau_start=1.0,
        gumbel_tau_end=1.0, dual_invariance=True, inv_L_weight=1.0,
        inv_P_weight=1.0, inv_var_weight=0.1, inv_var_gamma=1.0,
        vicreg_full=True, vicreg_cov_weight=0.2, club_enabled=True,
        club_weight=0.3, club_inner_steps=3, club_hidden=512, club_lr=1e-3,
        club_phoneme_enabled=False, grl_weight=0.0, grl_phoneme_weight=0.2,
        pair_alpha_arctic_w=0.0, pair_alpha_pert_w=1.0,
        pair_beta_libri_w=1.0, pairs_alpha_per_step=8, pairs_beta_per_step=8,
        inv_L_interp_frames=200, inv_f0_low=0.7, inv_f0_high=1.5,
        inv_formant_low=0.85, inv_formant_high=1.3, grad_log_every=200,
        rho=0.001, routing_spec_weight=0.01),
    "libri_club_pure": _merged(
        LIBRI_COMMON, hard_gumbel_routing=False, gumbel_tau_start=1.0,
        gumbel_tau_end=1.0, dual_invariance=True, inv_L_weight=1.0,
        inv_P_weight=1.0, inv_var_weight=0.1, inv_var_gamma=1.0,
        vicreg_full=True, vicreg_cov_weight=0.2, club_enabled=True,
        club_weight=0.3, club_inner_steps=3, club_hidden=512, club_lr=1e-3,
        club_phoneme_enabled=True, club_phoneme_weight=0.3,
        club_phoneme_inner_steps=3, club_phoneme_hidden=512,
        club_phoneme_lr=1e-3, club_phoneme_warmup_steps=1000,
        grl_weight=0.0, grl_phoneme_weight=0.0,
        pair_alpha_arctic_w=0.0, pair_alpha_pert_w=1.0,
        pair_beta_libri_w=1.0, pairs_alpha_per_step=8, pairs_beta_per_step=8,
        inv_L_interp_frames=200, inv_f0_low=0.7, inv_f0_high=1.5,
        inv_formant_low=0.85, inv_formant_high=1.3, grad_log_every=200,
        rho=0.001, routing_spec_weight=0.01),
}

MSP_EXPERIMENTS = frozenset(k for k in PRESETS if k.startswith("msp_"))
LIBRI_EXPERIMENTS = frozenset(k for k in PRESETS if k.startswith("libri_"))


def resolve_preset(name: str, profile: str = "full") -> dict:
    if name not in PRESETS:
        raise KeyError(f"unknown experiment {name!r}; choose from {', '.join(sorted(PRESETS))}")
    if profile not in {"pilot", "full"}:
        raise ValueError("profile must be 'pilot' or 'full'")
    out = deepcopy(PRESETS[name])
    if profile == "pilot":
        if name.startswith("msp_"):
            out.update(steps=1000, warmup_steps=100, ckpt_every=200)
        else:
            out.update(stage2_steps=1000, warmup_steps=100,
                       max_train_examples=2048, max_val_examples=128,
                       max_test_examples=128, ckpt_every=200)
    return out
