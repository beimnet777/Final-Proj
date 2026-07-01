from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from Disentanglement.experiment_presets import PRESETS, resolve_preset
from Disentanglement.experiment_runner import (
    _trainer_option_names, apply_overrides, main as runner_main,
    validate_experiment_config,
)
from Disentanglement.colab_bundle import prepare_msp, verify
from Disentanglement.training_runtime import (
    StatefulRandomSampler, checkpoint_payload, resolve_amp_precision, resolve_microbatch,
    restore_rng_state, restore_training_state, rng_state, validate_resume,
)
from Disentanglement.probe_robust.club import CLUBSampled, normalize_club_gradient


class PresetTests(unittest.TestCase):
    def test_precision_resolution_covers_every_policy(self):
        self.assertEqual((True, False), resolve_amp_precision(
            "auto", cuda_available=True, bf16_supported=True))
        self.assertEqual((False, True), resolve_amp_precision(
            "auto", cuda_available=True, bf16_supported=False))
        self.assertEqual((False, False), resolve_amp_precision(
            "auto", cuda_available=False, bf16_supported=False))
        self.assertEqual((False, True), resolve_amp_precision(
            "fp16", cuda_available=True, bf16_supported=True))
        self.assertEqual((False, False), resolve_amp_precision(
            "fp32", cuda_available=True, bf16_supported=True))
        with self.assertRaises(RuntimeError):
            resolve_amp_precision("bf16", cuda_available=True, bf16_supported=False)

    def test_all_named_presets_exist(self):
        expected = {
            "msp_baseline", "msp_no_pcgrad", "msp_no_invariance",
            "msp_soft_routing", "msp_no_cross_adversaries", "msp_no_adversaries",
            "libri_grl_stats_gelu", "libri_club_hybrid", "libri_club_pure",
        }
        self.assertEqual(expected, set(PRESETS))

    def test_stats_gelu_is_single_branch(self):
        p = resolve_preset("libri_grl_stats_gelu")
        self.assertTrue(p["grl_stats_pool"])
        self.assertFalse(p["grl_robust_sid"])
        self.assertEqual(0.00025, p["grl_grad_norm_target"])

    def test_club_variants(self):
        hybrid = resolve_preset("libri_club_hybrid")
        pure = resolve_preset("libri_club_pure")
        self.assertTrue(hybrid["club_enabled"])
        self.assertFalse(hybrid["club_phoneme_enabled"])
        self.assertEqual(0.2, hybrid["grl_phoneme_weight"])
        self.assertTrue(pure["club_phoneme_enabled"])
        self.assertEqual(0.0, pure["grl_phoneme_weight"])

    def test_microbatch_must_divide_effective(self):
        self.assertEqual((4, 4), resolve_microbatch("4", 16))
        with self.assertRaises(ValueError): resolve_microbatch("3", 16)

    def test_fine_grained_overrides_are_typed_and_strict(self):
        preset = resolve_preset("libri_grl_stats_gelu")
        changed = apply_overrides(preset, ["lr=0.0002", "stage2_steps=25",
                                           "dann_full_discriminator=false"])
        self.assertEqual(0.0002, changed["lr"])
        self.assertEqual(25, changed["stage2_steps"])
        self.assertFalse(changed["dann_full_discriminator"])
        with self.assertRaises(ValueError):
            apply_overrides(preset, ["learnging_rate=1e-4"])

    def test_libri_override_catalog_covers_all_requested_families(self):
        available = _trainer_option_names("libri_grl_stats_gelu")
        expected = {
            "lr", "lr_min", "lr_heads", "lr_sid_head", "lr_disc", "lr_routing",
            "weight_decay", "alpha", "beta", "grl_weight", "grl_phoneme_weight",
            "prosody_weight", "grl_prosody_weight", "emotion_weight",
            "grl_emotion_weight", "grl_grad_norm", "grl_grad_norm_target",
            "grl_p_grad_norm", "grl_p_grad_norm_target", "club_enabled",
            "club_weight", "club_lr", "club_inner_steps", "club_hidden",
            "club_grad_norm", "club_grad_norm_target",
            "club_phoneme_enabled", "club_phoneme_weight", "club_phoneme_lr",
            "club_phoneme_inner_steps", "club_phoneme_hidden",
            "club_phoneme_warmup_steps", "gumbel_tau_start", "gumbel_tau_end",
            "routing_spec_weight", "rho", "grad_clip", "n_disc_steps",
            "ckpt_every", "log_every", "grad_log_every", "eval_batch_size",
        }
        self.assertFalse(expected - available, expected - available)

    def test_club_gradient_normalization_config_is_strict(self):
        valid = resolve_preset("libri_club_hybrid")
        valid.update(club_grad_norm=True, club_grad_norm_target=0.005)
        validate_experiment_config(valid)
        for changes in (
            {"club_enabled": False},
            {"club_grad_norm_target": 0.0},
            {"club_weight": 0.0},
        ):
            invalid = dict(valid); invalid.update(changes)
            with self.assertRaises(ValueError):
                validate_experiment_config(invalid)


class CheckpointTests(unittest.TestCase):
    def test_rng_restore_canonicalizes_serialized_state_to_cpu_bytes(self):
        original = rng_state()
        expected = original["torch"].clone()
        serialized = dict(original)
        # Reproduces a non-ByteTensor representation; CUDA map_location is the
        # same failure class and is canonicalized through the tensor branch.
        serialized["torch"] = expected.tolist()
        try:
            torch.manual_seed(123456)
            restore_rng_state(serialized)
            self.assertEqual(torch.uint8, torch.get_rng_state().dtype)
            self.assertEqual("cpu", torch.get_rng_state().device.type)
            self.assertTrue(torch.equal(expected, torch.get_rng_state()))
        finally:
            restore_rng_state(original)

    def test_compact_roundtrip_and_strict_identity(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.sae = torch.nn.Linear(3, 2)
                self.encoder = torch.nn.Module(); self.encoder._spear = torch.nn.Linear(3, 3)
        model = Model(); opt = torch.optim.Adam(model.sae.parameters())
        cfg = SimpleNamespace(K=2, topk=1, n_routes=2, hard_gumbel_routing=True,
                              spear_model_id="fake", device="cpu")
        payload = checkpoint_payload(model=model, optimizer=opt, step=7,
                                     best_metric=0.5, cfg=cfg,
                                     dataset_hash="abc", preset="unit")
        self.assertFalse(any(k.startswith("encoder._spear") for k in payload["model_state"]))
        validate_resume(payload, dataset_hash="abc", preset="unit", cfg=cfg)
        with self.assertRaises(ValueError):
            validate_resume(payload, dataset_hash="different", preset="unit", cfg=cfg)
        normalized_cfg = SimpleNamespace(**vars(cfg), club_enabled=True,
                                         club_grad_norm=True,
                                         club_grad_norm_target=0.005)
        with self.assertRaisesRegex(ValueError, "club_grad_norm"):
            validate_resume(payload, dataset_hash="abc", preset="unit", cfg=normalized_cfg)
        for p in model.sae.parameters(): p.data.zero_()
        step, best = restore_training_state(payload, model=model, optimizer=opt)
        self.assertEqual(7, step); self.assertEqual(0.5, best)

    def test_sampler_cursor_roundtrip(self):
        sampler = StatefulRandomSampler(list(range(12)), seed=9)
        iterator = iter(sampler); prefix = [next(iterator) for _ in range(5)]
        state = sampler.state_dict(); suffix = list(iterator)
        restored = StatefulRandomSampler(list(range(12)), seed=999)
        restored.load_state_dict(state)
        self.assertEqual(suffix, list(iter(restored)))
        self.assertEqual(12, len(prefix + suffix))

    def test_club_bound_only_backpropagates_to_representation(self):
        club = CLUBSampled(4, 3, hidden=8)
        z = torch.randn(6, 4, requires_grad=True); y = torch.tensor([0, 1, 2, 0, 1, 2])
        club.inner_step(z.detach(), y, k=1)
        bound = club.mi_bound(z, y); bound.backward()
        self.assertIsNotNone(z.grad)
        self.assertTrue(all(p.grad is None for p in club.classifier.parameters()))
        self.assertIn("negative_label_collision", club.last_diagnostics)

    def test_club_gradient_normalization_preserves_direction_and_target(self):
        raw = torch.randn(2, 2, 5, requires_grad=True)
        transformed = normalize_club_gradient(
            raw, target=0.5, weight=0.3, accumulation=2, amp_scale=8.0)
        upstream = torch.randn_like(transformed)
        transformed.backward(upstream)
        expected_norm = 0.5 * 0.3 / 2 * 8.0
        norms = raw.grad.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.full_like(norms, expected_norm)))
        # No reversal: each normalized vector is positively collinear with the
        # unmodified upstream CLUB gradient.
        cosine = torch.nn.functional.cosine_similarity(raw.grad, upstream, dim=-1)
        self.assertTrue(torch.allclose(cosine, torch.ones_like(cosine), atol=1e-6))
        # Simulate GradScaler.unscale_: the requested effective magnitude is
        # invariant to the temporary FP16 scale.
        unscaled = raw.grad / 8.0
        expected_unscaled = 0.5 * 0.3 / 2
        self.assertTrue(torch.allclose(
            unscaled.norm(dim=-1),
            torch.full_like(norms, expected_unscaled),
        ))

    def test_club_gradient_normalization_keeps_zero_gradient_zero(self):
        z = torch.randn(2, 3, 4, requires_grad=True)
        transformed = normalize_club_gradient(z, target=0.005)
        (transformed * 0.0).sum().backward()
        self.assertEqual(0.0, float(z.grad.abs().sum()))


class RunnerTests(unittest.TestCase):
    def test_msp_dry_run_writes_contract(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root / "Transcripts").mkdir()
            (root / "manifest.csv").write_text(
                "FileName,wav,speaker_idx,emotion,emotion_idx,split,Act,Val,Dom\n"
                "a.wav,Audios/a.wav,0,neutral,0,train,1,1,1\n")
            (root / "Transcripts" / "a.txt").write_text("HELLO")
            out = root / "out"
            rc = runner_main([
                "--experiment", "msp_baseline", "--data_root", str(root),
                "--profile", "pilot", "--output_dir", str(out), "--dry_run",
                "--microbatch_size", "4",
            ])
            self.assertEqual(0, rc)
            config = json.loads((out / "resolved_config.yaml").read_text())
            self.assertEqual(4, config["microbatch_size"])
            self.assertEqual(4, config["gradient_accumulation_steps"])

    def test_notebook_is_valid_json(self):
        notebook = Path(__file__).resolve().parents[2] / "notebooks" / "Disentanglement_Colab.ipynb"
        payload = json.loads(notebook.read_text())
        self.assertEqual(4, payload["nbformat"])
        self.assertGreaterEqual(len(payload["cells"]), 8)
        source = "".join(
            line for cell in payload["cells"] for line in cell.get("source", [])
        )
        self.assertIn("averaged_perceptron_tagger_eng", source)
        self.assertIn("cmudict", source)

    def test_msp_bundle_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); audio = root / "source" / "Audios"; audio.mkdir(parents=True)
            (audio / "a.wav").write_bytes(b"not-decoded-during-pack")
            transcripts = root / "source" / "Transcripts"; transcripts.mkdir()
            (transcripts / "a.txt").write_text("HELLO")
            manifest = root / "manifest.csv"
            manifest.write_text(
                "FileName,wav,speaker_idx,emotion,emotion_idx,split,Act,Val,Dom\n"
                "a.wav,Audios/a.wav,0,neutral,0,train,1,1,1\n")
            archive = root / "msp.tar.gz"
            prepare_msp(SimpleNamespace(manifest=manifest, audio_root=root / "source",
                                        transcripts=transcripts, profile="full",
                                        seed=42, output=archive))
            extracted = root / "extracted"
            verify(SimpleNamespace(archive=archive, extract_to=extracted))
            payload = json.loads((extracted / "bundle_manifest.json").read_text())
            self.assertEqual("msp", payload["dataset"])
            self.assertTrue((extracted / "Audios" / "a.wav").exists())


if __name__ == "__main__": unittest.main()
