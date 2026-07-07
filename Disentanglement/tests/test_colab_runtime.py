from __future__ import annotations

import json
import importlib
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from Disentanglement.experiment_presets import PRESETS, resolve_preset
from Disentanglement.experiment_runner import (
    _trainer_option_names, apply_overrides, main as runner_main,
    validate_experiment_config,
)
from Disentanglement.colab_bundle import prepare_msp, verify
from Disentanglement.training_runtime import (
    StatefulRandomSampler, accumulate_task_grads, apply_task_gradient_caps_,
    checkpoint_payload, resolve_amp_precision, resolve_microbatch,
    restore_rng_state, restore_training_state, rng_state, validate_resume,
)
from Disentanglement.probe_robust.club import CLUBSampled, normalize_club_gradient

# Import the routing module without executing Disentanglement.model.__init__,
# which imports the optional Transformers/SPEAR stack unavailable in CPU-only
# unit-test environments.
_routing_spec = importlib.util.spec_from_file_location(
    "_routing_under_test", Path(__file__).resolve().parents[1] / "model" / "routing.py")
_routing_module = importlib.util.module_from_spec(_routing_spec)
assert _routing_spec.loader is not None
_routing_spec.loader.exec_module(_routing_module)
RoutingModule = _routing_module.RoutingModule


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
        self.assertFalse(p.get("grl_linear_stats", False))
        self.assertFalse(p.get("grl_linear_mean", False))
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
            "adversarial_task_grad_cap", "grl_shared_grad_cap_ratio",
            "grl_p_shared_grad_cap_ratio",
            "club_weight", "club_lr", "club_inner_steps", "club_hidden",
            "club_grad_norm", "club_grad_norm_target",
            "club_full_diagnostics", "club_diagnostics_every",
            "club_phoneme_enabled", "club_phoneme_weight", "club_phoneme_lr",
            "club_phoneme_inner_steps", "club_phoneme_hidden",
            "club_phoneme_warmup_steps", "gumbel_tau_start", "gumbel_tau_end",
            "routing_spec_weight", "rho", "grad_clip", "n_disc_steps",
            "ckpt_every", "log_every", "grad_log_every", "eval_batch_size",
        }
        self.assertFalse(expected - available, expected - available)

    def test_adversarial_cap_config_is_strict(self):
        valid = resolve_preset("libri_grl_stats_gelu")
        valid.update(adversarial_task_grad_cap=True,
                     grl_shared_grad_cap_ratio=2.0,
                     grl_p_shared_grad_cap_ratio=1.0)
        validate_experiment_config(valid)
        for key in ("grl_shared_grad_cap_ratio", "grl_p_shared_grad_cap_ratio"):
            broken = dict(valid); broken[key] = 0.0
            with self.assertRaisesRegex(ValueError, key):
                validate_experiment_config(broken)


class FrozenLearnedRoutingTests(unittest.TestCase):
    def _cfg(self, *, hard=True):
        return SimpleNamespace(
            K=4, D=3, n_routes=2, gumbel_tau_start=1.0,
            gumbel_tau_end=0.1, hard_gumbel_routing=hard,
            routing_init_std=0.0, routing_dynamic=False,
            fixed_routing=False,
        )

    def test_hard_freeze_keeps_loaded_argmax_and_disables_gumbel(self):
        routing = RoutingModule(self._cfg(hard=True))
        with torch.no_grad():
            routing.logits.copy_(torch.tensor([
                [3.0, -1.0], [-2.0, 4.0], [0.5, 0.1], [-1.0, 2.0],
            ]))
        # Constructing the optimizer first mirrors stage-2 resume: freezing must
        # not change parameter-group shape or invalidate its saved state.
        optimizer = torch.optim.AdamW([routing.logits], lr=1e-3)
        group_size = len(optimizer.param_groups[0]["params"])

        routing.train()
        routing.freeze_learned_routing()
        first = routing()[0]
        second = routing()[0]

        self.assertTrue(torch.equal(first, torch.tensor([1., 0., 1., 0.])))
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(first.requires_grad)
        self.assertEqual(group_size, len(optimizer.param_groups[0]["params"]))

    def test_soft_freeze_uses_deterministic_final_temperature(self):
        routing = RoutingModule(self._cfg(hard=False))
        with torch.no_grad():
            routing.logits.copy_(torch.tensor([
                [1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [-1.0, 1.0],
            ]))
        routing.train()
        routing.freeze_learned_routing()
        first = routing()[0]
        second = routing()[0]
        expected = torch.softmax(routing.logits.detach() / 0.1, dim=-1)[:, 0]
        self.assertTrue(torch.allclose(first, expected))
        self.assertTrue(torch.equal(first, second))

    def test_optimizer_continues_heads_without_moving_frozen_logits(self):
        routing = RoutingModule(self._cfg(hard=True))
        head = torch.nn.Parameter(torch.ones(4))
        optimizer = torch.optim.AdamW([
            {"params": [routing.logits], "lr": 1e-2},
            {"params": [head], "lr": 1e-2},
        ], weight_decay=0.1)
        routing.freeze_learned_routing()
        logits_before = routing.logits.detach().clone()
        head_before = head.detach().clone()

        loss = (routing()[0] * head).sum()
        loss.backward()
        optimizer.step()

        self.assertIsNone(routing.logits.grad)
        self.assertTrue(torch.equal(routing.logits, logits_before))
        self.assertFalse(torch.equal(head, head_before))

    def test_dynamic_freeze_is_rejected(self):
        cfg = self._cfg()
        cfg.routing_dynamic = True
        cfg.routing_dynamic_hidden = 2
        routing = RoutingModule(cfg)
        with self.assertRaisesRegex(ValueError, "static routing only"):
            routing.freeze_learned_routing()

class AdversarialTaskGradientCapTests(unittest.TestCase):
    def test_cap_changes_shared_gradient_but_not_unlisted_head(self):
        shared = torch.nn.Parameter(torch.zeros(2))
        head = torch.nn.Parameter(torch.zeros(1))
        reference = torch.tensor([3.0, 4.0])       # norm 5
        adversary = torch.tensor([30.0, 40.0])    # norm 50
        shared.grad = reference + adversary
        head.grad = torch.tensor([7.0])

        stats = apply_task_gradient_caps_(
            [shared], {"grl": [adversary]}, {"grl": 2.0})

        self.assertTrue(torch.allclose(shared.grad, torch.tensor([9.0, 12.0])))
        self.assertTrue(torch.equal(head.grad, torch.tensor([7.0])))
        self.assertAlmostEqual(5.0, stats["reference_norm"])
        self.assertAlmostEqual(50.0, stats["grl_raw_norm"])
        self.assertAlmostEqual(10.0, stats["grl_capped_norm"])
        self.assertAlmostEqual(0.2, stats["grl_scale"])

    def test_two_tasks_use_same_non_target_reference(self):
        shared = torch.nn.Parameter(torch.zeros(2))
        reference = torch.tensor([3.0, 4.0])
        speaker = torch.tensor([30.0, 40.0])
        phoneme = torch.tensor([0.0, 10.0])
        shared.grad = reference + speaker + phoneme

        stats = apply_task_gradient_caps_(
            [shared], {"grl": [speaker], "grl_p": [phoneme]},
            {"grl": 2.0, "grl_p": 1.0})

        self.assertTrue(torch.allclose(shared.grad, torch.tensor([9.0, 17.0])))
        self.assertAlmostEqual(10.0, stats["grl_capped_norm"])
        self.assertAlmostEqual(5.0, stats["grl_p_capped_norm"])

    def test_effective_batch_accumulation(self):
        first = [torch.tensor([1.0, 2.0]), None]
        second = [torch.tensor([3.0, 4.0]), torch.tensor([5.0])]
        buffer = accumulate_task_grads(None, first)
        buffer = accumulate_task_grads(buffer, second)
        self.assertTrue(torch.equal(buffer[0], torch.tensor([4.0, 6.0])))
        self.assertTrue(torch.equal(buffer[1], torch.tensor([5.0])))

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

    def test_club_full_diagnostics_config_is_strict(self):
        valid = resolve_preset("libri_club_hybrid")
        valid.update(club_full_diagnostics=True, club_diagnostics_every=100)
        validate_experiment_config(valid)
        for changes in (
            {"club_enabled": False},
            {"club_diagnostics_every": 0},
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
        linear_grl_cfg = SimpleNamespace(**vars(cfg), grl_linear_stats=True)
        with self.assertRaisesRegex(ValueError, "grl_linear_stats"):
            validate_resume(payload, dataset_hash="abc", preset="unit", cfg=linear_grl_cfg)
        linear_mean_cfg = SimpleNamespace(**vars(cfg), grl_linear_mean=True)
        with self.assertRaisesRegex(ValueError, "grl_linear_mean"):
            validate_resume(payload, dataset_hash="abc", preset="unit", cfg=linear_mean_cfg)
        capped_cfg = SimpleNamespace(**vars(cfg), adversarial_task_grad_cap=True,
                                     grl_shared_grad_cap_ratio=2.0,
                                     grl_p_shared_grad_cap_ratio=1.0)
        with self.assertRaisesRegex(ValueError, "adversarial_task_grad_cap"):
            validate_resume(payload, dataset_hash="abc", preset="unit", cfg=capped_cfg)
        fixed_cfg = SimpleNamespace(**vars(cfg), fixed_blocks=True,
                                    per_block_topk=True, K_L=1, K_P=1, K_U=0,
                                    topk_L=1, topk_P=0, topk_U=0)
        with self.assertRaisesRegex(ValueError, "fixed_blocks"):
            validate_resume(payload, dataset_hash="abc", preset="unit", cfg=fixed_cfg)
        # Unfrozen -> frozen is the intentional branching transition.
        freeze_cfg = SimpleNamespace(**vars(cfg), freeze_learned_routing_on_resume=True)
        validate_resume(payload, dataset_hash="abc", preset="unit", cfg=freeze_cfg)
        # Once a continuation checkpoint records frozen routing, silently
        # resuming it as trainable routing must fail.
        frozen_payload = dict(payload)
        frozen_payload["analysis_config"] = dict(payload["analysis_config"])
        frozen_payload["analysis_config"]["freeze_learned_routing_on_resume"] = True
        with self.assertRaisesRegex(ValueError, "freeze_learned_routing_on_resume"):
            validate_resume(frozen_payload, dataset_hash="abc", preset="unit", cfg=cfg)
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

    def test_sampler_restores_list_serialized_generator_state(self):
        sampler = StatefulRandomSampler(list(range(12)), seed=9)
        iterator = iter(sampler); prefix = [next(iterator) for _ in range(5)]
        state = sampler.state_dict(); suffix = list(iterator)
        state["generator_state"] = state["generator_state"].tolist()

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

    def test_club_full_diagnostics_capture_real_pre_and_post_updates(self):
        torch.manual_seed(4)
        club = CLUBSampled(4, 3, hidden=8, lr=1e-3)
        z = torch.randn(12, 4)
        y = torch.tensor([0, 1, 2] * 4)
        club.inner_step(z, y, k=3, capture_diagnostics=True)
        diag = club.last_inner_diagnostics
        expected = {
            "pre_ce", "pre_acc", "pre_entropy", "post_ce", "post_acc",
            "post_entropy", "pre_logit_std", "pre_logit_absmax",
            "post_logit_std", "post_logit_absmax", "batch_size",
            "unique_classes", "majority_fraction", "last_grad_norm",
            "parameter_norm", "update_norm", "lr", "inner_steps",
        }
        self.assertEqual(expected, set(diag))
        self.assertTrue(all(math.isfinite(value) for value in diag.values()))
        self.assertGreater(diag["last_grad_norm"], 0.0)
        self.assertGreater(diag["update_norm"], 0.0)
        self.assertEqual(12.0, diag["batch_size"])
        self.assertEqual(3.0, diag["unique_classes"])

    def test_club_controlled_negatives_are_repeatable_and_validated(self):
        club = CLUBSampled(4, 3, hidden=8)
        z = torch.randn(6, 4, requires_grad=True)
        y = torch.tensor([0, 1, 2, 0, 1, 2])
        negative = y.roll(1)
        first = club.mi_bound(z, y, negative_labels=negative,
                              update_diagnostics=False)
        second = club.mi_bound(z, y, negative_labels=negative,
                               update_diagnostics=False)
        self.assertTrue(torch.equal(first, second))
        with self.assertRaises(ValueError):
            club.mi_bound(z, y, negative_labels=negative[:-1])

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
    def test_pr_loader_uses_extracted_librispeech_when_local(self):
        repo = Path(__file__).resolve().parents[2]
        pr_dir = repo / "Probing" / "pr"
        if str(pr_dir) not in sys.path:
            sys.path.insert(0, str(pr_dir))

        fake_soundfile = types.ModuleType("soundfile")
        fake_soundfile.read = lambda source: (torch.zeros(16).numpy(), 16_000)
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("local PR loading unexpectedly contacted Hugging Face"))
        fake_datasets.Audio = lambda *a, **k: object()
        with patch.dict(sys.modules, {
            "soundfile": fake_soundfile,
            "datasets": fake_datasets,
        }):
            sys.modules.pop("pr_data", None)
            pr_data = importlib.import_module("pr_data")
            from pr_config import PRConfig

            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                for split in ("train-clean-100", "dev-clean", "test-clean"):
                    chapter = root / "LibriSpeech" / split / "1" / "2"
                    chapter.mkdir(parents=True)
                    (chapter / "1-2.trans.txt").write_text(
                        "1-2-0001 HELLO WORLD\n", encoding="utf-8")
                    (chapter / "1-2-0001.flac").write_bytes(b"local-placeholder")
                lexicon = root / "lexicon.txt"
                lexicon.write_text("HELLO HH AH0 L OW1\nWORLD W ER1 L D\n", encoding="utf-8")

                cfg = PRConfig(local_data=True,
                               librispeech_root=root / "LibriSpeech",
                               librispeech_lexicon=lexicon,
                               batch_size=1, eval_batch_size=1, num_workers=0)
                pr_data._LEXICON = None
                _, train_dl, val_dl, test_dl = pr_data.make_pr_dataloaders(cfg)
                self.assertEqual(1, len(train_dl.dataset))
                self.assertEqual(1, len(val_dl.dataset))
                self.assertEqual(1, len(test_dl.dataset))
                self.assertIsNone(train_dl.dataset.examples[0]["audio"]["bytes"])
                audio, _, text = train_dl.dataset[0]
                self.assertEqual(16, audio.numel())
                self.assertEqual("HELLO WORLD", text)

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
