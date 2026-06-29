from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from Disentanglement.experiment_presets import PRESETS, resolve_preset
from Disentanglement.experiment_runner import main as runner_main
from Disentanglement.colab_bundle import prepare_msp, verify
from Disentanglement.training_runtime import (
    StatefulRandomSampler, checkpoint_payload, resolve_microbatch,
    restore_training_state, validate_resume,
)
from Disentanglement.probe_robust.club import CLUBSampled


class PresetTests(unittest.TestCase):
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


class CheckpointTests(unittest.TestCase):
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
