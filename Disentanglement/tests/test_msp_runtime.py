from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import torch

from Disentanglement.msp.checkpoints import (
    checkpoint_model_state, load_sae_initialization,
)
from Disentanglement.msp.grad_conflict import PCGrad
from Disentanglement.experiment_runner import _msp_args


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sae = torch.nn.Linear(3, 2)
        self.sid_head = torch.nn.Linear(2, 4)


class MSPCheckpointTests(unittest.TestCase):
    def test_current_checkpoint_loads_only_sae(self):
        source = _TinyModel()
        target = _TinyModel()
        with torch.no_grad():
            source.sae.weight.fill_(7.0)
            source.sid_head.weight.fill_(9.0)
        sid_before = target.sid_head.weight.detach().clone()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "current.pt"
            torch.save({"model_state": source.state_dict(), "step": 12}, path)
            audit = load_sae_initialization(target, path)
        self.assertEqual("model_state", audit["source_format"])
        self.assertEqual(12, audit["source_step"])
        self.assertTrue(torch.equal(source.sae.weight, target.sae.weight))
        self.assertTrue(torch.equal(sid_before, target.sid_head.weight))

    def test_legacy_and_raw_formats_are_recognized(self):
        state = _TinyModel().state_dict()
        self.assertIs(state, checkpoint_model_state(state))
        self.assertIs(state, checkpoint_model_state({"model": state}))
        self.assertIs(state, checkpoint_model_state({"model_state": state}))

    def test_shape_mismatch_cannot_silently_initialize(self):
        target = _TinyModel()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.pt"
            torch.save({"model_state": {"sae.weight": torch.zeros(9, 9)}}, path)
            with self.assertRaisesRegex(ValueError, "no shape-compatible SAE tensors"):
                load_sae_initialization(target, path)

    def test_runner_can_explicitly_disable_default_grl_normalization(self):
        self.assertEqual(["--no-grl_grad_norm"], _msp_args({"grl_grad_norm": False}))
        self.assertEqual(["--grl_grad_norm"], _msp_args({"grl_grad_norm": True}))


class MSPGradientDiagnosticsTests(unittest.TestCase):
    def test_vector_diagnostics_does_not_advance_pcgrad_rng(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        pcgrad = PCGrad([parameter], seed=3)
        before = pcgrad.state_dict()["rng_state"]
        gradients = {"a": torch.tensor([1.0, 0.0]),
                     "b": torch.tensor([-1.0, 1.0])}
        diagnostics = pcgrad.vector_diagnostics(
            gradients, torch.tensor([0.0, 1.0]), torch.tensor([0.5, 0.5]))
        after = pcgrad.state_dict()["rng_state"]
        self.assertEqual(before, after)
        self.assertEqual(1, diagnostics["coop_conflicts"])
        self.assertIn("external_bundle", diagnostics["norms"])

    def test_unit_balance_applies_weights_after_normalization(self):
        gradients = {
            "recon": torch.tensor([3.0, 4.0]),
            "emotion": torch.tensor([0.0, 20.0]),
            "inactive": torch.zeros(2),
        }
        balanced, scales = PCGrad.unit_balance(
            gradients, {"recon": 1.0, "emotion": 0.5, "inactive": 0.25})
        self.assertAlmostEqual(1.0, float(balanced["recon"].norm()), places=6)
        self.assertAlmostEqual(0.5, float(balanced["emotion"].norm()), places=6)
        self.assertEqual(0.0, float(balanced["inactive"].norm()))
        self.assertAlmostEqual(0.2, scales["recon"], places=6)
        self.assertAlmostEqual(0.025, scales["emotion"], places=6)


class MSPOptionalAdversaryTests(unittest.TestCase):
    def test_missing_optional_adversary_output_is_zero_loss(self):
        dis_dir = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(dis_dir))
        from msp.train import _loss_if_present

        ref = torch.tensor(3.0)
        loss = _loss_if_present(
            {}, "pr_grl_logits", ref,
            lambda value: self.fail("loss function should not be called"))
        self.assertEqual(0.0, float(loss))
        self.assertEqual(ref.dtype, loss.dtype)
        self.assertEqual(ref.device, loss.device)

    def test_present_optional_adversary_output_calls_loss(self):
        dis_dir = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(dis_dir))
        from msp.train import _loss_if_present

        ref = torch.tensor(3.0)
        out = {"pr_grl_logits": torch.tensor([2.0, 4.0])}
        loss = _loss_if_present(out, "pr_grl_logits", ref, lambda value: value.mean())
        self.assertEqual(3.0, float(loss))


class MSPSeparatedOptimizerTests(unittest.TestCase):
    def test_frozen_discriminator_still_backpropagates_to_representation(self):
        dis_dir = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(dis_dir))
        from msp.train import _set_requires_grad

        discriminator = torch.nn.Linear(3, 2)
        representation = torch.randn(4, 3, requires_grad=True)
        _set_requires_grad(discriminator.parameters(), False)
        discriminator(representation).square().mean().backward()
        self.assertIsNotNone(representation.grad)
        self.assertGreater(float(representation.grad.norm()), 0.0)
        self.assertTrue(all(param.grad is None for param in discriminator.parameters()))

    def test_group_clipping_does_not_scale_another_group(self):
        dis_dir = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(dis_dir))
        from msp.train import _clip_group

        small = torch.nn.Parameter(torch.zeros(1))
        large = torch.nn.Parameter(torch.zeros(1))
        small.grad = torch.tensor([0.5])
        large.grad = torch.tensor([100.0])
        small_norm, small_scale = _clip_group([small], 1.0)
        large_norm, large_scale = _clip_group([large], 1.0)
        self.assertAlmostEqual(0.5, small_norm, places=6)
        self.assertEqual(1.0, small_scale)
        self.assertAlmostEqual(0.5, float(small.grad), places=6)
        self.assertAlmostEqual(100.0, large_norm, places=4)
        self.assertAlmostEqual(0.01, large_scale, places=6)
        self.assertAlmostEqual(1.0, float(large.grad), places=6)


if __name__ == "__main__":
    unittest.main()
