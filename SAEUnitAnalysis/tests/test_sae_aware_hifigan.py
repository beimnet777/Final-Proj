from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from SAEUnitAnalysis.train_sae_aware_direct_hifigan import (
    FrozenSAEReconstructor,
    _mixed_conditioning,
)


class SAEAwareHiFiGANTests(unittest.TestCase):
    def test_frozen_reconstructor_uses_exact_fixed_route_quotas(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "fixed.pt"
            enc = torch.tensor([
                [1.0, 0.0],
                [0.5, 0.0],
                [0.0, 1.0],
                [0.0, 0.5],
            ])
            torch.save({
                "model": {
                    "sae.enc_weight": enc,
                    "sae.dec_weight": enc.T.contiguous(),
                    "sae.b_pre": torch.zeros(2),
                    "block_idx": torch.tensor([0, 0, 1, 1]),
                },
                "analysis_config": {
                    "D": 2, "K": 4, "topk": 2,
                    "fixed_blocks": True, "per_block_topk": True,
                    "block_topk": [1, 1, 0], "sample_rate": 16000,
                    "spear_model_id": "test/spear", "spear_revision": "abc",
                    "spear_layernorm": False,
                },
            }, checkpoint)
            cache_manifest = {
                "input_dim": 2, "sample_rate": 16000,
                "spear_model_id": "test/spear", "spear_revision": "abc",
                "spear_layernorm": False,
            }
            reconstructor = FrozenSAEReconstructor(checkpoint, cache_manifest)
            features = torch.tensor([[[2.0, 3.0], [4.0, 1.0]]])
            reconstructed = reconstructor(features)
            torch.testing.assert_close(reconstructed, features)
            self.assertFalse(any(p.requires_grad for p in reconstructor.parameters()))

    def test_mixed_conditioning_selects_exact_fraction_without_blending(self):
        torch.manual_seed(42)
        original = torch.zeros(8, 2, 3)
        reconstructed = torch.ones_like(original)
        mixed, mask = _mixed_conditioning(original, reconstructed, 0.5)
        self.assertEqual(int(mask.sum()), 4)
        self.assertTrue(torch.equal(mixed[mask], reconstructed[mask]))
        self.assertTrue(torch.equal(mixed[~mask], original[~mask]))

    def test_csd3_script_requires_explicit_base_and_avoids_scratch(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "slurm"
            / "finetune_sae_aware_direct_hifigan_csd3.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('BASE_VOCODER_CHECKPOINT="${1:-${BASE_VOCODER_CHECKPOINT:-}}"', script)
        self.assertIn("--base-vocoder-checkpoint", script)
        self.assertIn("--sae-fraction", script)
        self.assertIn("train_sae_aware_direct_hifigan", script)
        self.assertIn("/rds/user/bbg25/hpc-work/Thesis/Final-Proj", script)
        self.assertNotIn("/scratch/", script)


if __name__ == "__main__":
    unittest.main()
