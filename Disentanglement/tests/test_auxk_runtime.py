from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from Disentanglement.model.sae import SparseAutoencoder


def _cfg(*, adaptive: bool) -> SimpleNamespace:
    return SimpleNamespace(
        D=2,
        K=4,
        topk=2,
        fixed_blocks=False,
        aux_k=3,
        aux_k_adaptive=adaptive,
        dead_steps_threshold=1,
    )


class AdaptiveAuxKTests(unittest.TestCase):
    def test_adaptive_aux_uses_available_dead_units_before_ceiling(self):
        sae = SparseAutoencoder(_cfg(adaptive=True))
        sae.steps_since_fired.copy_(torch.tensor([2.0, 0.0, 0.0, 0.0]))
        z_pre = torch.tensor([[[4.0, 3.0, 2.0, 1.0]]])

        reconstruction = sae.aux_reconstruct(z_pre, collect_stats=True)

        self.assertIsNotNone(reconstruction)
        self.assertEqual(1, sae.last_aux_stats["k_eff"])
        self.assertEqual(1, sae.last_aux_stats["unique"])

    def test_legacy_aux_waits_until_full_aux_k_is_dead(self):
        sae = SparseAutoencoder(_cfg(adaptive=False))
        sae.steps_since_fired.copy_(torch.tensor([2.0, 0.0, 0.0, 0.0]))
        z_pre = torch.tensor([[[4.0, 3.0, 2.0, 1.0]]])

        reconstruction = sae.aux_reconstruct(z_pre, collect_stats=True)

        self.assertIsNone(reconstruction)
        self.assertEqual(0, sae.last_aux_stats["k_eff"])

    def test_valid_frames_drive_dead_and_revival_events(self):
        sae = SparseAutoencoder(_cfg(adaptive=True))
        sae.steps_since_fired.copy_(torch.tensor([2.0, 2.0, 0.0, 0.0]))
        # Unit 0 fires on a valid frame; unit 1 fires only on padding.
        z_t = torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0]]])

        sae.update_dead(z_t, lengths=torch.tensor([1]))

        self.assertEqual(1, sae.last_dead_stats["revived"])
        self.assertEqual(0.0, float(sae.steps_since_fired[0]))
        self.assertGreater(float(sae.steps_since_fired[1]), sae.dead_threshold)

    def test_aux_diagnostics_report_route_coverage_on_valid_frames(self):
        sae = SparseAutoencoder(_cfg(adaptive=True))
        sae.steps_since_fired.fill_(2.0)
        z_pre = torch.tensor([[[4.0, 3.0, 2.0, 1.0],
                               [1.0, 2.0, 3.0, 4.0]]])
        routes = torch.tensor([0, 0, 1, 1])

        sae.aux_reconstruct(
            z_pre,
            lengths=torch.tensor([1]),
            route_idx=routes,
            collect_stats=True,
        )

        self.assertEqual(3, sae.last_aux_stats["k_eff"])
        self.assertEqual(3, sae.last_aux_stats["unique"])
        self.assertEqual((2, 1), sae.last_aux_stats["unique_by_route"])


if __name__ == "__main__":
    unittest.main()
