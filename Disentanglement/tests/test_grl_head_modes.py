import unittest
import sys
import types

import torch

# Importing Disentanglement.model executes its package initializer, whose frozen
# SPEAR wrapper imports transformers.  Head unit tests do not construct SPEAR,
# so provide the smallest test-only stand-in when that optional dependency is
# absent from a lightweight CPU environment.
if "transformers" not in sys.modules:
    try:
        __import__("transformers")
    except ImportError:
        transformers_stub = types.ModuleType("transformers")
        transformers_stub.AutoModel = object
        sys.modules["transformers"] = transformers_stub

from Disentanglement.config import DISConfig
from Disentanglement.model.heads import GRLHead


class StandaloneLinearStatsGRLTests(unittest.TestCase):
    def _config(self, **changes):
        cfg = DISConfig()
        cfg.K = 8
        cfg.num_speakers = 3
        for key, value in changes.items():
            setattr(cfg, key, value)
        return cfg

    def test_forward_is_signed_project_then_masked_mean_std_then_linear(self):
        head = GRLHead(self._config(grl_linear_stats=True))
        z_l = torch.randn(2, 4, 8)
        lengths = torch.tensor([4, 2])

        actual = head(z_l, lengths, lam=0.7)

        projected = head.projector(z_l)
        mask = (torch.arange(4).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1)
        mean = (projected * mask).sum(1) / lengths.unsqueeze(1)
        var = (((projected - mean.unsqueeze(1)) ** 2) * mask).sum(1) / lengths.unsqueeze(1)
        std = (var + 1e-5).sqrt()
        expected = head.fc(torch.cat([mean, std], dim=-1))

        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual((2, 3), tuple(actual.shape))
        torch.testing.assert_close(actual, expected)

    def test_mode_is_off_by_default(self):
        self.assertFalse(DISConfig().grl_linear_stats)

    def test_rejects_paired_robust_mode(self):
        with self.assertRaisesRegex(ValueError, "standalone adversary"):
            GRLHead(self._config(grl_linear_stats=True, grl_robust_sid=True))

    def test_rejects_nonlinear_stats_companion(self):
        with self.assertRaisesRegex(ValueError, "grl_stats_pool"):
            GRLHead(self._config(grl_linear_stats=True, grl_stats_pool=True))


if __name__ == "__main__":
    unittest.main()
