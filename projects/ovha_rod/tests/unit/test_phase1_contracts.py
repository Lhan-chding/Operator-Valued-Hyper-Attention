import ast
import unittest
from pathlib import Path

import torch

from ovha_rod.models.scoring import masked_expression_score, referent_focal_loss


ROOT = Path(__file__).resolve().parents[2]


class ScoringTests(unittest.TestCase):
    def test_expression_score_ignores_padding(self):
        logits = torch.tensor([[[1.0, 2.0, 999.0], [0.0, 0.0, -999.0]]])
        valid = torch.tensor([[1, 1, 0]], dtype=torch.bool)
        score = masked_expression_score(logits, valid, temperature=1.0)
        expected = torch.logsumexp(logits[..., :2], dim=-1)
        self.assertTrue(torch.allclose(score, expected))

    def test_referent_focal_loss_is_finite(self):
        score = torch.tensor([[2.0, -1.0, 0.5]], requires_grad=True)
        positive = torch.tensor([[1, 0, 0]], dtype=torch.bool)
        loss = referent_focal_loss(score, positive)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(score.grad.abs().sum()), 0.0)


class StaticIntegrationContractTests(unittest.TestCase):
    def test_required_phase1_files_exist_and_compile(self):
        required = [
            ROOT / "ovha_rod/models/detectors/ovha_grounding_dino.py",
            ROOT / "ovha_rod/models/dense_heads/ovha_grounding_dino_head.py",
            ROOT / "configs/ovha_rod_swin_t_5e_refcoco.py",
            ROOT / "configs/ovha_rod_swin_t_5e_refcoco_plus.py",
            ROOT / "configs/ovha_rod_swin_t_5e_refcocog.py",
            ROOT / "scripts/server_preflight.py",
            ROOT / "scripts/run_phase1_server.sh",
            ROOT / "scripts/two_batch_smoke.py",
            ROOT / "README.md",
            ROOT / "environment/mmdetection.lock",
        ]
        for path in required:
            with self.subTest(path=path):
                self.assertTrue(path.exists(), path)
                if path.suffix == ".py":
                    ast.parse(path.read_text())

    def test_configs_are_val_only_and_do_not_use_cached_candidates(self):
        for path in sorted((ROOT / "configs").glob("*.py")):
            source = path.read_text().lower()
            with self.subTest(path=path):
                self.assertNotIn("testa", source)
                self.assertNotIn("testb", source)
                self.assertNotIn("region_features.npy", source)
                self.assertNotIn("proposal_cache", source)
                self.assertNotIn("clip crop", source)
                self.assertIn("max_epochs=5", source.replace(" ", ""))

    def test_detector_keeps_parent_token_max_for_topk(self):
        source = (ROOT / "ovha_rod/models/detectors/ovha_grounding_dino.py").read_text()
        self.assertIn("enc_outputs_class.max(-1)[0]", source)
        self.assertNotIn("logsumexp(enc_outputs_class", source)

    def test_environment_is_pinned_to_a_commit(self):
        lock = (ROOT / "environment/mmdetection.lock").read_text()
        self.assertIn("cfd5d3a985b0249de009b67d04f37263e11cdf3d", lock)
        self.assertNotIn("@main", lock)

    def test_phase0_parent_configs_preserve_official_optimizer_schedule(self):
        for dataset in ("refcoco", "refcoco_plus", "refcocog"):
            path = ROOT / "configs" / f"phase0_parent_swin_t_5e_{dataset}.py"
            with self.subTest(path=path):
                self.assertTrue(path.exists(), path)
                source = path.read_text()
                self.assertNotIn("optim_wrapper", source)
                self.assertNotIn("param_scheduler", source)
                self.assertNotIn("model =", source)


if __name__ == "__main__":
    unittest.main()
