import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MultimodalTrainingProtocolTests(unittest.TestCase):
    def test_controlled_public_and_robustness_stage_protocols_are_explicit(self):
        from moat_ovha_torch.train.multimodal_protocol import (
            EXPECTED_STAGE_SEQUENCES,
            validate_training_protocol,
        )

        controlled = validate_training_protocol(
            {
                "task_type": "controlled_multimodal",
                "training_stages": ["T0", "T1", "T2", "T3", "T4"],
                "losses_by_stage": {
                    "T0": ["cache_validation"],
                    "T1": ["task_loss", "candidate_individual_loss"],
                    "T2": ["task_loss", "router_ce_true_active_operator", "adapter_kl_true_params"],
                    "T3": ["task_loss", "router_ce_true_active_operator"],
                    "T4": ["task_loss", "cato_alignment_ce", "lrio_rank_kl", "spo_prototype_kl", "tleo_lengthscale_huber", "rceo_reliability_huber"],
                },
                "adapter_params_by_candidate": _valid_adapter_params(),
            }
        )
        public = validate_training_protocol(
            {
                "task_type": "phrase_region_grounding",
                "training_stages": ["T0", "T5"],
                "losses_by_stage": {"T0": ["cache_validation"], "T5": ["task_loss", "public_alignment_ce", "candidate_individual_loss"]},
                "adapter_params_by_candidate": _valid_adapter_params(),
            }
        )
        robustness = validate_training_protocol(
            {
                "task_type": "robustness_eval",
                "training_stages": ["T0", "T6"],
                "losses_by_stage": {"T0": ["cache_validation"], "T6": ["robustness_evaluation_only"]},
                "adapter_params_by_candidate": _valid_adapter_params(),
            }
        )

        self.assertEqual(EXPECTED_STAGE_SEQUENCES["controlled_multimodal"], ("T0", "T1", "T2", "T3", "T4"))
        self.assertTrue(controlled.ok, controlled.errors)
        self.assertTrue(public.ok, public.errors)
        self.assertTrue(robustness.ok, robustness.errors)

    def test_public_protocol_rejects_hidden_losses_and_unmarked_weak_reliability(self):
        from moat_ovha_torch.train.multimodal_protocol import validate_training_protocol

        report = validate_training_protocol(
            {
                "task_type": "sentiment_emotion",
                "training_stages": ["T0", "T5"],
                "losses_by_stage": {
                    "T0": ["cache_validation"],
                    "T5": ["task_loss", "router_ce_true_active_operator", "adapter_kl_true_params", "rceo_unimodal_disagreement"],
                },
                "adapter_params_by_candidate": _valid_adapter_params(),
            }
        )

        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn("hidden loss is controlled-only", joined)
        self.assertIn("weak reliability loss must be explicitly marked", joined)

    def test_v1_adapter_params_reject_forbidden_v2_dynamic_params(self):
        from moat_ovha_torch.train.multimodal_protocol import validate_training_protocol

        params = _valid_adapter_params()
        params["CATO"] = list(params["CATO"]) + ["dynamic_sinkhorn_epsilon"]
        params["TLEO"] = list(params["TLEO"]) + ["repair_variance"]
        report = validate_training_protocol(
            {
                "task_type": "controlled_multimodal",
                "training_stages": ["T0", "T1", "T2", "T3", "T4"],
                "losses_by_stage": {"T0": ["cache_validation"], "T1": ["task_loss"], "T2": ["task_loss"], "T3": ["task_loss"], "T4": ["task_loss"]},
                "adapter_params_by_candidate": params,
            }
        )

        self.assertFalse(report.ok)
        self.assertIn("dynamic_sinkhorn_epsilon", "\n".join(report.errors))
        self.assertIn("repair_variance", "\n".join(report.errors))

    def test_training_protocol_cli_returns_json_and_nonzero_on_violation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_plan.json"
            path.write_text(
                json.dumps(
                    {
                        "task_type": "phrase_region_grounding",
                        "training_stages": ["T0", "T5"],
                        "losses_by_stage": {"T0": ["cache_validation"], "T5": ["task_loss", "true_alignment_ce"]},
                        "adapter_params_by_candidate": _valid_adapter_params(),
                    }
                )
            )
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "multimodal" / "validate_training_plan.py"), str(path)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 2)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["ok"])
        self.assertIn("hidden loss is controlled-only", "\n".join(payload["errors"]))


def _valid_adapter_params() -> dict[str, list[str]]:
    return {
        "TLEO": ["lengthscale", "local_temperature", "scale", "bias"],
        "SPO": ["prototype_temperature", "prototype_logits_shift", "scale", "bias"],
        "LRIO": ["rank_logits", "interaction_temperature", "scale", "bias"],
        "CATO": ["alignment_temperature", "transport_scale", "scale", "bias"],
    }


if __name__ == "__main__":
    unittest.main()
