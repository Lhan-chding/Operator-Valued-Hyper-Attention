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

    def test_stage_loss_contracts_are_required_for_public_dataset_task_names(self):
        from moat_ovha_torch.train.multimodal_protocol import validate_training_protocol

        refcoco = validate_training_protocol(
            {
                "task_type": "refcoco",
                "training_stages": ["T0", "T5"],
                "losses_by_stage": {
                    "T0": ["cache_validation"],
                    "T5": ["task_loss", "public_alignment_ce", "candidate_individual_loss"],
                },
                "adapter_params_by_candidate": _valid_adapter_params(),
            }
        )
        self.assertTrue(refcoco.ok, refcoco.errors)

        incomplete_public = validate_training_protocol(
            {
                "task_type": "cmu_mosi",
                "training_stages": ["T0", "T5"],
                "losses_by_stage": {
                    "T5": ["task_loss"],
                },
                "adapter_params_by_candidate": _valid_adapter_params(),
            }
        )
        self.assertFalse(incomplete_public.ok)
        joined_public = "\n".join(incomplete_public.errors)
        self.assertIn("losses_by_stage missing stage: T0", joined_public)
        self.assertIn("cmu_mosi T5 must include required loss/record: candidate_individual_loss", joined_public)

        incomplete_region = validate_training_protocol(
            {
                "task_type": "flickr30k_entities",
                "training_stages": ["T0", "T5"],
                "losses_by_stage": {
                    "T0": ["cache_validation"],
                    "T5": ["task_loss", "candidate_individual_loss"],
                },
                "adapter_params_by_candidate": _valid_adapter_params(),
            }
        )
        self.assertFalse(incomplete_region.ok)
        self.assertIn(
            "flickr30k_entities T5 must include public alignment loss: public_alignment_ce or public_contrastive_retrieval",
            "\n".join(incomplete_region.errors),
        )

        incomplete_controlled = validate_training_protocol(
            {
                "task_type": "controlled_multimodal",
                "training_stages": ["T0", "T1", "T2", "T3", "T4"],
                "losses_by_stage": {
                    "T0": ["cache_validation"],
                    "T1": ["task_loss", "candidate_individual_loss"],
                    "T2": ["task_loss"],
                    "T3": ["task_loss"],
                    "T4": ["task_loss"],
                },
                "adapter_params_by_candidate": _valid_adapter_params(),
            }
        )
        self.assertFalse(incomplete_controlled.ok)
        joined_controlled = "\n".join(incomplete_controlled.errors)
        self.assertIn("controlled_multimodal T2 must include required loss/record: router_ce_true_active_operator", joined_controlled)
        self.assertIn("controlled_multimodal T2 must include required loss/record: adapter_kl_true_params", joined_controlled)
        self.assertIn("controlled_multimodal T4 must include required loss/record: cato_alignment_ce", joined_controlled)
        self.assertIn("controlled_multimodal T4 must include required loss/record: rceo_reliability_huber", joined_controlled)

        incomplete_robustness = validate_training_protocol(
            {
                "task_type": "robustness_eval",
                "training_stages": ["T0", "T6"],
                "losses_by_stage": {
                    "T0": ["cache_validation"],
                    "T6": ["task_loss"],
                },
                "adapter_params_by_candidate": _valid_adapter_params(),
            }
        )
        self.assertFalse(incomplete_robustness.ok)
        self.assertIn(
            "robustness_eval T6 must include required loss/record: robustness_evaluation_only",
            "\n".join(incomplete_robustness.errors),
        )

    def test_public_protocol_rejects_hidden_losses_and_unmarked_weak_reliability(self):
        from moat_ovha_torch.train.multimodal_protocol import validate_training_protocol

        report = validate_training_protocol(
            {
                "task_type": "sentiment_emotion",
                "training_stages": ["T0", "T5"],
                "losses_by_stage": {
                    "T0": ["cache_validation"],
                    "T5": [
                        "task_loss",
                        "candidate_individual_loss",
                        "router_ce_true_active_operator",
                        "adapter_kl_true_params",
                        "rceo_unimodal_disagreement",
                    ],
                },
                "adapter_params_by_candidate": _valid_adapter_params(),
            }
        )

        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn("hidden loss is controlled-only", joined)
        self.assertIn("weak reliability loss must be explicitly marked", joined)

    def test_public_protocol_rejects_section13_controlled_only_loss_aliases(self):
        from moat_ovha_torch.train.multimodal_protocol import validate_training_protocol

        controlled_only_aliases = (
            "active_operator_ce",
            "router_ce_active_operator",
            "true_active_operator_ce",
            "adapter_huber_true_params",
            "true_adapter_params_kl",
            "true_adapter_params_huber",
            "cato_alignment_ce",
            "cato_alignment_kl",
            "cato_true_alignment_kl",
            "true_rank_logits_kl",
            "true_prototype_logits_kl",
            "tleo_log_lengthscale_huber",
            "true_lengthscale_huber",
            "true_reliability_huber",
            "rceo_reliability_monotonic_loss",
        )

        for loss_name in controlled_only_aliases:
            with self.subTest(loss_name=loss_name):
                report = validate_training_protocol(
                    {
                        "task_type": "sentiment_emotion",
                        "training_stages": ["T0", "T5"],
                        "losses_by_stage": {
                            "T0": ["cache_validation"],
                            "T5": ["task_loss", "candidate_individual_loss", loss_name],
                        },
                        "adapter_params_by_candidate": _valid_adapter_params(),
                    }
                )

                self.assertFalse(report.ok)
                self.assertIn(
                    f"hidden loss is controlled-only and forbidden for public data: {loss_name} in T5",
                    "\n".join(report.errors),
                )

    def test_controlled_protocol_accepts_section13_controlled_loss_aliases(self):
        from moat_ovha_torch.train.multimodal_protocol import validate_training_protocol

        report = validate_training_protocol(
            {
                "task_type": "controlled_multimodal",
                "training_stages": ["T0", "T1", "T2", "T3", "T4"],
                "losses_by_stage": {
                    "T0": ["cache_validation"],
                    "T1": ["task_loss", "candidate_individual_loss"],
                    "T2": [
                        "task_loss",
                        "router_ce_true_active_operator",
                        "adapter_kl_true_params",
                        "active_operator_ce",
                        "adapter_huber_true_params",
                    ],
                    "T3": ["task_loss", "router_ce_true_active_operator", "router_ce_active_operator"],
                    "T4": [
                        "task_loss",
                        "cato_alignment_ce",
                        "cato_alignment_kl",
                        "lrio_rank_kl",
                        "spo_prototype_kl",
                        "tleo_lengthscale_huber",
                        "tleo_log_lengthscale_huber",
                        "rceo_reliability_huber",
                        "rceo_reliability_monotonic_loss",
                    ],
                },
                "adapter_params_by_candidate": _valid_adapter_params(),
            }
        )

        self.assertTrue(report.ok, report.errors)

    def test_public_weak_losses_require_structured_marking_metadata(self):
        from moat_ovha_torch.train.multimodal_protocol import validate_training_protocol

        missing_metadata = validate_training_protocol(
            {
                "task_type": "sentiment_emotion",
                "training_stages": ["T0", "T5"],
                "losses_by_stage": {
                    "T0": ["cache_validation"],
                    "T5": ["task_loss", "candidate_individual_loss", "weak_rceo_unimodal_disagreement_marked"],
                },
                "adapter_params_by_candidate": _valid_adapter_params(),
            }
        )

        self.assertFalse(missing_metadata.ok)
        self.assertIn(
            "weak loss requires structured marking metadata: weak_rceo_unimodal_disagreement_marked",
            "\n".join(missing_metadata.errors),
        )

        marked = validate_training_protocol(
            {
                "task_type": "sentiment_emotion",
                "training_stages": ["T0", "T5"],
                "losses_by_stage": {
                    "T0": ["cache_validation"],
                    "T5": ["task_loss", "candidate_individual_loss", "weak_rceo_unimodal_disagreement_marked"],
                },
                "loss_metadata": {
                    "weak_rceo_unimodal_disagreement_marked": {
                        "supervision_type": "weak",
                        "source": "unimodal_entropy_calibration_v1",
                        "must_report_as": "weak",
                    }
                },
                "adapter_params_by_candidate": _valid_adapter_params(),
            }
        )

        self.assertTrue(marked.ok, marked.errors)

    def test_public_rceo_weak_losses_accept_marked_plan_signals(self):
        from moat_ovha_torch.train.multimodal_protocol import validate_training_protocol

        report = validate_training_protocol(
            {
                "task_type": "sentiment_emotion",
                "training_stages": ["T0", "T5"],
                "losses_by_stage": {
                    "T0": ["cache_validation"],
                    "T5": [
                        "task_loss",
                        "candidate_individual_loss",
                        "weak_unimodal_entropy_calibration_marked",
                        "weak_cross_modal_disagreement_marked",
                    ],
                },
                "loss_metadata": {
                    "weak_unimodal_entropy_calibration_marked": {
                        "supervision_type": "weak",
                        "source": "unimodal_entropy_calibration_v1",
                        "must_report_as": "weak",
                    },
                    "weak_cross_modal_disagreement_marked": {
                        "supervision_type": "weak",
                        "source": "cross_modal_disagreement_pseudo_label_v1",
                        "must_report_as": "weak",
                    },
                },
                "adapter_params_by_candidate": _valid_adapter_params(),
            }
        )

        self.assertTrue(report.ok, report.errors)

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

    def test_v1_adapter_params_require_exact_candidate_param_sets(self):
        from moat_ovha_torch.train.multimodal_protocol import validate_training_protocol

        params = _valid_adapter_params()
        params["LRIO"] = ["rank_logits", "scale", "bias"]
        params["SPO"] = list(params["SPO"]) + ["scale"]
        report = validate_training_protocol(
            {
                "task_type": "controlled_multimodal",
                "training_stages": ["T0", "T1", "T2", "T3", "T4"],
                "losses_by_stage": {
                    "T0": ["cache_validation"],
                    "T1": ["task_loss"],
                    "T2": ["task_loss"],
                    "T3": ["task_loss"],
                    "T4": ["task_loss"],
                },
                "adapter_params_by_candidate": params,
            }
        )

        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn("adapter params for LRIO missing required v1 param: interaction_temperature", joined)
        self.assertIn("adapter params for SPO contains duplicate v1 param: scale", joined)

    def test_training_protocol_cli_returns_json_and_nonzero_on_violation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_plan.json"
            path.write_text(
                json.dumps(
                    {
                        "task_type": "phrase_region_grounding",
                        "training_stages": ["T0", "T5"],
                        "losses_by_stage": {
                            "T0": ["cache_validation"],
                            "T5": ["task_loss", "candidate_individual_loss", "true_alignment_ce"],
                        },
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
