import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RefCOCOExternalAlignmentExperimentDesign(unittest.TestCase):
    def test_pro_plan_config_covers_required_protocols_baselines_metrics_and_artifacts(self):
        plan = json.loads((ROOT / "configs/multimodal_refcoco_external_alignment_experiments.json").read_text())

        self.assertEqual(plan["schema_version"], "refcoco-external-alignment-experiments-v0.1")
        self.assertEqual(plan["dataset"], "refcoco")
        self.assertEqual(plan["fixed_candidate_protocol"]["candidate_count"], 32)
        self.assertEqual(plan["fixed_candidate_protocol"]["feature_policy"], "frozen_clip_same_candidate")
        self.assertEqual(plan["seed_policy"]["main"], [201, 202, 203, 204, 205])

        tracks = {track["id"]: track for track in plan["comparison_tracks"]}
        self.assertEqual(
            set(tracks),
            {
                "A_external_open_box_reference",
                "B_groundingdino_same_candidate_scorer",
                "C_groundingdino_proposals_plus_reranker",
            },
        )
        self.assertEqual(tracks["A_external_open_box_reference"]["comparison_scope"], "separate_external_reference_table")
        self.assertTrue(tracks["A_external_open_box_reference"]["forbid_fixed_candidate_win_loss_claims"])
        self.assertEqual(
            tracks["B_groundingdino_same_candidate_scorer"]["candidate_score_formula"],
            "max_j IoU(candidate_i, predicted_box_j) * score_j",
        )
        self.assertIn("proposal_oracle_recall_at_k", tracks["C_groundingdino_proposals_plus_reranker"]["upper_bound_metrics"])
        self.assertIn("empty_proposal_rate", tracks["C_groundingdino_proposals_plus_reranker"]["upper_bound_metrics"])

        baseline_ids = {baseline["id"] for baseline in plan["strong_same_candidate_baselines"]}
        self.assertGreaterEqual(
            baseline_ids,
            {
                "box_aware_cross_attention_reranker",
                "clip_geometry_mlp",
                "lightweight_transvg_style_reranker",
                "groundingdino_same_candidate_scorer",
            },
        )

        required_metrics = {
            "recall_at_1",
            "recall_at_5",
            "mrr",
            "acc_at_0_5",
            "acc_at_0_7",
            "mean_iou",
            "nll",
        }
        self.assertGreaterEqual(set(plan["required_metrics"]), required_metrics)
        self.assertGreaterEqual(
            set(plan["artifact_contract"]["per_sample_prediction_fields"]),
            {
                "source_id",
                "split",
                "candidate_boxes",
                "candidate_mask",
                "target_index",
                "selected_index",
                "selected_iou",
                "operator_contributions",
                "baseline_scores",
            },
        )

        diagnostics = {row["id"]: row for row in plan["operator_subset_diagnostics"]}
        self.assertEqual(diagnostics["spatial_subset"]["target_operator"], "SRO")
        self.assertEqual(diagnostics["dense_distractor_subset"]["target_operator"], "TLEO")
        self.assertEqual(diagnostics["long_relational_subset"]["target_operator"], "CATO")
        self.assertEqual(diagnostics["attribute_subset"]["decision"], "evaluate_error_profile_before_AMO")

        table_ids = [table["id"] for table in plan["final_tables"]]
        self.assertEqual(
            table_ids,
            [
                "table_1_cmu_mosei_main",
                "table_2_refcoco_fixed_candidate_mechanism",
                "table_3_refcoco_operator_subset_diagnostics",
                "table_4_groundingdino_proposal_reranking",
                "table_5_external_open_box_reference",
                "table_6_cross_task_operator_admission",
            ],
        )

    def test_external_sota_runbook_embeds_refcoco_alignment_plan_without_mixing_scopes(self):
        from scripts.multimodal.build_external_sota_runbook import build_external_sota_runbook

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "runbook"
            payload, exit_code = build_external_sota_runbook(
                ROOT / "configs/multimodal_external_sota_references.json",
                output_dir,
                ROOT / "configs/multimodal_refcoco_external_alignment_experiments.json",
            )
            runbook = json.loads((output_dir / "external_sota_runbook.json").read_text())
            markdown = (output_dir / "external_sota_runbook.md").read_text()

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["ok"])
        self.assertEqual(runbook["refcoco_external_alignment_plan"]["schema_version"], "refcoco-external-alignment-experiments-v0.1")
        self.assertIn("A_external_open_box_reference", markdown)
        self.assertIn("separate_external_reference_table", markdown)
        self.assertIn("box_aware_cross_attention_reranker", markdown)
        self.assertIn("proposal_oracle_recall_at_k", markdown)
