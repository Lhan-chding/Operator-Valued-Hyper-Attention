from __future__ import annotations

import json
import importlib.util
import math
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MultimodalControlledReportingTests(unittest.TestCase):
    def test_controlled_report_requires_all_oracle_cells_and_gates(self):
        from moat_ovha_torch.eval.multimodal_controlled_report import (
            CONTROLLED_REQUIRED_GATES,
            ORACLE_MATRIX_CELLS,
            build_controlled_report,
        )

        rows = [
            _row("tleo_local_evidence", "TLEO", 0.010, 0.010),
            _row("spo_global_prototype", "SPO", 0.020, 0.020),
            _row("lrio_low_rank_interaction", "LRIO", 0.030, 0.030),
            _row("cato_alignment_transport", "CATO", 0.040, 0.040),
            _row("rceo_reliability_corruption", "LRIO", 0.050, 0.050, rceo=True),
            _row("mixed_relation_operator", "mixed", 0.060, 0.060, router_accuracy=0.85),
        ]
        report = build_controlled_report(rows)

        self.assertEqual(ORACLE_MATRIX_CELLS, ("learned_learned", "true_learned", "learned_true", "true_true"))
        for gate in CONTROLLED_REQUIRED_GATES:
            self.assertIn(gate, report["gate_table"])
        self.assertTrue(report["go_no_go"]["controlled_multimodal_passed"])
        self.assertEqual(report["families"]["mixed_relation_operator"]["router_accuracy"], 0.85)

    def test_controlled_report_blocks_public_when_required_family_or_gate_missing(self):
        from moat_ovha_torch.eval.multimodal_controlled_report import build_controlled_report

        report = build_controlled_report([_row("tleo_local_evidence", "TLEO", 0.2, 0.1, true_learned_loss=0.2)])

        self.assertFalse(report["go_no_go"]["controlled_multimodal_passed"])
        self.assertIn("missing controlled family", "\n".join(report["go_no_go"]["reasons"]))
        self.assertFalse(report["gate_table"]["TLEO collapse"]["passed"])

    def test_controlled_report_rejects_unknown_or_duplicate_families(self):
        from moat_ovha_torch.eval.multimodal_controlled_report import build_controlled_report

        rows = [
            _row("tleo_local_evidence", "TLEO", 0.010, 0.010),
            _row("spo_global_prototype", "SPO", 0.020, 0.020),
            _row("lrio_low_rank_interaction", "LRIO", 0.030, 0.030),
            _row("cato_alignment_transport", "CATO", 0.040, 0.040),
            _row("rceo_reliability_corruption", "LRIO", 0.050, 0.050, rceo=True),
            _row("mixed_relation_operator", "mixed", 0.060, 0.060, router_accuracy=0.85),
            _row("unplanned_relation_operator", "TLEO", 0.070, 0.070),
            _row("tleo_local_evidence", "TLEO", 0.011, 0.011),
        ]
        report = build_controlled_report(rows)

        self.assertFalse(report["go_no_go"]["controlled_multimodal_passed"])
        joined = "\n".join(report["go_no_go"]["reasons"])
        self.assertIn("unknown controlled family: unplanned_relation_operator", joined)
        self.assertIn("duplicate controlled family row: tleo_local_evidence", joined)

    def test_controlled_report_requires_candidate_specific_collapse_diagnostics(self):
        from moat_ovha_torch.eval.multimodal_controlled_report import build_controlled_report

        rows = [
            _row("tleo_local_evidence", "TLEO", 0.010, 0.010),
            _row("spo_global_prototype", "SPO", 0.020, 0.020, include_operator_diagnostics=False),
            _row("lrio_low_rank_interaction", "LRIO", 0.030, 0.030, include_operator_diagnostics=False),
            _row("cato_alignment_transport", "CATO", 0.040, 0.040, include_operator_diagnostics=False),
            _row("rceo_reliability_corruption", "LRIO", 0.050, 0.050, rceo=True),
            _row("mixed_relation_operator", "mixed", 0.060, 0.060, router_accuracy=0.85),
        ]
        report = build_controlled_report(rows)

        self.assertFalse(report["go_no_go"]["controlled_multimodal_passed"])
        joined = "\n".join(report["go_no_go"]["reasons"])
        self.assertIn("SPO collapse missing diagnostic: prototype_kl_delta", joined)
        self.assertIn("LRIO collapse missing diagnostic: rank_logits_kl_delta", joined)
        self.assertIn("CATO collapse missing diagnostic: alignment_entropy_delta", joined)
        self.assertIn("CATO collapse missing diagnostic: alignment_topk_delta", joined)

    def test_controlled_report_requires_oracle_gap_and_rceo_prior_effect_evidence(self):
        from moat_ovha_torch.eval.multimodal_controlled_report import build_controlled_report

        rows = [
            _row("tleo_local_evidence", "TLEO", 0.010, 0.010, include_oracle_gap_evidence=False),
            _row("spo_global_prototype", "SPO", 0.020, 0.020),
            _row("lrio_low_rank_interaction", "LRIO", 0.030, 0.030),
            _row("cato_alignment_transport", "CATO", 0.040, 0.040),
            _row("rceo_reliability_corruption", "LRIO", 0.050, 0.050, rceo=True, include_rceo_prior_effect=False),
            _row("mixed_relation_operator", "mixed", 0.060, 0.060, router_accuracy=0.85),
        ]
        report = build_controlled_report(rows)

        self.assertFalse(report["go_no_go"]["controlled_multimodal_passed"])
        joined = "\n".join(report["go_no_go"]["reasons"])
        self.assertIn("tleo_local_evidence missing oracle gap evidence: TLEO_oracle_gap", joined)
        self.assertIn("rceo_reliability_corruption missing RCEO prior effect", joined)

    def test_controlled_report_rejects_non_finite_oracle_matrix_and_gap_values(self):
        from moat_ovha_torch.eval.multimodal_controlled_report import build_controlled_report

        rows = [
            _row(
                "tleo_local_evidence",
                "TLEO",
                0.010,
                0.010,
                oracle_overrides={"true_true": {"loss": "nan"}},
                oracle_gap_overrides={"TLEO_oracle_gap": "nan"},
            ),
            _row(
                "spo_global_prototype",
                "SPO",
                0.020,
                0.020,
                oracle_overrides={"learned_true": {"loss": "inf"}},
            ),
            _row("lrio_low_rank_interaction", "LRIO", 0.030, 0.030),
            _row("cato_alignment_transport", "CATO", 0.040, 0.040),
            _row("rceo_reliability_corruption", "LRIO", 0.050, 0.050, rceo=True),
            _row("mixed_relation_operator", "mixed", 0.060, 0.060, router_accuracy=0.85),
        ]

        report = build_controlled_report(rows)

        self.assertFalse(report["go_no_go"]["controlled_multimodal_passed"])
        joined = "\n".join(report["go_no_go"]["reasons"])
        self.assertIn("tleo_local_evidence oracle_matrix.true_true.loss must be finite non-negative", joined)
        self.assertIn("spo_global_prototype oracle_matrix.learned_true.loss must be finite non-negative", joined)
        self.assertIn("tleo_local_evidence oracle gap must be finite non-negative: TLEO_oracle_gap", joined)

    def test_controlled_report_requires_strict_gate_metric_values(self):
        from moat_ovha_torch.eval.multimodal_controlled_report import build_controlled_report

        rows = [
            _row(
                "tleo_local_evidence",
                "TLEO",
                0.010,
                0.010,
                stackability_passed="true",
                no_operator_memory_delta="nan",
            ),
            _row("spo_global_prototype", "SPO", 0.020, 0.020),
            _row("lrio_low_rank_interaction", "LRIO", 0.030, 0.030, no_lrio_delta="nan"),
            _row("cato_alignment_transport", "CATO", 0.040, 0.040, no_hyper_adapter_delta=-0.01),
            _row(
                "rceo_reliability_corruption",
                "LRIO",
                0.050,
                0.050,
                rceo=True,
                rceo_reliability_monotonic="true",
                rceo_router_load_shift="nan",
                no_rceo_delta=-0.01,
            ),
            _row("mixed_relation_operator", "mixed", 0.060, 0.060, router_accuracy="nan"),
        ]

        report = build_controlled_report(rows)

        self.assertFalse(report["go_no_go"]["controlled_multimodal_passed"])
        joined = "\n".join(report["go_no_go"]["reasons"])
        self.assertIn("tleo_local_evidence stackability_passed must be explicit true", joined)
        self.assertIn("Router gate router_accuracy must be a finite probability", joined)
        self.assertIn("RCEO gate rceo_reliability_monotonic must be explicit true", joined)
        self.assertIn("RCEO gate rceo_router_load_shift must be finite positive", joined)
        self.assertIn("Memory gate no_operator_memory_delta must be finite positive for tleo_local_evidence", joined)
        self.assertIn("Adapter gate no_hyper_adapter_delta must be finite positive for cato_alignment_transport", joined)
        self.assertIn(
            "no_lrio_delta must be finite positive for lrio_low_rank_interaction",
            "\n".join(report["gate_table"]["no-LRIO ablation"]["reasons"]),
        )
        self.assertIn(
            "no_rceo_delta must be finite positive for rceo_reliability_corruption",
            "\n".join(report["gate_table"]["no-RCEO ablation"]["reasons"]),
        )

    def test_controlled_report_requires_router_decomposition_ablation_evidence(self):
        from moat_ovha_torch.eval.multimodal_controlled_report import build_controlled_report

        rows = [
            _row("tleo_local_evidence", "TLEO", 0.010, 0.010),
            _row("spo_global_prototype", "SPO", 0.020, 0.020),
            _row("lrio_low_rank_interaction", "LRIO", 0.030, 0.030),
            _row("cato_alignment_transport", "CATO", 0.040, 0.040),
            _row("rceo_reliability_corruption", "LRIO", 0.050, 0.050, rceo=True),
            _row("mixed_relation_operator", "mixed", 0.060, 0.060, router_accuracy=0.85),
        ]
        rows[0].pop("no_reliability_prior_delta")

        report = build_controlled_report(rows)

        self.assertFalse(report["go_no_go"]["controlled_multimodal_passed"])
        self.assertIn("Router decomposition ablations", report["gate_table"])
        joined = "\n".join(report["go_no_go"]["reasons"])
        self.assertIn("no_reliability_prior_delta must be finite positive for tleo_local_evidence", joined)

    def test_rceo_gate_requires_reliability_corruption_curve_evidence(self):
        from moat_ovha_torch.eval.multimodal_controlled_report import build_controlled_report

        rows = [
            _row("tleo_local_evidence", "TLEO", 0.010, 0.010),
            _row("spo_global_prototype", "SPO", 0.020, 0.020),
            _row("lrio_low_rank_interaction", "LRIO", 0.030, 0.030),
            _row("cato_alignment_transport", "CATO", 0.040, 0.040),
            _row(
                "rceo_reliability_corruption",
                "LRIO",
                0.050,
                0.050,
                rceo=True,
                include_rceo_reliability_curve=False,
            ),
            _row("mixed_relation_operator", "mixed", 0.060, 0.060, router_accuracy=0.85),
        ]

        report = build_controlled_report(rows)

        self.assertFalse(report["go_no_go"]["controlled_multimodal_passed"])
        joined = "\n".join(report["go_no_go"]["reasons"])
        self.assertIn("RCEO gate rceo_reliability_curve must include at least two corruption points", joined)

    def test_rceo_gate_rejects_reliability_curve_that_increases_with_corruption(self):
        from moat_ovha_torch.eval.multimodal_controlled_report import build_controlled_report

        rows = [
            _row("tleo_local_evidence", "TLEO", 0.010, 0.010),
            _row("spo_global_prototype", "SPO", 0.020, 0.020),
            _row("lrio_low_rank_interaction", "LRIO", 0.030, 0.030),
            _row("cato_alignment_transport", "CATO", 0.040, 0.040),
            _row(
                "rceo_reliability_corruption",
                "LRIO",
                0.050,
                0.050,
                rceo=True,
                rceo_reliability_curve=[
                    {"corruption_strength": 0.0, "mean_reliability": 0.70},
                    {"corruption_strength": 0.5, "mean_reliability": 0.80},
                ],
            ),
            _row("mixed_relation_operator", "mixed", 0.060, 0.060, router_accuracy=0.85),
        ]

        report = build_controlled_report(rows)

        self.assertFalse(report["go_no_go"]["controlled_multimodal_passed"])
        joined = "\n".join(report["go_no_go"]["reasons"])
        self.assertIn(
            "RCEO gate rceo_reliability_curve must be non-increasing as corruption_strength increases",
            joined,
        )

    def test_collapse_gate_uses_true_router_learned_adapter_oracle_cell(self):
        from moat_ovha_torch.eval.multimodal_controlled_report import build_controlled_report

        rows = [
            _row("tleo_local_evidence", "TLEO", 0.300, 0.010),
            _row("spo_global_prototype", "SPO", 0.300, 0.020),
            _row("lrio_low_rank_interaction", "LRIO", 0.300, 0.030),
            _row("cato_alignment_transport", "CATO", 0.300, 0.040),
            _row("rceo_reliability_corruption", "LRIO", 0.050, 0.050, rceo=True),
            _row("mixed_relation_operator", "mixed", 0.060, 0.060, router_accuracy=0.85),
        ]
        report = build_controlled_report(rows)

        for gate_name, expected_value in (
            ("TLEO collapse", 0.010),
            ("SPO collapse", 0.020),
            ("LRIO collapse", 0.030),
            ("CATO collapse", 0.040),
        ):
            with self.subTest(gate=gate_name):
                gate = report["gate_table"][gate_name]
                self.assertTrue(gate["passed"], gate)
                self.assertEqual(gate["value"], expected_value)
                self.assertEqual(gate["oracle_cell"], "true_learned")

    def test_controlled_report_emits_sentiment_entry_ablation_gates(self):
        from moat_ovha_torch.eval.multimodal_controlled_report import build_controlled_report
        from moat_ovha_torch.eval.multimodal_public_entry import validate_public_entry_requirements

        rows = [
            _row("tleo_local_evidence", "TLEO", 0.010, 0.010),
            _row("spo_global_prototype", "SPO", 0.020, 0.020),
            _row("lrio_low_rank_interaction", "LRIO", 0.030, 0.030, no_lrio_delta=0.12),
            _row("cato_alignment_transport", "CATO", 0.040, 0.040),
            _row("rceo_reliability_corruption", "LRIO", 0.050, 0.050, rceo=True, no_rceo_delta=0.14),
            _row(
                "mixed_relation_operator",
                "mixed",
                0.060,
                0.060,
                router_accuracy=0.85,
                no_lrio_delta=0.10,
                no_rceo_delta=0.11,
            ),
        ]

        report = build_controlled_report(rows, evidence_artifacts=_controlled_evidence_artifacts())
        entry = validate_public_entry_requirements("sentiment_emotion", report)

        self.assertTrue(report["gate_table"]["no-LRIO ablation"]["passed"])
        self.assertTrue(report["gate_table"]["no-RCEO ablation"]["passed"])
        self.assertTrue(entry.ok, entry.errors)

    def test_controlled_report_emits_region_text_alignment_entry_gate(self):
        from moat_ovha_torch.eval.multimodal_controlled_report import build_controlled_report
        from moat_ovha_torch.eval.multimodal_public_entry import validate_public_entry_requirements

        rows = [
            _row("tleo_local_evidence", "TLEO", 0.010, 0.010),
            _row("spo_global_prototype", "SPO", 0.020, 0.020),
            _row("lrio_low_rank_interaction", "LRIO", 0.030, 0.030),
            _row("cato_alignment_transport", "CATO", 0.040, 0.040),
            _row("rceo_reliability_corruption", "LRIO", 0.050, 0.050, rceo=True),
            _row("mixed_relation_operator", "mixed", 0.060, 0.060, router_accuracy=0.85),
        ]

        report = build_controlled_report(rows, evidence_artifacts=_controlled_evidence_artifacts())
        entry = validate_public_entry_requirements("phrase_region_grounding", report)

        self.assertTrue(report["gate_table"]["CATO alignment diagnostics"]["passed"])
        self.assertTrue(entry.ok, entry.errors)

    def test_oracle_report_conversion_produces_gate_ready_row_without_torch(self):
        from moat_ovha_torch.eval.multimodal_controlled_report import build_controlled_report
        from moat_ovha_torch.eval.multimodal_oracle import controlled_row_from_oracle_report

        oracle_report = {
            "learned_learned": {"mse": 0.05},
            "true_learned": {"mse": 0.05},
            "learned_true": {"mse": 0.05},
            "true_true": {"mse": 0.0},
            "TLEO_oracle_gap": 0.2,
            "SPO_oracle_gap": 0.2,
            "LRIO_oracle_gap": 0.2,
            "CATO_oracle_gap": 0.2,
            "rceo_prior_effect": 0.3,
        }
        rceo_row = controlled_row_from_oracle_report(
            "rceo_reliability_corruption",
            "LRIO",
            oracle_report,
            no_operator_memory_delta=0.1,
            no_hyper_adapter_delta=0.1,
            no_evidence_router_delta=0.1,
            no_reliability_prior_delta=0.1,
            memory_only_router_delta=0.1,
            evidence_only_router_delta=0.1,
            no_rceo_delta=0.2,
            diagnostics={
                "rank_logits_kl_delta": 0.1,
                "rceo_reliability_monotonic": True,
                "rceo_router_load_shift": 0.1,
                "rceo_reliability_curve": [
                    {"corruption_strength": 0.0, "mean_reliability": 0.9},
                    {"corruption_strength": 0.5, "mean_reliability": 0.7},
                ],
            },
        )
        rows = [
            _row("tleo_local_evidence", "TLEO", 0.010, 0.010),
            _row("spo_global_prototype", "SPO", 0.020, 0.020),
            _row("lrio_low_rank_interaction", "LRIO", 0.030, 0.030),
            _row("cato_alignment_transport", "CATO", 0.040, 0.040),
            rceo_row,
            _row("mixed_relation_operator", "mixed", 0.060, 0.060, router_accuracy=0.85),
        ]
        report = build_controlled_report(rows)

        self.assertEqual(rceo_row["oracle_matrix"]["true_true"]["loss"], 0.0)
        self.assertEqual(rceo_row["LRIO_oracle_gap"], 0.2)
        self.assertEqual(rceo_row["rceo_prior_effect"], 0.3)
        self.assertEqual(rceo_row["no_rceo_delta"], 0.2)
        self.assertEqual(rceo_row["no_evidence_router_delta"], 0.1)
        self.assertEqual(rceo_row["no_reliability_prior_delta"], 0.1)
        self.assertEqual(rceo_row["memory_only_router_delta"], 0.1)
        self.assertEqual(rceo_row["evidence_only_router_delta"], 0.1)
        self.assertTrue(report["go_no_go"]["controlled_multimodal_passed"], report["go_no_go"]["reasons"])

    def test_robustness_summary_requires_auc_reliability_and_load_shift(self):
        from moat_ovha_torch.eval.multimodal_robustness import summarize_robustness_rows

        rows = [
            {
                "model": "ovha_full",
                "corruption_type": "image_blur",
                "corruption_strength": 0.0,
                "score": 0.80,
                "rceo_reliability": 0.90,
                "router_load_by_candidate": {"CATO": 0.60, "SPO": 0.10, "TLEO": 0.20, "LRIO": 0.10},
                "candidate_loss": {"CATO": 0.10, "SPO": 0.24, "TLEO": 0.20, "LRIO": 0.18},
            },
            {
                "model": "ovha_full",
                "corruption_type": "image_blur",
                "corruption_strength": 0.5,
                "score": 0.70,
                "rceo_reliability": 0.65,
                "router_load_by_candidate": {"CATO": 0.40, "SPO": 0.25, "TLEO": 0.25, "LRIO": 0.10},
                "candidate_loss": {"CATO": 0.22, "SPO": 0.18, "TLEO": 0.21, "LRIO": 0.19},
            },
            {
                "model": "cross_attention_transformer",
                "corruption_type": "image_blur",
                "corruption_strength": 0.0,
                "score": 0.78,
                "rceo_reliability": 0.0,
                "router_load_by_candidate": {},
            },
            {
                "model": "cross_attention_transformer",
                "corruption_type": "image_blur",
                "corruption_strength": 0.5,
                "score": 0.60,
                "rceo_reliability": 0.0,
                "router_load_by_candidate": {},
            },
            {
                "model": "ovha_no_rceo",
                "corruption_type": "image_blur",
                "corruption_strength": 0.0,
                "score": 0.79,
                "rceo_reliability": 0.0,
                "router_load_by_candidate": {},
            },
            {
                "model": "ovha_no_rceo",
                "corruption_type": "image_blur",
                "corruption_strength": 0.5,
                "score": 0.60,
                "rceo_reliability": 0.0,
                "router_load_by_candidate": {},
            },
            {
                "model": "ovha_no_evidence_router",
                "corruption_type": "image_blur",
                "corruption_strength": 0.0,
                "score": 0.79,
                "rceo_reliability": 0.0,
                "router_load_by_candidate": {},
            },
            {
                "model": "ovha_no_evidence_router",
                "corruption_type": "image_blur",
                "corruption_strength": 0.5,
                "score": 0.59,
                "rceo_reliability": 0.0,
                "router_load_by_candidate": {},
            },
        ]
        summary = summarize_robustness_rows(rows, full_model="ovha_full", baseline_model="cross_attention_transformer")

        self.assertEqual(summary["clean_score"]["ovha_full"], 0.80)
        self.assertEqual(summary["corrupted_score"]["ovha_full"], 0.70)
        self.assertEqual(summary["clean_score"]["cross_attention_transformer"], 0.78)
        self.assertEqual(summary["corrupted_score"]["cross_attention_transformer"], 0.60)
        self.assertLess(summary["relative_drop"]["ovha_full"], summary["relative_drop"]["cross_attention_transformer"])
        self.assertTrue(summary["rceo_reliability_monotonic"])
        self.assertEqual(summary["rceo_reliability_shift"], -0.25)
        self.assertEqual(
            summary["rceo_reliability_curve"],
            [
                {"corruption_strength": 0.0, "mean_reliability": 0.90},
                {"corruption_strength": 0.5, "mean_reliability": 0.65},
            ],
        )
        self.assertLess(summary["operator_load_shift"]["CATO"], 0.0)
        self.assertEqual(summary["candidate_loss_shift"]["CATO"], 0.12)
        self.assertEqual(summary["candidate_loss_shift"]["SPO"], -0.06)
        self.assertIn("auc_over_corruption_strength", summary)
        self.assertTrue(summary["required_ablation_degradation"]["passed"], summary["required_ablation_degradation"]["reasons"])

    def test_robustness_summary_requires_no_rceo_and_no_evidence_router_degradation(self):
        from moat_ovha_torch.eval.multimodal_robustness import summarize_robustness_rows

        rows = [
            _robustness_row("ovha_full", 0.0, 0.80, reliability=0.90),
            _robustness_row("ovha_full", 0.5, 0.70, reliability=0.65),
            _robustness_row("cross_attention_transformer", 0.0, 0.78),
            _robustness_row("cross_attention_transformer", 0.5, 0.60),
        ]

        summary = summarize_robustness_rows(rows, full_model="ovha_full", baseline_model="cross_attention_transformer")

        self.assertFalse(summary["required_ablation_degradation"]["passed"])
        joined = "\n".join(summary["required_ablation_degradation"]["reasons"])
        self.assertIn("missing robustness ablation rows: ovha_no_rceo", joined)
        self.assertIn("missing robustness ablation rows: ovha_no_evidence_router", joined)

    def test_robustness_summary_aggregates_rceo_curve_by_corruption_strength(self):
        from moat_ovha_torch.eval.multimodal_robustness import summarize_robustness_rows

        rows = [
            _robustness_row("ovha_full", 0.0, 0.80, reliability=0.90),
            _robustness_row("ovha_full", 0.0, 0.78, reliability=0.80),
            _robustness_row("ovha_full", 0.5, 0.70, reliability=0.60),
            _robustness_row("ovha_full", 0.5, 0.68, reliability=0.50),
        ]

        summary = summarize_robustness_rows(rows, full_model="ovha_full", baseline_model="cross_attention_transformer")

        self.assertEqual([point["corruption_strength"] for point in summary["rceo_reliability_curve"]], [0.0, 0.5])
        self.assertAlmostEqual(summary["rceo_reliability_curve"][0]["mean_reliability"], 0.85)
        self.assertAlmostEqual(summary["rceo_reliability_curve"][1]["mean_reliability"], 0.55)
        self.assertAlmostEqual(summary["rceo_reliability_shift"], -0.30)
        self.assertTrue(summary["rceo_reliability_monotonic"])

    def test_robustness_summary_aggregates_auc_by_corruption_strength(self):
        from moat_ovha_torch.eval.multimodal_robustness import summarize_robustness_rows

        rows = [
            _robustness_row("ovha_full", 0.0, 0.50, reliability=0.90),
            _robustness_row("ovha_full", 0.0, 1.00, reliability=0.90),
            _robustness_row("ovha_full", 0.5, 0.00, reliability=0.65),
            _robustness_row("ovha_full", 0.5, 1.00, reliability=0.65),
        ]

        summary = summarize_robustness_rows(rows, full_model="ovha_full", baseline_model="cross_attention_transformer")

        self.assertAlmostEqual(summary["clean_score"]["ovha_full"], 0.75)
        self.assertAlmostEqual(summary["corrupted_score"]["ovha_full"], 0.50)
        self.assertAlmostEqual(summary["auc_over_corruption_strength"]["ovha_full"], 0.625)

    def test_robustness_summary_emits_rceo_reliability_calibration_curve(self):
        from moat_ovha_torch.eval.multimodal_robustness import summarize_robustness_rows

        rows = []
        for strength, predicted, observed in (
            (0.0, 0.10, 0.00),
            (0.1, 0.30, 0.40),
            (0.2, 0.50, 0.60),
            (0.3, 0.70, 0.80),
            (0.4, 0.90, 1.00),
        ):
            row = _robustness_row("ovha_full", strength, 0.80 - strength, reliability=predicted)
            row["rceo_observed_reliability"] = observed
            rows.append(row)

        summary = summarize_robustness_rows(rows, full_model="ovha_full", baseline_model="cross_attention_transformer")
        calibration = summary["rceo_reliability_calibration"]

        self.assertAlmostEqual(calibration["ece"], 0.10)
        self.assertEqual(calibration["bin_count"], 5)
        self.assertEqual(
            calibration["calibration_curve"],
            [
                {"bin": 0, "mean_confidence": 0.10, "observed_accuracy": 0.00, "count": 1},
                {"bin": 1, "mean_confidence": 0.30, "observed_accuracy": 0.40, "count": 1},
                {"bin": 2, "mean_confidence": 0.50, "observed_accuracy": 0.60, "count": 1},
                {"bin": 3, "mean_confidence": 0.70, "observed_accuracy": 0.80, "count": 1},
                {"bin": 4, "mean_confidence": 0.90, "observed_accuracy": 1.00, "count": 1},
            ],
        )

    def test_robustness_summary_reports_paired_drop_significance_by_seed(self):
        from moat_ovha_torch.eval.multimodal_robustness import summarize_robustness_rows

        rows = []
        for seed, clean in ((11, 1.00), (12, 0.98), (13, 0.96)):
            for model, drop in (("ovha_full", 0.10), ("cross_attention_transformer", 0.30)):
                clean_row = _robustness_row(model, 0.0, clean, reliability=0.90 if model == "ovha_full" else 0.0)
                clean_row["seed"] = seed
                corrupted_row = _robustness_row(model, 0.5, clean * (1.0 - drop), reliability=0.65 if model == "ovha_full" else 0.0)
                corrupted_row["seed"] = seed
                rows.extend((clean_row, corrupted_row))

        summary = summarize_robustness_rows(rows, full_model="ovha_full", baseline_model="cross_attention_transformer")
        significance = summary["robustness_significance"]

        self.assertEqual(
            significance["model_delta"],
            "cross_attention_transformer_relative_drop_minus_ovha_full_relative_drop",
        )
        self.assertEqual(significance["metric"], "relative_drop_delta")
        self.assertEqual(significance["delta_interpretation"], "positive means ovha_full drops less under robustness stress")
        self.assertEqual(significance["common_seed_count"], 3)
        self.assertAlmostEqual(significance["drop_delta"], 0.20)
        self.assertAlmostEqual(significance["paired_permutation_p"], 0.25)
        self.assertEqual(len(significance["per_seed_drop_delta"]), 3)
        for row in significance["per_seed_drop_delta"]:
            self.assertAlmostEqual(row["full_relative_drop"], 0.10)
            self.assertAlmostEqual(row["baseline_relative_drop"], 0.30)
            self.assertAlmostEqual(row["drop_delta"], 0.20)
        self.assertAlmostEqual(significance["paired_bootstrap_ci95"][0], 0.20)
        self.assertAlmostEqual(significance["paired_bootstrap_ci95"][1], 0.20)

    def test_robustness_summary_requires_plan_stress_family_coverage(self):
        from moat_ovha_torch.eval.multimodal_robustness import summarize_robustness_rows

        rows = [
            _robustness_row("ovha_full", 0.0, 0.80, reliability=0.90),
            _robustness_row("ovha_full", 0.5, 0.70, reliability=0.65),
            _robustness_row("cross_attention_transformer", 0.0, 0.78),
            _robustness_row("cross_attention_transformer", 0.5, 0.60),
            _robustness_row("ovha_no_rceo", 0.0, 0.79),
            _robustness_row("ovha_no_rceo", 0.5, 0.60),
            _robustness_row("ovha_no_evidence_router", 0.0, 0.79),
            _robustness_row("ovha_no_evidence_router", 0.5, 0.59),
        ]

        summary = summarize_robustness_rows(rows, full_model="ovha_full", baseline_model="cross_attention_transformer")

        self.assertFalse(summary["required_stress_coverage"]["passed"])
        joined = "\n".join(summary["required_stress_coverage"]["reasons"])
        self.assertIn("missing robustness stress target: missing_text", joined)
        self.assertIn("missing robustness stress target: missing_vision", joined)
        self.assertIn("missing robustness stress target: audio_noise", joined)
        self.assertIn("missing robustness stress target: hard_negative_caption_mismatch", joined)

    def test_robustness_summary_requires_each_plan_stress_transform(self):
        from moat_ovha_torch.eval.multimodal_robustness import summarize_robustness_rows

        rows = [
            _stress_row("missing_modality", missing_modalities=["text"]),
            _stress_row("missing_modality", missing_modalities=["vision"]),
            _stress_row("missing_modality", missing_modalities=["audio"]),
            _stress_row("image_blur"),
            _stress_row("audio_noise"),
            _stress_row("text_token_mask"),
            _stress_row("hard_negative_mismatch", mismatch_source_id="other-sample"),
            _stress_row("temporal_shift", temporal_shift_sec=1.2),
        ]

        summary = summarize_robustness_rows(
            rows,
            full_model="ovha_full",
            baseline_model="cross_attention_transformer",
            temporal_data=True,
        )

        self.assertFalse(summary["required_stress_coverage"]["passed"])
        joined = "\n".join(summary["required_stress_coverage"]["reasons"])
        self.assertIn("missing robustness stress target: image_crop", joined)
        self.assertIn("missing robustness stress target: image_occlusion", joined)
        self.assertIn("missing robustness stress target: audio_masking", joined)
        self.assertIn("missing robustness stress target: text_paraphrase", joined)
        self.assertIn("missing robustness stress target: hard_negative_caption_mismatch", joined)
        self.assertIn("missing robustness stress target: hard_negative_region_mismatch", joined)
        self.assertIn("missing robustness stress target: hard_negative_audio_mismatch", joined)

    def test_robustness_summary_requires_required_stress_metadata_fields(self):
        from moat_ovha_torch.eval.multimodal_robustness import summarize_robustness_rows

        rows = [
            _stress_row("missing_text"),
            _stress_row("missing_vision"),
            _stress_row("missing_audio"),
            _stress_row("image_blur"),
            _stress_row("image_crop"),
            _stress_row("image_occlusion"),
            _stress_row("audio_noise"),
            _stress_row("audio_masking"),
            _stress_row("text_token_mask"),
            _stress_row("text_paraphrase"),
            _stress_row("hard_negative_caption_mismatch", mismatch_source_id="other-caption"),
            _stress_row("hard_negative_region_mismatch", mismatch_source_id="other-region"),
            _stress_row("hard_negative_audio_mismatch", mismatch_source_id="other-audio"),
            _stress_row("temporal_shift"),
        ]

        summary = summarize_robustness_rows(
            rows,
            full_model="ovha_full",
            baseline_model="cross_attention_transformer",
            temporal_data=True,
        )

        self.assertFalse(summary["required_stress_coverage"]["passed"])
        joined = "\n".join(summary["required_stress_coverage"]["reasons"])
        self.assertIn("missing robustness stress target: missing_text", joined)
        self.assertIn("missing robustness stress target: missing_vision", joined)
        self.assertIn("missing robustness stress target: missing_audio", joined)
        self.assertIn("missing robustness stress target: temporal_shift", joined)

    def test_robustness_summary_requires_finite_corruption_strength_metadata(self):
        from moat_ovha_torch.eval.multimodal_robustness import summarize_robustness_rows

        rows = [
            _stress_row("missing_modality", missing_modalities=["text"]),
            _stress_row("missing_modality", missing_modalities=["vision"]),
            _stress_row("missing_modality", missing_modalities=["audio"]),
            _stress_row("image_blur"),
            _stress_row("image_crop"),
            _stress_row("image_occlusion"),
            _stress_row("audio_noise"),
            _stress_row("audio_masking"),
            _stress_row("text_token_mask"),
            _stress_row("text_paraphrase"),
            _stress_row("hard_negative_caption_mismatch", mismatch_source_id="other-caption"),
            _stress_row("hard_negative_region_mismatch", mismatch_source_id="other-region"),
            _stress_row("hard_negative_audio_mismatch", mismatch_source_id="other-audio"),
            _stress_row("temporal_shift", temporal_shift_sec=1.2),
        ]
        rows[3]["corruption_strength"] = float("nan")

        summary = summarize_robustness_rows(
            rows,
            full_model="ovha_full",
            baseline_model="cross_attention_transformer",
            temporal_data=True,
        )

        self.assertFalse(summary["required_stress_coverage"]["passed"])
        self.assertIn(
            "missing robustness stress target: image_blur",
            "\n".join(summary["required_stress_coverage"]["reasons"]),
        )

    def test_robustness_summary_requires_non_negative_corruption_strength_metadata(self):
        from moat_ovha_torch.eval.multimodal_robustness import summarize_robustness_rows

        rows = [
            _stress_row("missing_modality", missing_modalities=["text"]),
            _stress_row("missing_modality", missing_modalities=["vision"]),
            _stress_row("missing_modality", missing_modalities=["audio"]),
            _stress_row("image_blur"),
            _stress_row("image_crop"),
            _stress_row("image_occlusion"),
            _stress_row("audio_noise"),
            _stress_row("audio_masking"),
            _stress_row("text_token_mask"),
            _stress_row("text_paraphrase"),
            _stress_row("hard_negative_caption_mismatch", mismatch_source_id="other-caption"),
            _stress_row("hard_negative_region_mismatch", mismatch_source_id="other-region"),
            _stress_row("hard_negative_audio_mismatch", mismatch_source_id="other-audio"),
            _stress_row("temporal_shift", temporal_shift_sec=1.2),
        ]
        rows[4]["corruption_strength"] = -0.1

        summary = summarize_robustness_rows(
            rows,
            full_model="ovha_full",
            baseline_model="cross_attention_transformer",
            temporal_data=True,
        )

        self.assertFalse(summary["required_stress_coverage"]["passed"])
        self.assertIn(
            "missing robustness stress target: image_crop",
            "\n".join(summary["required_stress_coverage"]["reasons"]),
        )

    def test_robustness_summary_accepts_plan_stress_metadata_coverage(self):
        from moat_ovha_torch.eval.multimodal_robustness import summarize_robustness_rows

        rows = [
            _stress_row("missing_modality", missing_modalities=["text"]),
            _stress_row("missing_modality", missing_modalities=["vision"]),
            _stress_row("missing_modality", missing_modalities=["audio"]),
            _stress_row("image_blur"),
            _stress_row("image_crop"),
            _stress_row("image_occlusion"),
            _stress_row("audio_noise"),
            _stress_row("audio_masking"),
            _stress_row("text_token_mask"),
            _stress_row("text_paraphrase"),
            _stress_row("hard_negative_caption_mismatch", mismatch_source_id="other-caption"),
            _stress_row("hard_negative_region_mismatch", mismatch_source_id="other-region"),
            _stress_row("hard_negative_audio_mismatch", mismatch_source_id="other-audio"),
            _stress_row("temporal_shift", temporal_shift_sec=1.2),
        ]

        summary = summarize_robustness_rows(
            rows,
            full_model="ovha_full",
            baseline_model="cross_attention_transformer",
            temporal_data=True,
        )

        self.assertTrue(summary["required_stress_coverage"]["passed"], summary["required_stress_coverage"]["reasons"])
        self.assertIn("temporal_shift", summary["required_stress_coverage"]["observed"])

    def test_controlled_and_robustness_summary_scripts_emit_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            controlled_path = tmp_path / "controlled.jsonl"
            controlled_path.write_text(json.dumps(_row("tleo_local_evidence", "TLEO", 0.2, 0.1)) + "\n")
            robustness_path = tmp_path / "robustness.jsonl"
            robustness_path.write_text(
                json.dumps(
                    {
                        "model": "ovha_full",
                        "corruption_type": "image_blur",
                        "corruption_strength": 0.0,
                        "score": 0.8,
                        "rceo_reliability": 0.9,
                        "router_load_by_candidate": {"CATO": 0.6},
                    }
                )
                + "\n"
            )

            controlled = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "multimodal" / "summarize_controlled_report.py"), str(controlled_path)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            robustness = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "summarize_robustness.py"),
                    str(robustness_path),
                    "--baseline-model",
                    "cross_attention_transformer",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(controlled.returncode, 0, controlled.stderr)
        self.assertEqual(robustness.returncode, 0, robustness.stderr)
        controlled_payload = json.loads(controlled.stdout)
        self.assertIn("go_no_go", controlled_payload)
        artifacts = controlled_payload["evidence_artifacts"]
        self.assertEqual(artifacts["task"], "controlled_multimodal")
        self.assertIn("summarize_controlled_report.py", artifacts["generated_by"])
        self.assertRegex(artifacts["controlled_rows"]["sha256"], r"^[a-f0-9]{64}$")
        self.assertRegex(artifacts["diagnostics_report"]["sha256"], r"^[a-f0-9]{64}$")
        self.assertIn("relative_drop", json.loads(robustness.stdout))

    def test_controlled_smoke_writes_auditable_rows_and_diagnostics_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_root = Path(tmp) / "controlled_artifacts"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "run_controlled_smoke.py"),
                    "--artifact-root",
                    str(artifact_root),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            artifacts = payload["controlled_report"]["evidence_artifacts"]
            controlled_rows_path = Path(artifacts["controlled_rows"]["path"])
            diagnostics_report_path = Path(artifacts["diagnostics_report"]["path"])
            controlled_rows = [
                json.loads(line)
                for line in controlled_rows_path.read_text().splitlines()
                if line.strip()
            ]
            diagnostics_rows = [
                json.loads(line)
                for line in diagnostics_report_path.read_text().splitlines()
                if line.strip()
            ]

        self.assertEqual(payload["mode"], "oracle_smoke_only")
        self.assertEqual(len(controlled_rows), 6)
        self.assertEqual(len(diagnostics_rows), 6)
        self.assertNotEqual(controlled_rows_path, diagnostics_report_path)
        self.assertNotEqual(
            artifacts["controlled_rows"]["sha256"],
            artifacts["diagnostics_report"]["sha256"],
        )
        self.assertRegex(artifacts["controlled_rows"]["sha256"], r"^[a-f0-9]{64}$")
        self.assertRegex(artifacts["diagnostics_report"]["sha256"], r"^[a-f0-9]{64}$")
        self.assertTrue(payload["controlled_report"]["go_no_go"]["controlled_multimodal_passed"])
        self.assertEqual(
            {row["family"] for row in controlled_rows},
            {row["family"] for row in diagnostics_rows},
        )
        self.assertTrue(all(row["artifact_type"] == "controlled_diagnostics" for row in diagnostics_rows))
        self.assertTrue(all("active_operator" not in row for row in diagnostics_rows))
        self.assertIn("oracle_matrix", diagnostics_rows[0])
        self.assertIn("stackability_passed", diagnostics_rows[0])

    def test_controlled_training_smoke_runs_real_optimizer_steps_and_artifacts(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is required for controlled training smoke")

        with tempfile.TemporaryDirectory() as tmp:
            artifact_root = Path(tmp) / "controlled_training_artifacts"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "run_controlled_training_smoke.py"),
                    "--steps",
                    "6",
                    "--batch-size",
                    "2",
                    "--query-count",
                    "4",
                    "--d-model",
                    "16",
                    "--artifact-root",
                    str(artifact_root),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = json.loads(result.stdout)
            artifacts = payload["controlled_report"]["evidence_artifacts"]
            controlled_rows_path = Path(artifacts["controlled_rows"]["path"])
            diagnostics_report_path = Path(artifacts["diagnostics_report"]["path"])
            controlled_rows = [
                json.loads(line)
                for line in controlled_rows_path.read_text().splitlines()
                if line.strip()
            ]
            diagnostics_rows = [
                json.loads(line)
                for line in diagnostics_report_path.read_text().splitlines()
                if line.strip()
            ]

        training = payload["training"]
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["mode"], "trained_smoke")
        self.assertEqual(training["optimizer_steps"], 6)
        self.assertEqual(len(training["loss_history"]), 6)
        self.assertGreater(training["parameter_l2_delta"], 0.0)
        self.assertGreater(training["max_grad_norm"], 0.0)
        self.assertLess(training["mean_task_loss_after"], training["mean_task_loss_before"])
        self.assertEqual(len(payload["rows"]), 6)
        self.assertEqual(len(controlled_rows), 6)
        self.assertEqual(len(diagnostics_rows), 6)
        self.assertIn("go_no_go", payload["controlled_report"])
        self.assertEqual(artifacts["generated_by"], "scripts/multimodal/run_controlled_training_smoke.py")
        self.assertNotEqual(artifacts["controlled_rows"]["sha256"], artifacts["diagnostics_report"]["sha256"])
        self.assertTrue(all(row["training_mode"] == "trained_smoke" for row in controlled_rows))
        self.assertTrue(all(row["artifact_type"] == "controlled_training_diagnostics" for row in diagnostics_rows))
        self.assertTrue(all("active_operator" not in row for row in diagnostics_rows))
        controlled_by_family = {row["family"]: row for row in controlled_rows}
        diagnostics_by_family = {row["family"]: row for row in diagnostics_rows}
        ablation_keys = (
            "no_evidence_router_delta",
            "no_reliability_prior_delta",
            "memory_only_router_delta",
            "evidence_only_router_delta",
            "no_operator_memory_delta",
            "no_hyper_adapter_delta",
        )
        for family, row in controlled_by_family.items():
            measured = []
            for key in ablation_keys:
                self.assertIn(key, row, family)
                self.assertIn(key, diagnostics_by_family[family], family)
                value = float(row[key])
                self.assertTrue(math.isfinite(value), f"{family}.{key}")
                measured.append(abs(value))
            self.assertTrue(any(value > 1e-12 for value in measured), family)
        self.assertIn("prototype_kl_delta", controlled_by_family["spo_global_prototype"])
        self.assertTrue(math.isfinite(float(controlled_by_family["spo_global_prototype"]["prototype_kl_delta"])))
        self.assertIn("rank_logits_kl_delta", controlled_by_family["lrio_low_rank_interaction"])
        self.assertTrue(math.isfinite(float(controlled_by_family["lrio_low_rank_interaction"]["rank_logits_kl_delta"])))
        cato_row = controlled_by_family["cato_alignment_transport"]
        self.assertIn("alignment_entropy_delta", cato_row)
        self.assertIn("alignment_topk_delta", cato_row)
        self.assertTrue(math.isfinite(float(cato_row["alignment_entropy_delta"])))
        self.assertTrue(math.isfinite(float(cato_row["alignment_topk_delta"])))
        rceo_row = controlled_by_family["rceo_reliability_corruption"]
        self.assertIn("rceo_reliability_curve", rceo_row)
        self.assertGreaterEqual(len(rceo_row["rceo_reliability_curve"]), 2)
        self.assertEqual(rceo_row["rceo_reliability_monotonic"], True)
        self.assertTrue(math.isfinite(float(rceo_row["rceo_router_load_shift"])))
        self.assertIn("no_lrio_delta", controlled_by_family["lrio_low_rank_interaction"])
        self.assertIn("no_rceo_delta", rceo_row)

    def test_router_decomposition_diagnostics_use_active_operator_ce_not_task_loss(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is required for controlled training smoke")

        import torch

        from scripts.multimodal.run_controlled_training_smoke import _router_decomposition_ablation_deltas

        candidate_values = torch.zeros(1, 4, 4, 1)
        batch = SimpleNamespace(
            hidden={"true_active_operator": torch.arange(4).view(1, 4)},
            target_y=torch.zeros(1, 4, 1),
            target_mask=torch.ones(1, 4, dtype=torch.bool),
        )
        memory_logits = torch.nn.functional.one_hot(torch.arange(4).view(1, 4), num_classes=4).float() * 2.0
        evidence_logits = torch.nn.functional.one_hot(torch.arange(4).view(1, 4), num_classes=4).float() * 2.0
        reliability_logits = torch.zeros(1, 4, 4)
        output = SimpleNamespace(
            candidate_values=candidate_values,
            router_logits=memory_logits + evidence_logits + reliability_logits,
            router_logit_parts={
                "memory": memory_logits,
                "evidence": evidence_logits,
                "reliability": reliability_logits,
            },
        )

        deltas = _router_decomposition_ablation_deltas(output, batch, full_loss=0.0)

        self.assertGreater(deltas["memory_only_router_delta"], 0.0)
        self.assertGreater(deltas["evidence_only_router_delta"], 0.0)

    def test_controlled_training_smoke_consumes_formal_config_and_stage_protocol(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is required for controlled training smoke")

        with tempfile.TemporaryDirectory() as tmp:
            artifact_root = Path(tmp) / "controlled_training_artifacts"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "run_controlled_training_smoke.py"),
                    "--config",
                    str(ROOT / "configs" / "multimodal_controlled_v1_smoke.json"),
                    "--steps-per-stage",
                    "1",
                    "--batch-size",
                    "2",
                    "--query-count",
                    "4",
                    "--d-model",
                    "16",
                    "--artifact-root",
                    str(artifact_root),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = json.loads(result.stdout)
            rows = payload["rows"]

        training = payload["training"]
        stages = {stage["stage"]: stage for stage in training["stage_history"]}
        self.assertEqual(training["config_name"], "multimodal_controlled_v1_smoke")
        self.assertEqual(training["validated_training_stages"], ["T0", "T1", "T2", "T3", "T4"])
        self.assertEqual(training["optimizer_steps"], 4)
        self.assertEqual(len(training["loss_history"]), 4)
        self.assertEqual(stages["T0"]["loss_names_observed"], ["cache_validation"])
        self.assertEqual(stages["T0"]["optimizer_steps"], 0)
        self.assertEqual(stages["T1"]["loss_names_observed"], ["candidate_individual_loss", "task_loss"])
        self.assertEqual(
            stages["T2"]["loss_names_observed"],
            ["adapter_kl_true_params", "router_ce_true_active_operator", "task_loss"],
        )
        self.assertEqual(stages["T2"]["route_override_mode"], "true_router_weights")
        self.assertEqual(stages["T2"]["trainable_parameter_scope"], "oracle_router_adapter_candidate_warmup")
        self.assertEqual(
            stages["T2"]["trainable_parameter_groups"],
            [
                "candidate_primitives.CATO",
                "candidate_primitives.LRIO",
                "candidate_primitives.SPO",
                "candidate_primitives.TLEO",
                "joint_router_adapter.hyper_adapter.CATO",
                "joint_router_adapter.hyper_adapter.LRIO",
                "joint_router_adapter.hyper_adapter.SPO",
                "joint_router_adapter.hyper_adapter.TLEO",
            ],
        )
        self.assertIn("joint_router_adapter.router", stages["T2"]["frozen_parameter_groups"])
        self.assertIn("evidence_encoder", stages["T2"]["frozen_parameter_groups"])
        self.assertIn("memory_encoder", stages["T2"]["frozen_parameter_groups"])
        self.assertIn("reliability_prior", stages["T2"]["frozen_parameter_groups"])
        self.assertEqual(
            stages["T3"]["loss_names_observed"],
            ["router_ce_true_active_operator", "task_loss"],
        )
        self.assertEqual(stages["T3"]["trainable_parameter_scope"], "router_only_warmup")
        self.assertEqual(stages["T3"]["trainable_parameter_groups"], ["joint_router_adapter.router"])
        self.assertIn("candidate_primitives", stages["T3"]["frozen_parameter_groups"])
        self.assertIn("joint_router_adapter.hyper_adapter", stages["T3"]["frozen_parameter_groups"])
        self.assertEqual(stages["T3"]["families_seen"], ["mixed_relation_operator"])
        self.assertTrue(
            all(
                entry["family"] == "mixed_relation_operator"
                for entry in training["loss_history"]
                if entry["stage"] == "T3"
            )
        )
        self.assertEqual(
            stages["T4"]["loss_names_observed"],
            [
                "cato_alignment_ce",
                "lrio_rank_kl",
                "rceo_reliability_huber",
                "spo_prototype_kl",
                "task_loss",
                "tleo_lengthscale_huber",
            ],
        )
        self.assertEqual(stages["T4"]["oracle_matrix_monitoring"], "per_step")
        self.assertEqual(
            stages["T4"]["oracle_matrix_cells"],
            ["learned_learned", "true_learned", "learned_true", "true_true"],
        )
        self.assertEqual(stages["T4"]["oracle_matrix_snapshot_count"], 1)
        t4_rows = [entry for entry in training["loss_history"] if entry["stage"] == "T4"]
        self.assertTrue(t4_rows)
        for entry in t4_rows:
            snapshot = entry.get("oracle_matrix_snapshot")
            self.assertIsInstance(snapshot, dict)
            self.assertEqual(
                sorted(snapshot),
                ["learned_learned", "learned_true", "true_learned", "true_true"],
            )
            for cell in ("learned_learned", "true_learned", "learned_true", "true_true"):
                loss = snapshot[cell]["loss"]
                self.assertTrue(math.isfinite(loss), cell)
                self.assertGreaterEqual(loss, 0.0, cell)
        self.assertTrue(all(entry["stage"] in {"T1", "T2", "T3", "T4"} for entry in training["loss_history"]))
        self.assertTrue(
            all(
                entry.get("route_override_mode") == "true_router_weights"
                for entry in training["loss_history"]
                if entry["stage"] == "T2"
            )
        )
        self.assertTrue(
            all(
                entry.get("trainable_parameter_scope") == "oracle_router_adapter_candidate_warmup"
                for entry in training["loss_history"]
                if entry["stage"] == "T2"
            )
        )
        self.assertTrue(
            all(
                entry.get("trainable_parameter_scope") == "router_only_warmup"
                for entry in training["loss_history"]
                if entry["stage"] == "T3"
            )
        )
        self.assertTrue(all(row["training_config_name"] == "multimodal_controlled_v1_smoke" for row in rows))

    def test_controlled_training_t1_specialist_warmup_excludes_rceo_and_mixed_families(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is required for controlled training smoke")

        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "run_controlled_training_smoke.py"),
                    "--config",
                    str(ROOT / "configs" / "multimodal_controlled_v1_smoke.json"),
                    "--steps-per-stage",
                    "6",
                    "--batch-size",
                    "2",
                    "--query-count",
                    "4",
                    "--d-model",
                    "16",
                    "--artifact-root",
                    str(Path(tmp) / "controlled_training_artifacts"),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = json.loads(result.stdout)

        stages = {stage["stage"]: stage for stage in payload["training"]["stage_history"]}
        self.assertEqual(
            stages["T1"]["families_seen"],
            [
                "cato_alignment_transport",
                "lrio_low_rank_interaction",
                "spo_global_prototype",
                "tleo_local_evidence",
            ],
        )
        t1_rows = [row for row in payload["training"]["loss_history"] if row["stage"] == "T1"]
        self.assertTrue(t1_rows)
        self.assertNotIn("rceo_reliability_corruption", {row["family"] for row in t1_rows})
        self.assertNotIn("mixed_relation_operator", {row["family"] for row in t1_rows})
        expected_candidate_by_family = {
            "tleo_local_evidence": "TLEO",
            "spo_global_prototype": "SPO",
            "lrio_low_rank_interaction": "LRIO",
            "cato_alignment_transport": "CATO",
        }
        for row in t1_rows:
            specialist_candidate = expected_candidate_by_family[row["family"]]
            self.assertEqual(row["specialist_candidate"], specialist_candidate)
            self.assertEqual(row["route_override_mode"], "true_router_weights")
            self.assertEqual(row["trainable_parameter_scope"], "candidate_only_specialist_warmup")
            self.assertEqual(
                row["trainable_parameter_groups"],
                [
                    f"candidate_primitives.{specialist_candidate}",
                    f"joint_router_adapter.hyper_adapter.{specialist_candidate}",
                ],
            )


def _row(
    family: str,
    active_operator: str,
    full_loss: float,
    specialist_loss: float,
    *,
    router_accuracy: float = 0.9,
    rceo: bool = False,
    include_operator_diagnostics: bool = True,
    include_oracle_gap_evidence: bool = True,
    include_rceo_prior_effect: bool = True,
    true_learned_loss: float | None = None,
    no_lrio_delta: float | None = None,
    no_rceo_delta: float | None = None,
    oracle_overrides: dict[str, object] | None = None,
    oracle_gap_overrides: dict[str, object] | None = None,
    stackability_passed: object = True,
    no_operator_memory_delta: object = 0.1,
    no_hyper_adapter_delta: object = 0.1,
    no_evidence_router_delta: object = 0.1,
    no_reliability_prior_delta: object = 0.1,
    memory_only_router_delta: object = 0.1,
    evidence_only_router_delta: object = 0.1,
    router_accuracy_value: object | None = None,
    rceo_reliability_monotonic: object = True,
    rceo_router_load_shift: object = 0.1,
    include_rceo_reliability_curve: bool = True,
    rceo_reliability_curve: object | None = None,
) -> dict[str, object]:
    oracle_matrix = {
        "learned_learned": {"loss": full_loss},
        "true_learned": {"loss": specialist_loss if true_learned_loss is None else true_learned_loss},
        "learned_true": {"loss": full_loss},
        "true_true": {"loss": 0.0},
    }
    if oracle_overrides is not None:
        for cell, values in oracle_overrides.items():
            if isinstance(values, dict) and isinstance(oracle_matrix.get(cell), dict):
                oracle_matrix[cell] = {**oracle_matrix[cell], **values}
            else:
                oracle_matrix[cell] = values
    row = {
        "family": family,
        "active_operator": active_operator,
        "oracle_matrix": oracle_matrix,
        "specialist_loss": specialist_loss,
        "router_accuracy": router_accuracy if router_accuracy_value is None else router_accuracy_value,
        "stackability_passed": stackability_passed,
        "no_operator_memory_delta": no_operator_memory_delta,
        "no_hyper_adapter_delta": no_hyper_adapter_delta,
        "no_evidence_router_delta": no_evidence_router_delta,
        "no_reliability_prior_delta": no_reliability_prior_delta,
        "memory_only_router_delta": memory_only_router_delta,
        "evidence_only_router_delta": evidence_only_router_delta,
    }
    if include_oracle_gap_evidence:
        row.update(
            {
                "TLEO_oracle_gap": 0.1,
                "SPO_oracle_gap": 0.1,
                "LRIO_oracle_gap": 0.1,
                "CATO_oracle_gap": 0.1,
            }
        )
        if oracle_gap_overrides:
            row.update(oracle_gap_overrides)
    if rceo:
        row["rceo_reliability_monotonic"] = rceo_reliability_monotonic
        row["rceo_router_load_shift"] = rceo_router_load_shift
        if include_rceo_reliability_curve:
            row["rceo_reliability_curve"] = rceo_reliability_curve or [
                {"corruption_strength": 0.0, "mean_reliability": 0.9},
                {"corruption_strength": 0.5, "mean_reliability": 0.7},
            ]
        if include_rceo_prior_effect:
            row["rceo_prior_effect"] = 0.1
    if no_lrio_delta is not None:
        row["no_lrio_delta"] = no_lrio_delta
    if no_rceo_delta is not None:
        row["no_rceo_delta"] = no_rceo_delta
    if include_operator_diagnostics:
        if active_operator == "SPO":
            row["prototype_kl_delta"] = 0.1
        elif active_operator == "LRIO":
            row["rank_logits_kl_delta"] = 0.1
        elif active_operator == "CATO":
            row["alignment_entropy_delta"] = 0.1
            row["alignment_topk_delta"] = 0.1
    return row


def _robustness_row(model: str, strength: float, score: float, *, reliability: float = 0.0) -> dict[str, object]:
    return {
        "model": model,
        "corruption_type": "image_blur",
        "corruption_strength": strength,
        "score": score,
        "rceo_reliability": reliability,
        "router_load_by_candidate": {"CATO": 0.60 - strength * 0.40, "SPO": 0.10 + strength * 0.30},
    }


def _controlled_evidence_artifacts() -> dict[str, object]:
    return {
        "task": "controlled_multimodal",
        "generated_by": "scripts/multimodal/summarize_controlled_report.py",
        "controlled_rows": {"path": "artifacts/controlled_rows.jsonl", "sha256": "e" * 64},
        "diagnostics_report": {"path": "artifacts/controlled_diagnostics.jsonl", "sha256": "f" * 64},
    }


def _stress_row(
    corruption_type: str,
    *,
    missing_modalities: list[str] | None = None,
    mismatch_source_id: str | None = None,
    temporal_shift_sec: float | None = None,
) -> dict[str, object]:
    row = {
        "model": "ovha_full",
        "corruption_type": corruption_type,
        "corruption_strength": 0.5,
        "score": 0.70,
        "rceo_reliability": 0.65,
        "router_load_by_candidate": {"CATO": 0.40, "SPO": 0.25},
    }
    if missing_modalities is not None:
        row["missing_modalities"] = missing_modalities
    if mismatch_source_id is not None:
        row["mismatch_source_id"] = mismatch_source_id
    if temporal_shift_sec is not None:
        row["temporal_shift_sec"] = temporal_shift_sec
    return row


if __name__ == "__main__":
    unittest.main()
