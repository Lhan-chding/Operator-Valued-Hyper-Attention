import json
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

        report = build_controlled_report([_row("tleo_local_evidence", "TLEO", 0.2, 0.1)])

        self.assertFalse(report["go_no_go"]["controlled_multimodal_passed"])
        self.assertIn("missing controlled family", "\n".join(report["go_no_go"]["reasons"]))
        self.assertFalse(report["gate_table"]["TLEO collapse"]["passed"])

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
            diagnostics={
                "rank_logits_kl_delta": 0.1,
                "rceo_reliability_monotonic": True,
                "rceo_router_load_shift": 0.1,
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
            },
            {
                "model": "ovha_full",
                "corruption_type": "image_blur",
                "corruption_strength": 0.5,
                "score": 0.70,
                "rceo_reliability": 0.65,
                "router_load_by_candidate": {"CATO": 0.40, "SPO": 0.25, "TLEO": 0.25, "LRIO": 0.10},
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
        ]
        summary = summarize_robustness_rows(rows, full_model="ovha_full", baseline_model="cross_attention_transformer")

        self.assertLess(summary["relative_drop"]["ovha_full"], summary["relative_drop"]["cross_attention_transformer"])
        self.assertTrue(summary["rceo_reliability_monotonic"])
        self.assertLess(summary["operator_load_shift"]["CATO"], 0.0)
        self.assertIn("auc_over_corruption_strength", summary)

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
        self.assertIn("go_no_go", json.loads(controlled.stdout))
        self.assertIn("relative_drop", json.loads(robustness.stdout))


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
) -> dict[str, object]:
    row = {
        "family": family,
        "active_operator": active_operator,
        "oracle_matrix": {
            "learned_learned": {"loss": full_loss},
            "true_learned": {"loss": specialist_loss},
            "learned_true": {"loss": full_loss},
            "true_true": {"loss": 0.0},
        },
        "specialist_loss": specialist_loss,
        "router_accuracy": router_accuracy,
        "stackability_passed": True,
        "no_operator_memory_delta": 0.1,
        "no_hyper_adapter_delta": 0.1,
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
    if rceo:
        row["rceo_reliability_monotonic"] = True
        row["rceo_router_load_shift"] = 0.1
        if include_rceo_prior_effect:
            row["rceo_prior_effect"] = 0.1
    if include_operator_diagnostics:
        if active_operator == "SPO":
            row["prototype_kl_delta"] = 0.1
        elif active_operator == "LRIO":
            row["rank_logits_kl_delta"] = 0.1
        elif active_operator == "CATO":
            row["alignment_entropy_delta"] = 0.1
            row["alignment_topk_delta"] = 0.1
    return row


if __name__ == "__main__":
    unittest.main()
