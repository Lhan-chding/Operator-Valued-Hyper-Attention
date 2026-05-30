import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MultimodalClaimsAndModulePolicyTests(unittest.TestCase):
    def test_pde_claim_validator_accepts_feasibility_framing_and_rejects_main_claims(self):
        from moat_ovha_torch.eval.multimodal_claims import validate_claim_text

        note = (ROOT / "reports" / "pdebench_architecture_feasibility_note.md").read_text()
        accepted = validate_claim_text(note, evidence_scope="pdebench")
        rejected = validate_claim_text(
            "PDEBench proves OVHA is the strongest neural operator and outperforms FNO, DeepONet, and ICON.",
            evidence_scope="pdebench",
        )

        self.assertTrue(accepted.ok, accepted.errors)
        self.assertFalse(rejected.ok)
        self.assertIn("PDEBench may not support main benchmark superiority claims", "\n".join(rejected.errors))

    def test_multimodal_claim_validator_requires_gate_evidence_for_main_claim(self):
        from moat_ovha_torch.eval.multimodal_claims import validate_claim_text

        claim = "We propose a typed-token operator-valued attention framework for multimodal reasoning."
        missing = validate_claim_text(claim, evidence_scope="multimodal_main", evidence={})
        supported = validate_claim_text(
            claim,
            evidence_scope="multimodal_main",
            evidence={
                "controlled_multimodal_passed": True,
                "region_text_public_passed": True,
                "sentiment_public_passed": True,
                "robustness_passed": True,
                "multi_seed_statistics_passed": True,
            },
        )

        self.assertFalse(missing.ok)
        self.assertIn("missing evidence for multimodal main claim", "\n".join(missing.errors))
        self.assertTrue(supported.ok, supported.errors)

    def test_module_policy_rejects_v1_forbidden_candidates_and_wrong_support_positions(self):
        from moat_ovha_torch.models.multimodal.module_policy import validate_module_policy

        report = validate_module_policy(
            {
                "version": "v1",
                "candidate_stack": ["TLEO", "SPO", "LRIO", "CATO", "RCEO"],
                "support_modules": {"RCEO": "candidate_stack", "MMRO": "candidate_stack", "CTRO": "candidate_stack", "OMRO": "candidate_stack"},
                "task_modalities": ["text", "region"],
            }
        )

        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn("v1 candidate_stack must be exactly", joined)
        self.assertIn("RCEO must be router_prior_adapter_condition_diagnostics", joined)
        self.assertIn("MMRO is v2+ only", joined)
        self.assertIn("CTRO is v2+ only", joined)
        self.assertIn("OMRO is memory infrastructure", joined)

    def test_module_policy_allows_v2_temporal_tldo_only_for_temporal_tasks(self):
        from moat_ovha_torch.models.multimodal.module_policy import validate_module_policy

        non_temporal = validate_module_policy(
            {
                "version": "v2",
                "candidate_stack": ["TLEO", "SPO", "LRIO", "CATO", "TLDO"],
                "support_modules": {"RCEO": "router_prior_adapter_condition_diagnostics", "MMRO": "upstream_conditioner", "CTRO": "second_stage_residual", "OMRO": "memory_infrastructure"},
                "task_modalities": ["text", "region"],
            }
        )
        temporal = validate_module_policy(
            {
                "version": "v2",
                "candidate_stack": ["TLEO", "SPO", "LRIO", "CATO", "TLDO"],
                "support_modules": {"RCEO": "router_prior_adapter_condition_diagnostics", "MMRO": "upstream_conditioner", "CTRO": "second_stage_residual", "OMRO": "memory_infrastructure"},
                "task_modalities": ["text", "audio", "video"],
            }
        )

        self.assertFalse(non_temporal.ok)
        self.assertIn("TLDO requires temporal modality", "\n".join(non_temporal.errors))
        self.assertTrue(temporal.ok, temporal.errors)

    def test_claims_cli_emits_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "claim.md"
            path.write_text("PDEBench proves OVHA beats all FNO baselines.")
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "multimodal" / "validate_claims.py"), str(path), "--scope", "pdebench"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 2)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["ok"])
        self.assertIn("PDEBench may not support", "\n".join(payload["errors"]))


if __name__ == "__main__":
    unittest.main()
