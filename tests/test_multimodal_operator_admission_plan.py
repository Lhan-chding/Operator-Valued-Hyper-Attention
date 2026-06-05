import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class MultimodalOperatorAdmissionPlanTests(unittest.TestCase):
    @unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
    def test_tanso_base_and_shift_are_distinct_candidates_with_null_source_diagnostics(self):
        import torch

        from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA

        batch = _sentiment_batch(torch)
        model = MultimodalOVHA(
            field_dims={name: int(field.x.shape[-1]) for name, field in batch.fields.items()},
            query_dim=int(batch.query.x.shape[-1]),
            output_dim=int(batch.target_y.shape[-1]),
            d_model=8,
            memory_tokens=1,
            candidate_names=("TANSOBase", "TANSOShift"),
            use_evidence_router=False,
            composition_mode="base_plus_residual",
            base_candidate="TANSOBase",
            residual_candidates=("TANSOShift",),
            lrio_pairs=(("text", "audio"), ("text", "vision")),
        )

        output = model(batch)
        base_diag = output.diagnostics["candidate_diagnostics"]["TANSOBase"]
        shift_diag = output.diagnostics["candidate_diagnostics"]["TANSOShift"]

        self.assertEqual(base_diag["semantic_role"], "full_predictor")
        self.assertEqual(shift_diag["semantic_role"], "residual_delta")
        for diagnostics in (base_diag, shift_diag):
            self.assertIn("null_source_load", diagnostics)
            total_load = (
                diagnostics["audio_source_load"]
                + diagnostics["vision_source_load"]
                + diagnostics["null_source_load"]
            )
            self.assertTrue(torch.allclose(total_load, torch.ones_like(total_load), atol=1e-5))
            self.assertIn("null", diagnostics["source_gate_tensor"])
        self.assertEqual(output.diagnostics["composition"]["base_candidate"], "TANSOBase")
        self.assertEqual(output.diagnostics["composition"]["residual_candidates"], ("TANSOShift",))
        self.assertEqual(tuple(output.y_hat.shape), tuple(batch.target_y.shape))

    def test_cmu_tanso_primary_config_encodes_five_seed_official_protocol(self):
        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig

        config = MultimodalExperimentConfig.from_file(ROOT / "configs" / "multimodal_cmu_mosei_tanso_primary_main.json")

        self.assertEqual(config.main_model_name, "ovha_tanso_primary")
        self.assertEqual(config.candidate_names, ("TANSOBase",))
        self.assertEqual(config.composition_mode, "tanso_base")
        self.assertEqual(config.seeds, (301, 302, 303, 304, 305))
        self.assertFalse(config.allow_hidden_losses)
        self.assertEqual(config.loss_metadata["candidate_individual_loss"]["weight"], 0.0)
        self.assertEqual(config.loss_metadata["val_affine_calibration"]["stage"], "validation_postfit")

    def test_operator_admission_eval_reports_residual_utility_and_rejects_interference(self):
        rows = [
            {"sample_id": "a", "split": "val", "model": "base", "truth": 1.0, "prediction": 0.0},
            {"sample_id": "b", "split": "val", "model": "base", "truth": 0.0, "prediction": 1.0},
            {"sample_id": "a", "split": "val", "model": "candidate", "prediction": 0.8},
            {"sample_id": "b", "split": "val", "model": "candidate", "prediction": 0.8},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            predictions = tmp_path / "predictions.jsonl"
            output = tmp_path / "admission.json"
            predictions.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "operator_admission_eval.py"),
                    "--predictions-jsonl",
                    str(predictions),
                    "--base-model",
                    "base",
                    "--candidate-model",
                    "candidate",
                    "--output",
                    str(output),
                    "--bootstrap-samples",
                    "32",
                    "--seed",
                    "7",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(output.read_text())
        self.assertEqual(payload["admission_decision"], "diagnostic_only")
        self.assertGreater(payload["residual_alignment"], 0.0)
        self.assertGreater(payload["non_interference_delta"], 0.0)
        self.assertIn("paired_bootstrap_p", payload)


def _sentiment_batch(torch):
    from moat_ovha_torch.data.multimodal.typed_batch import (
        MultimodalEpisodeBatch,
        ProvenanceBank,
        QueryField,
        SupervisionBank,
        TokenField,
    )

    batch_size = 2
    query_count = 3
    token_count = 4
    dim = 5
    fields = {}
    for offset, name in enumerate(("text", "audio", "vision")):
        values = torch.arange(batch_size * token_count * dim, dtype=torch.float32).view(batch_size, token_count, dim)
        fields[name] = TokenField(
            modality=name,
            x=(values / 50.0) + float(offset),
            pos=torch.linspace(0.0, 1.0, token_count).view(1, token_count, 1).repeat(batch_size, 1, 2),
            mask=torch.ones(batch_size, token_count, dtype=torch.bool),
            quality=torch.ones(batch_size, token_count, 1),
            attrs=None,
        )
    query_x = torch.ones(batch_size, query_count, dim) * 0.25
    return MultimodalEpisodeBatch(
        fields=fields,
        query=QueryField(
            x=query_x,
            pos=torch.linspace(0.0, 1.0, query_count).view(1, query_count, 1).repeat(batch_size, 1, 2),
            query_type=torch.zeros(batch_size, query_count, dtype=torch.long),
            mask=torch.ones(batch_size, query_count, dtype=torch.bool),
        ),
        target_y=torch.zeros(batch_size, query_count, 1),
        target_mask=torch.ones(batch_size, query_count, dtype=torch.bool),
        task_type="sentiment_emotion",
        split="train",
        source_dataset="cmu_mosei",
        supervision=SupervisionBank(
            task_label=None,
            alignment_pairs=None,
            alignment_weights=None,
            bbox_targets=None,
            region_targets=None,
            timestamp_targets=None,
            modality_missing_mask=None,
            corruption_metadata=None,
            weak_labels=None,
            weak_label_confidence=None,
            pseudo_label_source=None,
        ),
        provenance=ProvenanceBank(
            source_id=["sample-0", "sample-1"],
            original_split=["train", "train"],
            raw_ref=["synthetic", "synthetic"],
            license_tag=["synthetic", "synthetic"],
            preprocessing_version="test",
            feature_extractor_version={"text": "test", "audio": "test", "vision": "test"},
            pseudo_label_version={},
        ),
        hidden=None,
    )
