import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
ROOT = Path(__file__).resolve().parents[1]


class MultimodalStrictAuditStaticContracts(unittest.TestCase):
    def test_cmu_public_configs_use_task_aware_spo_lrio_bank(self):
        for name in ("multimodal_cmu_mosei_public_main.json", "multimodal_cmu_mosei_public_smoke.json"):
            with self.subTest(config=name):
                payload = json.loads((ROOT / "configs" / name).read_text())

                self.assertEqual(payload["candidate_names"], ["SPO", "LRIO"])
                self.assertEqual(
                    payload["lrio_pairs"],
                    [["text", "audio"], ["text", "vision"]],
                )

    def test_multimodal_primitives_are_not_shared_linear_head_stubs(self):
        tleo = (ROOT / "moat_ovha_torch" / "models" / "multimodal" / "primitives" / "typed_local_evidence.py").read_text()
        lrio = (ROOT / "moat_ovha_torch" / "models" / "multimodal" / "primitives" / "low_rank_interaction.py").read_text()
        cato = (ROOT / "moat_ovha_torch" / "models" / "multimodal" / "primitives" / "alignment_transport.py").read_text()

        self.assertIn("self.query_proj = nn.ModuleDict", tleo)
        self.assertIn("self.key_proj = nn.ModuleDict", tleo)
        self.assertIn("self.value_proj = nn.ModuleDict", tleo)
        self.assertIn("self.source_gate = nn.ModuleDict", tleo)

        self.assertIn("self.branch = nn.ModuleDict", lrio)
        self.assertIn("self.trunk = nn.ModuleDict", lrio)
        self.assertIn("self.pair_gate = nn.ModuleDict", lrio)
        self.assertIn("active_pair_names", lrio)

        self.assertIn("self.query_proj = nn.ModuleDict", cato)
        self.assertIn("self.key_proj = nn.ModuleDict", cato)
        self.assertIn("self.value_proj = nn.ModuleDict", cato)
        self.assertIn("_source_transport_marginal_error", cato)


@unittest.skipUnless(TORCH_AVAILABLE, "Torch is not installed; strict multimodal audit tests skipped.")
class MultimodalStrictAuditContracts(unittest.TestCase):
    def test_untrained_controlled_router_does_not_use_task_type_hidden_truth(self):
        import torch

        from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA

        torch.manual_seed(7)
        batch = _batch(torch, task_type="cato_alignment_transport")
        model = MultimodalOVHA(
            field_dims={"text": 5, "audio": 4, "vision": 3},
            query_dim=6,
            output_dim=1,
            d_model=12,
            memory_tokens=2,
        )

        with torch.no_grad():
            output = model(batch)

        self.assertLess(
            float(output.router_weights.max().item()),
            0.70,
            "untrained controlled router must not become one-hot from task_type or hidden active-operator leakage",
        )
        self.assertEqual(
            float(output.evidence.diagnostics["controlled_family_relation_prior_rate"].item()),
            0.0,
        )

    def test_adapter_special_params_receive_task_loss_gradient(self):
        import torch

        from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA

        torch.manual_seed(11)
        batch = _batch(torch)
        model = MultimodalOVHA(
            field_dims={"text": 5, "audio": 4, "vision": 3},
            query_dim=6,
            output_dim=1,
            d_model=16,
            memory_tokens=2,
        )

        output = model(batch)
        loss = (output.y_hat - batch.target_y).square().mean()
        loss.backward()

        special_rows = {
            "TLEO": slice(2, 4),
            "SPO": slice(2, 7),
            "LRIO": slice(2, 7),
            "CATO": slice(2, 4),
        }
        for candidate, row_slice in special_rows.items():
            with self.subTest(candidate=candidate):
                grad = model.joint_router_adapter.hyper_adapter.heads[candidate].weight.grad
                self.assertIsNotNone(grad)
                self.assertGreater(
                    float(grad[row_slice].norm().item()),
                    1e-8,
                    f"{candidate} adapter special parameters must affect candidate output, not just diagnostics",
                )

    def test_rceo_is_pure_reliability_prior_not_evidence_logit_duplicate(self):
        import torch

        from moat_ovha_torch.models.multimodal.evidence import MultimodalEvidenceEncoder
        from moat_ovha_torch.models.multimodal.reliability_prior import RCEOReliabilityPrior

        torch.manual_seed(13)
        batch = _batch(torch)
        encoder = MultimodalEvidenceEncoder(
            field_dims={"text": 5, "audio": 4, "vision": 3},
            query_dim=6,
            d_model=10,
        )
        evidence = encoder(batch)
        perturbed = evidence.__class__(
            query_features=evidence.query_features,
            global_features=evidence.global_features,
            local_features=evidence.local_features,
            prototype_features=evidence.prototype_features,
            low_rank_features=evidence.low_rank_features,
            alignment_features=evidence.alignment_features,
            candidate_evidence_logits=evidence.candidate_evidence_logits + 100.0,
            local_entropy=evidence.local_entropy,
            alignment_entropy=evidence.alignment_entropy,
            field_features=evidence.field_features,
            diagnostics=evidence.diagnostics,
        )
        rceo = RCEOReliabilityPrior(d_model=10)

        bias = rceo(batch, evidence).operator_logit_bias
        perturbed_bias = rceo(batch, perturbed).operator_logit_bias

        self.assertTrue(
            torch.allclose(bias, perturbed_bias, atol=1e-6),
            "RCEO must not reintroduce candidate evidence logits when router evidence is disabled",
        )

    def test_lrio_preserves_ordered_pair_direction(self):
        from moat_ovha_torch.models.multimodal.primitives.low_rank_interaction import LRIOPrimitive

        primitive = LRIOPrimitive(d_model=8, output_dim=1, pairs=(("text", "audio"), ("audio", "text")))

        self.assertEqual(primitive.pairs, (("text", "audio"), ("audio", "text")))
        self.assertIn("text__audio", primitive.branch)
        self.assertIn("audio__text", primitive.branch)

    def test_rceo_exposes_pair_reliability_for_lrio_pairs(self):
        import torch

        from moat_ovha_torch.models.multimodal.evidence import MultimodalEvidenceEncoder
        from moat_ovha_torch.models.multimodal.reliability_prior import RCEOReliabilityPrior

        batch = _batch(torch)
        encoder = MultimodalEvidenceEncoder(
            field_dims={"text": 5, "audio": 4, "vision": 3},
            query_dim=6,
            d_model=10,
        )
        reliability = RCEOReliabilityPrior(
            d_model=10,
            candidate_names=("SPO", "LRIO"),
            lrio_pairs=(("text", "audio"), ("text", "vision")),
        )(batch, encoder(batch))

        self.assertEqual(reliability.pair_names, ("text__audio", "text__vision"))
        self.assertEqual(tuple(reliability.pair_reliability.shape), (3, 2))
        expected = torch.stack(
            [
                reliability.modality_reliability[:, 0] * reliability.modality_reliability[:, 1],
                reliability.modality_reliability[:, 0] * reliability.modality_reliability[:, 2],
            ],
            dim=-1,
        )
        self.assertTrue(torch.allclose(reliability.pair_reliability, expected))

    def test_lrio_diagnostics_include_pair_load_reliability_and_rank_entropy(self):
        import torch

        from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA

        torch.manual_seed(17)
        batch = _batch(torch)
        model = MultimodalOVHA(
            field_dims={"text": 5, "audio": 4, "vision": 3},
            query_dim=6,
            output_dim=1,
            d_model=12,
            memory_tokens=2,
            candidate_names=("SPO", "LRIO"),
            lrio_pairs=(("text", "audio"), ("text", "vision")),
        )

        with torch.no_grad():
            output = model(batch)

        lrio = output.diagnostics["candidate_diagnostics"]["LRIO"]
        self.assertEqual(set(lrio["pair_load"]), {"text__audio", "text__vision"})
        self.assertEqual(set(lrio["pair_reliability"]), {"text__audio", "text__vision"})
        self.assertEqual(set(lrio["pair_rank_entropy"]), {"text__audio", "text__vision"})
        self.assertGreaterEqual(lrio["pair_reliability"]["text__audio"], 0.0)
        self.assertLessEqual(lrio["pair_reliability"]["text__audio"], 1.0)

    def test_sentiment_public_batch_uses_neutral_query_not_text_mean(self):
        import torch

        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout
        from scripts.multimodal.run_public_smoke import _load_public_batch

        with tempfile.TemporaryDirectory() as tmp:
            layout = _write_public_cache_fixture(Path(tmp), task_type="sentiment_regression")
            config = MultimodalExperimentConfig.from_file(ROOT / "configs" / "multimodal_cmu_mosei_public_smoke.json")

            batch = _load_public_batch(layout, config, "train", torch.device("cpu"))

        text_mean = batch.fields["text"].x.mean(dim=1, keepdim=True).expand_as(batch.query.x)
        self.assertTrue(torch.all(batch.query.x == 0.0))
        self.assertFalse(torch.allclose(batch.query.x, text_mean))

    def test_public_main_sentiment_metrics_use_model_diagnostics_not_router_proxy(self):
        import torch

        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
        from scripts.multimodal.run_public_main import _public_main_metrics

        config = MultimodalExperimentConfig.from_file(ROOT / "configs" / "multimodal_cmu_mosei_public_main.json")
        batch = _batch(torch)
        prediction = batch.target_y.clone()
        diagnostics = {
            "candidate_diagnostics": {
                "LRIO": {"rank_entropy": torch.tensor(0.42)},
                "SPO": {"prototype_entropy": torch.tensor(0.73)},
            },
            "reliability": {"modality_reliability_mean": torch.tensor(0.66)},
        }

        metrics = _public_main_metrics(
            config,
            batch,
            prediction=prediction,
            router_load_by_candidate={"SPO": 0.95, "LRIO": 0.05},
            router_entropy=torch.tensor(0.0),
            candidate_loss={},
            diagnostics=diagnostics,
        )

        self.assertAlmostEqual(metrics["lrio_rank_entropy"], 0.42, places=5)
        self.assertAlmostEqual(metrics["spo_prototype_entropy"], 0.73, places=5)
        self.assertAlmostEqual(metrics["rceo_reliability_calibration"]["mean_predicted_reliability"], 0.66, places=5)
        self.assertEqual(metrics["rceo_reliability_calibration"]["source"], "model_reliability_prior")

    def test_robustness_row_uses_model_reliability_not_corruption_strength_proxy(self):
        import torch

        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
        from scripts.multimodal.run_public_main import _robustness_row

        config = MultimodalExperimentConfig.from_file(ROOT / "configs" / "multimodal_cmu_mosei_public_main.json")
        row = _robustness_row(
            config,
            _batch(torch),
            model_name="ovha_full",
            seed=201,
            raw_metric_path=Path("raw_metrics.jsonl"),
            corruption_type="audio_noise",
            score=0.8,
            router_load_by_candidate={"SPO": 0.5, "LRIO": 0.5},
            candidate_loss={},
            model_reliability=0.33,
        )

        self.assertAlmostEqual(row["rceo_reliability"], 0.33)
        self.assertNotEqual(row["rceo_reliability"], 1.0 - row["corruption_strength"])

    def test_diagnostic_validator_respects_active_candidate_subset(self):
        from moat_ovha_torch.eval.multimodal_diagnostics import validate_diagnostic_row

        row = {
            "active_candidate_names": ["SPO", "LRIO"],
            "router_entropy": 0.5,
            "router_load_by_candidate": {"SPO": 0.6, "LRIO": 0.4},
            "router_memory_logit_norm": 0.1,
            "router_evidence_logit_norm": 0.2,
            "router_reliability_logit_norm": 0.3,
            "router_logit_parts": {"memory": 0.0, "evidence": 0.0, "reliability": 0.0},
            "candidate_loss": {"SPO": 0.2, "LRIO": 0.3},
            "adapter_params": {"SPO_temperature": 1.0, "LRIO_rank_entropy": 0.5},
            "memory_slot_norm": {"SPO": 1.1, "LRIO": 1.2},
            "candidate_diagnostics": {
                "SPO": {
                    "prototype_entropy": 0.5,
                    "top_prototype": 1.0,
                    "prototype_temperature": 1.0,
                    "candidate_loss": 0.2,
                },
                "LRIO": {
                    "rank_entropy": 0.6,
                    "rank_top_k": 1.0,
                    "pair_interaction_strength": 0.4,
                    "candidate_loss": 0.3,
                },
                "RCEO": {
                    "modality_reliability": 0.8,
                    "reliability_bias_norm": 0.1,
                    "corruption_response": 0.2,
                },
            },
            "stackability_passed": True,
        }

        report = validate_diagnostic_row(row)

        self.assertTrue(report.ok, report.errors)


def _batch(torch, *, task_type: str = "sentiment_regression"):
    from moat_ovha_torch.data.multimodal.typed_batch import (
        MultimodalEpisodeBatch,
        ProvenanceBank,
        QueryField,
        SupervisionBank,
        TokenField,
    )

    batch_size = 3
    query_count = 2
    fields = {
        "text": TokenField(
            modality="text",
            x=torch.randn(batch_size, 4, 5),
            pos=torch.linspace(0.0, 1.0, 4).view(1, 4, 1).repeat(batch_size, 1, 1),
            mask=torch.ones(batch_size, 4, dtype=torch.bool),
            quality=torch.tensor([[1.0], [0.8], [0.6]]),
        ),
        "audio": TokenField(
            modality="audio",
            x=torch.randn(batch_size, 3, 4),
            pos=torch.linspace(0.0, 1.0, 3).view(1, 3, 1).repeat(batch_size, 1, 1),
            mask=torch.ones(batch_size, 3, dtype=torch.bool),
            quality=torch.tensor([[0.9], [0.7], [0.5]]),
        ),
        "vision": TokenField(
            modality="vision",
            x=torch.randn(batch_size, 5, 3),
            pos=torch.linspace(0.0, 1.0, 5).view(1, 5, 1).repeat(batch_size, 1, 1),
            mask=torch.ones(batch_size, 5, dtype=torch.bool),
            quality=torch.tensor([[0.7], [0.9], [0.4]]),
        ),
    }
    return MultimodalEpisodeBatch(
        fields=fields,
        query=QueryField(
            x=torch.randn(batch_size, query_count, 6),
            pos=torch.linspace(0.2, 0.8, query_count).view(1, query_count, 1).repeat(batch_size, 1, 1),
            query_type=torch.zeros(batch_size, query_count, 1),
            mask=torch.ones(batch_size, query_count, dtype=torch.bool),
        ),
        target_y=torch.randn(batch_size, query_count, 1),
        target_mask=torch.ones(batch_size, query_count, dtype=torch.bool),
        task_type=task_type,
        split="train",
        source_dataset="unit_test",
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
            source_id=[f"sample_{idx}" for idx in range(batch_size)],
            original_split=["train"] * batch_size,
            raw_ref=[f"raw-{idx}" for idx in range(batch_size)],
            license_tag=["unit-test"] * batch_size,
            preprocessing_version="unit-test",
            feature_extractor_version={"text": "unit", "audio": "unit", "vision": "unit"},
            pseudo_label_version={},
        ),
        hidden={"true_active_operator": torch.zeros(batch_size, query_count, dtype=torch.long)},
    )


def _write_public_cache_fixture(root: Path, *, task_type: str):
    import numpy as np

    from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout

    layout = MultimodalCacheLayout(root, "cmu_mosei", "unit")
    cache_root = layout.root
    for directory in ("token_fields", "positions", "masks", "supervision", "provenance"):
        (cache_root / directory).mkdir(parents=True, exist_ok=True)

    batch_size = 3
    specs = {
        "text": (4, 6),
        "audio": (3, 5),
        "vision": (2, 4),
    }
    manifest: dict[str, dict[str, str]] = {}
    for offset, (modality, (token_count, dim)) in enumerate(specs.items(), start=1):
        x = np.arange(batch_size * token_count * dim, dtype=np.float32).reshape(batch_size, token_count, dim) + offset
        pos = np.linspace(0.0, 1.0, token_count, dtype=np.float32).reshape(1, token_count, 1).repeat(batch_size, axis=0)
        mask = np.ones((batch_size, token_count), dtype=bool)
        x_path = f"token_fields/{modality}_x_train.npy"
        pos_path = f"positions/{modality}_pos_train.npy"
        mask_path = f"masks/{modality}_mask_train.npy"
        np.save(cache_root / x_path, x)
        np.save(cache_root / pos_path, pos)
        np.save(cache_root / mask_path, mask)
        manifest[modality] = {"x": x_path, "pos": pos_path, "mask": mask_path}

    np.save(cache_root / "supervision" / "task_labels_train.npy", np.array([[-1.0], [0.0], [1.0]], dtype=np.float32))
    np.save(cache_root / "supervision" / "missing_modality_mask_train.npy", np.zeros((batch_size, 3), dtype=np.float32))
    (cache_root / "token_fields" / "manifest_train.json").write_text(json.dumps(manifest, sort_keys=True))
    (cache_root / "data_card.json").write_text(
        json.dumps(
            {
                "dataset_name": "cmu_mosei",
                "cache_version": "unit",
                "modalities": ["text", "audio", "vision"],
                "tasks": [task_type],
                "operator_supervision": {
                    "TLEO": "not used",
                    "SPO": "semantic prototype",
                    "LRIO": "paired modality interaction",
                    "CATO": "not used",
                    "RCEO": "reliability",
                },
                "leakage_controls": {"hidden_metadata_excluded": True},
            },
            sort_keys=True,
        )
    )
    records = [
        {
            "source_id": f"sample_{index}",
            "split": "train",
            "original_split": "train",
            "raw_ref": f"raw_{index}",
            "license_tag": "unit-test",
            "preprocessing_version": "unit-test",
        }
        for index in range(batch_size)
    ]
    (cache_root / "provenance" / "sample_records_train.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n"
    )
    (cache_root / "provenance" / "feature_versions.json").write_text(
        json.dumps({"text": "unit", "audio": "unit", "vision": "unit"}, sort_keys=True)
    )
    (cache_root / "provenance" / "pseudo_label_versions.json").write_text(json.dumps({"version": "none"}))
    return layout


if __name__ == "__main__":
    unittest.main()
