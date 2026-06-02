import importlib.util
import json
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
                    [["text", "audio"], ["text", "vision"], ["audio", "vision"]],
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


if __name__ == "__main__":
    unittest.main()
