import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class MultimodalMainlineStaticContractTests(unittest.TestCase):
    def test_required_protocol_docs_and_pde_feasibility_note_exist(self):
        expected = [
            ROOT / "reports" / "pdebench_architecture_feasibility_note.md",
            ROOT / "docs" / "data_protocol_multimodal.md",
            ROOT / "moat_ovha_torch" / "data" / "multimodal" / "typed_batch.py",
            ROOT / "moat_ovha_torch" / "data" / "multimodal" / "cache_schema.py",
            ROOT / "moat_ovha_torch" / "data" / "multimodal" / "adapters" / "base.py",
            ROOT / "moat_ovha_torch" / "data" / "multimodal" / "adapters" / "controlled_synthetic.py",
            ROOT / "moat_ovha_torch" / "data" / "multimodal" / "adapters" / "refcoco.py",
            ROOT / "moat_ovha_torch" / "data" / "multimodal" / "adapters" / "flickr30k_entities.py",
            ROOT / "moat_ovha_torch" / "data" / "multimodal" / "adapters" / "visual_genome.py",
            ROOT / "moat_ovha_torch" / "data" / "multimodal" / "adapters" / "cmu_mosei.py",
            ROOT / "moat_ovha_torch" / "data" / "multimodal" / "adapters" / "meld.py",
            ROOT / "moat_ovha_torch" / "models" / "multimodal" / "ovha_multimodal.py",
            ROOT / "scripts" / "multimodal" / "validate_cache.py",
        ]
        for path in expected:
            with self.subTest(path=path):
                self.assertTrue(path.exists(), path)

    def test_step1_public_dataset_adapters_export_and_fail_fast_on_missing_raw(self):
        from moat_ovha_torch.data.multimodal.adapters import (
            CMUMOSEIAdapter,
            Flickr30kEntitiesAdapter,
            MELDAdapter,
            MissingMultimodalDataError,
            RefCOCOAdapter,
            VisualGenomeAdapter,
        )

        adapters = (
            RefCOCOAdapter(),
            Flickr30kEntitiesAdapter(),
            VisualGenomeAdapter(),
            CMUMOSEIAdapter(),
            MELDAdapter(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp)
            for adapter in adapters:
                with self.subTest(adapter=adapter.name):
                    with self.assertRaises(MissingMultimodalDataError):
                        adapter.discover_raw(raw_root)

    def test_build_cache_cli_registers_step1_dataset_adapters(self):
        module = _load_script_module(ROOT / "scripts" / "multimodal" / "build_cache.py")

        self.assertEqual(
            set(module.ADAPTERS),
            {
                "controlled_multimodal",
                "refcoco",
                "flickr30k_entities",
                "visual_genome",
                "cmu_mosei",
                "meld",
            },
        )
        self.assertEqual(module._modalities_for("visual_genome"), ["text", "region"])
        self.assertEqual(module._tasks_for("visual_genome"), ["phrase_region_grounding"])
        self.assertEqual(module._modalities_for("meld"), ["text", "audio", "vision"])
        self.assertEqual(module._tasks_for("meld"), ["sentiment_regression", "emotion_classification"])

    def test_public_dataset_adapters_expose_cache_required_supervision_shards(self):
        from moat_ovha_torch.data.multimodal.adapters import (
            CMUMOSEIAdapter,
            Flickr30kEntitiesAdapter,
            MELDAdapter,
            RefCOCOAdapter,
            VisualGenomeAdapter,
        )

        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp)
            for adapter in (RefCOCOAdapter(), Flickr30kEntitiesAdapter(), VisualGenomeAdapter()):
                with self.subTest(adapter=adapter.name):
                    supervision = adapter.extract_supervision({"cache_root": cache_root}, "train")
                    self.assertEqual(supervision.alignment_pairs_path, cache_root / "alignment_pairs_train.parquet")
                    self.assertEqual(supervision.bbox_targets_path, cache_root / "bbox_targets_train.npy")
                    self.assertEqual(supervision.region_targets_path, cache_root / "region_targets_train.npy")

            for adapter in (CMUMOSEIAdapter(), MELDAdapter()):
                with self.subTest(adapter=adapter.name):
                    supervision = adapter.extract_supervision({"cache_root": cache_root}, "train")
                    self.assertEqual(supervision.task_label_path, cache_root / "sentiment_train.npy")
                    self.assertEqual(
                        supervision.modality_missing_mask_path,
                        cache_root / "missing_modality_mask_train.npy",
                    )
                    self.assertEqual(supervision.corruption_metadata_path, cache_root / "corruption_train.parquet")

    def test_public_dataset_adapters_use_dataset_specific_raw_manifests(self):
        from moat_ovha_torch.data.multimodal.adapters import (
            Flickr30kEntitiesAdapter,
            MELDAdapter,
            MissingMultimodalDataError,
        )

        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp)

            with self.assertRaises(MissingMultimodalDataError) as flickr_error:
                Flickr30kEntitiesAdapter().discover_raw(raw_root)
            flickr_message = str(flickr_error.exception)
            self.assertIn("annotations/phrase_regions.json", flickr_message)
            self.assertIn("annotations/captions.json", flickr_message)
            self.assertNotIn("annotations/instances.json", flickr_message)
            self.assertNotIn("annotations/refs.json", flickr_message)

            with self.assertRaises(MissingMultimodalDataError) as meld_error:
                MELDAdapter().discover_raw(raw_root)
            meld_message = str(meld_error.exception)
            self.assertIn("labels/emotion.npy", meld_message)
            self.assertIn("metadata/dialogues.json", meld_message)
            self.assertNotIn("labels/sentiment.npy", meld_message)

    def test_pde_feasibility_note_uses_non_main_claim_framing(self):
        note = (ROOT / "reports" / "pdebench_architecture_feasibility_note.md").read_text()

        self.assertIn("architecture feasibility evidence", note)
        self.assertIn("not used as the main top-conference benchmark claim", note)
        self.assertIn("multimodal typed-token relation-operator tasks", note)

    def test_model_source_emits_step14_candidate_specific_diagnostics(self):
        model_source = (ROOT / "moat_ovha_torch" / "models" / "multimodal" / "ovha_multimodal.py").read_text()
        self.assertIn('"candidate_diagnostics": _candidate_diagnostics(', model_source)
        self.assertIn("candidate_outputs", model_source)
        self.assertIn("candidate_losses", model_source)

        primitive_requirements = {
            "typed_local_evidence.py": ("lengthscale", "local_entropy", "local_window_size"),
            "semantic_prototype.py": ("prototype_entropy", "top_prototype", "prototype_temperature"),
            "low_rank_interaction.py": ("rank_entropy", "rank_top_k", "pair_interaction_strength"),
            "alignment_transport.py": ("alignment_entropy", "top_k_alignment", "transport_marginal_error"),
        }
        primitive_root = ROOT / "moat_ovha_torch" / "models" / "multimodal" / "primitives"
        for filename, required_keys in primitive_requirements.items():
            source = (primitive_root / filename).read_text()
            for key in required_keys:
                with self.subTest(filename=filename, key=key):
                    self.assertIn(key, source)

        rceo_source = (ROOT / "moat_ovha_torch" / "models" / "multimodal" / "reliability_prior.py").read_text()
        for key in ("modality_reliability", "reliability_bias_norm", "corruption_response"):
            with self.subTest(module="RCEO", key=key):
                self.assertIn(key, rceo_source)

    def test_model_source_emits_step14_flat_adapter_param_diagnostics(self):
        model_source = (ROOT / "moat_ovha_torch" / "models" / "multimodal" / "ovha_multimodal.py").read_text()

        for key in ("TLEO_lengthscale", "SPO_temperature", "LRIO_rank_entropy", "CATO_alignment_temperature"):
            with self.subTest(key=key):
                self.assertIn(f'"{key}"', model_source)
        self.assertIn('"adapter_params_detail": _adapter_param_details(', model_source)
        self.assertNotIn("diagnostics[name] = {}", model_source)

    def test_cache_schema_requires_data_card_checksums_and_provenance(self):
        from moat_ovha_torch.data.multimodal.cache_schema import (
            MultimodalCacheLayout,
            required_cache_files,
            validate_cache_layout,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout = MultimodalCacheLayout(root, "refcoco", "v0.1")
            expected = required_cache_files(layout)

            self.assertIn(layout.root / "data_card.json", expected)
            self.assertIn(layout.root / "checksums.json", expected)
            self.assertIn(layout.root / "provenance" / "source_ids_train.txt", expected)
            self.assertIn(layout.root / "provenance" / "sample_records_train.jsonl", expected)
            self.assertIn(layout.root / "provenance" / "feature_versions.json", expected)

            report = validate_cache_layout(layout, splits=("train",))
            self.assertFalse(report.ok)
            self.assertIn("data_card.json", "\n".join(report.errors))

    def test_typed_batch_shape_contract_rejects_bad_rank_and_shared_dimension_mismatch(self):
        from moat_ovha_torch.data.multimodal.typed_batch import (
            MultimodalEpisodeBatch,
            ProvenanceBank,
            QueryField,
            SupervisionBank,
            TokenField,
            validate_multimodal_batch_contract,
        )

        batch = _static_batch(
            fields={
                "text": TokenField(
                    modality="text",
                    x=_Shape((2, 6, 4)),
                    pos=_Shape((2, 5, 2)),
                    mask=_Shape((2, 6)),
                    quality=_Shape((2, 6, 1)),
                )
            },
            query=QueryField(x=_Shape((2, 4)), pos=_Shape((2, 5, 2)), query_type=_Shape((2, 5)), mask=_Shape((2, 5))),
            target_y=_Shape((2, 5)),
            target_mask=_Shape((2, 4)),
        )

        report = validate_multimodal_batch_contract(batch)

        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn("query.x must have rank 3", joined)
        self.assertIn("target_y must have rank 3", joined)
        self.assertIn("target_mask shape must be [B,Q]", joined)
        self.assertIn("fields.text.pos first two dims must match fields.text.x", joined)

    def test_model_inputs_runs_shape_contract_before_returning_public_inputs(self):
        from moat_ovha_torch.data.multimodal.typed_batch import QueryField

        batch = _static_batch(
            query=QueryField(x=_Shape((2, 5, 4)), pos=_Shape((2, 5, 2)), query_type=_Shape((2, 4)), mask=_Shape((2, 5)))
        )

        with self.assertRaisesRegex(ValueError, "query.query_type shape must be"):
            batch.model_inputs()

    def test_typed_batch_rejects_malformed_provenance_values(self):
        from moat_ovha_torch.data.multimodal.typed_batch import ProvenanceBank, validate_multimodal_batch_contract

        batch = _static_batch(
            provenance=ProvenanceBank(
                source_id=["sample-0", 2],
                original_split=["train", ""],
                raw_ref=["shape", None],
                license_tag=["test", []],
                preprocessing_version="",
                feature_extractor_version={"text": 123, "region": ""},
                pseudo_label_version={1: "teacher-v1", "teacher": 456},
            )
        )

        report = validate_multimodal_batch_contract(batch)

        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn("provenance.source_id entries must be non-empty strings", joined)
        self.assertIn("provenance.original_split entries must be non-empty strings", joined)
        self.assertIn("provenance.raw_ref entries must be non-empty strings", joined)
        self.assertIn("provenance.license_tag entries must be non-empty strings", joined)
        self.assertIn("provenance.preprocessing_version must be a non-empty string", joined)
        self.assertIn("provenance.feature_extractor_version must map strings to non-empty strings", joined)
        self.assertIn("provenance.pseudo_label_version must map strings to non-empty strings", joined)

    def test_typed_batch_rejects_unmarked_weak_or_pseudo_supervision(self):
        from moat_ovha_torch.data.multimodal.typed_batch import SupervisionBank, validate_multimodal_batch_contract

        batch = _static_batch(
            supervision=SupervisionBank(
                task_label=None,
                alignment_pairs=None,
                alignment_weights=None,
                bbox_targets=None,
                region_targets=None,
                timestamp_targets=None,
                modality_missing_mask=None,
                corruption_metadata=None,
                weak_labels={"caption_sentiment": _Shape((2, 5, 1)), "alignment_hint": _Shape((2, 5, 1))},
                weak_label_confidence={"caption_sentiment": _Shape((2, 5, 1))},
                pseudo_label_source={"caption_sentiment": "", 2: "teacher-v1"},
            )
        )

        report = validate_multimodal_batch_contract(batch)

        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn("supervision.weak_label_confidence keys must match weak_labels keys", joined)
        self.assertIn("supervision.pseudo_label_source keys must match weak_labels keys", joined)
        self.assertIn("supervision.pseudo_label_source must map strings to non-empty strings", joined)

    def test_typed_batch_rejects_misaligned_weak_label_confidence_shapes(self):
        from moat_ovha_torch.data.multimodal.typed_batch import SupervisionBank, validate_multimodal_batch_contract

        batch = _static_batch(
            supervision=SupervisionBank(
                task_label=None,
                alignment_pairs=None,
                alignment_weights=None,
                bbox_targets=None,
                region_targets=None,
                timestamp_targets=None,
                modality_missing_mask=None,
                corruption_metadata=None,
                weak_labels={"caption_sentiment": _Shape((2, 5, 1)), "alignment_hint": _Shape((1, 5, 1))},
                weak_label_confidence={"caption_sentiment": _Shape((2, 4, 1)), "alignment_hint": _Shape((1, 5, 1))},
                pseudo_label_source={"caption_sentiment": "teacher-v1", "alignment_hint": "teacher-v1"},
            )
        )

        report = validate_multimodal_batch_contract(batch)

        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn("supervision.weak_labels.alignment_hint first two dims must match [B,Q]", joined)
        self.assertIn("supervision.weak_label_confidence.caption_sentiment shape must match weak label", joined)

    def test_typed_batch_rejects_misaligned_alignment_supervision(self):
        from moat_ovha_torch.data.multimodal.typed_batch import SupervisionBank, validate_multimodal_batch_contract

        batch = _static_batch(
            supervision=SupervisionBank(
                task_label=None,
                alignment_pairs=_Shape((1, 5, 3)),
                alignment_weights=_Shape((2, 4)),
                bbox_targets=None,
                region_targets=None,
                timestamp_targets=None,
                modality_missing_mask=None,
                corruption_metadata=None,
                weak_labels=None,
                weak_label_confidence=None,
                pseudo_label_source=None,
            )
        )

        report = validate_multimodal_batch_contract(batch)

        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn("supervision.alignment_pairs first two dims must match [B,Q]", joined)
        self.assertIn("supervision.alignment_pairs last dimension must be 2", joined)
        self.assertIn("supervision.alignment_weights shape must be [B,Q]", joined)

    def test_typed_batch_rejects_misaligned_missing_and_corruption_supervision(self):
        from moat_ovha_torch.data.multimodal.typed_batch import SupervisionBank, TokenField, validate_multimodal_batch_contract

        fields = {
            "text": TokenField(
                modality="text",
                x=_Shape((2, 6, 4)),
                pos=_Shape((2, 6, 2)),
                mask=_Shape((2, 6)),
            ),
            "region": TokenField(
                modality="region",
                x=_Shape((2, 8, 4)),
                pos=_Shape((2, 8, 2)),
                mask=_Shape((2, 8)),
            ),
        }
        batch = _static_batch(
            fields=fields,
            supervision=SupervisionBank(
                task_label=None,
                alignment_pairs=None,
                alignment_weights=None,
                bbox_targets=None,
                region_targets=None,
                timestamp_targets=None,
                modality_missing_mask=_Shape((2, 1)),
                corruption_metadata={"": _Shape((2,)), "corruption_strength": _Shape((1,)), "corruption_type": "gaussian"},
                weak_labels=None,
                weak_label_confidence=None,
                pseudo_label_source=None,
            ),
        )

        report = validate_multimodal_batch_contract(batch)

        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn("supervision.modality_missing_mask shape must be [B,M]", joined)
        self.assertIn("supervision.corruption_metadata keys must be non-empty strings", joined)
        self.assertIn("supervision.corruption_metadata.corruption_strength first dimension must match batch size 2", joined)
        self.assertIn("supervision.corruption_metadata.corruption_type must expose a tensor-like shape", joined)

    def test_typed_batch_rejects_hidden_token_field_metadata(self):
        from moat_ovha_torch.data.multimodal.typed_batch import TokenField, validate_multimodal_batch_contract

        batch = _static_batch(
            fields={
                "text": TokenField(
                    modality="text",
                    x=_Shape((2, 6, 4)),
                    pos=_Shape((2, 6, 2)),
                    mask=_Shape((2, 6)),
                    attrs={"true_active_operator": _Shape((2, 6, 1)), "": _Shape((2, 6, 1))},
                ),
                "corruption_strength": TokenField(
                    modality="corruption_strength",
                    x=_Shape((2, 6, 4)),
                    pos=_Shape((2, 6, 2)),
                    mask=_Shape((2, 6)),
                ),
            }
        )

        report = validate_multimodal_batch_contract(batch)

        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn(
            "fields.corruption_strength must not expose controlled or hidden metadata as model input",
            joined,
        )
        self.assertIn("fields.text.attrs keys must be non-empty strings", joined)
        self.assertIn(
            "fields.text.attrs must not expose controlled or hidden metadata as model input: "
            "true_active_operator",
            joined,
        )

    def test_typed_batch_rejects_invalid_episode_identity_and_split_provenance(self):
        from moat_ovha_torch.data.multimodal.typed_batch import ProvenanceBank, validate_multimodal_batch_contract

        batch = _static_batch(
            task_type="",
            split="val",
            source_dataset=123,
            provenance=ProvenanceBank(
                source_id=["sample-0", "sample-1"],
                original_split=["val", "test"],
                raw_ref=["shape", "shape"],
                license_tag=["test", "test"],
                preprocessing_version="test",
                feature_extractor_version={"text": "test"},
                pseudo_label_version={},
            ),
        )

        report = validate_multimodal_batch_contract(batch)

        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn("task_type must be a non-empty string", joined)
        self.assertIn("source_dataset must be a non-empty string", joined)
        self.assertIn("provenance.original_split entries must match batch split val", joined)


@unittest.skipUnless(TORCH_AVAILABLE, "Torch is not installed; multimodal tensor contract tests skipped.")
class MultimodalMainlineTorchContractTests(unittest.TestCase):
    def test_typed_batch_hides_controlled_truth_from_model_inputs(self):
        import torch

        from moat_ovha_torch.data.multimodal.typed_batch import (
            MultimodalEpisodeBatch,
            ProvenanceBank,
            QueryField,
            SupervisionBank,
            TokenField,
        )

        batch = _batch(torch)
        model_inputs = batch.model_inputs()

        self.assertIsInstance(batch.fields["text"], TokenField)
        self.assertIsInstance(batch.query, QueryField)
        self.assertIsInstance(batch.supervision, SupervisionBank)
        self.assertIsInstance(batch.provenance, ProvenanceBank)
        self.assertIn("fields", model_inputs)
        self.assertNotIn("hidden", model_inputs)
        self.assertNotIn("true_active_operator", str(model_inputs))

    def test_multimodal_ovha_stack_contract_and_diagnostics(self):
        import torch

        from moat_ovha_torch.models.multimodal.ovha_multimodal import (
            MULTIMODAL_CANDIDATE_NAMES,
            MultimodalOVHA,
        )

        model = MultimodalOVHA(
            field_dims={"text": 4, "region": 4},
            query_dim=4,
            output_dim=3,
            d_model=8,
        )
        output = model(_batch(torch))

        self.assertEqual(MULTIMODAL_CANDIDATE_NAMES, ("TLEO", "SPO", "LRIO", "CATO"))
        self.assertEqual(tuple(output.y_hat.shape), (2, 5, 3))
        self.assertEqual(tuple(output.candidate_values.shape), (2, 5, 4, 3))
        self.assertEqual(tuple(output.router_weights.shape), (2, 5, 4))
        self.assertEqual(set(output.candidate_outputs), set(MULTIMODAL_CANDIDATE_NAMES))
        self.assertNotIn("RCEO", output.candidate_outputs)
        self.assertTrue(torch.allclose(output.router_weights.sum(dim=-1), torch.ones(2, 5), atol=1e-6))
        self.assertTrue(
            torch.allclose(
                output.router_logits,
                output.router_logit_parts["memory"]
                + output.router_logit_parts["evidence"]
                + output.router_logit_parts["reliability"],
                atol=1e-6,
            )
        )
        self.assertIn("router_logit_parts", output.diagnostics)
        self.assertIn("candidate_loss", output.diagnostics)
        self.assertIn("stackability_passed", output.diagnostics)
        self.assertTrue(output.diagnostics["stackability_passed"])

    def test_stackability_guard_rejects_non_candidate_and_bad_shape(self):
        import torch

        from moat_ovha_torch.models.multimodal.operator_bank import assert_stackable
        from moat_ovha_torch.models.multimodal.primitives.base import CandidateOutput

        good = CandidateOutput(torch.zeros(2, 5, 3), torch.zeros(2, 5, 4), {})
        bad_shape = CandidateOutput(torch.zeros(2, 5, 1), torch.zeros(2, 5, 4), {})

        with self.assertRaisesRegex(ValueError, "Only TLEO / SPO / LRIO / CATO"):
            assert_stackable({"TLEO": good, "SPO": good, "LRIO": good, "RCEO": good}, 2, 5, 3)

        with self.assertRaisesRegex(ValueError, "expected"):
            assert_stackable({"TLEO": good, "SPO": good, "LRIO": good, "CATO": bad_shape}, 2, 5, 3)

    def test_controlled_synthetic_families_have_hidden_truth_but_no_input_leakage(self):
        import torch

        from moat_ovha_torch.data.multimodal.adapters.controlled_synthetic import (
            CONTROLLED_MULTIMODAL_FAMILIES,
            ControlledSyntheticMultimodalAdapter,
        )

        adapter = ControlledSyntheticMultimodalAdapter(seed=123, output_dim=3)
        for family in CONTROLLED_MULTIMODAL_FAMILIES:
            with self.subTest(family=family):
                batch = adapter.sample_batch(family=family, batch_size=2, query_count=4, device="cpu")
                self.assertIn("true_active_operator", batch.hidden)
                self.assertIn("true_router_weights", batch.hidden)
                self.assertIn("true_adapter_params", batch.hidden)
                self.assertEqual(tuple(batch.target_y.shape), (2, 4, 3))
                self.assertFalse(torch.equal(batch.query.query_type, batch.hidden["true_active_operator"]))
                self.assertNotIn("hidden", batch.model_inputs())
                self.assertNotIn("true_active_operator", str(batch.model_inputs()))
                self.assertTrue(torch.allclose(batch.hidden["true_router_weights"].sum(dim=-1), torch.ones(2, 4)))

    def test_oracle_matrix_true_true_reconstructs_controlled_targets(self):
        import torch

        from moat_ovha_torch.data.multimodal.adapters.controlled_synthetic import ControlledSyntheticMultimodalAdapter
        from moat_ovha_torch.eval.multimodal_oracle import evaluate_oracle_matrix

        batch = ControlledSyntheticMultimodalAdapter(seed=77, output_dim=2).sample_batch(
            family="mixed_relation_operator",
            batch_size=2,
            query_count=5,
            device="cpu",
        )
        report = evaluate_oracle_matrix(batch)

        self.assertLess(float(report["true_true"]["mse"]), 1e-8)
        self.assertIn("TLEO_oracle_gap", report)
        self.assertIn("CATO_oracle_gap", report)


def _batch(torch):
    fields = {
        "text": _field(torch, offset=0.0),
        "region": _field(torch, offset=1.0),
    }
    query = torch.linspace(0.0, 1.0, 5).view(1, 5, 1).repeat(2, 1, 4)
    target = torch.zeros(2, 5, 3)
    from moat_ovha_torch.data.multimodal.typed_batch import (
        MultimodalEpisodeBatch,
        ProvenanceBank,
        QueryField,
        SupervisionBank,
    )

    return MultimodalEpisodeBatch(
        fields=fields,
        query=QueryField(
            x=query,
            pos=query[..., :2],
            query_type=torch.zeros(2, 5, dtype=torch.long),
            mask=torch.ones(2, 5, dtype=torch.bool),
        ),
        target_y=target,
        target_mask=torch.ones(2, 5, dtype=torch.bool),
        task_type="phrase_region_grounding",
        split="train",
        source_dataset="controlled_multimodal",
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
            feature_extractor_version={"text": "frozen-test", "region": "frozen-test"},
            pseudo_label_version={},
        ),
        hidden={"true_active_operator": torch.zeros(2, 5, dtype=torch.long)},
    )


def _field(torch, offset):
    from moat_ovha_torch.data.multimodal.typed_batch import TokenField

    x = torch.arange(2 * 6 * 4, dtype=torch.float32).view(2, 6, 4) / 100.0 + offset
    pos = torch.linspace(0.0, 1.0, 6).view(1, 6, 1).repeat(2, 1, 2)
    return TokenField(
        modality="text" if offset == 0.0 else "region",
        x=x,
        pos=pos,
        mask=torch.ones(2, 6, dtype=torch.bool),
        quality=torch.ones(2, 6, 1),
        attrs=None,
    )


def _load_script_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Shape:
    def __init__(self, shape):
        self.shape = shape


def _static_batch(**overrides):
    from moat_ovha_torch.data.multimodal.typed_batch import (
        MultimodalEpisodeBatch,
        ProvenanceBank,
        QueryField,
        SupervisionBank,
        TokenField,
    )

    values = {
        "fields": {
            "text": TokenField(
                modality="text",
                x=_Shape((2, 6, 4)),
                pos=_Shape((2, 6, 2)),
                mask=_Shape((2, 6)),
                quality=_Shape((2, 6, 1)),
            )
        },
        "query": QueryField(x=_Shape((2, 5, 4)), pos=_Shape((2, 5, 2)), query_type=_Shape((2, 5)), mask=_Shape((2, 5))),
        "target_y": _Shape((2, 5, 3)),
        "target_mask": _Shape((2, 5)),
        "task_type": "phrase_region_grounding",
        "split": "train",
        "source_dataset": "shape-test",
        "supervision": SupervisionBank(
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
        "provenance": ProvenanceBank(
            source_id=["sample-0", "sample-1"],
            original_split=["train", "train"],
            raw_ref=["shape", "shape"],
            license_tag=["test", "test"],
            preprocessing_version="test",
            feature_extractor_version={"text": "test"},
            pseudo_label_version={},
        ),
        "hidden": None,
    }
    values.update(overrides)
    return MultimodalEpisodeBatch(**values)


if __name__ == "__main__":
    unittest.main()
