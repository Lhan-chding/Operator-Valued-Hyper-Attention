import json
import tempfile
import unittest
from pathlib import Path


class MultimodalCacheHardeningTests(unittest.TestCase):
    def test_cache_validator_rejects_missing_failed_sample_manifest(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(
                layout.root,
                train_ids=["train-source"],
                test_ids=["test-source"],
                mismatched_features=False,
                include_failed_manifests=False,
            )
            _write_complete_checksums(layout.root)

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn("missing required cache artifact: provenance/failed_samples_train.jsonl", joined)
        self.assertIn("missing required cache artifact: provenance/failed_samples_test.jsonl", joined)

    def test_cache_validator_rejects_missing_sample_provenance_manifest(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(
                layout.root,
                train_ids=["train-source"],
                test_ids=["test-source"],
                mismatched_features=False,
                include_sample_records=False,
            )
            _write_complete_checksums(layout.root)

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn("missing required cache artifact: provenance/sample_records_train.jsonl", joined)
        self.assertIn("missing required cache artifact: provenance/sample_records_test.jsonl", joined)

    def test_cache_validator_rejects_malformed_failed_sample_manifest_entry(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(
                layout.root,
                train_ids=["train-source"],
                test_ids=["test-source"],
                mismatched_features=False,
                invalid_failed_manifest=True,
            )
            _write_complete_checksums(layout.root)

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn(
            "failed_samples_train.jsonl line 1 missing required keys",
            "\n".join(report.errors),
        )

    def test_cache_validator_rejects_non_string_failed_sample_fields(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(
                layout.root,
                train_ids=["train-source"],
                test_ids=["test-source"],
                mismatched_features=False,
            )
            (layout.root / "provenance" / "failed_samples_train.jsonl").write_text(
                json.dumps({"source_id": 123, "split": "train", "reason": ["decode_failed"]}, sort_keys=True)
                + "\n"
            )
            _write_complete_checksums(layout.root)

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn(
            "failed_samples_train.jsonl line 1 required fields must be non-empty strings: source_id, reason",
            "\n".join(report.errors),
        )

    def test_cache_validator_rejects_failed_sample_that_is_retained(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(
                layout.root,
                train_ids=["train-source"],
                test_ids=["test-source"],
                mismatched_features=False,
                failed_sample_mode="overlaps_retained_source",
            )
            _write_complete_checksums(layout.root)

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn(
            "failed_samples_train.jsonl source_id must not also appear in retained split source ids: train-source",
            "\n".join(report.errors),
        )

    def test_cache_validator_rejects_malformed_sample_provenance_manifest_entry(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(
                layout.root,
                train_ids=["train-source"],
                test_ids=["test-source"],
                mismatched_features=False,
                sample_record_mode="missing_required_keys",
            )
            _write_complete_checksums(layout.root)

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn(
            "sample_records_train.jsonl line 1 missing required keys: raw_ref, license_tag",
            "\n".join(report.errors),
        )

    def test_cache_validator_rejects_non_string_sample_provenance_fields(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(
                layout.root,
                train_ids=["123"],
                test_ids=["test-source"],
                mismatched_features=False,
            )
            (layout.root / "provenance" / "sample_records_train.jsonl").write_text(
                json.dumps(
                    {"source_id": 123, "split": "train", "raw_ref": 456, "license_tag": ["bad"]},
                    sort_keys=True,
                )
                + "\n"
            )
            _write_complete_checksums(layout.root)

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn(
            "sample_records_train.jsonl line 1 required fields must be non-empty strings: "
            "source_id, raw_ref, license_tag",
            "\n".join(report.errors),
        )

    def test_cache_validator_rejects_sample_provenance_split_mismatch(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(
                layout.root,
                train_ids=["train-source"],
                test_ids=["test-source"],
                mismatched_features=False,
                sample_record_mode="split_mismatch",
            )
            _write_complete_checksums(layout.root)

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn("sample_records_train.jsonl line 1 split must match train", "\n".join(report.errors))

    def test_cache_validator_rejects_sample_provenance_source_id_mismatch(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(
                layout.root,
                train_ids=["train-source"],
                test_ids=["test-source"],
                mismatched_features=False,
                sample_record_mode="source_id_mismatch",
            )
            _write_complete_checksums(layout.root)

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn(
            "provenance/sample_records_train.jsonl source_id set must match provenance/source_ids_train.txt",
            "\n".join(report.errors),
        )

    def test_cache_validator_rejects_duplicate_sample_provenance_source_id(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(
                layout.root,
                train_ids=["train-source"],
                test_ids=["test-source"],
                mismatched_features=False,
                sample_record_mode="duplicate_source_id",
            )
            _write_complete_checksums(layout.root)

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn("duplicate source_id within sample_records_train.jsonl: train-source", "\n".join(report.errors))

    def test_cache_validator_rejects_missing_token_field_manifest(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(
                layout.root,
                train_ids=["train-source"],
                test_ids=["test-source"],
                mismatched_features=False,
                include_token_manifests=False,
            )
            _write_complete_checksums(layout.root)

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn("missing required cache artifact: token_fields/manifest_train.json", joined)
        self.assertIn("missing required cache artifact: token_fields/manifest_test.json", joined)

    def test_cache_validator_rejects_incomplete_token_field_manifest(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(
                layout.root,
                train_ids=["train-source"],
                test_ids=["test-source"],
                mismatched_features=False,
                invalid_token_manifest=True,
            )
            _write_complete_checksums(layout.root)

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn(
            "token_fields/manifest_train.json entry for region missing required keys: pos",
            "\n".join(report.errors),
        )

    def test_cache_validator_rejects_undeclared_token_field_manifest_modality(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(
                layout.root,
                train_ids=["train-source"],
                test_ids=["test-source"],
                mismatched_features=False,
                extra_token_manifest_modality="hidden_metadata",
            )
            _write_complete_checksums(layout.root)

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn(
            "token_fields/manifest_train.json contains undeclared modality: hidden_metadata",
            "\n".join(report.errors),
        )

    def test_cache_validator_rejects_incomplete_operator_supervision_data_card(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(
                layout.root,
                train_ids=["train-source"],
                test_ids=["test-source"],
                mismatched_features=False,
                missing_operator_supervision=("LRIO", "RCEO"),
            )
            _write_complete_checksums(layout.root)

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn("data_card.json operator_supervision missing required operator: LRIO", joined)
        self.assertIn("data_card.json operator_supervision missing required operator: RCEO", joined)

    def test_cache_validator_rejects_data_card_dataset_name_mismatch(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(
                layout.root,
                train_ids=["train-source"],
                test_ids=["test-source"],
                mismatched_features=False,
                data_card_overrides={"dataset_name": "wrong_dataset"},
            )
            _write_complete_checksums(layout.root)

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn(
            "data_card.json dataset_name must match cache layout: expected refcoco",
            "\n".join(report.errors),
        )

    def test_cache_validator_rejects_data_card_cache_version_mismatch(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(
                layout.root,
                train_ids=["train-source"],
                test_ids=["test-source"],
                mismatched_features=False,
                data_card_overrides={"cache_version": "v9"},
            )
            _write_complete_checksums(layout.root)

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn(
            "data_card.json cache_version must match cache layout: expected v0.1",
            "\n".join(report.errors),
        )

    def test_cache_validator_rejects_non_object_data_card(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(layout.root, train_ids=["train-source"], test_ids=["test-source"], mismatched_features=False)
            (layout.root / "data_card.json").write_text(json.dumps([]) + "\n")
            _write_complete_checksums(layout.root)

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn("data_card.json must be a JSON object", "\n".join(report.errors))

    def test_cache_validator_rejects_non_object_leakage_controls(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(
                layout.root,
                train_ids=["train-source"],
                test_ids=["test-source"],
                mismatched_features=False,
                data_card_overrides={"leakage_controls": []},
            )
            _write_complete_checksums(layout.root)

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn("data_card.json leakage_controls must be an object", "\n".join(report.errors))

    def test_cache_validator_rejects_empty_data_card_modalities(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(
                layout.root,
                train_ids=["train-source"],
                test_ids=["test-source"],
                mismatched_features=False,
                data_card_overrides={"modalities": []},
            )
            _write_complete_checksums(layout.root)

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn(
            "data_card.json modalities must be a non-empty list of strings",
            "\n".join(report.errors),
        )

    def test_cache_validator_rejects_duplicate_data_card_tasks(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(
                layout.root,
                train_ids=["train-source"],
                test_ids=["test-source"],
                mismatched_features=False,
                data_card_overrides={"tasks": ["phrase_region_grounding", "phrase_region_grounding"]},
            )
            _write_complete_checksums(layout.root)

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn(
            "data_card.json tasks must not contain duplicate entries",
            "\n".join(report.errors),
        )

    def test_cache_validator_rejects_source_ids_that_disagree_with_splits_manifest(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(layout.root, train_ids=["train-source"], test_ids=["test-source"], mismatched_features=False)
            (layout.root / "provenance" / "source_ids_train.txt").write_text("unlisted-train-source\n")
            _write_complete_checksums(layout.root)

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn(
            "provenance/source_ids_train.txt must match splits.json train entries",
            "\n".join(report.errors),
        )

    def test_cache_validator_rejects_duplicate_source_ids_in_splits_manifest(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(layout.root, train_ids=["train-source"], test_ids=["test-source"], mismatched_features=False)
            (layout.root / "splits.json").write_text(
                json.dumps({"train": ["train-source", "train-source"], "test": ["test-source"]}, sort_keys=True) + "\n"
            )
            _write_complete_checksums(layout.root)

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn("splits.json train contains duplicate source_id: train-source", "\n".join(report.errors))

    def test_cache_validator_rejects_non_string_source_ids_in_splits_manifest(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(layout.root, train_ids=["None"], test_ids=["test-source"], mismatched_features=False)
            (layout.root / "splits.json").write_text(
                json.dumps({"train": [None], "test": ["test-source"]}, sort_keys=True) + "\n"
            )
            _write_complete_checksums(layout.root)

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn("splits.json train source_id entries must be non-empty strings", "\n".join(report.errors))

    def test_cache_validator_rejects_checksum_mismatch(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(layout.root, train_ids=["train-source"], test_ids=["test-source"], mismatched_features=False)
            _write_complete_checksums(layout.root)
            checksums_path = layout.root / "checksums.json"
            checksums = json.loads(checksums_path.read_text())
            checksums["data_card.json"] = "0" * 64
            checksums_path.write_text(json.dumps(checksums, sort_keys=True) + "\n")

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn("checksums.json hash mismatch for artifact: data_card.json", "\n".join(report.errors))

    def test_cache_validator_rejects_empty_checksum_manifest(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(layout.root, train_ids=["train-source"], test_ids=["test-source"], mismatched_features=False)
            _write_complete_checksums(layout.root)
            (layout.root / "checksums.json").write_text("{}\n")

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn("checksums.json must be a non-empty object of artifact hashes", "\n".join(report.errors))

    def test_cache_validator_rejects_non_object_checksum_manifest(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(layout.root, train_ids=["train-source"], test_ids=["test-source"], mismatched_features=False)
            _write_complete_checksums(layout.root)
            (layout.root / "checksums.json").write_text("[]\n")

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn("checksums.json must be a non-empty object of artifact hashes", "\n".join(report.errors))

    def test_cache_validator_rejects_checksum_manifest_path_escape(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, file_sha256, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(layout.root, train_ids=["train-source"], test_ids=["test-source"], mismatched_features=False)
            escaped = Path(tmp) / "escaped.txt"
            escaped.write_text("outside cache\n")
            _write_complete_checksums(layout.root)
            checksums_path = layout.root / "checksums.json"
            checksums = json.loads(checksums_path.read_text())
            checksums["../escaped.txt"] = file_sha256(escaped)
            checksums_path.write_text(json.dumps(checksums, sort_keys=True) + "\n")

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn("checksums.json key must stay within cache root: ../escaped.txt", "\n".join(report.errors))

    def test_cache_validator_rejects_checksum_manifest_missing_artifact_entry(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(layout.root, train_ids=["train-source"], test_ids=["test-source"], mismatched_features=False)
            _write_complete_checksums(layout.root)
            checksums_path = layout.root / "checksums.json"
            checksums = json.loads(checksums_path.read_text())
            checksums["token_fields/missing_train.npy"] = "0" * 64
            checksums_path.write_text(json.dumps(checksums, sort_keys=True) + "\n")

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn(
            "checksums.json references missing artifact: token_fields/missing_train.npy",
            "\n".join(report.errors),
        )

    def test_cache_validator_rejects_checksum_manifest_directory_entry(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(layout.root, train_ids=["train-source"], test_ids=["test-source"], mismatched_features=False)
            _write_complete_checksums(layout.root)
            checksums_path = layout.root / "checksums.json"
            checksums = json.loads(checksums_path.read_text())
            checksums["token_fields"] = "0" * 64
            checksums_path.write_text(json.dumps(checksums, sort_keys=True) + "\n")

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn("checksums.json key must point to a file artifact: token_fields", "\n".join(report.errors))

    def test_cache_validator_rejects_same_feature_policy_without_ovha_reference(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(layout.root, train_ids=["train-source"], test_ids=["test-source"], mismatched_features=False)
            (layout.root / "provenance" / "feature_versions.json").write_text(
                json.dumps(
                    {
                        "text": "clip-text-v1",
                        "region": "clip-region-v1",
                        "baselines": {
                            "cross_attention_transformer": {"text": "clip-text-v1", "region": "clip-region-v1"}
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            _write_complete_checksums(layout.root)

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn("feature_versions.json baselines must include ovha_full reference", "\n".join(report.errors))

    def test_cache_validator_rejects_ovha_reference_missing_modality_feature_version(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(layout.root, train_ids=["train-source"], test_ids=["test-source"], mismatched_features=False)
            (layout.root / "provenance" / "feature_versions.json").write_text(
                json.dumps(
                    {
                        "text": "clip-text-v1",
                        "region": "clip-region-v1",
                        "baselines": {
                            "cross_attention_transformer": {"text": "clip-text-v1"},
                            "ovha_full": {"text": "clip-text-v1"},
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            _write_complete_checksums(layout.root)

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn(
            "feature_versions.json baselines.ovha_full missing feature extractor version for modality: region",
            "\n".join(report.errors),
        )

    def test_cache_validator_rejects_ovha_reference_that_disagrees_with_cache_modality_version(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(layout.root, train_ids=["train-source"], test_ids=["test-source"], mismatched_features=False)
            (layout.root / "provenance" / "feature_versions.json").write_text(
                json.dumps(
                    {
                        "text": "clip-text-v1",
                        "region": "clip-region-v2",
                        "baselines": {
                            "cross_attention_transformer": {"text": "clip-text-v1", "region": "clip-region-v1"},
                            "ovha_full": {"text": "clip-text-v1", "region": "clip-region-v1"},
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            _write_complete_checksums(layout.root)

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn(
            "feature_versions.json modality region must match baselines.ovha_full reference",
            "\n".join(report.errors),
        )

    def test_cache_validator_rejects_baseline_feature_mismatch_against_ovha_reference(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(layout.root, train_ids=["train-source"], test_ids=["test-source"], mismatched_features=False)
            (layout.root / "provenance" / "feature_versions.json").write_text(
                json.dumps(
                    {
                        "text": "clip-text-v1",
                        "region": "clip-region-v1",
                        "baselines": {
                            "cross_attention_transformer": {"text": "clip-text-v2", "region": "clip-region-v1"},
                            "ovha_full": {"text": "clip-text-v1", "region": "clip-region-v1"},
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            _write_complete_checksums(layout.root)

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn(
            "same_features_for_baselines is true but cross_attention_transformer differs from ovha_full",
            "\n".join(report.errors),
        )

    def test_cache_validator_detects_source_overlap_checksum_gap_and_feature_mismatch(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(layout.root, train_ids=["same-source"], test_ids=["same-source"], mismatched_features=True)

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn("source_id appears in multiple splits", joined)
        self.assertIn("checksums.json missing hash for required artifact", joined)
        self.assertIn("same_features_for_baselines is true but cross_attention_transformer differs from ovha_full", joined)

    def test_cache_validator_accepts_complete_minimal_cache(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(layout.root, train_ids=["train-source"], test_ids=["test-source"], mismatched_features=False)
            _write_complete_checksums(layout.root)

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertTrue(report.ok, report.errors)
        self.assertEqual(report.warnings, [])

    def test_cache_validator_rejects_pseudo_label_test_leakage(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(layout.root, train_ids=["train-source"], test_ids=["test-source"], mismatched_features=False)
            _write_complete_checksums(layout.root)
            (layout.root / "provenance" / "pseudo_label_versions.json").write_text(
                json.dumps({"generated_from_splits": ["train", "test"], "version": "bad"}) + "\n"
            )

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn("pseudo labels must not be generated from test split", "\n".join(report.errors))

    def test_cache_validator_rejects_non_object_pseudo_label_versions(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(layout.root, train_ids=["train-source"], test_ids=["test-source"], mismatched_features=False)
            _write_complete_checksums(layout.root)
            (layout.root / "provenance" / "pseudo_label_versions.json").write_text(json.dumps(["train"]) + "\n")

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn("pseudo_label_versions.json must be an object", "\n".join(report.errors))

    def test_cache_validator_rejects_string_pseudo_label_generated_from_splits(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(layout.root, train_ids=["train-source"], test_ids=["test-source"], mismatched_features=False)
            _write_complete_checksums(layout.root)
            (layout.root / "provenance" / "pseudo_label_versions.json").write_text(
                json.dumps({"generated_from_splits": "test", "version": "bad"}) + "\n"
            )

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn(
            "pseudo_label_versions.json generated_from_splits must be a list of split names",
            "\n".join(report.errors),
        )

    def test_cache_validator_rejects_pseudo_labels_without_version(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            layout = MultimodalCacheLayout(Path(tmp), "refcoco", "v0.1")
            _write_minimal_cache(layout.root, train_ids=["train-source"], test_ids=["test-source"], mismatched_features=False)
            _write_complete_checksums(layout.root)
            (layout.root / "provenance" / "pseudo_label_versions.json").write_text(
                json.dumps({"generated_from_splits": ["train"]}) + "\n"
            )

            report = validate_cache_layout(layout, splits=("train", "test"))

        self.assertFalse(report.ok)
        self.assertIn("pseudo_label_versions.json version must be a non-empty string", "\n".join(report.errors))


def _write_minimal_cache(
    root: Path,
    *,
    train_ids: list[str],
    test_ids: list[str],
    mismatched_features: bool,
    include_failed_manifests: bool = True,
    invalid_failed_manifest: bool = False,
    failed_sample_mode: str = "valid",
    include_sample_records: bool = True,
    sample_record_mode: str = "valid",
    include_token_manifests: bool = True,
    invalid_token_manifest: bool = False,
    missing_operator_supervision: tuple[str, ...] = (),
    data_card_overrides: dict[str, object] | None = None,
    extra_token_manifest_modality: str | None = None,
) -> None:
    for folder in ("provenance", "masks", "positions", "supervision", "token_fields"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    data_card = {
        "dataset_name": "refcoco",
        "cache_version": "v0.1",
        "modalities": ["text", "region"],
        "tasks": ["phrase_region_grounding"],
        "operator_supervision": {
            "TLEO": "local region/token structure",
            "SPO": "class/prototype/task label",
            "LRIO": "paired modality interaction",
            "CATO": "phrase-region or source-target alignment",
            "RCEO": "quality/corruption/missing metadata",
        },
        "leakage_controls": {
            "split_by_source_id": True,
            "deduplicate_by_source_id": True,
            "pseudo_labels_generated_without_test_labels": True,
            "same_features_for_baselines": True,
        },
    }
    for operator in missing_operator_supervision:
        data_card["operator_supervision"].pop(operator, None)
    if data_card_overrides:
        data_card = {**data_card, **data_card_overrides}
    (root / "data_card.json").write_text(json.dumps(data_card, sort_keys=True) + "\n")
    (root / "splits.json").write_text(json.dumps({"train": train_ids, "test": test_ids}, sort_keys=True) + "\n")
    (root / "samples.parquet").write_text("placeholder manifest\n")
    (root / "checksums.json").write_text(json.dumps({"data_card.json": "placeholder"}) + "\n")
    (root / "provenance" / "source_ids_train.txt").write_text("\n".join(train_ids) + "\n")
    (root / "provenance" / "source_ids_test.txt").write_text("\n".join(test_ids) + "\n")
    feature_versions = {
        "text": "clip-text-v1",
        "region": "clip-region-v1",
        "baselines": {
            "cross_attention_transformer": {"text": "clip-text-v2" if mismatched_features else "clip-text-v1", "region": "clip-region-v1"},
            "ovha_full": {"text": "clip-text-v1", "region": "clip-region-v1"},
        },
    }
    (root / "provenance" / "feature_versions.json").write_text(json.dumps(feature_versions, sort_keys=True) + "\n")
    (root / "provenance" / "pseudo_label_versions.json").write_text(json.dumps({"generated_from_splits": ["train"], "version": "ok"}) + "\n")
    for split in ("train", "test"):
        (root / "masks" / f"text_mask_{split}.npy").write_text("placeholder mask\n")
        (root / "supervision" / f"task_labels_{split}.npy").write_text("placeholder labels\n")
        if include_failed_manifests:
            payload = {}
            if invalid_failed_manifest and split == "train":
                payload = {"source_id": f"failed-{split}", "split": split}
            elif failed_sample_mode == "overlaps_retained_source" and split == "train":
                payload = {"source_id": train_ids[0], "split": split, "reason": "decode_failed"}
            line = json.dumps(payload, sort_keys=True) + "\n" if payload else ""
            (root / "provenance" / f"failed_samples_{split}.jsonl").write_text(line)
        if include_sample_records:
            source_ids = train_ids if split == "train" else test_ids
            _write_sample_records(root, split, source_ids, mode=sample_record_mode if split == "train" else "valid")
        if include_token_manifests:
            _write_token_field_manifest(
                root,
                split,
                invalid_token_manifest=invalid_token_manifest and split == "train",
                extra_modality=extra_token_manifest_modality if split == "train" else None,
            )


def _write_complete_checksums(root: Path) -> None:
    from moat_ovha_torch.data.multimodal.cache_schema import file_sha256, required_cache_files

    class _Layout:
        pass

    layout = _Layout()
    layout.root = root
    checksums = {}
    for path in required_cache_files(layout, splits=("train", "test")):
        if path.exists():
            checksums[str(path.relative_to(root))] = file_sha256(path)
    for path in sorted(root.rglob("*")):
        if path.is_file():
            checksums.setdefault(str(path.relative_to(root)), file_sha256(path))
    (root / "checksums.json").write_text(json.dumps(checksums, sort_keys=True) + "\n")


def _write_sample_records(root: Path, split: str, source_ids: list[str], *, mode: str = "valid") -> None:
    records = []
    for source_id in source_ids:
        record = {
            "source_id": "unlisted-source" if mode == "source_id_mismatch" else source_id,
            "split": "val" if mode == "split_mismatch" else split,
            "raw_ref": f"raw://{source_id}",
            "license_tag": "test-license",
        }
        if mode == "missing_required_keys":
            record.pop("raw_ref")
            record.pop("license_tag")
        records.append(record)
        if mode == "duplicate_source_id":
            records.append({**record})
    lines = [json.dumps(record, sort_keys=True) for record in records]
    (root / "provenance" / f"sample_records_{split}.jsonl").write_text("\n".join(lines) + "\n")


def _write_token_field_manifest(
    root: Path,
    split: str,
    *,
    invalid_token_manifest: bool,
    extra_modality: str | None = None,
) -> None:
    manifest = {}
    modalities = ("text", "region", extra_modality) if extra_modality else ("text", "region")
    for modality in modalities:
        assert modality is not None
        x_path = root / "token_fields" / f"{modality}_{split}.npy"
        pos_path = root / "positions" / f"{modality}_pos_{split}.npy"
        mask_path = root / "masks" / f"{modality}_mask_{split}.npy"
        x_path.write_text("placeholder token field\n")
        pos_path.write_text("placeholder positions\n")
        mask_path.write_text("placeholder mask\n")
        entry = {
            "x": str(x_path.relative_to(root)),
            "pos": str(pos_path.relative_to(root)),
            "mask": str(mask_path.relative_to(root)),
        }
        if invalid_token_manifest and modality == "region":
            entry.pop("pos")
        manifest[modality] = entry
    (root / "token_fields" / f"manifest_{split}.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")


if __name__ == "__main__":
    unittest.main()
