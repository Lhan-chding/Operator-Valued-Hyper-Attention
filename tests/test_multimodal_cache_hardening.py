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


def _write_minimal_cache(
    root: Path,
    *,
    train_ids: list[str],
    test_ids: list[str],
    mismatched_features: bool,
    include_failed_manifests: bool = True,
    invalid_failed_manifest: bool = False,
    include_token_manifests: bool = True,
    invalid_token_manifest: bool = False,
    missing_operator_supervision: tuple[str, ...] = (),
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
            payload = {"source_id": f"failed-{split}", "split": split} if invalid_failed_manifest and split == "train" else {}
            line = json.dumps(payload, sort_keys=True) + "\n" if payload else ""
            (root / "provenance" / f"failed_samples_{split}.jsonl").write_text(line)
        if include_token_manifests:
            _write_token_field_manifest(root, split, invalid_token_manifest=invalid_token_manifest and split == "train")


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


def _write_token_field_manifest(root: Path, split: str, *, invalid_token_manifest: bool) -> None:
    manifest = {}
    for modality in ("text", "region"):
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
