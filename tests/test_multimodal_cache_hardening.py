import json
import tempfile
import unittest
from pathlib import Path


class MultimodalCacheHardeningTests(unittest.TestCase):
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
        self.assertIn("same_features_for_baselines is true but baseline feature versions differ", joined)

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


def _write_minimal_cache(root: Path, *, train_ids: list[str], test_ids: list[str], mismatched_features: bool) -> None:
    for folder in ("provenance", "masks", "supervision"):
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


def _write_complete_checksums(root: Path) -> None:
    from moat_ovha_torch.data.multimodal.cache_schema import required_cache_files

    class _Layout:
        pass

    layout = _Layout()
    layout.root = root
    checksums = {}
    for path in required_cache_files(layout, splits=("train", "test")):
        checksums[str(path.relative_to(root))] = "placeholder"
    (root / "checksums.json").write_text(json.dumps(checksums, sort_keys=True) + "\n")


if __name__ == "__main__":
    unittest.main()
