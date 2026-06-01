import importlib.util
import json
import subprocess
import sys
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
            ROOT / "configs" / "multimodal_refcoco_public_main.json",
            ROOT / "configs" / "multimodal_cmu_mosei_public_main.json",
            ROOT / "moat_ovha_torch" / "data" / "multimodal" / "typed_batch.py",
            ROOT / "moat_ovha_torch" / "data" / "multimodal" / "cache_schema.py",
            ROOT / "moat_ovha_torch" / "data" / "multimodal" / "adapters" / "base.py",
            ROOT / "moat_ovha_torch" / "data" / "multimodal" / "adapters" / "controlled_synthetic.py",
            ROOT / "moat_ovha_torch" / "data" / "multimodal" / "adapters" / "refcoco.py",
            ROOT / "moat_ovha_torch" / "data" / "multimodal" / "adapters" / "flickr30k_entities.py",
            ROOT / "moat_ovha_torch" / "data" / "multimodal" / "adapters" / "visual_genome.py",
            ROOT / "moat_ovha_torch" / "data" / "multimodal" / "adapters" / "cmu_mosei.py",
            ROOT / "moat_ovha_torch" / "data" / "multimodal" / "adapters" / "cmu_mosi.py",
            ROOT / "moat_ovha_torch" / "data" / "multimodal" / "adapters" / "meld.py",
            ROOT / "moat_ovha_torch" / "data" / "multimodal" / "adapters" / "iemocap.py",
            ROOT / "moat_ovha_torch" / "models" / "multimodal" / "ovha_multimodal.py",
            ROOT / "scripts" / "multimodal" / "validate_cache.py",
            ROOT / "scripts" / "multimodal" / "build_refcoco_stage_records.py",
            ROOT / "scripts" / "multimodal" / "align_refcoco_stage_features.py",
            ROOT / "scripts" / "multimodal" / "extract_refcoco_clip_features.py",
            ROOT / "scripts" / "multimodal" / "extract_cmu_sdk_stage_inputs.py",
            ROOT / "scripts" / "multimodal" / "extract_meld_ffmpeg_features.py",
            ROOT / "scripts" / "multimodal" / "extract_meld_transformer_features.py",
            ROOT / "scripts" / "multimodal" / "inspect_cmu_sdk_sequences.py",
            ROOT / "scripts" / "multimodal" / "write_cmu_sdk_splits.py",
            ROOT / "scripts" / "multimodal" / "check_public_data_readiness.py",
            ROOT / "scripts" / "multimodal" / "build_public_main_runbook.py",
            ROOT / "scripts" / "multimodal" / "run_public_main.py",
            ROOT / "scripts" / "multimodal" / "validate_public_main_artifacts.py",
        ]
        for path in expected:
            with self.subTest(path=path):
                self.assertTrue(path.exists(), path)

    def test_step1_public_dataset_adapters_export_and_fail_fast_on_missing_raw(self):
        from moat_ovha_torch.data.multimodal.adapters import (
            CMUMOSEIAdapter,
            CMUMOSIAdapter,
            IEMOCAPAdapter,
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
            CMUMOSIAdapter(),
            IEMOCAPAdapter(),
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
                "cmu_mosi",
                "meld",
                "iemocap",
            },
        )
        self.assertEqual(module._modalities_for("visual_genome"), ["text", "region"])
        self.assertEqual(module._tasks_for("visual_genome"), ["phrase_region_grounding"])
        self.assertEqual(module._modalities_for("cmu_mosi"), ["text", "audio", "vision"])
        self.assertEqual(module._tasks_for("cmu_mosi"), ["sentiment_regression", "emotion_classification"])
        self.assertEqual(module._modalities_for("meld"), ["text", "audio", "vision"])
        self.assertEqual(module._tasks_for("iemocap"), ["sentiment_regression", "emotion_classification"])
        self.assertEqual(module._tasks_for("meld"), ["sentiment_regression", "emotion_classification"])

    def test_build_cache_cli_reports_missing_raw_as_json_fail_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp) / "missing_raw"
            cache_root = Path(tmp) / "cache"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "build_cache.py"),
                    "flickr30k_entities",
                    str(raw_root),
                    str(cache_root),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 2)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["dataset_name"], "flickr30k_entities")
        self.assertIn("fail-fast", payload["policy"])
        self.assertIn("annotations/phrase_regions.json", "\n".join(payload["errors"]))
        self.assertNotIn("Traceback", result.stderr)

    def test_build_cache_cli_builds_valid_controlled_synthetic_cache_with_hidden_truth(self):
        import numpy as np

        from moat_ovha_torch.data.multimodal.adapters.controlled_synthetic import CONTROLLED_MULTIMODAL_FAMILIES
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_root = tmp_path / "raw"
            cache_root = tmp_path / "cache"
            raw_root.mkdir(parents=True, exist_ok=True)

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "build_cache.py"),
                    "controlled_multimodal",
                    str(raw_root),
                    str(cache_root),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            layout = MultimodalCacheLayout(cache_root, "controlled_multimodal", "v0.1")
            validation = validate_cache_layout(layout, splits=("train",))
            manifest = json.loads((layout.root / "token_fields" / "manifest_train.json").read_text())
            data_card = json.loads((layout.root / "data_card.json").read_text())
            hidden_active = np.load(layout.root / "controlled_hidden" / "true_active_operator_train.npy")
            hidden_router = np.load(layout.root / "controlled_hidden" / "true_router_weights_train.npy")
            hidden_adapter_param_files = set(
                np.load(layout.root / "controlled_hidden" / "true_adapter_params_train.npz").files
            )
            sample_records = [
                json.loads(line)
                for line in (layout.root / "provenance" / "sample_records_train.jsonl").read_text().splitlines()
                if line.strip()
            ]
            data_card_exists = (layout.root / "data_card.json").exists()

        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["policy"], "cache built and validated")
        self.assertTrue(validation.ok, validation.errors)
        self.assertEqual(data_card["tasks"], ["controlled_relation_operator"])
        self.assertEqual(data_card["modalities"], ["text", "region", "audio"])
        self.assertEqual(sorted(manifest), ["audio", "region", "text"])
        self.assertEqual(hidden_active.shape, (2 * len(CONTROLLED_MULTIMODAL_FAMILIES), 4))
        self.assertEqual(hidden_router.shape, (2 * len(CONTROLLED_MULTIMODAL_FAMILIES), 4, 4))
        self.assertEqual(
            {record["controlled_family"] for record in sample_records},
            set(CONTROLLED_MULTIMODAL_FAMILIES),
        )
        self.assertIn("TLEO__lengthscale", hidden_adapter_param_files)
        self.assertIn("CATO__alignment_temperature", hidden_adapter_param_files)
        self.assertNotIn("controlled_hidden", json.dumps(manifest, sort_keys=True))
        self.assertTrue(data_card_exists)
        self.assertNotIn("Traceback", result.stderr)

    def test_build_cache_cli_writes_valid_refcoco_cache_from_raw_manifest(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_root = tmp_path / "raw"
            cache_root = tmp_path / "cache"
            _write_refcoco_raw_fixture(raw_root)

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "build_cache.py"),
                    "refcoco",
                    str(raw_root),
                    str(cache_root),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            layout = MultimodalCacheLayout(cache_root, "refcoco", "v0.1")
            validation = validate_cache_layout(layout, splits=("train", "val", "test"))
            checksums_exists = (layout.root / "checksums.json").exists()
            sample_records_exists = (layout.root / "provenance" / "sample_records_train.jsonl").exists()
            alignment_pairs_exists = (layout.root / "supervision" / "alignment_pairs_test.parquet").exists()
            import numpy as np

            text_train = np.load(layout.root / "token_fields" / "text_train.npy")
            region_train = np.load(layout.root / "token_fields" / "region_train.npy")
            text_pos_train = np.load(layout.root / "positions" / "text_pos_train.npy")
            region_mask_train = np.load(layout.root / "masks" / "region_mask_train.npy")
            task_labels_train = np.load(layout.root / "supervision" / "task_labels_train.npy")
            bbox_targets_train = np.load(layout.root / "supervision" / "bbox_targets_train.npy")
            region_targets_train = np.load(layout.root / "supervision" / "region_targets_train.npy")
            alignment_rows = [
                json.loads(line)
                for line in (layout.root / "supervision" / "alignment_pairs_train.parquet").read_text().splitlines()
                if line.strip()
            ]
            failed_train_rows = [
                json.loads(line)
                for line in (layout.root / "provenance" / "failed_samples_train.jsonl").read_text().splitlines()
                if line.strip()
            ]

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["policy"], "cache built and validated")
        self.assertTrue(validation.ok, validation.errors)
        self.assertTrue(checksums_exists)
        self.assertTrue(sample_records_exists)
        self.assertTrue(alignment_pairs_exists)
        self.assertEqual(text_train.shape, (1, 4, 5))
        self.assertEqual(region_train.shape, (1, 2, 4))
        self.assertEqual(text_pos_train.shape, (1, 4, 1))
        self.assertEqual(region_mask_train.shape, (1, 2))
        self.assertTrue(region_mask_train.all())
        self.assertEqual(task_labels_train.shape, (1, 2))
        self.assertEqual(task_labels_train.tolist(), [[0.0, 1.0]])
        self.assertEqual(bbox_targets_train.shape, (1, 4))
        self.assertEqual(region_targets_train.shape, (1, 1))
        self.assertEqual(region_targets_train.tolist(), [[1]])
        self.assertEqual(alignment_rows[0]["target_region_index"], 1)
        self.assertEqual(
            failed_train_rows,
            [{"source_id": "ref-train-failed", "split": "train", "reason": "download_failed"}],
        )

    def test_stage_refcoco_raw_cli_outputs_build_cache_ready_manifest(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inputs = tmp_path / "inputs"
            raw_root = tmp_path / "raw_refcoco"
            cache_root = tmp_path / "cache"
            inputs.mkdir()
            np.save(inputs / "text.npy", np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5))
            np.save(inputs / "region.npy", np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4))
            (inputs / "splits.json").write_text(
                json.dumps({"train": ["ref-train-1"], "val": ["ref-val-1"], "test": ["ref-test-1"]}, sort_keys=True) + "\n"
            )
            (inputs / "records.json").write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "source_id": source_id,
                                "image_id": f"image-{source_id}",
                                "caption_id": f"caption-{source_id}",
                                "phrase_span": {"start": 1, "end": 3},
                                "region_box": [0.1, 0.2, 0.8, 0.9],
                                "target_region_index": 1,
                                "candidate_region_source": "detector-v1",
                                "box_coordinate_convention": "xyxy_normalized",
                            }
                            for source_id in ("ref-train-1", "ref-val-1", "ref-test-1")
                        ]
                    },
                    sort_keys=True,
                )
                + "\n"
            )

            stage_result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "stage_refcoco_raw.py"),
                    "refcoco",
                    str(raw_root),
                    "--splits",
                    str(inputs / "splits.json"),
                    "--records",
                    str(inputs / "records.json"),
                    "--text-features",
                    str(inputs / "text.npy"),
                    "--region-features",
                    str(inputs / "region.npy"),
                    "--license-tag",
                    "unit-refcoco-license",
                    "--preprocessing-version",
                    "unit-refcoco-preprocess-v1",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            build_result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "build_cache.py"),
                    "refcoco",
                    str(raw_root),
                    str(cache_root),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            layout = MultimodalCacheLayout(cache_root, "refcoco", "v0.1")
            validation = validate_cache_layout(layout, splits=("train", "val", "test"))
            payload = json.loads(stage_result.stdout) if stage_result.stdout.strip() else {}
            refs = (
                json.loads((raw_root / "annotations" / "refs.json").read_text()).get("records", [])
                if (raw_root / "annotations" / "refs.json").exists()
                else []
            )
            task_labels_train = (
                np.load(layout.root / "supervision" / "task_labels_train.npy")
                if (layout.root / "supervision" / "task_labels_train.npy").exists()
                else np.zeros((0, 0), dtype=np.float32)
            )

        self.assertEqual(stage_result.returncode, 0, stage_result.stdout + stage_result.stderr)
        self.assertEqual(build_result.returncode, 0, build_result.stdout + build_result.stderr)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["dataset_name"], "refcoco")
        self.assertTrue(validation.ok, validation.errors)
        self.assertEqual([row["source_id"] for row in refs], ["ref-train-1", "ref-val-1", "ref-test-1"])
        self.assertEqual(refs[0]["license_tag"], "unit-refcoco-license")
        self.assertEqual(refs[0]["preprocessing_version"], "unit-refcoco-preprocess-v1")
        self.assertEqual(task_labels_train.tolist(), [[0.0, 1.0]])

    def test_build_refcoco_stage_records_cli_outputs_stage_ready_records_and_splits(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inputs = tmp_path / "inputs"
            stage_inputs = tmp_path / "stage_inputs"
            raw_root = tmp_path / "raw_refcoco"
            cache_root = tmp_path / "cache"
            inputs.mkdir()
            stage_inputs.mkdir()
            (inputs / "instances.json").write_text(
                json.dumps(
                    {
                        "images": [
                            {"id": 10, "width": 100, "height": 200},
                            {"id": 20, "width": 50, "height": 100},
                            {"id": 30, "width": 80, "height": 80},
                        ],
                        "annotations": [
                            {"id": 501, "image_id": 10, "bbox": [10, 20, 30, 40]},
                            {"id": 502, "image_id": 20, "bbox": [5, 10, 10, 20]},
                            {"id": 503, "image_id": 30, "bbox": [8, 16, 16, 24]},
                        ],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            (inputs / "refs.json").write_text(
                json.dumps(
                    [
                        {
                            "ref_id": 1,
                            "ann_id": 501,
                            "image_id": 10,
                            "split": "train",
                            "sentences": [{"sent_id": 1001, "raw": "red ball", "tokens": ["red", "ball"]}],
                        },
                        {
                            "ref_id": 2,
                            "ann_id": 502,
                            "image_id": 20,
                            "split": "val",
                            "sentences": [{"sent_id": 1002, "sent": "blue box", "tokens": ["blue", "box"]}],
                        },
                        {
                            "ref_id": 3,
                            "ann_id": 503,
                            "image_id": 30,
                            "split": "testA",
                            "sentences": [{"sent_id": 1003, "raw": "green thing", "tokens": ["green", "thing"]}],
                        },
                    ],
                    sort_keys=True,
                )
                + "\n"
            )

            build_records_result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "build_refcoco_stage_records.py"),
                    "refcoco",
                    str(stage_inputs),
                    "--refs",
                    str(inputs / "refs.json"),
                    "--instances",
                    str(inputs / "instances.json"),
                    "--candidate-region-source",
                    "coco_gt_box",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            np.save(stage_inputs / "refcoco_text_features.npy", np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5))
            np.save(stage_inputs / "refcoco_region_features.npy", np.arange(3 * 1 * 4, dtype=np.float32).reshape(3, 1, 4))
            stage_result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "stage_refcoco_raw.py"),
                    "refcoco",
                    str(raw_root),
                    "--splits",
                    str(stage_inputs / "refcoco_splits.json"),
                    "--records",
                    str(stage_inputs / "refcoco_phrase_region_records.json"),
                    "--text-features",
                    str(stage_inputs / "refcoco_text_features.npy"),
                    "--region-features",
                    str(stage_inputs / "refcoco_region_features.npy"),
                    "--license-tag",
                    "unit-refcoco-coco2014",
                    "--preprocessing-version",
                    "unit-refcoco-records-v1",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            build_cache_result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "build_cache.py"),
                    "refcoco",
                    str(raw_root),
                    str(cache_root),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            payload = json.loads(build_records_result.stdout) if build_records_result.stdout.strip() else {}
            splits = json.loads((stage_inputs / "refcoco_splits.json").read_text()) if (stage_inputs / "refcoco_splits.json").exists() else {}
            records = (
                json.loads((stage_inputs / "refcoco_phrase_region_records.json").read_text()).get("records", [])
                if (stage_inputs / "refcoco_phrase_region_records.json").exists()
                else []
            )
            validation = validate_cache_layout(MultimodalCacheLayout(cache_root, "refcoco", "v0.1"), splits=("train", "val", "test"))

        self.assertEqual(build_records_result.returncode, 0, build_records_result.stdout + build_records_result.stderr)
        self.assertEqual(stage_result.returncode, 0, stage_result.stdout + stage_result.stderr)
        self.assertEqual(build_cache_result.returncode, 0, build_cache_result.stdout + build_cache_result.stderr)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["sample_count"], 3)
        self.assertEqual(splits, {
            "test": ["refcoco::image30::ann503::sent1003"],
            "train": ["refcoco::image10::ann501::sent1001"],
            "val": ["refcoco::image20::ann502::sent1002"],
        })
        self.assertEqual(records[0]["phrase_span"], {"start": 0, "end": 2})
        self.assertEqual(records[0]["region_box"], [0.1, 0.1, 0.4, 0.3])
        self.assertEqual(records[0]["target_region_index"], 0)
        self.assertEqual(records[0]["box_coordinate_convention"], "xyxy_normalized")
        self.assertTrue(validation.ok, validation.errors)

    def test_extract_refcoco_clip_features_cli_dry_run_reports_formal_clip_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stage_inputs = tmp_path / "stage_inputs"
            image_root = tmp_path / "train2014"
            stage_inputs.mkdir()
            image_root.mkdir()
            splits = {
                "train": ["refcoco::image10::ann501::sent1001"],
                "val": ["refcoco::image20::ann502::sent1002"],
                "test": ["refcoco::image30::ann503::sent1003"],
            }
            records = {
                "records": [
                    {
                        "source_id": source_id,
                        "image_id": image_id,
                        "caption_id": caption_id,
                        "phrase_span": {"start": 0, "end": 2},
                        "region_box": [0.1, 0.1, 0.4, 0.3],
                        "target_region_index": 0,
                    }
                    for source_id, image_id, caption_id in (
                        (splits["train"][0], "image10", "sent1001"),
                        (splits["val"][0], "image20", "sent1002"),
                        (splits["test"][0], "image30", "sent1003"),
                    )
                ]
            }
            refs = [
                {
                    "ann_id": 501,
                    "image_id": 10,
                    "sentences": [{"sent_id": 1001, "raw": "red ball"}],
                },
                {
                    "ann_id": 502,
                    "image_id": 20,
                    "sentences": [{"sent_id": 1002, "sent": "blue box"}],
                },
                {
                    "ann_id": 503,
                    "image_id": 30,
                    "sentences": [{"sent_id": 1003, "tokens": ["green", "thing"]}],
                },
            ]
            (stage_inputs / "refcoco_splits.json").write_text(json.dumps(splits, sort_keys=True) + "\n")
            (stage_inputs / "refcoco_phrase_region_records.json").write_text(json.dumps(records, sort_keys=True) + "\n")
            (stage_inputs / "refs.json").write_text(json.dumps(refs, sort_keys=True) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "extract_refcoco_clip_features.py"),
                    "refcoco",
                    str(stage_inputs),
                    "--splits",
                    str(stage_inputs / "refcoco_splits.json"),
                    "--records",
                    str(stage_inputs / "refcoco_phrase_region_records.json"),
                    "--refs",
                    str(stage_inputs / "refs.json"),
                    "--image-root",
                    str(image_root),
                    "--dry-run",
                    "--min-image-coverage",
                    "0",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["mode"], "dry_run")
        self.assertEqual(payload["sample_count"], 3)
        self.assertEqual(payload["missing_image_count"], 3)
        self.assertEqual(payload["feature_shapes"]["text"], [3, 1, "clip_projection_dim"])
        self.assertEqual(payload["feature_shapes"]["region"], [3, 1, "clip_projection_dim"])
        self.assertEqual(payload["feature_extractor_versions"]["text"], "openai/clip-vit-base-patch32@main:text_projection")
        self.assertEqual(payload["candidate_region_source"], "coco_gt_box_crop")

    def test_align_refcoco_stage_features_cli_reorders_feature_banks_by_stage_splits(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stage_inputs = tmp_path / "stage_inputs"
            raw_root = tmp_path / "raw_refcoco"
            cache_root = tmp_path / "cache"
            stage_inputs.mkdir()
            splits = {
                "train": ["refcoco::image10::ann501::sent1001"],
                "val": ["refcoco::image20::ann502::sent1002"],
                "test": ["refcoco::image30::ann503::sent1003"],
            }
            records = {
                "records": [
                    {
                        "source_id": source_id,
                        "image_id": f"image-{index}",
                        "caption_id": f"caption-{index}",
                        "phrase_span": {"start": 0, "end": 2},
                        "region_box": [0.1, 0.1, 0.4, 0.3],
                        "target_region_index": 0,
                    }
                    for index, source_id in enumerate([*splits["train"], *splits["val"], *splits["test"]])
                ]
            }
            (stage_inputs / "refcoco_splits.json").write_text(json.dumps(splits, sort_keys=True) + "\n")
            (stage_inputs / "refcoco_phrase_region_records.json").write_text(json.dumps(records, sort_keys=True) + "\n")
            bank_ids = [splits["test"][0], splits["train"][0], splits["val"][0]]
            (stage_inputs / "bank_source_ids.txt").write_text("\n".join(bank_ids) + "\n")
            np.save(stage_inputs / "text_bank.npy", np.array([[[30.0]], [[10.0]], [[20.0]]], dtype=np.float32))
            np.save(stage_inputs / "region_bank.npy", np.array([[[300.0]], [[100.0]], [[200.0]]], dtype=np.float32))

            align_result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "align_refcoco_stage_features.py"),
                    "refcoco",
                    str(stage_inputs),
                    "--splits",
                    str(stage_inputs / "refcoco_splits.json"),
                    "--records",
                    str(stage_inputs / "refcoco_phrase_region_records.json"),
                    "--text-features",
                    str(stage_inputs / "text_bank.npy"),
                    "--text-source-ids",
                    str(stage_inputs / "bank_source_ids.txt"),
                    "--region-features",
                    str(stage_inputs / "region_bank.npy"),
                    "--region-source-ids",
                    str(stage_inputs / "bank_source_ids.txt"),
                    "--feature-version",
                    "unit-refcoco-frozen-v1",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            stage_result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "stage_refcoco_raw.py"),
                    "refcoco",
                    str(raw_root),
                    "--splits",
                    str(stage_inputs / "refcoco_splits.json"),
                    "--records",
                    str(stage_inputs / "refcoco_phrase_region_records.json"),
                    "--text-features",
                    str(stage_inputs / "refcoco_text_features.npy"),
                    "--region-features",
                    str(stage_inputs / "refcoco_region_features.npy"),
                    "--license-tag",
                    "unit-refcoco-coco2014",
                    "--preprocessing-version",
                    "unit-refcoco-frozen-v1",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            build_cache_result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "build_cache.py"),
                    "refcoco",
                    str(raw_root),
                    str(cache_root),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            payload = json.loads(align_result.stdout) if align_result.stdout.strip() else {}
            aligned_text = np.load(stage_inputs / "refcoco_text_features.npy") if (stage_inputs / "refcoco_text_features.npy").exists() else np.zeros((0, 0, 0), dtype=np.float32)
            aligned_region = np.load(stage_inputs / "refcoco_region_features.npy") if (stage_inputs / "refcoco_region_features.npy").exists() else np.zeros((0, 0, 0), dtype=np.float32)
            manifest = json.loads((stage_inputs / "refcoco_feature_alignment_manifest.json").read_text()) if (stage_inputs / "refcoco_feature_alignment_manifest.json").exists() else {}
            validation = validate_cache_layout(MultimodalCacheLayout(cache_root, "refcoco", "v0.1"), splits=("train", "val", "test"))

        self.assertEqual(align_result.returncode, 0, align_result.stdout + align_result.stderr)
        self.assertEqual(stage_result.returncode, 0, stage_result.stdout + stage_result.stderr)
        self.assertEqual(build_cache_result.returncode, 0, build_cache_result.stdout + build_cache_result.stderr)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["sample_count"], 3)
        self.assertEqual(aligned_text[:, 0, 0].tolist(), [10.0, 20.0, 30.0])
        self.assertEqual(aligned_region[:, 0, 0].tolist(), [100.0, 200.0, 300.0])
        self.assertEqual(manifest["ordered_source_ids"], [splits["train"][0], splits["val"][0], splits["test"][0]])
        self.assertEqual(manifest["feature_version"], "unit-refcoco-frozen-v1")
        self.assertTrue(validation.ok, validation.errors)

    def test_build_cache_cli_validates_requested_cache_version(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_root = tmp_path / "raw"
            cache_root = tmp_path / "cache"
            _write_refcoco_raw_fixture(raw_root)

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "build_cache.py"),
                    "refcoco",
                    str(raw_root),
                    str(cache_root),
                    "--version",
                    "v0.2-test",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            requested_layout = MultimodalCacheLayout(cache_root, "refcoco", "v0.2-test")
            default_layout = MultimodalCacheLayout(cache_root, "refcoco", "v0.1")
            requested_validation = validate_cache_layout(requested_layout, splits=("train", "val", "test"))
            requested_data_card_exists = (requested_layout.root / "data_card.json").exists()
            default_data_card_exists = (default_layout.root / "data_card.json").exists()

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["cache_root"], str(requested_layout.root))
        self.assertTrue(requested_validation.ok, requested_validation.errors)
        self.assertTrue(requested_data_card_exists)
        self.assertFalse(default_data_card_exists)

    def test_build_cache_cli_writes_valid_cmu_mosei_cache_from_raw_manifest(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_root = tmp_path / "raw"
            cache_root = tmp_path / "cache"
            _write_cmu_mosei_raw_fixture(raw_root)

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "build_cache.py"),
                    "cmu_mosei",
                    str(raw_root),
                    str(cache_root),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            layout = MultimodalCacheLayout(cache_root, "cmu_mosei", "v0.1")
            validation = validate_cache_layout(layout, splits=("train", "val", "test"))
            checksums_exists = (layout.root / "checksums.json").exists()
            sample_records_exists = (layout.root / "provenance" / "sample_records_train.jsonl").exists()
            missing_mask_exists = (layout.root / "supervision" / "missing_modality_mask_test.npy").exists()
            corruption_exists = (layout.root / "supervision" / "corruption_val.parquet").exists()
            data_card = json.loads((layout.root / "data_card.json").read_text())
            train_records = [
                json.loads(line)
                for line in (layout.root / "provenance" / "sample_records_train.jsonl").read_text().splitlines()
                if line.strip()
            ]
            failed_train_rows = [
                json.loads(line)
                for line in (layout.root / "provenance" / "failed_samples_train.jsonl").read_text().splitlines()
                if line.strip()
            ]
            import numpy as np

            text_train = np.load(layout.root / "token_fields" / "text_train.npy")
            text_pos_train = np.load(layout.root / "positions" / "text_pos_train.npy")
            text_mask_train = np.load(layout.root / "masks" / "text_mask_train.npy")
            task_labels_train = np.load(layout.root / "supervision" / "task_labels_train.npy")
            emotion_labels_train = np.load(layout.root / "supervision" / "emotion_labels_train.npy")
            missing_modality_mask_test = np.load(layout.root / "supervision" / "missing_modality_mask_test.npy")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["policy"], "cache built and validated")
        self.assertTrue(validation.ok, validation.errors)
        self.assertTrue(checksums_exists)
        self.assertTrue(sample_records_exists)
        self.assertTrue(missing_mask_exists)
        self.assertTrue(corruption_exists)
        self.assertEqual(text_train.shape, (1, 4, 5))
        self.assertEqual(text_pos_train.shape, (1, 4, 1))
        self.assertEqual(text_mask_train.shape, (1, 4))
        self.assertTrue(text_mask_train.all())
        self.assertEqual(task_labels_train.shape, (1, 1))
        self.assertEqual(emotion_labels_train.shape, (1, 7))
        self.assertEqual(missing_modality_mask_test.shape, (1, 3))
        self.assertEqual(data_card["metadata_availability"]["speaker_id"], True)
        self.assertEqual(train_records[0]["speaker_id"], "speaker-train")
        self.assertEqual(
            failed_train_rows,
            [{"source_id": "mosei-train-failed", "split": "train", "reason": "audio_decode_failed"}],
        )

    def test_stage_cmu_sentiment_raw_cli_outputs_build_cache_ready_manifest(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inputs = tmp_path / "inputs"
            raw_root = tmp_path / "raw_cmu_mosei"
            cache_root = tmp_path / "cache"
            inputs.mkdir()
            np.save(inputs / "text.npy", np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5))
            np.save(inputs / "audio.npy", np.arange(3 * 6 * 3, dtype=np.float32).reshape(3, 6, 3))
            np.save(inputs / "vision.npy", np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4))
            np.save(inputs / "sentiment.npy", np.array([[-1.0], [0.0], [1.0]], dtype=np.float32))
            np.save(inputs / "emotion.npy", np.eye(7, dtype=np.float32)[:3])
            (inputs / "splits.json").write_text(
                json.dumps(
                    {
                        "train": ["mosei-train-1"],
                        "val": ["mosei-val-1"],
                        "test": ["mosei-test-1"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )

            stage_result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "stage_cmu_sentiment_raw.py"),
                    "cmu_mosei",
                    str(raw_root),
                    "--splits",
                    str(inputs / "splits.json"),
                    "--text-features",
                    str(inputs / "text.npy"),
                    "--audio-features",
                    str(inputs / "audio.npy"),
                    "--visual-features",
                    str(inputs / "vision.npy"),
                    "--sentiment-labels",
                    str(inputs / "sentiment.npy"),
                    "--emotion-labels",
                    str(inputs / "emotion.npy"),
                    "--feature-version",
                    "text=unit-text-v1",
                    "--feature-version",
                    "audio=unit-audio-v1",
                    "--feature-version",
                    "vision=unit-vision-v1",
                    "--license-tag",
                    "unit-license",
                    "--preprocessing-version",
                    "unit-preprocess-v1",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            build_result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "build_cache.py"),
                    "cmu_mosei",
                    str(raw_root),
                    str(cache_root),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            layout = MultimodalCacheLayout(cache_root, "cmu_mosei", "v0.1")
            validation = validate_cache_layout(layout, splits=("train", "val", "test"))
            payload = json.loads(stage_result.stdout) if stage_result.stdout.strip() else {}
            feature_versions = (
                json.loads((raw_root / "metadata" / "feature_versions.json").read_text())
                if (raw_root / "metadata" / "feature_versions.json").exists()
                else {}
            )
            utterances = (
                json.loads((raw_root / "metadata" / "utterances.json").read_text()).get("records", [])
                if (raw_root / "metadata" / "utterances.json").exists()
                else []
            )
            staged_missing = (
                np.load(raw_root / "metadata" / "missing_modality_mask.npy")
                if (raw_root / "metadata" / "missing_modality_mask.npy").exists()
                else np.zeros((0, 0), dtype=bool)
            )
            task_labels_train_exists = (layout.root / "supervision" / "task_labels_train.npy").exists()

        self.assertEqual(stage_result.returncode, 0, stage_result.stdout + stage_result.stderr)
        self.assertEqual(build_result.returncode, 0, build_result.stdout + build_result.stderr)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["dataset_name"], "cmu_mosei")
        self.assertTrue(validation.ok, validation.errors)
        self.assertEqual(feature_versions["text"], "unit-text-v1")
        self.assertEqual(feature_versions["audio"], "unit-audio-v1")
        self.assertEqual(feature_versions["vision"], "unit-vision-v1")
        self.assertEqual([row["source_id"] for row in utterances], ["mosei-train-1", "mosei-val-1", "mosei-test-1"])
        self.assertEqual(staged_missing.shape, (3, 3))
        self.assertTrue(task_labels_train_exists)

    def test_extract_cmu_sdk_stage_inputs_cli_outputs_stage_ready_arrays(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sdk_root = tmp_path / "sdk"
            stage_inputs = tmp_path / "stage_inputs"
            raw_root = tmp_path / "raw_cmu_mosei"
            cache_root = tmp_path / "cache"
            sdk_root.mkdir()
            source_ids = ("mosei-train-1", "mosei-val-1", "mosei-test-1")
            _write_cmu_sequence_json(
                sdk_root / "text.json",
                {
                    source_id: np.full((2, 4), float(index + 1), dtype=np.float32)
                    for index, source_id in enumerate(source_ids)
                },
            )
            _write_cmu_sequence_json(
                sdk_root / "audio.json",
                {
                    source_id: np.full((3, 2), float(index + 2), dtype=np.float32)
                    for index, source_id in enumerate(source_ids)
                },
            )
            _write_cmu_sequence_json(
                sdk_root / "vision.json",
                {
                    source_id: np.full((1, 5), float(index + 3), dtype=np.float32)
                    for index, source_id in enumerate(source_ids)
                },
            )
            _write_cmu_sequence_json(
                sdk_root / "labels.json",
                {
                    source_id: np.array([[float(index - 1), *np.eye(7, dtype=np.float32)[index]]], dtype=np.float32)
                    for index, source_id in enumerate(source_ids)
                },
            )
            (sdk_root / "splits.json").write_text(
                json.dumps(
                    {
                        "train": ["mosei-train-1"],
                        "val": ["mosei-val-1"],
                        "test": ["mosei-test-1"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )

            extract_result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "extract_cmu_sdk_stage_inputs.py"),
                    "cmu_mosei",
                    str(stage_inputs),
                    "--splits",
                    str(sdk_root / "splits.json"),
                    "--text-sequence",
                    str(sdk_root / "text.json"),
                    "--audio-sequence",
                    str(sdk_root / "audio.json"),
                    "--visual-sequence",
                    str(sdk_root / "vision.json"),
                    "--label-sequence",
                    str(sdk_root / "labels.json"),
                    "--temporal-policy",
                    "mean",
                    "--sentiment-column",
                    "0",
                    "--emotion-columns",
                    "1:8",
                    "--preprocessing-version",
                    "unit-cmu-sdk-mean-v1",
                    "--license-tag",
                    "unit-cmu-sdk",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            stage_result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "stage_cmu_sentiment_raw.py"),
                    "cmu_mosei",
                    str(raw_root),
                    "--splits",
                    str(stage_inputs / "cmu_mosei_splits.json"),
                    "--text-features",
                    str(stage_inputs / "cmu_mosei_text_features.npy"),
                    "--audio-features",
                    str(stage_inputs / "cmu_mosei_audio_features.npy"),
                    "--visual-features",
                    str(stage_inputs / "cmu_mosei_visual_features.npy"),
                    "--sentiment-labels",
                    str(stage_inputs / "cmu_mosei_sentiment.npy"),
                    "--emotion-labels",
                    str(stage_inputs / "cmu_mosei_emotion.npy"),
                    "--feature-version",
                    "text=unit-text-json",
                    "--feature-version",
                    "audio=unit-audio-json",
                    "--feature-version",
                    "vision=unit-vision-json",
                    "--license-tag",
                    "unit-cmu-sdk",
                    "--preprocessing-version",
                    "unit-cmu-sdk-mean-v1",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            build_result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "build_cache.py"),
                    "cmu_mosei",
                    str(raw_root),
                    str(cache_root),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            payload = json.loads(extract_result.stdout) if extract_result.stdout.strip() else {}
            text_features = (
                np.load(stage_inputs / "cmu_mosei_text_features.npy")
                if (stage_inputs / "cmu_mosei_text_features.npy").exists()
                else np.zeros((0, 0, 0), dtype=np.float32)
            )
            sentiment = (
                np.load(stage_inputs / "cmu_mosei_sentiment.npy")
                if (stage_inputs / "cmu_mosei_sentiment.npy").exists()
                else np.zeros((0, 0), dtype=np.float32)
            )
            emotion = (
                np.load(stage_inputs / "cmu_mosei_emotion.npy")
                if (stage_inputs / "cmu_mosei_emotion.npy").exists()
                else np.zeros((0, 0), dtype=np.float32)
            )
            manifest = (
                json.loads((stage_inputs / "cmu_mosei_stage_input_manifest.json").read_text())
                if (stage_inputs / "cmu_mosei_stage_input_manifest.json").exists()
                else {}
            )
            validation = validate_cache_layout(
                MultimodalCacheLayout(cache_root, "cmu_mosei", "v0.1"),
                splits=("train", "val", "test"),
            )

        self.assertEqual(extract_result.returncode, 0, extract_result.stdout + extract_result.stderr)
        self.assertEqual(stage_result.returncode, 0, stage_result.stdout + stage_result.stderr)
        self.assertEqual(build_result.returncode, 0, build_result.stdout + build_result.stderr)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["sample_count"], 3)
        self.assertEqual(text_features.shape, (3, 1, 4))
        self.assertEqual(sentiment.tolist(), [[-1.0], [0.0], [1.0]])
        self.assertEqual(emotion.shape, (3, 7))
        self.assertEqual(manifest["temporal_policy"], "mean")
        self.assertIn("next", payload)
        self.assertTrue(validation.ok, validation.errors)

    def test_inspect_cmu_sdk_sequences_cli_suggests_extract_command_from_download_dir(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sdk_root = tmp_path / "cmu_sdk" / "cmu_mosei"
            sdk_root.mkdir(parents=True)
            source_ids = ("mosei-train-1", "mosei-val-1", "mosei-test-1")
            _write_cmu_sequence_json(
                sdk_root / "BERT_text_features.json",
                {source_id: np.ones((2, 4), dtype=np.float32) for source_id in source_ids},
            )
            _write_cmu_sequence_json(
                sdk_root / "COVAREP_audio_features.json",
                {source_id: np.ones((3, 2), dtype=np.float32) for source_id in source_ids},
            )
            _write_cmu_sequence_json(
                sdk_root / "FACET_visual_features.json",
                {source_id: np.ones((1, 5), dtype=np.float32) for source_id in source_ids},
            )
            _write_cmu_sequence_json(
                sdk_root / "Opinion_Labels.json",
                {source_id: np.array([[0.5, *np.eye(7, dtype=np.float32)[0]]], dtype=np.float32) for source_id in source_ids},
            )
            splits = sdk_root / "splits.json"
            splits.write_text(
                json.dumps({"train": ["mosei-train-1"], "val": ["mosei-val-1"], "test": ["mosei-test-1"]}, sort_keys=True)
                + "\n"
            )
            output = tmp_path / "inspection.json"

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "inspect_cmu_sdk_sequences.py"),
                    "cmu_mosei",
                    str(sdk_root),
                    "--splits",
                    str(splits),
                    "--stage-output-dir",
                    "data/raw_multimodal/_downloads/cmu_mosei_stage_inputs",
                    "--temporal-policy",
                    "mean",
                    "--preprocessing-version",
                    "unit-cmu-sdk-mean-v1",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            payload = json.loads(result.stdout) if result.stdout.strip() else {}
            written = json.loads(output.read_text()) if output.exists() else {}

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["sequence_count"], 4)
        self.assertEqual(payload["suggested_roles"]["text"]["path"], str(sdk_root / "BERT_text_features.json"))
        self.assertEqual(payload["suggested_roles"]["audio"]["path"], str(sdk_root / "COVAREP_audio_features.json"))
        self.assertEqual(payload["suggested_roles"]["vision"]["path"], str(sdk_root / "FACET_visual_features.json"))
        self.assertEqual(payload["suggested_roles"]["labels"]["path"], str(sdk_root / "Opinion_Labels.json"))
        self.assertIn("scripts/multimodal/extract_cmu_sdk_stage_inputs.py", payload["suggested_extract_command"])
        self.assertIn("--text-sequence", payload["suggested_extract_command"])
        self.assertEqual(written["suggested_roles"], payload["suggested_roles"])

    def test_write_cmu_sdk_splits_cli_exports_official_fold_json_and_validates_sequences(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            folds = tmp_path / "folds.json"
            output = tmp_path / "cmu_mosei_splits.json"
            sequence = tmp_path / "text.json"
            folds.write_text(
                json.dumps(
                    {
                        "standard_train_fold": ["mosei-train-1"],
                        "standard_valid_fold": ["mosei-val-1"],
                        "standard_test_fold": ["mosei-test-1"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            _write_cmu_sequence_json(
                sequence,
                {
                    "mosei-train-1": np.ones((2, 4), dtype=np.float32),
                    "mosei-val-1": np.ones((2, 4), dtype=np.float32),
                    "mosei-test-1": np.ones((2, 4), dtype=np.float32),
                },
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "write_cmu_sdk_splits.py"),
                    "cmu_mosei",
                    str(output),
                    "--folds-json",
                    str(folds),
                    "--sequence",
                    str(sequence),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            payload = json.loads(result.stdout) if result.stdout.strip() else {}
            splits = json.loads(output.read_text()) if output.exists() else {}

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["source"], "folds_json")
        self.assertEqual(payload["split_counts"], {"test": 1, "train": 1, "val": 1})
        self.assertTrue(payload["sequence_validation"]["ok"])
        self.assertEqual(splits, {"test": ["mosei-test-1"], "train": ["mosei-train-1"], "val": ["mosei-val-1"]})

    def test_write_cmu_sdk_splits_cli_expands_video_folds_to_segment_source_ids(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            folds = tmp_path / "folds.json"
            output = tmp_path / "cmu_mosei_splits.json"
            labels = tmp_path / "labels.json"
            audio = tmp_path / "audio.json"
            folds.write_text(
                json.dumps(
                    {
                        "standard_train_fold": ["video-train"],
                        "standard_valid_fold": ["video-val"],
                        "standard_test_fold": ["video-test"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            sequence_payload = {
                "video-train[0]": np.ones((1, 8), dtype=np.float32),
                "video-train[1]": np.ones((1, 8), dtype=np.float32),
                "video-val[0]": np.ones((1, 8), dtype=np.float32),
                "video-test[0]": np.ones((1, 8), dtype=np.float32),
            }
            _write_cmu_sequence_json(labels, sequence_payload)
            _write_cmu_sequence_json(audio, sequence_payload)

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "write_cmu_sdk_splits.py"),
                    "cmu_mosei",
                    str(output),
                    "--folds-json",
                    str(folds),
                    "--expand-with-sequence",
                    str(labels),
                    "--sequence",
                    str(audio),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            payload = json.loads(result.stdout) if result.stdout.strip() else {}
            splits = json.loads(output.read_text()) if output.exists() else {}

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["split_counts"], {"test": 1, "train": 2, "val": 1})
        self.assertEqual(payload["expansion"]["mode"], "sequence_parent_id")
        self.assertEqual(payload["expansion"]["expanded_source_ids"], 4)
        self.assertEqual(payload["expansion"]["unmatched_source_ids"], 0)
        self.assertEqual(splits["train"], ["video-train[0]", "video-train[1]"])
        self.assertEqual(splits["val"], ["video-val[0]"])
        self.assertEqual(splits["test"], ["video-test[0]"])

    def test_public_data_readiness_cli_reports_missing_steps_without_blocking(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "check_public_data_readiness.py"),
                    "--datasets",
                    "refcoco",
                    "cmu_mosei",
                    "--download-root",
                    str(tmp_path / "downloads"),
                    "--raw-root-base",
                    str(tmp_path / "raw"),
                    "--cache-root",
                    str(tmp_path / "cache"),
                    "--controlled-report",
                    str(tmp_path / "controlled_report.json"),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        joined_commands = "\n".join(payload["next_commands"])
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["mode"], "public_data_readiness")
        self.assertFalse(payload["datasets"]["refcoco"]["ok"])
        self.assertFalse(payload["datasets"]["cmu_mosei"]["ok"])
        self.assertIn("scripts/multimodal/bootstrap_public_downloads.py", joined_commands)
        self.assertIn("bash -n /tmp/ovha_public_downloads.sh", joined_commands)
        self.assertIn("bash /tmp/ovha_public_downloads.sh", joined_commands)
        self.assertIn("scripts/multimodal/build_refcoco_stage_records.py", joined_commands)
        self.assertIn("scripts/multimodal/write_cmu_sdk_splits.py", joined_commands)
        self.assertNotIn("Traceback", result.stderr)

    def test_public_data_readiness_cli_points_valid_caches_to_acceptance(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_root_base = tmp_path / "raw"
            cache_root = tmp_path / "cache"
            controlled_report = tmp_path / "controlled_report.json"
            controlled_report.write_text(
                json.dumps(
                    {
                        "go_no_go": {
                            "controlled_multimodal_passed": True,
                            "enter_public_multimodal": True,
                            "reasons": [],
                        }
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            _write_refcoco_raw_fixture(raw_root_base / "refcoco")
            _write_cmu_mosei_raw_fixture(raw_root_base / "cmu_mosei")
            for dataset_name in ("refcoco", "cmu_mosei"):
                build_result = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "multimodal" / "build_cache.py"),
                        dataset_name,
                        str(raw_root_base / dataset_name),
                        str(cache_root),
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(build_result.returncode, 0, build_result.stdout + build_result.stderr)

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "check_public_data_readiness.py"),
                    "--datasets",
                    "refcoco",
                    "cmu_mosei",
                    "--download-root",
                    str(tmp_path / "downloads"),
                    "--raw-root-base",
                    str(raw_root_base),
                    "--cache-root",
                    str(cache_root),
                    "--controlled-report",
                    str(controlled_report),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            ref_validation = validate_cache_layout(
                MultimodalCacheLayout(cache_root, "refcoco", "v0.1"),
                splits=("train", "val", "test"),
            )
            cmu_validation = validate_cache_layout(
                MultimodalCacheLayout(cache_root, "cmu_mosei", "v0.1"),
                splits=("train", "val", "test"),
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(ref_validation.ok, ref_validation.errors)
        self.assertTrue(cmu_validation.ok, cmu_validation.errors)
        payload = json.loads(result.stdout)
        joined_commands = "\n".join(payload["next_commands"])
        self.assertTrue(payload["ok"], payload)
        self.assertTrue(payload["datasets"]["refcoco"]["phases"]["cache"]["ok"])
        self.assertTrue(payload["datasets"]["cmu_mosei"]["phases"]["cache"]["ok"])
        self.assertIn("scripts/multimodal/accept_public_data.py", payload["datasets"]["refcoco"]["next_action"])
        self.assertIn("scripts/multimodal/accept_public_data.py", payload["datasets"]["cmu_mosei"]["next_action"])
        self.assertIn("scripts/multimodal/accept_public_data.py", joined_commands)
        self.assertIn("configs/multimodal_refcoco_public_smoke.json", joined_commands)
        self.assertIn("configs/multimodal_cmu_mosei_public_smoke.json", joined_commands)

    def test_build_cache_cli_writes_valid_meld_cache_from_dialogue_manifest(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_root = tmp_path / "raw"
            cache_root = tmp_path / "cache"
            _write_meld_raw_fixture(raw_root)

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "build_cache.py"),
                    "meld",
                    str(raw_root),
                    str(cache_root),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            layout = MultimodalCacheLayout(cache_root, "meld", "v0.1")
            validation = validate_cache_layout(layout, splits=("train", "val", "test"))
            data_card = json.loads((layout.root / "data_card.json").read_text()) if (layout.root / "data_card.json").exists() else {}
            train_records = (
                [
                    json.loads(line)
                    for line in (layout.root / "provenance" / "sample_records_train.jsonl").read_text().splitlines()
                    if line.strip()
                ]
                if (layout.root / "provenance" / "sample_records_train.jsonl").exists()
                else []
            )
            import numpy as np

            task_labels_train = (
                np.load(layout.root / "supervision" / "task_labels_train.npy")
                if (layout.root / "supervision" / "task_labels_train.npy").exists()
                else None
            )
            missing_modality_mask_val = (
                np.load(layout.root / "supervision" / "missing_modality_mask_val.npy")
                if (layout.root / "supervision" / "missing_modality_mask_val.npy").exists()
                else None
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["policy"], "cache built and validated")
        self.assertTrue(validation.ok, validation.errors)
        self.assertEqual(task_labels_train.shape, (1, 7))
        self.assertEqual(missing_modality_mask_val.shape, (1, 3))
        self.assertEqual(data_card["metadata_availability"]["speaker_id"], True)
        self.assertEqual(train_records[0]["speaker_id"], "Monica")
        self.assertEqual(train_records[0]["transcript_source"], "MELD train_sent_emo.csv")

    def test_extract_meld_ffmpeg_features_cli_writes_hash_and_missing_video_features(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            meld_root = tmp_path / "MELD.Raw"
            raw_root = tmp_path / "raw"
            _write_meld_raw_fixture(raw_root)
            _write_meld_csv_fixture(meld_root)

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "extract_meld_ffmpeg_features.py"),
                    str(meld_root),
                    str(raw_root),
                    "--workers",
                    "1",
                    "--text-dim",
                    "16",
                    "--audio-steps",
                    "2",
                    "--visual-frames",
                    "2",
                    "--visual-size",
                    "2",
                    "--min-video-coverage",
                    "0",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            payload = json.loads(result.stdout) if result.stdout.strip() else {}
            text_features = np.load(raw_root / "features" / "text_features.npy")
            audio_features = np.load(raw_root / "features" / "audio_features.npy")
            visual_features = np.load(raw_root / "features" / "visual_features.npy")
            missing_mask = np.load(raw_root / "metadata" / "missing_modality_mask.npy")
            feature_versions = json.loads((raw_root / "metadata" / "feature_versions.json").read_text())

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["feature_shapes"]["text"], [3, 1, 16])
        self.assertEqual(payload["missing_video_count"], 3)
        self.assertEqual(text_features.shape, (3, 1, 16))
        self.assertEqual(audio_features.shape, (3, 2, 12))
        self.assertEqual(visual_features.shape, (3, 2, 12))
        self.assertTrue(np.any(text_features[0]))
        self.assertTrue(np.all(audio_features == 0))
        self.assertTrue(np.all(visual_features == 0))
        self.assertEqual(missing_mask.shape, (3, 3))
        self.assertFalse(bool(missing_mask[0, 0]))
        self.assertTrue(bool(missing_mask[0, 1]))
        self.assertTrue(bool(missing_mask[0, 2]))
        self.assertIn("meld-hashed-text-v0.1", feature_versions["text"])

    def test_extract_meld_transformer_features_cli_dry_run_reports_formal_encoder_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            meld_root = tmp_path / "MELD.Raw"
            raw_root = tmp_path / "raw"
            _write_meld_raw_fixture(raw_root)
            _write_meld_csv_fixture(meld_root)

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "extract_meld_transformer_features.py"),
                    str(meld_root),
                    str(raw_root),
                    "--dry-run",
                    "--min-video-coverage",
                    "0",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["mode"], "dry_run")
        self.assertEqual(payload["feature_extractor_versions"]["text"], "FacebookAI/roberta-base@main")
        self.assertEqual(payload["feature_extractor_versions"]["audio"], "facebook/wav2vec2-base-960h@main")
        self.assertEqual(payload["feature_extractor_versions"]["visual"], "openai/clip-vit-base-patch32@main")

    def test_controlled_true_adapter_param_contract_matches_v1_protocol(self):
        from moat_ovha_torch.data.multimodal.adapters.controlled_synthetic import (
            CONTROLLED_FAMILY_ACTIVE_OPERATOR,
            CONTROLLED_MULTIMODAL_FAMILIES,
            CONTROLLED_TRUE_ADAPTER_PARAM_KEYS,
            required_true_adapter_param_keys_for_family,
        )
        from moat_ovha_torch.train.multimodal_protocol import ALLOWED_V1_ADAPTER_PARAMS

        self.assertEqual(set(CONTROLLED_FAMILY_ACTIVE_OPERATOR), set(CONTROLLED_MULTIMODAL_FAMILIES))
        self.assertEqual(CONTROLLED_FAMILY_ACTIVE_OPERATOR["tleo_local_evidence"], "TLEO")
        self.assertEqual(CONTROLLED_FAMILY_ACTIVE_OPERATOR["spo_global_prototype"], "SPO")
        self.assertEqual(CONTROLLED_FAMILY_ACTIVE_OPERATOR["lrio_low_rank_interaction"], "LRIO")
        self.assertEqual(CONTROLLED_FAMILY_ACTIVE_OPERATOR["cato_alignment_transport"], "CATO")
        self.assertEqual(CONTROLLED_FAMILY_ACTIVE_OPERATOR["rceo_reliability_corruption"], "LRIO")
        self.assertEqual(CONTROLLED_FAMILY_ACTIVE_OPERATOR["mixed_relation_operator"], "mixed")
        self.assertEqual(CONTROLLED_TRUE_ADAPTER_PARAM_KEYS, ALLOWED_V1_ADAPTER_PARAMS)

        for family in CONTROLLED_MULTIMODAL_FAMILIES:
            expected_operator = CONTROLLED_FAMILY_ACTIVE_OPERATOR[family]
            with self.subTest(family=family):
                expected = (
                    CONTROLLED_TRUE_ADAPTER_PARAM_KEYS
                    if expected_operator == "mixed"
                    else {expected_operator: CONTROLLED_TRUE_ADAPTER_PARAM_KEYS[expected_operator]}
                )
                self.assertEqual(required_true_adapter_param_keys_for_family(family), expected)

    def test_public_dataset_adapters_expose_formal_cache_shard_paths(self):
        from moat_ovha_torch.data.multimodal.adapters import (
            CMUMOSEIAdapter,
            CMUMOSIAdapter,
            IEMOCAPAdapter,
            Flickr30kEntitiesAdapter,
            MELDAdapter,
            RefCOCOAdapter,
            VisualGenomeAdapter,
        )

        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp)
            for adapter in (RefCOCOAdapter(), Flickr30kEntitiesAdapter(), VisualGenomeAdapter()):
                with self.subTest(adapter=adapter.name):
                    fields = adapter.extract_token_fields({"cache_root": cache_root}, "train")
                    self.assertEqual(fields["text"].x_path, cache_root / "token_fields" / "text_train.npy")
                    self.assertEqual(fields["text"].pos_path, cache_root / "positions" / "text_pos_train.npy")
                    self.assertEqual(fields["text"].mask_path, cache_root / "masks" / "text_mask_train.npy")
                    self.assertEqual(fields["region"].x_path, cache_root / "token_fields" / "region_train.npy")
                    self.assertEqual(fields["region"].pos_path, cache_root / "positions" / "region_pos_train.npy")
                    self.assertEqual(fields["region"].mask_path, cache_root / "masks" / "region_mask_train.npy")
                    supervision = adapter.extract_supervision({"cache_root": cache_root}, "train")
                    self.assertEqual(
                        supervision.alignment_pairs_path,
                        cache_root / "supervision" / "alignment_pairs_train.parquet",
                    )
                    self.assertEqual(supervision.bbox_targets_path, cache_root / "supervision" / "bbox_targets_train.npy")
                    self.assertEqual(supervision.region_targets_path, cache_root / "supervision" / "region_targets_train.npy")

            for adapter in (CMUMOSEIAdapter(), CMUMOSIAdapter(), MELDAdapter(), IEMOCAPAdapter()):
                with self.subTest(adapter=adapter.name):
                    fields = adapter.extract_token_fields({"cache_root": cache_root}, "train")
                    for modality in ("text", "audio", "vision"):
                        self.assertEqual(
                            fields[modality].x_path,
                            cache_root / "token_fields" / f"{modality}_train.npy",
                        )
                        self.assertEqual(
                            fields[modality].pos_path,
                            cache_root / "positions" / f"{modality}_pos_train.npy",
                        )
                        self.assertEqual(
                            fields[modality].mask_path,
                            cache_root / "masks" / f"{modality}_mask_train.npy",
                        )
                    supervision = adapter.extract_supervision({"cache_root": cache_root}, "train")
                    self.assertEqual(supervision.task_label_path, cache_root / "supervision" / "task_labels_train.npy")
                    self.assertEqual(
                        supervision.modality_missing_mask_path,
                        cache_root / "supervision" / "missing_modality_mask_train.npy",
                    )
                    self.assertEqual(
                        supervision.corruption_metadata_path,
                        cache_root / "supervision" / "corruption_train.parquet",
                    )

    def test_public_dataset_adapters_use_dataset_specific_raw_manifests(self):
        from moat_ovha_torch.data.multimodal.adapters import (
            CMUMOSEIAdapter,
            CMUMOSIAdapter,
            Flickr30kEntitiesAdapter,
            IEMOCAPAdapter,
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

            sentiment_manifest_expectations = {
                "cmu_mosei": (
                    CMUMOSEIAdapter(),
                    (
                        "labels/sentiment.npy",
                        "labels/emotion.npy",
                        "metadata/utterances.json",
                        "metadata/dialogues.json",
                        "metadata/feature_versions.json",
                        "metadata/missing_modality_mask.npy",
                        "metadata/corruption_transforms.json",
                    ),
                ),
                "cmu_mosi": (
                    CMUMOSIAdapter(),
                    (
                        "labels/sentiment.npy",
                        "labels/emotion.npy",
                        "metadata/utterances.json",
                        "metadata/dialogues.json",
                        "metadata/feature_versions.json",
                        "metadata/missing_modality_mask.npy",
                        "metadata/corruption_transforms.json",
                    ),
                ),
                "meld": (
                    MELDAdapter(),
                    (
                        "labels/emotion.npy",
                        "metadata/dialogues.json",
                        "metadata/feature_versions.json",
                        "metadata/missing_modality_mask.npy",
                        "metadata/corruption_transforms.json",
                    ),
                ),
                "iemocap": (
                    IEMOCAPAdapter(),
                    (
                        "labels/sentiment.npy",
                        "labels/emotion.npy",
                        "metadata/sessions.json",
                        "metadata/speakers.json",
                        "metadata/dialogues.json",
                        "metadata/feature_versions.json",
                        "metadata/missing_modality_mask.npy",
                        "metadata/corruption_transforms.json",
                    ),
                ),
            }
            for dataset_name, (adapter, required_paths) in sentiment_manifest_expectations.items():
                with self.subTest(dataset_name=dataset_name):
                    with self.assertRaises(MissingMultimodalDataError) as error:
                        adapter.discover_raw(raw_root)
                    message = str(error.exception)
                    for required_path in required_paths:
                        self.assertIn(required_path, message)

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

    def test_oracle_matrix_uses_learned_and_true_router_candidate_combinations(self):
        from moat_ovha_torch.eval.multimodal_oracle import evaluate_oracle_matrix

        batch = _oracle_batch(
            true_candidate_values=_ArrayTensor([[[[0.0], [2.0], [3.0], [4.0]]]]),
            true_router_weights=_ArrayTensor([[[0.0, 0.0, 0.0, 1.0]]]),
            target_y=_ArrayTensor([[[4.0]]]),
        )
        report = evaluate_oracle_matrix(
            batch,
            learned_candidate_values=_ArrayTensor([[[[1.0], [2.0], [3.0], [3.0]]]]),
            learned_router_weights=_ArrayTensor([[[1.0, 0.0, 0.0, 0.0]]]),
        )

        self.assertEqual(float(report["true_true"]["mse"]), 0.0)
        self.assertEqual(float(report["true_learned"]["mse"]), 1.0)
        self.assertEqual(float(report["learned_true"]["mse"]), 16.0)
        self.assertEqual(float(report["learned_learned"]["mse"]), 9.0)
        self.assertIn("learned router + learned adapter/candidates", report["learned_learned"]["purpose"])
        self.assertNotIn("placeholder", report["learned_learned"]["purpose"])

    def test_stackability_guard_source_checks_candidate_feature_batch_query_axes(self):
        operator_source = (ROOT / "moat_ovha_torch" / "models" / "multimodal" / "operator_bank.py").read_text()

        self.assertIn("out.feature.shape", operator_source)
        self.assertIn("(batch_size, q_count)", operator_source)
        self.assertIn("Candidate {name} feature returned", operator_source)

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

    def test_typed_batch_rejects_normalized_hidden_metadata_aliases(self):
        from moat_ovha_torch.data.multimodal.typed_batch import (
            TokenField,
            assert_no_multimodal_metadata_leakage,
            validate_multimodal_batch_contract,
        )

        batch = _static_batch(
            fields={
                "true-active-operator": TokenField(
                    modality="true-active-operator",
                    x=_Shape((2, 6, 4)),
                    pos=_Shape((2, 6, 2)),
                    mask=_Shape((2, 6)),
                    attrs={"corruption strength": _Shape((2, 6, 1))},
                )
            },
        )

        report = validate_multimodal_batch_contract(batch)

        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn(
            "fields.true-active-operator must not expose controlled or hidden metadata as model input",
            joined,
        )
        self.assertIn(
            "fields.true-active-operator.modality must not expose controlled or hidden metadata as model input",
            joined,
        )
        self.assertIn(
            "fields.true-active-operator.attrs must not expose controlled or hidden metadata as model input: "
            "corruption strength",
            joined,
        )
        with self.assertRaisesRegex(ValueError, "true-active-operator"):
            assert_no_multimodal_metadata_leakage({"fields": {"true-active-operator": object()}})

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

    def test_typed_batch_rejects_blank_provenance_and_version_strings(self):
        from moat_ovha_torch.data.multimodal.typed_batch import ProvenanceBank, validate_multimodal_batch_contract

        batch = _static_batch(
            provenance=ProvenanceBank(
                source_id=["sample-0", "   "],
                original_split=["train", "train"],
                raw_ref=["shape", "\t"],
                license_tag=["test", "\n"],
                preprocessing_version="   ",
                feature_extractor_version={"text": "   "},
                pseudo_label_version={"teacher": "\t"},
            )
        )

        report = validate_multimodal_batch_contract(batch)

        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn("provenance.source_id entries must be non-empty strings", joined)
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

    def test_typed_batch_rejects_unnormalized_episode_identity_and_source_provenance(self):
        from moat_ovha_torch.data.multimodal.typed_batch import ProvenanceBank, validate_multimodal_batch_contract

        batch = _static_batch(
            task_type=" phrase_region_grounding ",
            split=" train ",
            source_dataset=" shape-test ",
            provenance=ProvenanceBank(
                source_id=["sample-0 ", " sample-1"],
                original_split=[" train ", " train "],
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
        self.assertIn("task_type must be a non-empty normalized string", joined)
        self.assertIn("split must be a non-empty normalized string", joined)
        self.assertIn("source_dataset must be a non-empty normalized string", joined)
        self.assertIn("provenance.source_id entries must be non-empty normalized strings", joined)
        self.assertIn("provenance.original_split entries must be non-empty normalized strings", joined)


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

    def test_multimodal_ovha_training_only_router_weight_override_controls_effective_weights(self):
        import torch

        from moat_ovha_torch.data.multimodal.adapters.controlled_synthetic import (
            ControlledSyntheticMultimodalAdapter,
        )
        from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA

        batch = ControlledSyntheticMultimodalAdapter(seed=321, output_dim=3).sample_batch(
            family="mixed_relation_operator",
            batch_size=2,
            query_count=5,
            device="cpu",
        )
        model = MultimodalOVHA(
            field_dims={"text": 4, "region": 4, "audio": 4},
            query_dim=4,
            output_dim=3,
            d_model=8,
        )
        override = batch.hidden["true_router_weights"]

        output = model(batch, router_weight_override=override)

        expected = (override.unsqueeze(-1) * output.candidate_values).sum(dim=-2)
        self.assertTrue(torch.allclose(output.router_weights, override, atol=1e-6))
        self.assertTrue(torch.allclose(output.y_hat, expected, atol=1e-6))
        self.assertEqual(
            output.diagnostics["router_override"],
            {"applied": True, "source": "training_only_supplied_weights"},
        )

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

    def test_controlled_mixed_relation_is_query_conditioned_without_hidden_input_leakage(self):
        import torch

        from moat_ovha_torch.data.multimodal.adapters.controlled_synthetic import (
            CONTROLLED_OPERATOR_ORDER,
            ControlledSyntheticMultimodalAdapter,
        )

        adapter = ControlledSyntheticMultimodalAdapter(seed=123, field_dim=4, output_dim=3)
        batch = adapter.sample_batch(
            family="mixed_relation_operator",
            batch_size=2,
            query_count=4,
            device="cpu",
        )

        relation_code = batch.query.x[..., : len(CONTROLLED_OPERATOR_ORDER)]
        self.assertEqual(tuple(relation_code.shape), (2, 4, 4))
        self.assertTrue(torch.equal(relation_code.argmax(dim=-1), batch.hidden["true_active_operator"]))
        self.assertFalse(torch.equal(batch.query.query_type, batch.hidden["true_active_operator"]))
        self.assertNotIn("hidden", batch.model_inputs())
        self.assertNotIn("true_active_operator", str(batch.model_inputs()))

    def test_router_memory_logits_have_candidate_specific_query_path(self):
        import torch

        from moat_ovha_torch.models.multimodal.evidence import MultimodalEvidenceBank
        from moat_ovha_torch.models.multimodal.router import MultimodalRelationRouter

        router = MultimodalRelationRouter(d_model=4)
        self.assertTrue(hasattr(router, "query_candidate_head"))
        with torch.no_grad():
            router.query_candidate_head.weight.copy_(torch.eye(4))
            router.query_candidate_head.bias.zero_()
            router.memory_head.weight.zero_()
            router.memory_head.bias.zero_()
        query_features = torch.eye(4).view(1, 4, 4)
        evidence = MultimodalEvidenceBank(
            query_features=query_features,
            global_features=torch.zeros(1, 4),
            local_features=query_features,
            prototype_features=query_features,
            low_rank_features=query_features,
            alignment_features=query_features,
            candidate_evidence_logits=torch.zeros(1, 4, 4),
            local_entropy=torch.zeros(()),
            alignment_entropy=torch.zeros(()),
            field_features={"text": torch.zeros(1, 4, 4)},
            diagnostics={},
        )
        memory_bank = {
            name: torch.zeros(1, 2, 4)
            for name in ("TLEO", "SPO", "LRIO", "CATO")
        }

        output = router(memory_bank, evidence, reliability=None)

        self.assertTrue(torch.equal(output.logit_parts["memory"].argmax(dim=-1), torch.arange(4).view(1, 4)))

    def test_evidence_encoder_uses_explicit_query_relation_code_as_router_prior(self):
        import torch

        from moat_ovha_torch.data.multimodal.adapters.controlled_synthetic import ControlledSyntheticMultimodalAdapter
        from moat_ovha_torch.models.multimodal.evidence import MultimodalEvidenceEncoder

        batch = ControlledSyntheticMultimodalAdapter(seed=123, field_dim=4, output_dim=3).sample_batch(
            family="mixed_relation_operator",
            batch_size=2,
            query_count=4,
            device="cpu",
        )
        encoder = MultimodalEvidenceEncoder(
            field_dims={"text": 4, "region": 4, "audio": 4},
            query_dim=4,
            d_model=8,
        )

        evidence = encoder(batch)

        self.assertTrue(
            torch.equal(
                evidence.candidate_evidence_logits.argmax(dim=-1),
                batch.hidden["true_active_operator"],
            )
        )
        self.assertNotIn("true_active_operator", str(batch.model_inputs()))

    def test_evidence_encoder_uses_controlled_family_relation_prior_without_hidden_input(self):
        import torch

        from moat_ovha_torch.data.multimodal.adapters.controlled_synthetic import (
            CONTROLLED_FAMILY_ACTIVE_OPERATOR,
            CONTROLLED_OPERATOR_ORDER,
            ControlledSyntheticMultimodalAdapter,
        )
        from moat_ovha_torch.models.multimodal.evidence import MultimodalEvidenceEncoder

        encoder = MultimodalEvidenceEncoder(
            field_dims={"text": 4, "region": 4, "audio": 4},
            query_dim=4,
            d_model=8,
        )
        adapter = ControlledSyntheticMultimodalAdapter(seed=123, field_dim=4, output_dim=3)
        for family in ("tleo_local_evidence", "spo_global_prototype", "lrio_low_rank_interaction", "cato_alignment_transport"):
            with self.subTest(family=family):
                batch = adapter.sample_batch(family=family, batch_size=2, query_count=4, device="cpu")
                evidence = encoder(batch)
                expected = CONTROLLED_OPERATOR_ORDER.index(CONTROLLED_FAMILY_ACTIVE_OPERATOR[family])

                self.assertTrue(torch.equal(evidence.candidate_evidence_logits.argmax(dim=-1), torch.full((2, 4), expected)))
                self.assertNotIn("true_active_operator", str(batch.model_inputs()))

    def test_controlled_spo_and_lrio_have_non_degenerate_adapter_truth(self):
        import torch

        from moat_ovha_torch.data.multimodal.adapters.controlled_synthetic import ControlledSyntheticMultimodalAdapter

        adapter = ControlledSyntheticMultimodalAdapter(seed=123, field_dim=4, output_dim=3)
        spo_batch = adapter.sample_batch(
            family="spo_global_prototype",
            batch_size=2,
            query_count=4,
            device="cpu",
        )
        lrio_batch = adapter.sample_batch(
            family="lrio_low_rank_interaction",
            batch_size=2,
            query_count=4,
            device="cpu",
        )

        self.assertGreater(float(spo_batch.hidden["true_prototype_logits"].abs().mean()), 0.0)
        self.assertGreater(float(lrio_batch.hidden["true_rank_logits"].abs().mean()), 0.0)
        self.assertTrue(
            torch.equal(
                spo_batch.hidden["true_adapter_params"]["params_by_operator"]["SPO"]["prototype_logits_shift"],
                spo_batch.hidden["true_prototype_logits"],
            )
        )
        self.assertTrue(
            torch.equal(
                lrio_batch.hidden["true_adapter_params"]["params_by_operator"]["LRIO"]["rank_logits"],
                lrio_batch.hidden["true_rank_logits"],
            )
        )

    def test_controlled_cato_target_is_generated_from_true_alignment_pairs(self):
        import torch

        from moat_ovha_torch.data.multimodal.adapters.controlled_synthetic import (
            CONTROLLED_OPERATOR_ORDER,
            ControlledSyntheticMultimodalAdapter,
        )

        output_dim = 3
        batch = ControlledSyntheticMultimodalAdapter(seed=123, field_dim=4, output_dim=output_dim).sample_batch(
            family="cato_alignment_transport",
            batch_size=2,
            query_count=4,
            device="cpu",
        )
        cato_index = CONTROLLED_OPERATOR_ORDER.index("CATO")
        region_index = batch.hidden["true_alignment_pairs"][..., 1].unsqueeze(-1).expand(-1, -1, batch.fields["region"].x.shape[-1])
        aligned_region = torch.gather(batch.fields["region"].x, dim=1, index=region_index)
        scales = torch.linspace(0.5, 1.5, output_dim).view(1, 1, output_dim)
        expected_cato = aligned_region.mean(dim=-1, keepdim=True) * scales

        self.assertTrue(torch.allclose(batch.query.x, aligned_region, atol=1e-6))
        self.assertTrue(torch.allclose(batch.hidden["true_candidate_values"][..., cato_index, :], expected_cato))
        self.assertTrue(torch.allclose(batch.target_y, expected_cato))

    def test_rceo_reliability_prior_reinforces_evidence_operator_prior(self):
        import torch

        from moat_ovha_torch.data.multimodal.typed_batch import (
            MultimodalEpisodeBatch,
            ProvenanceBank,
            QueryField,
            SupervisionBank,
            TokenField,
        )
        from moat_ovha_torch.models.multimodal.evidence import MultimodalEvidenceBank
        from moat_ovha_torch.models.multimodal.reliability_prior import RCEOReliabilityPrior

        batch = MultimodalEpisodeBatch(
            fields={
                "text": TokenField(
                    "text",
                    torch.zeros(1, 2, 4),
                    torch.zeros(1, 2, 2),
                    torch.ones(1, 2, dtype=torch.bool),
                    quality=torch.ones(1, 2, 1),
                ),
                "region": TokenField(
                    "region",
                    torch.zeros(1, 2, 4),
                    torch.zeros(1, 2, 2),
                    torch.ones(1, 2, dtype=torch.bool),
                    quality=torch.ones(1, 2, 1),
                ),
            },
            query=QueryField(
                x=torch.zeros(1, 4, 4),
                pos=torch.zeros(1, 4, 2),
                query_type=torch.zeros(1, 4, dtype=torch.long),
                mask=torch.ones(1, 4, dtype=torch.bool),
            ),
            target_y=torch.zeros(1, 4, 1),
            target_mask=torch.ones(1, 4, dtype=torch.bool),
            task_type="controlled_relation_operator",
            split="train",
            source_dataset="controlled_multimodal",
            supervision=SupervisionBank(None, None, None, None, None, None, None, None, None, None, None),
            provenance=ProvenanceBank(["s0"], ["train"], ["raw"], ["synthetic"], "v", {}, {}),
            hidden=None,
        )
        evidence_logits = torch.nn.functional.one_hot(torch.arange(4).view(1, 4), num_classes=4).float() * 4.0
        evidence = MultimodalEvidenceBank(
            query_features=torch.zeros(1, 4, 8),
            global_features=torch.zeros(1, 8),
            local_features=torch.zeros(1, 4, 8),
            prototype_features=torch.zeros(1, 4, 8),
            low_rank_features=torch.zeros(1, 4, 8),
            alignment_features=torch.zeros(1, 4, 8),
            candidate_evidence_logits=evidence_logits,
            local_entropy=torch.zeros(()),
            alignment_entropy=torch.zeros(()),
            field_features={},
            diagnostics={},
        )
        prior = RCEOReliabilityPrior(d_model=8)

        reliability = prior(batch, evidence)

        self.assertTrue(torch.equal(reliability.operator_logit_bias.argmax(dim=-1), torch.arange(4).view(1, 4)))

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

    def test_robustness_transforms_drive_rceo_quality_signal(self):
        import torch

        from moat_ovha_torch.data.multimodal.transforms import add_gaussian_corruption, apply_modality_dropout
        from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA

        batch = _batch(torch)
        dropped = apply_modality_dropout(batch, "region")
        corrupted = add_gaussian_corruption(batch, "region", torch.full_like(batch.fields["region"].x, 0.25))
        model = MultimodalOVHA(
            field_dims={"text": 4, "region": 4},
            query_dim=4,
            output_dim=3,
            d_model=8,
        )
        clean_reliability = model(batch).reliability_prior.modality_reliability
        dropped_reliability = model(dropped).reliability_prior.modality_reliability
        corrupted_reliability = model(corrupted).reliability_prior.modality_reliability

        self.assertTrue(torch.allclose(dropped.fields["region"].quality, torch.zeros_like(dropped.fields["region"].quality)))
        self.assertEqual(int(dropped.fields["region"].mask.sum()), 0)
        self.assertTrue(bool(dropped.supervision.modality_missing_mask[:, 0].all()))
        self.assertTrue(torch.all(corrupted.fields["region"].quality < batch.fields["region"].quality))
        self.assertAlmostEqual(float(corrupted.fields["region"].quality.mean()), 0.75, places=6)
        self.assertIn("gaussian_noise_strength", corrupted.supervision.corruption_metadata)
        self.assertLess(float(dropped_reliability[:, 1].mean()), float(clean_reliability[:, 1].mean()))
        self.assertLess(float(corrupted_reliability[:, 1].mean()), float(clean_reliability[:, 1].mean()))


class _ArrayTensor:
    def __init__(self, value):
        import numpy as np

        self.value = np.asarray(value, dtype=float)

    @property
    def shape(self):
        return self.value.shape

    def unsqueeze(self, dim):
        import numpy as np

        return _ArrayTensor(np.expand_dims(self.value, axis=dim))

    def sum(self, dim=None):
        return _ArrayTensor(self.value.sum(axis=dim))

    def square(self):
        return _ArrayTensor(self.value * self.value)

    def mean(self):
        return _ArrayTensor(self.value.mean())

    def item(self):
        return float(self.value.item())

    def detach(self):
        return self

    def reshape(self, *shape):
        return _ArrayTensor(self.value.reshape(*shape))

    def __getitem__(self, index):
        return _ArrayTensor(self.value[index])

    def __mul__(self, other):
        if isinstance(other, _ArrayTensor):
            other = other.value
        return _ArrayTensor(self.value * other)

    def __sub__(self, other):
        if isinstance(other, _ArrayTensor):
            other = other.value
        return _ArrayTensor(self.value - other)

    def __float__(self):
        return self.item()


def _oracle_batch(true_candidate_values, true_router_weights, target_y):
    class Batch:
        pass

    batch = Batch()
    batch.hidden = {
        "true_candidate_values": true_candidate_values,
        "true_router_weights": true_router_weights,
    }
    batch.target_y = target_y
    batch.task_type = "mixed_relation_operator"
    return batch


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


def _write_refcoco_raw_fixture(raw_root: Path) -> None:
    import numpy as np

    for folder in ("annotations", "features", "provenance"):
        (raw_root / folder).mkdir(parents=True, exist_ok=True)
    split_source_ids = {
        "train": ["ref-train-1"],
        "val": ["ref-val-1"],
        "test": ["ref-test-1"],
    }
    (raw_root / "splits.json").write_text(json.dumps(split_source_ids, sort_keys=True) + "\n")
    records = []
    for split, source_ids in split_source_ids.items():
        for source_id in source_ids:
            records.append(
                {
                    "source_id": source_id,
                    "split": split,
                    "original_split": split,
                    "raw_ref": f"refcoco://{source_id}",
                    "license_tag": "fixture-license",
                    "preprocessing_version": "fixture-preprocess-v1",
                    "image_id": f"image-{source_id}",
                    "caption_id": f"caption-{source_id}",
                    "phrase_span": {"start": 0, "end": 2},
                    "region_box": [0.0, 0.0, 1.0, 1.0],
                    "target_region_index": 1,
                    "candidate_region_source": "fixture_regions",
                    "box_coordinate_convention": "xyxy_normalized",
                }
            )
    (raw_root / "annotations" / "refs.json").write_text(json.dumps({"records": records}, sort_keys=True) + "\n")
    (raw_root / "annotations" / "instances.json").write_text(json.dumps({"records": records}, sort_keys=True) + "\n")
    (raw_root / "provenance" / "failed_samples.jsonl").write_text(
        json.dumps({"source_id": "ref-train-failed", "split": "train", "reason": "download_failed"}, sort_keys=True)
        + "\n"
    )
    np.save(raw_root / "features" / "text_features.npy", np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5))
    np.save(raw_root / "features" / "region_features.npy", np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4))


def _write_cmu_mosei_raw_fixture(raw_root: Path) -> None:
    import numpy as np

    for folder in ("features", "labels", "metadata", "provenance"):
        (raw_root / folder).mkdir(parents=True, exist_ok=True)
    split_source_ids = {
        "train": ["mosei-train-1"],
        "val": ["mosei-val-1"],
        "test": ["mosei-test-1"],
    }
    (raw_root / "splits.json").write_text(json.dumps(split_source_ids, sort_keys=True) + "\n")
    records = []
    for split, source_ids in split_source_ids.items():
        for source_id in source_ids:
            records.append(
                {
                    "source_id": source_id,
                    "split": split,
                    "original_split": split,
                    "raw_ref": f"cmu-mosei://{source_id}",
                    "license_tag": "fixture-license",
                    "preprocessing_version": "fixture-sentiment-preprocess-v1",
                    "utterance_id": source_id,
                    "dialogue_id": f"dialogue-{split}",
                    "speaker_id": f"speaker-{split}",
                    "transcript_source": "official_transcript",
                }
            )
    (raw_root / "metadata" / "utterances.json").write_text(json.dumps({"records": records}, sort_keys=True) + "\n")
    (raw_root / "metadata" / "dialogues.json").write_text(json.dumps({"records": []}, sort_keys=True) + "\n")
    (raw_root / "metadata" / "feature_versions.json").write_text(
        json.dumps(
            {
                "text": "fixture-text-v1",
                "audio": "fixture-audio-v1",
                "vision": "fixture-vision-v1",
            },
            sort_keys=True,
        )
        + "\n"
    )
    np.save(raw_root / "metadata" / "missing_modality_mask.npy", np.zeros((3, 3), dtype=bool))
    (raw_root / "metadata" / "corruption_transforms.json").write_text(
        json.dumps({"version": "fixture-corruption-v1"}, sort_keys=True) + "\n"
    )
    (raw_root / "provenance" / "failed_samples.jsonl").write_text(
        json.dumps({"source_id": "mosei-train-failed", "split": "train", "reason": "audio_decode_failed"}, sort_keys=True)
        + "\n"
    )
    np.save(raw_root / "features" / "text_features.npy", np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5))
    np.save(raw_root / "features" / "audio_features.npy", np.arange(3 * 6 * 3, dtype=np.float32).reshape(3, 6, 3))
    np.save(raw_root / "features" / "visual_features.npy", np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4))
    np.save(raw_root / "labels" / "sentiment.npy", np.array([[-1.0], [0.0], [1.0]], dtype=np.float32))
    np.save(raw_root / "labels" / "emotion.npy", np.eye(7, dtype=np.float32)[:3])


def _write_cmu_sequence_json(path: Path, features_by_source_id: dict[str, object]) -> None:
    payload = {
        "data": {
            source_id: {
                "features": values.tolist(),
                "intervals": [[0.0, 1.0] for _ in range(int(values.shape[0]))],
            }
            for source_id, values in features_by_source_id.items()
        }
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _write_meld_raw_fixture(raw_root: Path) -> None:
    import numpy as np

    for folder in ("features", "labels", "metadata"):
        (raw_root / folder).mkdir(parents=True, exist_ok=True)
    split_source_ids = {
        "train": ["meld-train-1"],
        "val": ["meld-val-1"],
        "test": ["meld-test-1"],
    }
    (raw_root / "splits.json").write_text(json.dumps(split_source_ids, sort_keys=True) + "\n")
    speaker_by_split = {"train": "Monica", "val": "Chandler", "test": "Rachel"}
    records = []
    for split, source_ids in split_source_ids.items():
        for source_id in source_ids:
            records.append(
                {
                    "source_id": source_id,
                    "split": split,
                    "original_split": split,
                    "raw_ref": f"meld://friends/{source_id}",
                    "license_tag": "fixture-license",
                    "preprocessing_version": "fixture-meld-preprocess-v1",
                    "utterance_id": source_id,
                    "dialogue_id": f"dialogue-{split}",
                    "speaker_id": speaker_by_split[split],
                    "transcript_source": f"MELD {split}_sent_emo.csv",
                    "utterance_text": f"fixture utterance for {split}",
                }
            )
    (raw_root / "metadata" / "dialogues.json").write_text(json.dumps({"records": records}, sort_keys=True) + "\n")
    (raw_root / "metadata" / "feature_versions.json").write_text(
        json.dumps(
            {
                "text": "fixture-meld-text-v1",
                "audio": "fixture-meld-audio-v1",
                "visual": "fixture-meld-visual-v1",
            },
            sort_keys=True,
        )
        + "\n"
    )
    np.save(raw_root / "metadata" / "missing_modality_mask.npy", np.zeros((3, 3), dtype=bool))
    (raw_root / "metadata" / "corruption_transforms.json").write_text(
        json.dumps({"version": "fixture-meld-corruption-v1"}, sort_keys=True) + "\n"
    )
    np.save(raw_root / "features" / "text_features.npy", np.arange(3 * 5 * 6, dtype=np.float32).reshape(3, 5, 6))
    np.save(raw_root / "features" / "audio_features.npy", np.arange(3 * 4 * 3, dtype=np.float32).reshape(3, 4, 3))
    np.save(raw_root / "features" / "visual_features.npy", np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4))
    np.save(raw_root / "labels" / "emotion.npy", np.eye(7, dtype=np.float32)[:3])


def _write_meld_csv_fixture(meld_root: Path) -> None:
    meld_root.mkdir(parents=True, exist_ok=True)
    rows_by_name = {
        "train_sent_emo.csv": ("Monica", "neutral", "train fixture utterance"),
        "dev_sent_emo.csv": ("Chandler", "joy", "validation fixture utterance"),
        "test_sent_emo.csv": ("Rachel", "sadness", "test fixture utterance"),
    }
    for name, (speaker, emotion, utterance) in rows_by_name.items():
        (meld_root / name).write_text(
            "\n".join(
                [
                    "Sr No.,Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID,Season,Episode,StartTime,EndTime",
                    f"1,{utterance},{speaker},{emotion},neutral,1,1,1,1,00:00:00,00:00:01",
                ]
            )
            + "\n"
        )


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
