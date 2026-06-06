import argparse
import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
ROOT = Path(__file__).resolve().parents[1]


class RefCOCOStrictAuditPlanContracts(unittest.TestCase):
    def test_refcoco_standard_splits_preserve_testa_testb(self):
        from scripts.multimodal.align_refcoco_stage_features import _ordered_source_ids
        from scripts.multimodal.build_refcoco_stage_records import _normalize_split
        from scripts.multimodal.stage_refcoco_raw import _ordered_source_ids as _stage_ordered_source_ids

        self.assertEqual(_normalize_split("testA"), "testA")
        self.assertEqual(_normalize_split("test_a"), "testA")
        self.assertEqual(_normalize_split("testB"), "testB")
        self.assertEqual(_normalize_split("test_b"), "testB")
        self.assertEqual(_normalize_split("test"), "test")

        splits = {"train": ["tr"], "val": ["va"], "testA": ["ta"], "testB": ["tb"]}
        self.assertEqual(_ordered_source_ids(splits), ["tr", "va", "ta", "tb"])
        self.assertEqual(_stage_ordered_source_ids(splits), ["tr", "va", "ta", "tb"])

    def test_candidate_regions_are_deterministically_balanced_and_record_seed(self):
        from scripts.multimodal.build_refcoco_stage_records import (
            _candidate_regions_balanced,
            _record_for_sentence,
            _target_slot_histogram_by_valid_count,
        )

        base = _stage_base()
        sentence = {"sent_id": 9, "tokens": ["left", "person"]}
        slots = []
        for index in range(64):
            source_id = f"refcoco::sample::{index}"
            _, ann_ids, target_slot, seed = _candidate_regions_balanced(
                base,
                max_candidate_regions=4,
                source_id=source_id,
            )
            self.assertEqual(len(ann_ids), 4)
            self.assertEqual(ann_ids[target_slot], base["ann_id"])
            self.assertEqual(
                _candidate_regions_balanced(base, max_candidate_regions=4, source_id=source_id),
                _candidate_regions_balanced(base, max_candidate_regions=4, source_id=source_id),
            )
            self.assertIsInstance(seed, int)
            slots.append(target_slot)

        counts = np.bincount(np.asarray(slots), minlength=4)
        self.assertLessEqual(int(counts.max() - counts.min()), 16)

        record = _record_for_sentence(
            "refcoco",
            base,
            sentence,
            "train",
            candidate_region_source="coco_gt_box",
            box_coordinate_convention="xyxy_normalized",
            max_candidate_regions=4,
        )
        self.assertIn("candidate_permutation_seed", record)
        self.assertEqual(record["candidate_region_annotation_ids"][record["target_region_index"]], base["ann_id"])

        histogram = _target_slot_histogram_by_valid_count(
            [
                {
                    "source_id": f"sample-{index}",
                    "candidate_region_annotation_ids": [10, 20, 30, 40],
                    "target_region_index": slot,
                }
                for index, slot in enumerate(slots)
            ]
        )
        self.assertIn("4", histogram["by_valid_count"])
        self.assertEqual(histogram["by_valid_count"]["4"]["sample_count"], len(slots))
        self.assertLessEqual(histogram["by_valid_count"]["4"]["max_deviation"], 16)

    def test_validate_refcoco_candidate_order_rejects_old_sorted_records(self):
        from argparse import Namespace

        from scripts.multimodal.validate_refcoco_candidate_order import validate_refcoco_candidate_order

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "refs.json"
            records.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "source_id": "legacy",
                                "candidate_region_annotation_ids": [10, 20, 30],
                                "target_region_index": 1,
                            }
                        ]
                    }
                )
                + "\n"
            )

            payload = validate_refcoco_candidate_order(Namespace(records=records, fail_on_sorted=True, max_sorted_fraction=0.95))

        self.assertFalse(payload["ok"])
        self.assertIn("missing candidate_permutation_seed", "\n".join(payload["errors"]))
        self.assertIn("sorted candidate_region_annotation_ids", "\n".join(payload["errors"]))

    def test_validate_refcoco_candidate_order_accepts_seeded_permuted_records(self):
        from argparse import Namespace

        from scripts.multimodal.validate_refcoco_candidate_order import validate_refcoco_candidate_order

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "refs.json"
            records.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "source_id": "balanced",
                                "candidate_region_annotation_ids": [30, 20, 10],
                                "target_region_index": 1,
                                "candidate_permutation_seed": 123,
                            }
                        ]
                    }
                )
                + "\n"
            )

            payload = validate_refcoco_candidate_order(Namespace(records=records, fail_on_sorted=True, max_sorted_fraction=0.95))

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["sorted_fraction"], 0.0)

    def test_refcoco_cache_writes_region_geometry_positions_and_candidate_boxes(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch is required for public batch loading checks")
        import torch

        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout
        from moat_ovha_torch.data.multimodal.adapters.base import RawDatasetManifest
        from moat_ovha_torch.data.multimodal.adapters.refcoco import RefCOCOAdapter
        from scripts.multimodal.run_public_smoke import _load_public_batch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            cache = root / "cache"
            _write_refcoco_raw(raw)
            manifest = RawDatasetManifest(
                dataset_name="refcoco",
                raw_root=raw,
                files={
                    "annotations/instances.json": raw / "annotations" / "instances.json",
                    "annotations/refs.json": raw / "annotations" / "refs.json",
                    "features/text_features.npy": raw / "features" / "text_features.npy",
                    "features/region_features.npy": raw / "features" / "region_features.npy",
                    "splits.json": raw / "splits.json",
                },
            )
            RefCOCOAdapter().write_cache(manifest, cache, "train", "v0.1")
            layout = MultimodalCacheLayout(cache, "refcoco", "v0.1")
            config = MultimodalExperimentConfig.from_mapping(_refcoco_config(cache))

            region_pos = np.load(layout.root / "positions" / "region_pos_train.npy")
            candidate_boxes = np.load(layout.root / "supervision" / "candidate_region_boxes_train.npy")
            histogram = json.loads((layout.root / "supervision" / "target_slot_histogram_by_valid_count_train.json").read_text())
            sample_record = json.loads((layout.root / "provenance" / "sample_records_train.jsonl").read_text().splitlines()[0])
            batch = _load_public_batch(layout, config, "train", torch.device("cpu"))

        self.assertEqual(region_pos.shape, (2, 4, 9))
        self.assertEqual(histogram["by_valid_count"]["3"]["sample_count"], 2)
        self.assertIn("candidate_permutation_seed", sample_record)
        expected_first_box = candidate_boxes[0, 0]
        x1, y1, x2, y2 = expected_first_box
        expected_pos = np.asarray(
            [x1, y1, x2, y2, 0.5 * (x1 + x2), 0.5 * (y1 + y2), x2 - x1, y2 - y1, (x2 - x1) * (y2 - y1)],
            dtype=np.float32,
        )
        self.assertTrue(np.allclose(region_pos[0, 0], expected_pos))
        self.assertIsNotNone(batch.supervision.candidate_region_boxes)
        self.assertTrue(torch.allclose(batch.fields["region"].pos, batch.supervision.candidate_region_boxes.new_tensor(region_pos)))
        self.assertEqual(batch.fields["region"].attrs["position_semantics"], "xyxy_cxcywh_area")

    def test_refcoco_candidate_box_upgrade_patches_existing_cache_without_rebuilding_features(self):
        from argparse import Namespace

        from moat_ovha_torch.data.multimodal.adapters.base import RawDatasetManifest
        from moat_ovha_torch.data.multimodal.adapters.refcoco import RefCOCOAdapter
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout
        from scripts.multimodal.upgrade_refcoco_candidate_boxes import upgrade_refcoco_candidate_boxes

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            cache = root / "cache"
            _write_refcoco_raw(raw)
            manifest = RawDatasetManifest(
                dataset_name="refcoco",
                raw_root=raw,
                files={
                    "annotations/instances.json": raw / "annotations" / "instances.json",
                    "annotations/refs.json": raw / "annotations" / "refs.json",
                    "features/text_features.npy": raw / "features" / "text_features.npy",
                    "features/region_features.npy": raw / "features" / "region_features.npy",
                    "splits.json": raw / "splits.json",
                },
            )
            RefCOCOAdapter().write_cache(manifest, cache, "train", "v0.1")
            layout = MultimodalCacheLayout(cache, "refcoco", "v0.1")
            candidate_path = layout.root / "supervision" / "candidate_region_boxes_train.npy"
            region_pos_path = layout.root / "positions" / "region_pos_train.npy"
            candidate_path.unlink()
            np.save(region_pos_path, np.zeros_like(np.load(region_pos_path)))

            payload = upgrade_refcoco_candidate_boxes(
                Namespace(raw_root=raw, cache_root=cache, dataset_name="refcoco", version="v0.1", splits=["train"])
            )

            self.assertTrue(payload["ok"])
            candidate_boxes = np.load(candidate_path)
            region_pos = np.load(region_pos_path)
            self.assertEqual(candidate_boxes.shape, (2, 4, 4))
            self.assertTrue(np.allclose(candidate_boxes[0, 1], [0.3, 0.0, 0.5, 0.2]))
            self.assertTrue(np.allclose(region_pos[0, 1, :4], candidate_boxes[0, 1]))
            self.assertTrue(validate_cache_layout(layout, splits=("train",)).ok)

    def test_region_task_uses_cross_entropy_iou_metrics_and_no_target_standardization(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch is required for loss checks")
        import torch

        from scripts.multimodal.run_public_main import _standardize_batch_targets, _target_standardizer
        from scripts.multimodal.run_public_smoke import _region_text_metrics, _task_loss

        batch = _region_batch(torch)
        prediction = torch.tensor([[[0.0, 3.0, -1.0]], [[4.0, 0.0, -1.0]]])

        self.assertAlmostEqual(float(_task_loss(prediction, batch)), float(torch.nn.functional.cross_entropy(prediction[:, 0, :], torch.tensor([1, 0]))), places=6)
        metrics = _region_text_metrics(prediction, batch)
        self.assertEqual(metrics["region_recall_at_1"], 1.0)
        self.assertEqual(metrics["region_recall_at_5"], 1.0)
        self.assertEqual(metrics["candidate_iou_at_0_5"], 1.0)
        self.assertGreater(metrics["mean_candidate_iou"], 0.99)

        mean, std = _target_standardizer(batch)
        standardized = _standardize_batch_targets(batch, mean, std)
        self.assertTrue(torch.equal(standardized.target_y, batch.target_y))
        self.assertIsNone(mean)
        self.assertIsNone(std)

    def test_region_public_main_raw_metric_uses_refcoco_accuracy_not_loss_as_score(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch is required for raw metric checks")
        import torch

        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
        from scripts.multimodal.run_public_main import _ovha_raw_metric_row
        from scripts.multimodal.run_public_smoke import _task_loss

        batch = _region_batch(torch)
        output = _minimal_output(torch, batch)
        config = MultimodalExperimentConfig.from_mapping(_refcoco_config(Path("/tmp/cache")))

        row = _ovha_raw_metric_row(
            config,
            batch,
            output,
            seed=1,
            training_steps=10,
            parameter_count=123,
            raw_metrics_path=Path("/tmp/raw_metrics.jsonl"),
            hardware={"accelerator": "cpu", "device": "cpu", "wall_clock_hours": 0.0},
            model_name="ovha_refcoco_prso_sro_tleo_cato_primary",
        )

        self.assertEqual(row["metric_name"], "acc_at_0_5")
        self.assertTrue(row["higher_is_better"])
        self.assertEqual(row["score"], 1.0)
        self.assertEqual(row["public_metrics"]["acc_at_0_5"], 1.0)
        self.assertAlmostEqual(row["task_loss"], float(_task_loss(output.y_hat, batch)), places=6)

    def test_region_grounding_metrics_are_canonical_and_fail_without_candidate_boxes(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch is required for metric checks")
        import torch

        from dataclasses import replace

        from moat_ovha_torch.eval.grounding_metrics import METRICS_SOURCE, grounding_candidate_metrics

        batch = _region_batch(torch)
        prediction = torch.tensor([[[0.0, 3.0, -1.0]], [[4.0, 0.0, -1.0]]])
        metrics = grounding_candidate_metrics(
            prediction,
            batch.supervision.region_targets,
            batch.supervision.candidate_region_boxes,
            batch.supervision.bbox_targets,
            batch.target_mask,
        )

        self.assertEqual(metrics["metrics_source"], METRICS_SOURCE)
        self.assertEqual(metrics["recall_at_1"], 1.0)
        self.assertEqual(metrics["acc_at_0_5"], 1.0)
        self.assertGreater(metrics["mean_iou"], 0.99)
        self.assertGreater(metrics["cross_entropy"], 0.0)
        self.assertEqual(metrics["mrr"], 1.0)

        missing_boxes = replace(batch.supervision, candidate_region_boxes=None)
        broken = replace(batch, supervision=missing_boxes)
        with self.assertRaisesRegex(ValueError, "candidate_region_boxes"):
            grounding_candidate_metrics(
                prediction,
                broken.supervision.region_targets,
                broken.supervision.candidate_region_boxes,
                broken.supervision.bbox_targets,
                broken.target_mask,
            )

    def test_region_task_disables_affine_regression_calibration(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch is required for calibration checks")
        import torch

        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
        from scripts.multimodal.run_public_main import _apply_task_calibration, _fit_task_calibrator

        batch = _region_batch(torch)
        config = replace(MultimodalExperimentConfig.from_mapping(_refcoco_config(Path("/tmp/cache"))), task_type="phrase_region_grounding")
        prediction = torch.tensor([[[0.0, 3.0, -1.0]], [[4.0, 0.0, -1.0]]])
        calibrator = _fit_task_calibrator(config, prediction, batch)

        self.assertEqual(calibrator["method"], "none")
        self.assertNotEqual(calibrator["objective"], "validation_mse_closed_form")
        self.assertTrue(torch.equal(_apply_task_calibration(prediction, calibrator), prediction))

    def test_region_task_does_not_duplicate_public_alignment_ce(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch is required for loss checks")
        import torch

        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
        from scripts.multimodal.run_public_smoke import _public_loss_components

        batch = _region_batch(torch)
        output = _minimal_output(torch, batch)
        config = MultimodalExperimentConfig.from_mapping(_refcoco_config(Path("/tmp/cache")))

        losses = _public_loss_components(output, batch, config)

        self.assertIn("task_loss", losses)
        self.assertNotIn("public_alignment_ce", losses)

    def test_rceo_clean_reliability_uses_modality_presence_not_padding_density(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch is required for RCEO checks")
        import torch

        from moat_ovha_torch.models.multimodal.evidence import MultimodalEvidenceBank
        from moat_ovha_torch.models.multimodal.reliability_prior import RCEOReliabilityPrior

        batch = _region_batch(torch).model_inputs()
        evidence = MultimodalEvidenceBank(
            query_features=torch.zeros(2, 1, 8),
            global_features=torch.zeros(2, 8),
            local_features=torch.zeros(2, 1, 8),
            prototype_features=torch.zeros(2, 1, 8),
            low_rank_features=torch.zeros(2, 1, 8),
            alignment_features=torch.zeros(2, 1, 8),
            candidate_evidence_logits=torch.zeros(2, 1, 4),
            local_entropy=torch.zeros(()),
            alignment_entropy=torch.zeros(()),
            field_features={name: torch.zeros(field.x.shape[0], field.x.shape[1], 8) for name, field in batch.fields.items()},
            diagnostics={},
        )

        prior = RCEOReliabilityPrior(d_model=8, candidate_names=("CATO", "TLEO"))(batch, evidence)

        self.assertTrue(torch.allclose(prior.modality_reliability, torch.ones_like(prior.modality_reliability)))
        self.assertLess(float(prior.diagnostics["candidate_valid_fraction"].detach()), 1.0)
        self.assertLess(float(prior.diagnostics["text_token_valid_fraction"].detach()), 1.0)

    def test_refcoco_operator_bank_exposes_direct_region_logit_operators_and_residual_composition(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch is required for operator checks")
        import torch

        from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA

        torch.manual_seed(7)
        batch = _region_batch(torch)
        model = MultimodalOVHA(
            field_dims={"text": 5, "region": 4},
            query_dim=5,
            output_dim=3,
            d_model=12,
            memory_tokens=2,
            candidate_names=("PRSO", "SRO", "TLEO", "CATO"),
            composition_mode="base_plus_residual",
            base_candidate="PRSO",
            residual_candidates=("SRO", "TLEO", "CATO"),
        )

        output = model(batch)

        self.assertEqual(tuple(output.y_hat.shape), (2, 1, 3))
        self.assertEqual(output.diagnostics["composition"]["mode"], "base_plus_residual")
        self.assertEqual(output.diagnostics["composition"]["base_candidate"], "PRSO")
        for name in ("PRSO", "SRO", "TLEO", "CATO"):
            self.assertEqual(tuple(output.candidate_outputs[name].value.shape), (2, 1, 3))
            self.assertIn("direct_region_logits", output.candidate_outputs[name].diagnostics)
        self.assertEqual(output.diagnostics["candidate_diagnostics"]["CATO"]["source_modality"], ("region",))
        self.assertEqual(output.diagnostics["candidate_diagnostics"]["TLEO"]["region_local_context"], True)


def _stage_base():
    annotations = [
        {"id": 10, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
        {"id": 20, "image_id": 1, "category_id": 1, "bbox": [20, 0, 10, 10]},
        {"id": 30, "image_id": 1, "category_id": 2, "bbox": [0, 20, 10, 10]},
        {"id": 40, "image_id": 1, "category_id": 3, "bbox": [20, 20, 10, 10]},
        {"id": 50, "image_id": 1, "category_id": 4, "bbox": [40, 20, 10, 10]},
    ]
    return {
        "ref_id": 5,
        "ann_id": 20,
        "image_id": 1,
        "bbox": [0.2, 0.0, 0.3, 0.1],
        "candidate_annotations": annotations,
        "image_width": 100,
        "image_height": 100,
    }


def _write_refcoco_raw(root: Path) -> None:
    (root / "annotations").mkdir(parents=True)
    (root / "features").mkdir(parents=True)
    records = [
        _record("a", "train", target=1),
        _record("b", "train", target=2),
    ]
    (root / "annotations" / "refs.json").write_text('{"records": [' + ",".join(record for record in records) + "]}\n")
    (root / "annotations" / "instances.json").write_text('{"images": [], "annotations": []}\n')
    (root / "splits.json").write_text('{"train": ["a", "b"]}\n')
    np.save(root / "features" / "text_features.npy", np.ones((2, 3, 5), dtype=np.float32))
    np.save(root / "features" / "region_features.npy", np.ones((2, 4, 4), dtype=np.float32))
    np.save(root / "features" / "text_mask.npy", np.asarray([[True, True, False], [True, False, False]]))
    np.save(root / "features" / "region_mask.npy", np.asarray([[True, True, True, False], [True, True, True, False]]))


def _record(source_id: str, split: str, *, target: int) -> str:
    payload = {
        "source_id": source_id,
        "split": split,
        "original_split": split,
        "raw_ref": f"ref://{source_id}",
        "license_tag": "unit",
        "preprocessing_version": "unit",
        "image_id": "image1",
        "caption_id": f"sent{source_id}",
        "phrase_span": {"start": 0, "end": 2},
        "region_box": [[0.0, 0.0, 0.2, 0.2], [0.3, 0.0, 0.5, 0.2], [0.0, 0.3, 0.2, 0.5]][target],
        "candidate_region_boxes": [
            [0.0, 0.0, 0.2, 0.2],
            [0.3, 0.0, 0.5, 0.2],
            [0.0, 0.3, 0.2, 0.5],
        ],
        "candidate_region_annotation_ids": [10, 20, 30],
        "target_region_index": target,
        "candidate_permutation_seed": 1000 + target,
        "candidate_region_source": "coco_gt_box",
        "box_coordinate_convention": "xyxy_normalized",
    }
    import json

    return json.dumps(payload, sort_keys=True)


def _refcoco_config(cache_root: Path) -> dict:
    return {
        "name": "unit_refcoco",
        "dataset_name": "refcoco",
        "task_type": "phrase_region_grounding",
        "cache_version": "v0.1",
        "cache_root": str(cache_root),
        "output_dir": str(cache_root / "out"),
        "seeds": [1, 2, 3],
        "training_stages": ["T0", "T5"],
        "candidate_names": ["PRSO", "SRO", "TLEO", "CATO"],
        "candidate_pool_names": ["PRSO", "SRO", "TLEO", "CATO"],
        "baseline_names": [
            "index_prior_only",
            "text_only",
            "region_only",
            "concat_fusion",
            "cato_only",
            "ovha_no_cato",
            "ovha_no_rceo",
            "ovha_no_evidence_router",
        ],
        "eval_splits": ["train"],
        "eval_episode_count": 2,
        "require_public_alignment_labels": True,
        "losses_by_stage": {"T0": ["cache_validation"], "T5": ["task_loss", "public_alignment_ce"]},
        "adapter_params_by_candidate": {
            "PRSO": ["alignment_temperature", "scale", "bias"],
            "SRO": ["scale", "bias"],
            "TLEO": ["lengthscale", "local_temperature", "scale", "bias"],
            "CATO": ["alignment_temperature", "transport_scale", "scale", "bias"],
        },
        "composition_mode": "base_plus_residual",
        "base_candidate": "PRSO",
        "residual_candidates": ["SRO", "TLEO", "CATO"],
    }


def _region_batch(torch):
    from moat_ovha_torch.data.multimodal.typed_batch import (
        MultimodalEpisodeBatch,
        ProvenanceBank,
        QueryField,
        SupervisionBank,
        TokenField,
    )

    candidate_boxes = torch.tensor(
        [
            [[0.0, 0.0, 0.2, 0.2], [0.3, 0.0, 0.5, 0.2], [0.0, 0.3, 0.2, 0.5]],
            [[0.0, 0.0, 0.2, 0.2], [0.3, 0.0, 0.5, 0.2], [0.0, 0.3, 0.2, 0.5]],
        ],
        dtype=torch.float32,
    )
    region_pos = torch.cat(
        [
            candidate_boxes,
            0.5 * (candidate_boxes[..., :2] + candidate_boxes[..., 2:]),
            (candidate_boxes[..., 2:] - candidate_boxes[..., :2]).clamp_min(0.0),
            ((candidate_boxes[..., 2] - candidate_boxes[..., 0]) * (candidate_boxes[..., 3] - candidate_boxes[..., 1])).unsqueeze(-1),
        ],
        dim=-1,
    )
    target_y = torch.tensor([[[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]]], dtype=torch.float32)
    return MultimodalEpisodeBatch(
        fields={
            "text": TokenField(
                "text",
                torch.randn(2, 3, 5),
                torch.arange(3, dtype=torch.float32).view(1, 3, 1).expand(2, -1, -1),
                torch.tensor([[True, True, False], [True, False, False]]),
            ),
            "region": TokenField(
                "region",
                torch.randn(2, 3, 4),
                region_pos,
                torch.tensor([[True, True, False], [True, True, True]]),
                attrs={"position_semantics": "xyxy_cxcywh_area"},
            ),
        },
        query=QueryField(
            x=torch.randn(2, 1, 5),
            pos=torch.zeros(2, 1, 1),
            query_type=torch.zeros(2, 1, dtype=torch.long),
            mask=torch.ones(2, 1, dtype=torch.bool),
        ),
        target_y=target_y,
        target_mask=torch.ones(2, 1, dtype=torch.bool),
        task_type="phrase_region_grounding",
        split="test",
        source_dataset="refcoco",
        supervision=SupervisionBank(
            task_label=target_y,
            alignment_pairs=None,
            alignment_weights=None,
            bbox_targets=torch.tensor([[0.3, 0.0, 0.5, 0.2], [0.0, 0.0, 0.2, 0.2]], dtype=torch.float32),
            candidate_region_boxes=candidate_boxes,
            region_targets=torch.tensor([[1], [0]], dtype=torch.long),
            timestamp_targets=None,
            modality_missing_mask=None,
            corruption_metadata=None,
            weak_labels=None,
            weak_label_confidence=None,
            pseudo_label_source=None,
        ),
        provenance=ProvenanceBank(
            source_id=["a", "b"],
            original_split=["test", "test"],
            raw_ref=["ref://a", "ref://b"],
            license_tag=["unit", "unit"],
            preprocessing_version="unit",
            feature_extractor_version={"text": "unit", "region": "unit"},
            pseudo_label_version={"version": "none"},
        ),
    )


def _minimal_output(torch, batch):
    from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHAOutput
    from moat_ovha_torch.models.multimodal.primitives.base import CandidateOutput

    prediction = torch.tensor([[[0.0, 3.0, -1.0]], [[4.0, 0.0, -1.0]]], dtype=batch.target_y.dtype)
    candidate = CandidateOutput(
        value=prediction,
        feature=torch.zeros(2, 1, 1),
        diagnostics={"direct_region_logits": True},
    )
    return MultimodalOVHAOutput(
        y_hat=prediction,
        candidate_values=prediction.unsqueeze(2),
        router_weights=torch.ones(2, 1, 1),
        router_logits=torch.ones(2, 1, 1),
        router_logit_parts={},
        candidate_outputs={"PRSO": candidate},
        reliability_prior=None,
        diagnostics={},
        evidence=None,
    )
