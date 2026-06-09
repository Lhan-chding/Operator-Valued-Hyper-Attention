from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

import numpy as np


class GroundingDINOSameCandidateScoringTest(unittest.TestCase):
    def test_scores_predictions_against_fixed_refcoco_candidates(self) -> None:
        from scripts.multimodal.score_groundingdino_same_candidates import (
            score_groundingdino_same_candidates,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            _write_refcoco_cache(cache_root)
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps(
                    {
                        "source_id": "sample-a",
                        "boxes": [[0.31, 0.01, 0.49, 0.19]],
                        "scores": [0.9],
                        "phrases": ["target"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )

            payload = score_groundingdino_same_candidates(
                Namespace(
                    predictions=predictions,
                    cache_root=cache_root,
                    dataset_name="refcoco",
                    version="v0.1",
                    split="testA",
                    output_dir=root / "out",
                    prediction_box_format="xyxy_normalized",
                    max_predictions_per_sample=None,
                )
            )
            summary_exists = (Path(payload["output_dir"]) / "summary.json").exists()
            rows = [
                json.loads(line)
                for line in (Path(payload["output_dir"]) / "per_sample_scores.jsonl").read_text().splitlines()
                if line.strip()
            ]

        self.assertEqual(payload["sample_count"], 2)
        self.assertEqual(payload["prediction_count"], 1)
        self.assertEqual(payload["missing_prediction_count"], 1)
        self.assertEqual(payload["empty_prediction_count"], 1)
        self.assertAlmostEqual(payload["recall_at_1"], 0.5)
        self.assertAlmostEqual(payload["acc_at_0_5"], 0.5)
        self.assertAlmostEqual(payload["mean_iou"], 0.5)
        self.assertTrue(summary_exists)
        self.assertEqual(rows[0]["source_id"], "sample-a")
        self.assertEqual(rows[0]["selected_index"], 1)
        self.assertEqual(rows[0]["target_index"], 1)
        self.assertTrue(rows[0]["hit_at_1"])
        self.assertEqual(rows[1]["source_id"], "sample-b")
        self.assertEqual(rows[1]["selected_index"], -1)
        self.assertFalse(rows[1]["hit_at_1"])

    def test_accepts_groundingdino_cxcywh_normalized_boxes(self) -> None:
        from scripts.multimodal.score_groundingdino_same_candidates import (
            score_groundingdino_same_candidates,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            _write_refcoco_cache(cache_root)
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps(
                    {
                        "source_id": "sample-a",
                        "boxes": [[0.4, 0.1, 0.2, 0.2]],
                        "scores": [1.0],
                    },
                    sort_keys=True,
                )
                + "\n"
            )

            payload = score_groundingdino_same_candidates(
                Namespace(
                    predictions=predictions,
                    cache_root=cache_root,
                    dataset_name="refcoco",
                    version="v0.1",
                    split="testA",
                    output_dir=root / "out",
                    prediction_box_format="cxcywh_normalized",
                    max_predictions_per_sample=None,
                )
            )

        self.assertAlmostEqual(payload["recall_at_1"], 0.5)
        self.assertAlmostEqual(payload["acc_at_0_5"], 0.5)

    def test_prediction_input_helpers_resolve_refcoco_image_and_caption(self) -> None:
        from scripts.multimodal.run_groundingdino_refcoco_predictions import (
            _image_path_from_record,
            _normalize_caption,
        )

        path = _image_path_from_record(
            {"image_id": "image42"},
            image_root=Path("/datasets/coco/train2014"),
            image_template="COCO_train2014_{image_number:012d}.jpg",
        )

        self.assertEqual(path, Path("/datasets/coco/train2014/COCO_train2014_000000000042.jpg"))
        self.assertEqual(_normalize_caption("  The left person  "), "the left person .")
        self.assertEqual(_normalize_caption("cat . dog ."), "cat . dog .")

    def test_builds_expression_map_from_unc_refcoco_refs(self) -> None:
        from scripts.multimodal.build_refcoco_expression_map import build_refcoco_expression_map

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs = root / "refs.json"
            output = root / "expressions.jsonl"
            refs.write_text(
                json.dumps(
                    [
                        {
                            "image_id": 42,
                            "ann_id": 7,
                            "sentences": [
                                {"sent_id": 100, "tokens": ["left", "person"]},
                                {"sent_id": 101, "raw": "person in blue"},
                            ],
                        }
                    ],
                    sort_keys=True,
                )
                + "\n"
            )

            payload = build_refcoco_expression_map(
                Namespace(dataset_name="refcoco", refs=refs, output=output)
            )
            rows = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]

        self.assertEqual(payload["expression_count"], 2)
        self.assertEqual(rows[0]["source_id"], "refcoco::image42::ann7::sent100")
        self.assertEqual(rows[0]["expression"], "left person")
        self.assertEqual(rows[1]["expression"], "person in blue")

    def test_scores_groundingdino_openbox_predictions(self) -> None:
        from scripts.multimodal.score_groundingdino_openbox import score_groundingdino_openbox

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            _write_refcoco_cache(cache_root)
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "source_id": "sample-a",
                                "boxes": [[0.31, 0.01, 0.49, 0.19], [0.0, 0.0, 0.2, 0.2]],
                                "scores": [0.9, 0.4],
                            },
                            sort_keys=True,
                        ),
                        json.dumps({"source_id": "sample-b", "boxes": [], "scores": []}, sort_keys=True),
                    ]
                )
                + "\n"
            )

            payload = score_groundingdino_openbox(
                Namespace(
                    predictions=predictions,
                    cache_root=cache_root,
                    dataset_name="refcoco",
                    version="v0.1",
                    split="testA",
                    output_dir=root / "openbox",
                    prediction_box_format="xyxy_normalized",
                    max_predictions_per_sample=None,
                )
            )
            rows = [
                json.loads(line)
                for line in (Path(payload["output_dir"]) / "per_sample_openbox_scores.jsonl").read_text().splitlines()
                if line.strip()
            ]

        self.assertEqual(payload["sample_count"], 2)
        self.assertEqual(payload["prediction_count"], 2)
        self.assertEqual(payload["empty_prediction_count"], 1)
        self.assertAlmostEqual(payload["acc_at_0_5"], 0.5)
        self.assertGreater(payload["mean_iou"], 0.4)
        self.assertEqual(rows[0]["selected_prediction_index"], 0)
        self.assertTrue(rows[0]["acc_at_0_5"])
        self.assertEqual(rows[1]["selected_prediction_index"], -1)

    def test_summarizes_refcoco_subset_diagnostics(self) -> None:
        from scripts.multimodal.summarize_refcoco_subset_diagnostics import summarize_refcoco_subset_diagnostics

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            _write_refcoco_cache(cache_root)
            scores = root / "per_sample_scores.jsonl"
            scores.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "source_id": "sample-a",
                                "selected_iou": 1.0,
                                "hit_at_1": True,
                                "hit_at_5": True,
                                "prediction_count": 1,
                            },
                            sort_keys=True,
                        ),
                        json.dumps(
                            {
                                "source_id": "sample-b",
                                "selected_iou": 0.0,
                                "hit_at_1": False,
                                "hit_at_5": False,
                                "prediction_count": 0,
                            },
                            sort_keys=True,
                        ),
                    ]
                )
                + "\n"
            )
            expressions = root / "expressions.jsonl"
            expressions.write_text(
                "\n".join(
                    [
                        json.dumps({"source_id": "sample-a", "expression": "left red person"}, sort_keys=True),
                        json.dumps({"source_id": "sample-b", "expression": "person near dog"}, sort_keys=True),
                    ]
                )
                + "\n"
            )

            payload = summarize_refcoco_subset_diagnostics(
                Namespace(
                    per_sample_scores=scores,
                    expressions_jsonl=expressions,
                    cache_root=cache_root,
                    dataset_name="refcoco",
                    version="v0.1",
                    split="testA",
                    output_dir=root / "diagnostics",
                    min_count=1,
                )
            )

        by_group = {row["group"]: row for row in payload["groups"]}
        self.assertAlmostEqual(by_group["all"]["acc_at_0_5"], 0.5)
        self.assertAlmostEqual(by_group["spatial_expression"]["acc_at_0_5"], 0.5)
        self.assertAlmostEqual(by_group["attribute_expression"]["acc_at_0_5"], 1.0)
        self.assertAlmostEqual(by_group["relational_expression"]["acc_at_0_5"], 0.0)
        self.assertEqual(by_group["empty_prediction"]["sample_count"], 1)


def _write_refcoco_cache(cache_root: Path) -> None:
    root = cache_root / "refcoco" / "v0.1"
    (root / "provenance").mkdir(parents=True)
    (root / "supervision").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)
    source_ids = ["sample-a", "sample-b"]
    (root / "provenance" / "source_ids_testA.txt").write_text("\n".join(source_ids) + "\n")
    (root / "provenance" / "sample_records_testA.jsonl").write_text(
        "\n".join(
            json.dumps({"source_id": source_id, "split": "testA"}, sort_keys=True)
            for source_id in source_ids
        )
        + "\n"
    )
    candidate_boxes = np.asarray(
        [
            [
                [0.0, 0.0, 0.2, 0.2],
                [0.3, 0.0, 0.5, 0.2],
                [0.0, 0.3, 0.2, 0.5],
            ],
            [
                [0.1, 0.1, 0.4, 0.4],
                [0.5, 0.5, 0.8, 0.8],
                [0.0, 0.0, 0.0, 0.0],
            ],
        ],
        dtype=np.float32,
    )
    np.save(root / "supervision" / "candidate_region_boxes_testA.npy", candidate_boxes)
    np.save(root / "supervision" / "bbox_targets_testA.npy", np.asarray([[0.3, 0.0, 0.5, 0.2], [0.1, 0.1, 0.4, 0.4]], dtype=np.float32))
    np.save(root / "supervision" / "region_targets_testA.npy", np.asarray([[1], [0]], dtype=np.int64))
    np.save(root / "masks" / "region_mask_testA.npy", np.asarray([[True, True, True], [True, True, False]]))
