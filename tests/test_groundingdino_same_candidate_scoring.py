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

    def test_scores_groundingdino_proposal_oracle(self) -> None:
        from scripts.multimodal.score_groundingdino_proposal_oracle import score_groundingdino_proposal_oracle

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
                                "boxes": [
                                    [0.0, 0.0, 0.2, 0.2],
                                    [0.3, 0.0, 0.5, 0.2],
                                    [0.28, 0.0, 0.52, 0.22],
                                ],
                                "scores": [0.7, 0.6, 0.9],
                            },
                            sort_keys=True,
                        ),
                        json.dumps({"source_id": "sample-b", "boxes": [], "scores": []}, sort_keys=True),
                    ]
                )
                + "\n"
            )

            payload = score_groundingdino_proposal_oracle(
                Namespace(
                    predictions=predictions,
                    cache_root=cache_root,
                    dataset_name="refcoco",
                    version="v0.1",
                    split="testA",
                    output_dir=root / "proposal_oracle",
                    prediction_box_format="xyxy_normalized",
                    max_predictions_per_sample=None,
                    top_k=[1, 2, 3],
                    iou_threshold=0.5,
                )
            )
            rows = [
                json.loads(line)
                for line in (Path(payload["output_dir"]) / "per_sample_proposal_oracle.jsonl").read_text().splitlines()
                if line.strip()
            ]

        self.assertEqual(payload["artifact_type"], "groundingdino_proposal_oracle_summary")
        self.assertEqual(payload["sample_count"], 2)
        self.assertEqual(payload["empty_proposal_count"], 1)
        self.assertAlmostEqual(payload["proposal_oracle_recall_at_k"]["1"], 0.5)
        self.assertAlmostEqual(payload["proposal_oracle_recall_at_k"]["2"], 0.5)
        self.assertAlmostEqual(payload["proposal_oracle_recall_at_k"]["3"], 0.5)
        self.assertAlmostEqual(payload["positive_candidate_rate"], 0.5)
        self.assertEqual(payload["bucket_counts"]["easy"], 1)
        self.assertEqual(payload["bucket_counts"]["hard"], 1)
        self.assertEqual(rows[0]["best_proposal_rank"], 3)
        self.assertEqual(rows[0]["difficulty_bucket"], "easy")
        self.assertEqual(rows[1]["best_proposal_rank"], None)
        self.assertEqual(rows[1]["difficulty_bucket"], "hard")

    def test_scores_groundingdino_proposals_with_precomputed_clip_similarity(self) -> None:
        from scripts.multimodal.score_groundingdino_proposal_clip_similarity import (
            score_precomputed_groundingdino_proposal_scores,
        )
        from scripts.multimodal.score_groundingdino_same_candidates import _load_predictions

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            _write_refcoco_cache(cache_root)
            predictions_path = root / "predictions.jsonl"
            predictions_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "source_id": "sample-a",
                                "boxes": [[0.0, 0.0, 0.2, 0.2], [0.3, 0.0, 0.5, 0.2]],
                                "scores": [0.9, 0.8],
                            },
                            sort_keys=True,
                        ),
                        json.dumps(
                            {
                                "source_id": "sample-b",
                                "boxes": [[0.5, 0.5, 0.8, 0.8], [0.1, 0.1, 0.4, 0.4]],
                                "scores": [0.9, 0.8],
                            },
                            sort_keys=True,
                        ),
                    ]
                )
                + "\n"
            )
            predictions = _load_predictions(
                predictions_path,
                box_format="xyxy_normalized",
                max_predictions_per_sample=None,
            )
            payload = score_precomputed_groundingdino_proposal_scores(
                Namespace(
                    predictions=predictions_path,
                    cache_root=cache_root,
                    dataset_name="refcoco",
                    version="v0.1",
                    split="testA",
                    output_dir=root / "proposal_clip",
                    prediction_box_format="xyxy_normalized",
                    max_predictions_per_sample=None,
                    iou_threshold=0.5,
                    model_name="proposal_clip_test",
                    clip_model="test-clip",
                ),
                source_ids=["sample-a", "sample-b"],
                bbox_targets=np.asarray([[0.3, 0.0, 0.5, 0.2], [0.1, 0.1, 0.4, 0.4]], dtype=np.float32),
                predictions=predictions,
                proposal_similarity_scores={"sample-a": [0.1, 0.9], "sample-b": [0.9, 0.1]},
                crop_failure_count=0,
            )
            rows = [
                json.loads(line)
                for line in (Path(payload["output_dir"]) / "per_sample_scores.jsonl").read_text().splitlines()
                if line.strip()
            ]

        self.assertEqual(payload["artifact_type"], "groundingdino_proposal_reranker_summary")
        self.assertEqual(payload["model"], "proposal_clip_test")
        self.assertAlmostEqual(payload["recall_at_1"], 0.5)
        self.assertAlmostEqual(payload["recall_at_5"], 1.0)
        self.assertAlmostEqual(payload["acc_at_0_5"], 0.5)
        self.assertAlmostEqual(payload["mrr"], 0.75)
        self.assertEqual(rows[0]["selected_proposal_index"], 1)
        self.assertEqual(rows[0]["first_positive_rank"], 1)
        self.assertEqual(rows[1]["selected_proposal_index"], 0)
        self.assertEqual(rows[1]["first_positive_rank"], 2)

    def test_external_alignment_strong_rerankers_forward_on_region_batch(self) -> None:
        import torch
        from moat_ovha_torch.data.multimodal.typed_batch import (
            MultimodalEpisodeBatch,
            ProvenanceBank,
            QueryField,
            SupervisionBank,
            TokenField,
        )
        from scripts.multimodal.run_public_main import _make_region_reranker

        candidate_boxes = torch.tensor(
            [
                [[0.0, 0.0, 0.2, 0.2], [0.3, 0.0, 0.5, 0.2], [0.0, 0.0, 0.0, 0.0]],
                [[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.8, 0.8], [0.2, 0.2, 0.3, 0.3]],
            ],
            dtype=torch.float32,
        )
        width_height = (candidate_boxes[..., 2:] - candidate_boxes[..., :2]).clamp_min(0.0)
        region_pos = torch.cat(
            [
                candidate_boxes,
                0.5 * (candidate_boxes[..., :2] + candidate_boxes[..., 2:]),
                width_height,
                (width_height[..., 0] * width_height[..., 1]).unsqueeze(-1),
            ],
            dim=-1,
        )
        target_y = torch.zeros(2, 1, 3)
        batch = MultimodalEpisodeBatch(
            fields={
                "text": TokenField(
                    "text",
                    torch.randn(2, 4, 5),
                    torch.zeros(2, 4, 1),
                    torch.tensor([[True, True, True, False], [True, True, False, False]]),
                ),
                "region": TokenField(
                    "region",
                    torch.randn(2, 3, 6),
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
            split="testA",
            source_dataset="refcoco",
            supervision=SupervisionBank(
                task_label=target_y,
                alignment_pairs=None,
                alignment_weights=None,
                bbox_targets=candidate_boxes[:, 0],
                region_targets=torch.tensor([[1], [0]], dtype=torch.long),
                timestamp_targets=None,
                modality_missing_mask=None,
                corruption_metadata=None,
                weak_labels=None,
                weak_label_confidence=None,
                pseudo_label_source=None,
                candidate_region_boxes=candidate_boxes,
            ),
            provenance=ProvenanceBank(
                source_id=["sample-a", "sample-b"],
                original_split=["testA", "testA"],
                raw_ref=["a", "b"],
                license_tag=["test", "test"],
                preprocessing_version="test",
                feature_extractor_version={"text": "test", "region": "test"},
                pseudo_label_version={},
            ),
        )

        for baseline_name in (
            "clip_geometry_mlp",
            "box_aware_cross_attention_reranker",
            "lightweight_transvg_style_reranker",
        ):
            with self.subTest(baseline_name=baseline_name):
                model = _make_region_reranker(baseline_name, batch, d_model=8)
                logits = model(batch)

                self.assertEqual(tuple(logits.shape), (2, 1, 3))
                self.assertLess(float(logits.detach()[0, 0, 2]), -1e8)

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
                    input_format="score_rows",
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

    def test_summarizes_public_main_prediction_subset_diagnostics(self) -> None:
        from scripts.multimodal.summarize_refcoco_subset_diagnostics import summarize_refcoco_subset_diagnostics

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            _write_refcoco_cache(cache_root)
            predictions = root / "per_sample_predictions.jsonl"
            predictions.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "sample_id": "sample-a",
                                "split": "testA",
                                "model": "test_reranker",
                                "seed": 1,
                                "prediction_full": [[0.1, 0.9, 0.2]],
                            },
                            sort_keys=True,
                        ),
                        json.dumps(
                            {
                                "sample_id": "sample-b",
                                "split": "testA",
                                "model": "test_reranker",
                                "seed": 1,
                                "prediction_full": [[0.1, 0.7, -1.0]],
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
                    per_sample_scores=predictions,
                    input_format="public_main_predictions",
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
        self.assertAlmostEqual(by_group["all"]["recall_at_1"], 0.5)
        self.assertAlmostEqual(by_group["all"]["mean_iou"], 0.5)
        self.assertAlmostEqual(by_group["attribute_expression"]["acc_at_0_5"], 1.0)
        self.assertAlmostEqual(by_group["relational_expression"]["acc_at_0_5"], 0.0)

    def test_summarizes_refcoco_paired_significance_from_public_predictions(self) -> None:
        from scripts.multimodal.summarize_refcoco_paired_significance import summarize_refcoco_paired_significance

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            _write_refcoco_cache(cache_root)
            full = root / "full_predictions.jsonl"
            baseline = root / "baseline_predictions.jsonl"
            full.write_text(
                "\n".join(
                    [
                        json.dumps({"sample_id": "sample-a", "seed": 1, "prediction_full": [[0.1, 0.9, 0.2]]}, sort_keys=True),
                        json.dumps({"sample_id": "sample-b", "seed": 1, "prediction_full": [[0.9, 0.2, -1.0]]}, sort_keys=True),
                    ]
                )
                + "\n"
            )
            baseline.write_text(
                "\n".join(
                    [
                        json.dumps({"sample_id": "sample-a", "seed": 1, "prediction_full": [[0.9, 0.1, 0.2]]}, sort_keys=True),
                        json.dumps({"sample_id": "sample-b", "seed": 1, "prediction_full": [[0.1, 0.9, -1.0]]}, sort_keys=True),
                    ]
                )
                + "\n"
            )

            payload = summarize_refcoco_paired_significance(
                Namespace(
                    full=full,
                    baseline=[f"strong={baseline}"],
                    full_name="ovha",
                    input_format="public_main_predictions",
                    cache_root=cache_root,
                    dataset_name="refcoco",
                    version="v0.1",
                    split="testA",
                    metric="acc_at_0_5",
                    bootstrap_samples=200,
                    permutation_samples=200,
                    seed=7,
                    output_dir=root / "paired",
                )
            )

        comparison = payload["comparisons"][0]
        self.assertEqual(comparison["paired_observation_count"], 2)
        self.assertAlmostEqual(comparison["full_mean"], 1.0)
        self.assertAlmostEqual(comparison["baseline_mean"], 0.0)
        self.assertGreater(comparison["mean_delta"], 0.0)

    def test_scores_clip_crop_similarity_against_same_candidates(self) -> None:
        from scripts.multimodal.score_clip_crop_same_candidates import score_precomputed_clip_crop_same_candidates

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            _write_refcoco_cache(cache_root)
            payload = score_precomputed_clip_crop_same_candidates(
                Namespace(
                    cache_root=cache_root,
                    dataset_name="refcoco",
                    version="v0.1",
                    split="testA",
                    output_dir=root / "clip_crop",
                    model_name="clip_crop_similarity_test",
                    clip_model="openai/clip-vit-base-patch32",
                ),
                similarity_scores=np.asarray(
                    [
                        [0.1, 0.9, 0.2],
                        [0.2, 0.8, -1.0],
                    ],
                    dtype=np.float32,
                ),
                crop_failure_count=0,
            )
            rows = [
                json.loads(line)
                for line in (Path(payload["output_dir"]) / "per_sample_scores.jsonl").read_text().splitlines()
                if line.strip()
            ]
            raw_metrics = [
                json.loads(line)
                for line in (Path(payload["output_dir"]) / "raw_metrics.jsonl").read_text().splitlines()
                if line.strip()
            ]

        self.assertEqual(payload["artifact_type"], "clip_crop_same_candidate_summary")
        self.assertEqual(payload["sample_count"], 2)
        self.assertEqual(payload["crop_failure_count"], 0)
        self.assertAlmostEqual(payload["recall_at_1"], 0.5)
        self.assertAlmostEqual(payload["acc_at_0_5"], 0.5)
        self.assertAlmostEqual(payload["mean_iou"], 0.5)
        self.assertEqual(rows[0]["selected_index"], 1)
        self.assertTrue(rows[0]["hit_at_1"])
        self.assertEqual(rows[1]["selected_index"], 1)
        self.assertFalse(rows[1]["hit_at_1"])
        self.assertTrue(all(row["model"] == "clip_crop_similarity_test" for row in raw_metrics))
        self.assertTrue(all(row["same_candidate_source"] for row in raw_metrics))
        self.assertFalse(any(row["same_feature_source"] for row in raw_metrics))


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
