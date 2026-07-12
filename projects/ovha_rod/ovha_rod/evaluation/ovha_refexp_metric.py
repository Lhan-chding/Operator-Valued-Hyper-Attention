from __future__ import annotations

from typing import Dict

import numpy as np

from mmdet.evaluation.functional import bbox_overlaps
from mmdet.evaluation.metrics.refexp_metric import RefExpMetric
from mmdet.registry import METRICS


@METRICS.register_module()
class OVHARefExpMetric(RefExpMetric):
    """Official RefExpMetric plus Top-1 Acc@.75 and mean IoU.

    The inherited metrics and ``mean_precision`` key are preserved so parent
    checkpoint selection remains comparable.  The added fields close the
    Phase-1 boundary-localization reporting contract.
    """

    def process(self, data_batch: dict, data_samples) -> None:
        del data_batch
        for data_sample in data_samples:
            prediction = data_sample["pred_instances"]
            result = {
                "img_id": data_sample["img_id"],
                "bboxes": prediction["bboxes"].cpu().numpy(),
                "scores": prediction["scores"].cpu().numpy(),
            }
            if "encoder_query_boxes" in prediction:
                result["encoder_query_boxes"] = (
                    prediction["encoder_query_boxes"].cpu().numpy())
            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        output = super().compute_metrics(results)
        names = ("refcoco", "refcoco+", "refcocog")
        ious: dict[str, list[float]] = {name: [] for name in names}
        encoder_ious: dict[str, list[float]] = {name: [] for name in names}
        for result in results:
            image_id = result["img_id"]
            annotation_ids = self.coco.getAnnIds(imgIds=image_id)
            if len(annotation_ids) != 1:
                raise ValueError(
                    f"RefExp sample {image_id} must have exactly one annotation")
            image_info = self.coco.loadImgs(image_id)[0]
            dataset_name = image_info["dataset_name"]
            annotation = self.coco.loadAnns(annotation_ids[0])[0]
            x, y, width, height = annotation["bbox"]
            target = np.asarray([[x, y, x + width, y + height]], dtype=np.float32)
            predictions = result["bboxes"]
            top1_iou = 0.0
            if len(predictions):
                overlaps = bbox_overlaps(predictions[:1], target)
                top1_iou = float(overlaps[0, 0])
            ious.setdefault(dataset_name, []).append(top1_iou)
            if "encoder_query_boxes" in result:
                encoder_overlaps = bbox_overlaps(
                    result["encoder_query_boxes"], target)
                encoder_ious.setdefault(dataset_name, []).append(
                    float(encoder_overlaps.max()) if encoder_overlaps.size else 0.0)

        populated = []
        for name in names:
            values = ious[name]
            if not values:
                output[f"{name}_acc@0.75"] = 0.0
                output[f"{name}_miou"] = 0.0
                continue
            array = np.asarray(values, dtype=np.float64)
            output[f"{name}_acc@0.75"] = float((array >= 0.75).mean())
            output[f"{name}_miou"] = float(array.mean())
            if encoder_ious[name] and len(encoder_ious[name]) != len(values):
                raise ValueError(
                    f"encoder query boxes missing for {name}: "
                    f"{len(encoder_ious[name])}/{len(values)} samples")
            if encoder_ious[name]:
                encoder_array = np.asarray(encoder_ious[name], dtype=np.float64)
                output[f"{name}_encoder_oracle@0.5"] = float(
                    (encoder_array >= 0.5).mean())
                output[f"{name}_encoder_oracle@0.75"] = float(
                    (encoder_array >= 0.75).mean())
                output[f"{name}_encoder_max_iou"] = float(encoder_array.mean())
            populated.append(array)
        if populated:
            combined = np.concatenate(populated)
            output["acc@0.75"] = float((combined >= 0.75).mean())
            output["miou"] = float(combined.mean())
        else:
            output["acc@0.75"] = 0.0
            output["miou"] = 0.0
        return output
