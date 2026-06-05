# COCO / RefCOCO Operator Data Audit

Date: 2026-06-05

This audit is the pre-entry gate for using COCO-family data as OVHA operator-bank evidence. It prevents COCO detection/category data, RefCOCO referring-expression data, GroundingDINO pseudo labels, and same-feature internal baselines from being mixed into one claim.

## Dataset Classification

Classify the available files before training:

| Data shape | Valid task definition | Allowed claim |
|---|---|---|
| image + object boxes + category names | category text query to object box selection | weak category grounding only |
| image + caption + boxes, no phrase-to-box annotation | caption noun/category to matching category box | weak phrase-region proxy |
| referring expression + target box | referring expression to target box | RefCOCO-style grounding |
| region features + text query + target region index | same-feature region-text grounding | internal OVHA CATO/TLEO table |
| GroundingDINO detections or pseudo labels | pseudo-label teacher or hard-negative source | external/teacher-controlled evidence only |

## Operator Scope

CMU-MOSEI uses `TANSO`, `LRIO`, and `RCEO`.

COCO / RefCOCO / Flickr30k region-text grounding uses `CATO`, `TLEO`, `SPO`, and `RCEO`.

Do not use TANSO as a COCO grounding primary operator unless the task explicitly contains temporal or nonverbal shift structure.

## Same-Feature Tables

Internal same-feature tables may include:

| family | rows |
|---|---|
| sanity probes | `text_only`, `region_only`, `concat_fusion` |
| OVHA operator rows | `CATO`, `TLEO`, `SPO`, `CATO+TLEO`, admitted bank |
| diagnostics | alignment entropy, CATO load, TLEO local evidence load, Acc@0.5 IoU, Recall@K |

GroundingDINO, GroundingDINO-1.5, GLIP, MDETR, TransVG, LAVT, and SeqTR must stay in an external-reference table unless their exact frozen features and training protocol are converted into the same-feature cache protocol.

## Ubuntu File Audit Commands

Run these commands on the Ubuntu machine that holds the processed COCO / RefCOCO data, then send back the generated JSON and the first few listed paths:

```bash
DATA_ROOT=/path/to/your/coco_refcoco_root
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["DATA_ROOT"]).expanduser().resolve()
patterns = {
    "json": ["*.json"],
    "jsonl": ["*.jsonl"],
    "npy": ["*.npy"],
    "npz": ["*.npz"],
    "parquet": ["*.parquet"],
    "pt_pth": ["*.pt", "*.pth"],
    "images": ["*.jpg", "*.jpeg", "*.png"],
}
summary = {"root": str(root), "exists": root.exists(), "groups": {}}
for group, globs in patterns.items():
    paths = []
    for glob in globs:
        paths.extend(root.rglob(glob))
    paths = sorted(paths)
    summary["groups"][group] = {
        "count": len(paths),
        "examples": [str(path.relative_to(root)) for path in paths[:20]],
    }
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
```

If you already know the annotation directory, also run:

```bash
ANN_DIR=/path/to/annotations
python - <<'PY'
import json
import os
from pathlib import Path

ann_dir = Path(os.environ["ANN_DIR"]).expanduser().resolve()
for path in sorted(ann_dir.rglob("*.json"))[:50]:
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        print(json.dumps({"path": str(path), "error": str(exc)}, ensure_ascii=False))
        continue
    if isinstance(data, dict):
        keys = sorted(data)[:30]
        sizes = {key: len(value) for key, value in data.items() if isinstance(value, list)}
    elif isinstance(data, list):
        keys = ["<list>"]
        sizes = {"list": len(data)}
    else:
        keys = [type(data).__name__]
        sizes = {}
    print(json.dumps({"path": str(path), "top_keys": keys, "list_sizes": sizes}, ensure_ascii=False))
PY
```

## Entry Decision

Use the audit output as follows:

| Evidence found | Next step |
|---|---|
| `refs`, `sentences`, `ann_id`, `split`, `bbox` or equivalent | build RefCOCO records with `scripts/multimodal/stage_refcoco_raw.py` |
| COCO `annotations`, `categories`, `images` only | build weak category grounding records and mark the claim as weak grounding |
| GroundingDINO outputs only | use as pseudo-label teacher or hard-negative generator, not same-feature baseline |
| precomputed region/text features with target index | validate cache schema and run internal CATO/TLEO same-feature table |
