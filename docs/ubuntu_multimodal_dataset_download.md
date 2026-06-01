# Ubuntu 多模态数据下载命令

本文档对应 `OVHA_next_step_topconf_final_plan_multimodal_mainline_cn.md` 的 Step 1/4/5 数据准备阶段。目标是先把公开数据源稳定下载到 Ubuntu 主机，再由后续 feature extraction / cache builder 产出本仓库 adapter 需要的 raw manifest：

```text
data/raw_multimodal/<dataset>/
  annotations/...
  features/*_features.npy
  labels/...
  metadata/...
  splits.json
```

注意：当前 adapter 不把原始图片/视频直接当正式 cache。下载完成后仍需要冻结特征提取与 provenance/checksum 生成，才能进入 `scripts/multimodal/build_cache.py` 和 `validate_cache.py`。

## 1. 复用已有 PDEBench `.venv`

可以继续用 `~/work/Operator-Valued-Hyper-Attention/.venv`。需要补的是下载/解压/音视频/Parquet/Hugging Face 工具，不需要重建 Python 环境。

```bash
cd ~/work/Operator-Valued-Hyper-Attention
source .venv/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install -U \
  huggingface_hub hf_xet datasets kaggle gdown requests tqdm \
  pandas pyarrow fastparquet h5py scipy numpy pillow opencv-python-headless \
  soundfile librosa
```

系统工具：

```bash
sudo NEEDRESTART_MODE=l DEBIAN_FRONTEND=noninteractive apt-get update
sudo NEEDRESTART_MODE=l DEBIAN_FRONTEND=noninteractive apt-get install -y \
  aria2 git git-lfs unzip p7zip-full pigz zstd ffmpeg libsndfile1
git lfs install
```

Hugging Face 官方当前推荐 `hf-xet`；`HF_HUB_ENABLE_HF_TRANSFER` 已是旧变量。高带宽机器用：

```bash
export HF_HOME="$PWD/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_ETAG_TIMEOUT=30
```

如果 Ubuntu 主机访问 Hugging Face 很慢，可以临时使用镜像端点；下载完建议 `unset HF_ENDPOINT` 回到官方源复核：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

如果要减少手工复制错误，优先用仓库脚本生成 Ubuntu 下载脚本。它只生成脚本，不会在本机直接启动下载；先 `bash -n` 检查，再在 Ubuntu 数据主机上执行：

```bash
cd ~/work/Operator-Valued-Hyper-Attention
source .venv/bin/activate

python scripts/multimodal/bootstrap_public_downloads.py \
  --datasets refcoco cmu_mosei \
  --repo-root "$PWD" \
  --download-root data/raw_multimodal/_downloads \
  --include-system-packages \
  --use-hf-mirror \
  --output /tmp/ovha_public_downloads.sh

bash -n /tmp/ovha_public_downloads.sh
bash /tmp/ovha_public_downloads.sh
```

如果不用 Hugging Face 镜像，去掉 `--use-hf-mirror`。如果已经装好系统包，去掉 `--include-system-packages`。

## 2. 建议目录

根分区空间不足时，不要把全量 COCO / Visual Genome 放在 `/`。优先挂到更大的盘，例如 `/data/ovha_datasets`；没有额外挂盘时才用 repo 内目录。

```bash
cd ~/work/Operator-Valued-Hyper-Attention
source .venv/bin/activate

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
PY

df -h .
mkdir -p data/raw_multimodal/_downloads
mkdir -p data/raw_multimodal/{refcoco,flickr30k_entities,visual_genome,cmu_mosei,cmu_mosi,meld}
```

如果 `/` 只剩几百 GB，建议把下载目录放到大盘并软链回 repo：

```bash
mkdir -p /data/ovha_datasets/raw_multimodal
ln -sfn /data/ovha_datasets/raw_multimodal data/raw_multimodal
```

## 2.1 下载后必须整理成的 raw manifest

本仓库的 adapter 当前不直接读取 COCO 图片、MELD 视频或 CMU SDK 原始 computational sequences；它们先要被固定特征提取流程整理成下面的 raw manifest。这个结构是进入 `scripts/multimodal/build_cache.py` 的输入。

第一优先级建议先做 `cmu_mosei` 和 `refcoco`：

```text
data/raw_multimodal/cmu_mosei/
  features/text_features.npy
  features/audio_features.npy
  features/visual_features.npy
  labels/sentiment.npy
  labels/emotion.npy
  metadata/utterances.json
  metadata/dialogues.json
  metadata/feature_versions.json
  metadata/missing_modality_mask.npy
  metadata/corruption_transforms.json
  splits.json

data/raw_multimodal/refcoco/
  annotations/instances.json
  annotations/refs.json
  features/text_features.npy
  features/region_features.npy
  splits.json
```

其他数据集对应文件：

```text
data/raw_multimodal/cmu_mosi/
  features/text_features.npy
  features/audio_features.npy
  features/visual_features.npy
  labels/sentiment.npy
  labels/emotion.npy
  metadata/utterances.json
  metadata/dialogues.json
  metadata/feature_versions.json
  metadata/missing_modality_mask.npy
  metadata/corruption_transforms.json
  splits.json

data/raw_multimodal/meld/
  features/text_features.npy
  features/audio_features.npy
  features/visual_features.npy
  labels/emotion.npy
  metadata/dialogues.json
  metadata/feature_versions.json
  metadata/missing_modality_mask.npy
  metadata/corruption_transforms.json
  splits.json

data/raw_multimodal/flickr30k_entities/
  annotations/phrase_regions.json
  annotations/captions.json
  features/text_features.npy
  features/region_features.npy
  splits.json

data/raw_multimodal/visual_genome/
  annotations/region_descriptions.json
  annotations/objects.json
  annotations/relationships.json
  features/text_features.npy
  features/region_features.npy
  splits.json
```

可以先用 `build_cache.py` 让 adapter 直接列出缺失项：

```bash
python scripts/multimodal/build_cache.py cmu_mosei data/raw_multimodal/cmu_mosei data/multimodal_cache --version v0.1
python scripts/multimodal/build_cache.py refcoco data/raw_multimodal/refcoco data/multimodal_cache --version v0.1
```

下载、转换、stage raw manifest、build cache 之间如果不确定当前缺哪一步，直接跑状态检查器。它只输出 JSON 和下一条建议命令，不作为训练 gate：

```bash
python scripts/multimodal/check_public_data_readiness.py \
  --datasets refcoco cmu_mosei \
  --download-root data/raw_multimodal/_downloads \
  --raw-root-base data/raw_multimodal \
  --cache-root data/multimodal_cache \
  --controlled-report outputs/multimodal/controlled_v1_smoke/seed_101/controlled_report.json
```

当 formal cache 已经有效时，它会直接给出对应 `accept_public_data.py` 命令；如果 RefCOCO / CMU-MOSEI 还停在下载或 feature freeze 阶段，它会指向 `build_refcoco_stage_records.py`、`align_refcoco_stage_features.py`、`write_cmu_sdk_splits.py`、`inspect_cmu_sdk_sequences.py` 或 `extract_cmu_sdk_stage_inputs.py` 中下一条应执行的命令。

当缺失项全部补齐后，同一命令会写入正式 cache，再执行：

```bash
python scripts/multimodal/validate_cache.py data/multimodal_cache cmu_mosei v0.1 --splits train val test
python scripts/multimodal/validate_cache.py data/multimodal_cache refcoco v0.1 --splits train val test
```

如果你已经从 CMU SDK / MultiBench / 自己的冻结特征流程拿到了按同一 sample 顺序排列的 `.npy` 特征和标签，可以先用 staging 脚本生成 `cmu_mosei` raw manifest，再交给 cache builder：

```bash
python scripts/multimodal/stage_cmu_sentiment_raw.py \
  cmu_mosei \
  data/raw_multimodal/cmu_mosei \
  --splits data/raw_multimodal/_downloads/cmu_mosei_splits.json \
  --text-features data/raw_multimodal/_downloads/cmu_mosei_text_features.npy \
  --audio-features data/raw_multimodal/_downloads/cmu_mosei_audio_features.npy \
  --visual-features data/raw_multimodal/_downloads/cmu_mosei_visual_features.npy \
  --sentiment-labels data/raw_multimodal/_downloads/cmu_mosei_sentiment.npy \
  --emotion-labels data/raw_multimodal/_downloads/cmu_mosei_emotion.npy \
  --feature-version text=cmu-sdk-text-v1 \
  --feature-version audio=cmu-sdk-audio-v1 \
  --feature-version vision=cmu-sdk-vision-v1 \
  --license-tag cmu-mosei \
  --preprocessing-version cmu-mosei-frozen-features-v0.1

python scripts/multimodal/build_cache.py \
  cmu_mosei \
  data/raw_multimodal/cmu_mosei \
  data/multimodal_cache \
  --version v0.1
```

`--splits` 的 source_id 顺序必须和所有 `.npy` 第一维一致，例如：

```json
{
  "train": ["videoA::utterance0001", "videoA::utterance0002"],
  "val": ["videoB::utterance0001"],
  "test": ["videoC::utterance0001"]
}
```

RefCOCO / phrase-region grounding 同理。拿到按同一 sample 顺序排列的 text / region frozen features 和 phrase-region records 后，先 stage raw manifest：

```bash
python scripts/multimodal/stage_refcoco_raw.py \
  refcoco \
  data/raw_multimodal/refcoco \
  --splits data/raw_multimodal/_downloads/refcoco_splits.json \
  --records data/raw_multimodal/_downloads/refcoco_phrase_region_records.json \
  --text-features data/raw_multimodal/_downloads/refcoco_text_features.npy \
  --region-features data/raw_multimodal/_downloads/refcoco_region_features.npy \
  --license-tag refcoco-coco2014 \
  --preprocessing-version refcoco-frozen-features-v0.1

python scripts/multimodal/build_cache.py \
  refcoco \
  data/raw_multimodal/refcoco \
  data/multimodal_cache \
  --version v0.1
```

`--records` 至少需要这些字段：

```json
{
  "records": [
    {
      "source_id": "image123::caption0::phrase4",
      "image_id": "image123",
      "caption_id": "caption0",
      "phrase_span": {"start": 3, "end": 6},
      "region_box": [0.1, 0.2, 0.7, 0.9],
      "target_region_index": 1,
      "candidate_region_source": "detector-or-annotation-version",
      "box_coordinate_convention": "xyxy_normalized"
    }
  ]
}
```

## 3. Region-text grounding 数据

### 3.1 RefCOCO / RefCOCO+ / RefCOCOg

RefCOCO 系列需要 COCO 2014 train images、COCO 2014 annotations 和 UNC refer annotations。TensorFlow Datasets 也明确标注该数据需要 manual download。

```bash
cd ~/work/Operator-Valued-Hyper-Attention
mkdir -p data/raw_multimodal/_downloads/refcoco

aria2c -c -x16 -s16 -k1M \
  -d data/raw_multimodal/_downloads/refcoco \
  -o train2014.zip \
  http://images.cocodataset.org/zips/train2014.zip

aria2c -c -x16 -s16 -k1M \
  -d data/raw_multimodal/_downloads/refcoco \
  -o annotations_trainval2014.zip \
  http://images.cocodataset.org/annotations/annotations_trainval2014.zip

aria2c -c -x16 -s16 -k1M \
  -d data/raw_multimodal/_downloads/refcoco \
  -o refcoco.zip \
  https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip

aria2c -c -x16 -s16 -k1M \
  -d data/raw_multimodal/_downloads/refcoco \
  -o refcoco_plus.zip \
  https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip

aria2c -c -x16 -s16 -k1M \
  -d data/raw_multimodal/_downloads/refcoco \
  -o refcocog.zip \
  https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip
```

如果 UNC 直链失败，用 Web Archive fallback：

```bash
aria2c -c -x16 -s16 -k1M \
  -d data/raw_multimodal/_downloads/refcoco \
  -o refcoco.zip \
  https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip
```

解压：

```bash
mkdir -p data/raw_multimodal/_downloads/refcoco/extracted
unzip -q data/raw_multimodal/_downloads/refcoco/annotations_trainval2014.zip \
  -d data/raw_multimodal/_downloads/refcoco/extracted
unzip -q data/raw_multimodal/_downloads/refcoco/refcoco.zip \
  -d data/raw_multimodal/_downloads/refcoco/extracted
unzip -q data/raw_multimodal/_downloads/refcoco/refcoco_plus.zip \
  -d data/raw_multimodal/_downloads/refcoco/extracted
unzip -q data/raw_multimodal/_downloads/refcoco/refcocog.zip \
  -d data/raw_multimodal/_downloads/refcoco/extracted
```

解压后先把 UNC refer annotations 和 COCO instances 统一转成本仓库 `stage_refcoco_raw.py` 需要的 phrase-region records / splits。下面的 `--refs` 路径要按解压后的实际 `refs*.json` / `refs*.p` 文件替换：

```bash
python scripts/multimodal/build_refcoco_stage_records.py \
  refcoco \
  data/raw_multimodal/_downloads/refcoco_stage_inputs \
  --refs data/raw_multimodal/_downloads/refcoco/extracted/replace_with_refcoco_refs.json \
  --instances data/raw_multimodal/_downloads/refcoco/extracted/annotations/instances_train2014.json \
  --candidate-region-source coco_gt_box
```

这个步骤只生成 `refcoco_phrase_region_records.json` 和 `refcoco_splits.json`；正式训练前仍需要用冻结的 text / region feature extractor 产出与这些 records 顺序完全一致的 `refcoco_text_features.npy` 和 `refcoco_region_features.npy`。

如果外部 frozen feature extractor 输出的是 feature bank，而不是已经按 `refcoco_splits.json` 排好序的数组，必须先用 source_id 列表严格重排：

```bash
python scripts/multimodal/align_refcoco_stage_features.py \
  refcoco \
  data/raw_multimodal/_downloads/refcoco_stage_inputs \
  --splits data/raw_multimodal/_downloads/refcoco_stage_inputs/refcoco_splits.json \
  --records data/raw_multimodal/_downloads/refcoco_stage_inputs/refcoco_phrase_region_records.json \
  --text-features data/raw_multimodal/_downloads/refcoco_text_feature_bank.npy \
  --text-source-ids data/raw_multimodal/_downloads/refcoco_text_feature_source_ids.txt \
  --region-features data/raw_multimodal/_downloads/refcoco_region_feature_bank.npy \
  --region-source-ids data/raw_multimodal/_downloads/refcoco_region_feature_source_ids.txt \
  --feature-version refcoco-frozen-features-v0.1
```

该命令会写出 `refcoco_text_features.npy`、`refcoco_region_features.npy` 和 `refcoco_feature_alignment_manifest.json`，然后再执行前面的 `stage_refcoco_raw.py`。

### 3.2 Flickr30k Entities

优先用 GitHub 拉取 Entities annotations，用 Hugging Face/Xet 下载 Flickr30k images/captions。HF 数据集不是唯一权威源，正式实验前要记录具体 repo、revision 和 license。

```bash
cd ~/work/Operator-Valued-Hyper-Attention
mkdir -p data/raw_multimodal/_downloads/flickr30k_entities

git clone --depth 1 \
  https://github.com/BryanPlummer/flickr30k_entities.git \
  data/raw_multimodal/_downloads/flickr30k_entities/annotations_repo

hf download cjc/flickr30k \
  --repo-type dataset \
  --local-dir data/raw_multimodal/_downloads/flickr30k_entities/hf_flickr30k \
  --local-dir-use-symlinks False
```

若 `cjc/flickr30k` 不稳定，可替换为：

```bash
hf download nlphuji/flickr30k \
  --repo-type dataset \
  --local-dir data/raw_multimodal/_downloads/flickr30k_entities/hf_flickr30k_nlphuji \
  --local-dir-use-symlinks False
```

### 3.3 Visual Genome subset

Visual Genome 全量较大，当前计划中只建议作为 Region-text grounding 的补充 subset，不要在根分区空间不足时下载全量。

```bash
cd ~/work/Operator-Valued-Hyper-Attention
mkdir -p data/raw_multimodal/_downloads/visual_genome

aria2c -c -x16 -s16 -k1M \
  -d data/raw_multimodal/_downloads/visual_genome \
  -o image_data.json.zip \
  https://visualgenome.org/static/data/dataset/image_data.json.zip

aria2c -c -x16 -s16 -k1M \
  -d data/raw_multimodal/_downloads/visual_genome \
  -o region_descriptions.json.zip \
  https://visualgenome.org/static/data/dataset/region_descriptions.json.zip
```

只在空间确认足够后再下载 images：

```bash
aria2c -c -x16 -s16 -k1M \
  -d data/raw_multimodal/_downloads/visual_genome \
  -o images.zip \
  https://visualgenome.org/static/data/dataset/images.zip

aria2c -c -x16 -s16 -k1M \
  -d data/raw_multimodal/_downloads/visual_genome \
  -o images2.zip \
  https://visualgenome.org/static/data/dataset/images2.zip
```

## 4. Sentiment / emotion 数据

### 4.1 CMU-MOSEI / CMU-MOSI

正式来源优先使用 CMU Multimodal SDK；CMU SDK 文档说明 `mmdatasdk` 负责下载和处理 computational sequences。MultiBench README 也说明 MOSI/MOSEI 的 ready-to-go aligned data 可从 MultiBench 获取。

```bash
cd ~/work/Operator-Valued-Hyper-Attention
source .venv/bin/activate

python -m pip install -U git+https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
mkdir -p data/raw_multimodal/_downloads/cmu_sdk
```

下载 MOSEI/MOSI 的 high-level features 和 labels：

```bash
python - <<'PY'
from pathlib import Path
import mmdatasdk

targets = {
    "cmu_mosei": mmdatasdk.cmu_mosei,
    "cmu_mosi": mmdatasdk.cmu_mosi,
}
for name, spec in targets.items():
    out = Path("data/raw_multimodal/_downloads/cmu_sdk") / name
    out.mkdir(parents=True, exist_ok=True)
    recipe = {}
    for attr in ("highlevel", "labels"):
        value = getattr(spec, attr, {})
        if isinstance(value, dict):
            recipe.update(value)
    print(f"Downloading {name} sequences -> {out}")
    mmdatasdk.mmdataset(recipe, str(out))
PY
```

如果 CMU SDK 速度慢，可先用 HF/Xet 下载公开预处理版本作为开发缓存，但正式主表要保留来源和版本，不要混写成原始 CMU SDK 数据：

```bash
hf download reeha-parkar/cmu-mosei-comp-seq \
  --repo-type dataset \
  --local-dir data/raw_multimodal/_downloads/cmu_mosei_hf_comp_seq \
  --local-dir-use-symlinks False
```

CMU SDK 下载完成后，先检查下载目录里的 `.csd` / `.h5` / `.json` sequence 文件。这个检查脚本会列出 sample count、feature shape，并按文件名给出 text/audio/vision/labels 候选和下一条 extract 命令；正式执行前仍要人工审阅候选是否符合你选定的特征方案：

```bash
python scripts/multimodal/write_cmu_sdk_splits.py \
  cmu_mosei \
  data/raw_multimodal/_downloads/cmu_mosei_splits.json

python scripts/multimodal/inspect_cmu_sdk_sequences.py \
  cmu_mosei \
  data/raw_multimodal/_downloads/cmu_sdk/cmu_mosei \
  --splits data/raw_multimodal/_downloads/cmu_mosei_splits.json \
  --stage-output-dir data/raw_multimodal/_downloads/cmu_mosei_stage_inputs \
  --temporal-policy mean \
  --preprocessing-version cmu-mosei-cmu-sdk-mean-v0.1 \
  --output data/raw_multimodal/_downloads/cmu_mosei_sequence_inspection.json
```

然后把 computational sequences 显式转成 `stage_cmu_sentiment_raw.py` 需要的 `.npy` 输入。可以直接审阅并复制 inspection JSON 里的 `suggested_extract_command`；如果手工写命令，下面的四个 `.csd` 文件名需要按实际下载目录替换：

```bash
TEXT_FEATURE_CSD=data/raw_multimodal/_downloads/cmu_sdk/cmu_mosei/replace_with_text_feature.csd
AUDIO_FEATURE_CSD=data/raw_multimodal/_downloads/cmu_sdk/cmu_mosei/replace_with_audio_feature.csd
VISUAL_FEATURE_CSD=data/raw_multimodal/_downloads/cmu_sdk/cmu_mosei/replace_with_visual_feature.csd
LABELS_CSD=data/raw_multimodal/_downloads/cmu_sdk/cmu_mosei/replace_with_labels.csd

python scripts/multimodal/extract_cmu_sdk_stage_inputs.py \
  cmu_mosei \
  data/raw_multimodal/_downloads/cmu_mosei_stage_inputs \
  --splits data/raw_multimodal/_downloads/cmu_mosei_splits.json \
  --text-sequence "$TEXT_FEATURE_CSD" \
  --audio-sequence "$AUDIO_FEATURE_CSD" \
  --visual-sequence "$VISUAL_FEATURE_CSD" \
  --label-sequence "$LABELS_CSD" \
  --temporal-policy mean \
  --sentiment-column 0 \
  --emotion-columns 1: \
  --license-tag cmu-multimodal-sdk \
  --preprocessing-version cmu-mosei-cmu-sdk-mean-v0.1
```

`--temporal-policy strict` 要求每个样本的 sequence shape 完全一致；`mean` 会做 utterance-level mean pooling，适合先打通 public acceptance / smoke，不应直接包装成最终顶会主表的强特征方案。

然后把 stage inputs 写成本仓库 raw manifest：

```bash
python scripts/multimodal/stage_cmu_sentiment_raw.py \
  cmu_mosei \
  data/raw_multimodal/cmu_mosei \
  --splits data/raw_multimodal/_downloads/cmu_mosei_stage_inputs/cmu_mosei_splits.json \
  --text-features data/raw_multimodal/_downloads/cmu_mosei_stage_inputs/cmu_mosei_text_features.npy \
  --audio-features data/raw_multimodal/_downloads/cmu_mosei_stage_inputs/cmu_mosei_audio_features.npy \
  --visual-features data/raw_multimodal/_downloads/cmu_mosei_stage_inputs/cmu_mosei_visual_features.npy \
  --sentiment-labels data/raw_multimodal/_downloads/cmu_mosei_stage_inputs/cmu_mosei_sentiment.npy \
  --emotion-labels data/raw_multimodal/_downloads/cmu_mosei_stage_inputs/cmu_mosei_emotion.npy \
  --feature-version text=cmu-mosei-cmu-sdk-mean-v0.1:text \
  --feature-version audio=cmu-mosei-cmu-sdk-mean-v0.1:audio \
  --feature-version vision=cmu-mosei-cmu-sdk-mean-v0.1:vision \
  --license-tag cmu-multimodal-sdk \
  --preprocessing-version cmu-mosei-cmu-sdk-mean-v0.1
```

### 4.2 MELD

MELD 官方页面提供两个下载入口，其中一个就是 Hugging Face `declare-lab/MELD`。

```bash
cd ~/work/Operator-Valued-Hyper-Attention
mkdir -p data/raw_multimodal/_downloads/meld

aria2c -c -x16 -s16 -k1M \
  -d data/raw_multimodal/_downloads/meld \
  -o MELD.Raw.tar.gz \
  https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz

mkdir -p data/raw_multimodal/_downloads/meld/extracted
tar -xzf data/raw_multimodal/_downloads/meld/MELD.Raw.tar.gz \
  -C data/raw_multimodal/_downloads/meld/extracted
```

## 5. 下载后校验与 cache 初始化

下载只是第一步。正式进入 OVHA 训练前，必须生成本仓库 adapter 期望的 frozen features / labels / metadata / splits，然后运行 fail-fast 初始化与 cache 校验。

先检查 adapter 期望的 raw 文件：

```bash
python scripts/multimodal/build_cache.py refcoco data/raw_multimodal/refcoco data/multimodal_cache --version v0.1
python scripts/multimodal/build_cache.py cmu_mosei data/raw_multimodal/cmu_mosei data/multimodal_cache --version v0.1
python scripts/multimodal/build_cache.py meld data/raw_multimodal/meld data/multimodal_cache --version v0.1
```

如果 raw manifest 还不完整，这些命令应返回 JSON 错误并列出缺失文件；这不是失败，而是防止 silent drop。

完整 cache 生成后再跑：

```bash
python scripts/multimodal/validate_cache.py data/multimodal_cache refcoco v0.1 --splits train val test
python scripts/multimodal/validate_cache.py data/multimodal_cache cmu_mosei v0.1 --splits train val test
python scripts/multimodal/validate_cache.py data/multimodal_cache meld v0.1 --splits train val test
```

如果 controlled-v1 gate 已经有训练型报告，优先用一条 acceptance 命令把 raw manifest、formal cache、controlled public-entry gate 和 T5 public smoke 串起来。当前仓库已归档的 controlled 报告路径是：

```text
outputs/multimodal/controlled_v1_smoke/seed_101/controlled_report.json
```

RefCOCO public acceptance：

```bash
python scripts/multimodal/accept_public_data.py \
  configs/multimodal_refcoco_public_smoke.json \
  --raw-root data/raw_multimodal/refcoco \
  --cache-root data/multimodal_cache \
  --controlled-report outputs/multimodal/controlled_v1_smoke/seed_101/controlled_report.json \
  --train-smoke-steps 1 \
  --train-baseline-smoke-steps 1 \
  --train-all-config-seeds \
  --train-split train \
  --eval-smoke-split val \
  --artifact-root outputs/multimodal/refcoco_public_acceptance_multiseed
```

CMU-MOSEI public acceptance：

```bash
python scripts/multimodal/accept_public_data.py \
  configs/multimodal_cmu_mosei_public_smoke.json \
  --raw-root data/raw_multimodal/cmu_mosei \
  --cache-root data/multimodal_cache \
  --controlled-report outputs/multimodal/controlled_v1_smoke/seed_101/controlled_report.json \
  --train-smoke-steps 1 \
  --train-baseline-smoke-steps 1 \
  --train-all-config-seeds \
  --train-split train \
  --eval-smoke-split val \
  --artifact-root outputs/multimodal/cmu_mosei_public_acceptance_multiseed
```

这一步会跑配置里的 3 个开发种子，并为同特征 baseline 训练一轮 smoke probe，产出 raw metrics、diagnostics、statistics preview 和 robustness preview。它仍然只是 public smoke acceptance，不是顶会主表；主表必须继续补齐强 baseline、完整 multi-seed、统计检验和真实 robustness stress。

生成真实 public 主实验的后处理 runbook。这个 runbook 不会把 smoke artifact 当主表；它假设你已经用完整训练流程产出了真正的 `raw_metrics.jsonl`、`diagnostics.jsonl` 和 `robustness_rows.jsonl`，然后把 RefCOCO / CMU-MOSEI 的 public gate bundle 与最终 topconf entry manifest 串起来：

```bash
python scripts/multimodal/build_public_main_runbook.py \
  --output-dir outputs/multimodal/public_main_runbook \
  --cache-root data/multimodal_cache \
  --controlled-report outputs/multimodal/controlled_v1_smoke/seed_101/controlled_report.json

bash -n outputs/multimodal/public_main_runbook/public_main_commands.sh
bash outputs/multimodal/public_main_runbook/public_main_commands.sh
```

默认真实主实验输入路径是：

```text
outputs/multimodal/refcoco_main/raw_metrics.jsonl
outputs/multimodal/refcoco_main/diagnostics.jsonl
outputs/multimodal/refcoco_main/robustness_rows.jsonl
outputs/multimodal/cmu_mosei_main/raw_metrics.jsonl
outputs/multimodal/cmu_mosei_main/diagnostics.jsonl
outputs/multimodal/cmu_mosei_main/robustness_rows.jsonl
```

如果你的完整训练产物在别的目录，用 `--region-raw-metrics`、`--region-diagnostics`、`--region-robustness-rows`、`--sentiment-raw-metrics`、`--sentiment-diagnostics`、`--sentiment-robustness-rows` 覆盖。不要把 `public_smoke_raw_metrics.jsonl` 或 `public_smoke_statistics_preview.json` 传给这个 runbook。

正式 public 训练得到 raw metrics / diagnostics / robustness rows 后，用同一个 bundle 入口生成 public gate report。RefCOCO / region-text 示例：

```bash
python scripts/multimodal/build_public_gate_report.py region_text \
  --raw-metrics outputs/multimodal/refcoco_main/raw_metrics.jsonl \
  --diagnostics outputs/multimodal/refcoco_main/diagnostics.jsonl \
  --robustness-rows outputs/multimodal/refcoco_main/robustness_rows.jsonl \
  --task phrase_region_grounding \
  --split test \
  --output-dir outputs/multimodal/refcoco_main/gate_bundle
```

CMU-MOSEI / sentiment-emotion 示例：

```bash
python scripts/multimodal/build_public_gate_report.py sentiment \
  --raw-metrics outputs/multimodal/cmu_mosei_main/raw_metrics.jsonl \
  --diagnostics outputs/multimodal/cmu_mosei_main/diagnostics.jsonl \
  --robustness-rows outputs/multimodal/cmu_mosei_main/robustness_rows.jsonl \
  --task sentiment_emotion \
  --split test \
  --output-dir outputs/multimodal/cmu_mosei_main/gate_bundle
```

两个 gate bundle 都通过后，用 bundle 目录直接做最终主实验入口验证：

```bash
python scripts/multimodal/validate_topconf_entry.py \
  --controlled-report outputs/multimodal/controlled_v1_smoke/seed_101/controlled_report.json \
  --region-gate-bundle outputs/multimodal/refcoco_main/gate_bundle \
  --sentiment-gate-bundle outputs/multimodal/cmu_mosei_main/gate_bundle \
  --cache-target refcoco data/multimodal_cache refcoco v0.1 val,test \
  --cache-target cmu_mosei data/multimodal_cache cmu_mosei v0.1 val,test
```

为了归档复现，推荐用脚本生成并立即验证 manifest。脚本会把路径写成相对 `topconf_entry_manifest.json` 所在目录的形式：

```bash
python scripts/multimodal/build_topconf_entry_manifest.py \
  --output outputs/multimodal/topconf_entry_manifest.json \
  --controlled-report outputs/multimodal/controlled_v1_smoke/seed_101/controlled_report.json \
  --region-gate-bundle outputs/multimodal/refcoco_main/gate_bundle \
  --sentiment-gate-bundle outputs/multimodal/cmu_mosei_main/gate_bundle \
  --cache-target refcoco data/multimodal_cache refcoco v0.1 val,test \
  --cache-target cmu_mosei data/multimodal_cache cmu_mosei v0.1 val,test \
  --validate
```

## 6. 速度与稳定性建议

- 大文件优先 `aria2c -c -x16 -s16 -k1M`，支持断点续传。
- Hugging Face 优先 `hf download` + `hf_xet` + `HF_XET_HIGH_PERFORMANCE=1`。
- 国内网络慢时用 `HF_ENDPOINT=https://hf-mirror.com`，但正式结果归档里要记录是否使用镜像。
- 下载全部 Visual Genome / COCO 前先 `df -h .`，当前服务器根分区曾接近满载，不建议盲目全量下载。
- 不要用 test split 生成训练 pseudo-label；不要把 `true_active_operator`、`corruption_strength` 等 hidden/control metadata 放进模型输入。

## 7. 参考来源

- Hugging Face Hub 环境变量与 Xet: https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables
- MELD 官方下载页: https://affective-meld.github.io/
- CMU Multimodal SDK: https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK
- MultiBench MOSI/MOSEI processed data 说明: https://github.com/pliang279/MultiBench
- RefCOCO TFDS manual download 说明: https://www.tensorflow.org/datasets/catalog/ref_coco
- Flickr30k Entities annotations: https://github.com/BryanPlummer/flickr30k_entities
- Visual Genome data readme: https://homes.cs.washington.edu/~ranjay/visualgenome/api_readme.html
