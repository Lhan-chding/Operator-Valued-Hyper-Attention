# OVHA 项目完整通俗说明：原理、代码实现、数据流程与当前进展

---

## 1. 一句话说明这个项目

普通模型通常是：

```text
输入 token -> 输出向量
```

OVHA 想做的是：

```text
输入上下文 + 当前查询 -> 选择/组合一组可解释的 operator primitive -> 输出预测
```

> 模型不是死记一种模式，而是先准备几种“做事方式”，再根据当前样本判断该用哪几种、每种用多少、参数怎么调。

这些“做事方式”在代码里叫 `candidate primitive`，当前多模态主线固定为四个：

| 名称 | 全名 | 通俗解释 |
|---|---|---|
| `TLEO` | Typed Local Evidence Operator | 看局部证据，类似“离查询最近、最相关的局部 token/区域在说什么”。 |
| `SPO` | Semantic Prototype Operator | 看语义原型，类似“这个样本更像哪类语义模板”。 |
| `LRIO` | Low-Rank Interaction Operator | 看跨模态低秩交互，类似“文本和图像/音频之间有哪些压缩后的组合关系”。 |
| `CATO` | Cross-modal Alignment Transport Operator | 看跨模态对齐/搬运，类似“文本短语应该对齐到哪个图像区域，或者某个模态信息该搬到哪个查询位置”。 |

另有一个很重要但不直接进入最终四路候选堆叠的机制：

| 名称 | 全名 | 通俗解释 |
|---|---|---|
| `RCEO` | Reliability / Corruption Evidence Operator | 可靠性先验，判断某个模态是否缺失、坏掉、质量低，然后影响 router 的选择。 |

---

## 2. 项目的长期边界

- PDEBench：早期验证 operator learning 能不能跑通。
- controlled synthetic / controlled-v2：验证 router、adapter、memory 这些机制是否真的有效。
- RefCOCO / CMU-MOSEI / MELD：当前多模态公开数据主线。
- 未来还可以接图像、文本、音频、轨迹、连续场、密集预测等任务。

项目真正想证明的是：

```text
Operator-Valued Attention
  + Operator Memory
  + Hyper-Adapter
  + Primitive Routing
  + 可解释 operator decomposition
```

可以作为一种通用的跨模态/跨域 operator-valued attention 框架。

---

## 3. 代码目录总览

当前仓库大致可以分成几层。

```text
README.md
docs/
theory_notes/
reports/
configs/
moat_ovha/
moat_ovha_torch/
scripts/
tests/
data/
outputs/
```

各目录含义：

| 路径 | 作用 |
|---|---|
| `README.md` | 早期 Phase 1 项目入口说明。 |
| `docs/` | 项目协议、阶段说明、数据协议、运行策略和中文说明。 |
| `theory_notes/` | OVHA 数学定义、特例和理论笔记。 |
| `reports/` | 阶段性报告，比如 PDEBench 只是架构可行性证据。 |
| `configs/` | 所有实验配置。现在多模态主线主要看 `multimodal_*_public_main.json`。 |
| `moat_ovha/` | 早期零依赖 Python 原型。 |
| `moat_ovha_torch/` | PyTorch 实现主目录。当前真实训练、模型和多模态逻辑主要在这里。 |
| `scripts/` | 训练、数据准备、缓存构建、验收、报告生成脚本。 |
| `tests/` | 合同测试、协议测试、训练/评估测试。 |
| `data/` | 本地数据目录，通常不进 git。 |
| `outputs/` | 训练、验证、报告输出目录，通常不进 git。 |

---

## 4. 项目演进路线

### 4.1 Phase 1：最小原型

早期代码在：

```text
moat_ovha/
```

目标是证明最小 OVHA 思想能写成可运行代码。

这一阶段重点不是大规模训练，而是把概念立住：

- memory
- hyper-adapter
- router
- primitive
- OVHA layer
- toy operator zoo
- baseline / ablation

### 4.2 Phase 1.5：PyTorch 化

主目录：

```text
moat_ovha_torch/
```

这一阶段把早期原型升级成 PyTorch 训练栈：

- `Phase15Config` 管理实验参数。
- `train/trainer.py` 做训练。
- `eval/evaluator.py` 做评估。
- `models/ovha.py` 是单域 operator 版本的 OVHA。
- `data/component_stress_zoo.py` 生成 controlled stress 数据。

这一阶段还有 PDEBench、FNO、DeepONet 等对接尝试。

### 4.3 Phase 1.6 / 1.7：机制诊断和 controlled gate

主要解决的问题：

1. 训练和评估必须加载真实 checkpoint，不能拿未训练模型乱报结果。
2. controlled-v2 要能证明 router、memory、adapter 是否真的学到结构。
3. public benchmark 进入前必须有 controlled go/no-go 报告。

关键词：

| 词 | 含义 |
|---|---|
| `controlled` | 人造受控任务，知道隐藏真值，用来验证机制。 |
| `oracle` | 上帝视角，只能用于诊断，不能当模型输入。 |
| `go/no-go` | 是否允许进入下一阶段的门槛报告。 |
| `ablation` | 去掉某个模块，看性能是否下降，用来证明模块有必要。 |

### 4.4 当前主线：多模态公开数据

当前真正推进的是：

```text
RefCOCO     -> phrase-region grounding
CMU-MOSEI   -> sentiment/emotion
MELD        -> sentiment/emotion，已完成缓存，但当前 public main 重点先跑 RefCOCO 和 CMU-MOSEI
```

其中：

- RefCOCO 用 COCO2014 图片和 RefCOCO 指代表达。
- CMU-MOSEI 用 CMU MultimodalSDK 的官方 computational sequences。
- MELD 用 RoBERTa / Wav2Vec2 / CLIP 抽取文本、音频、视觉特征。

---

## 5. 数据在项目里怎么表示

### 5.1 多模态 batch 的核心数据结构

代码位置：

```text
moat_ovha_torch/data/multimodal/typed_batch.py
```

核心类：

```python
MultimodalEpisodeBatch
```

它可以理解成一次训练/评估用的一批样本。

字段如下：

| 字段 | 通俗解释 |
|---|---|
| `fields` | 各个模态的 token，比如 text、region、audio、vision。 |
| `query` | 当前要回答的问题/查询。 |
| `target_y` | 真实答案，训练时用来算 loss。 |
| `target_mask` | 哪些答案位置有效。 |
| `task_type` | 任务类型，比如 `phrase_region_grounding` 或 `sentiment_emotion`。 |
| `split` | 数据划分，比如 `train`、`val`、`test`。 |
| `source_dataset` | 数据集名字，比如 `refcoco`、`cmu_mosei`。 |
| `supervision` | 监督信息，比如标签、对齐、bbox、缺失模态 mask。 |
| `provenance` | 来源记录，比如 source_id、license、预处理版本。 |
| `hidden` | 隐藏真值，只能 controlled 诊断用，不能进模型输入。 |

### 5.2 `TokenField`

代码：

```python
@dataclass(frozen=True)
class TokenField:
    modality: str
    x: Any
    pos: Any
    mask: Any
    quality: Any | None = None
    attrs: dict[str, Any] | None = None
```

通俗解释：

| 字段 | 含义 |
|---|---|
| `modality` | 模态名，例如 `text`、`region`、`audio`、`vision`。 |
| `x` | 这个模态的特征向量，形状一般是 `[B, N, D]`。 |
| `pos` | token 的位置或顺序信息，形状一般是 `[B, N, P]`。 |
| `mask` | 哪些 token 有效，形状一般是 `[B, N]`。 |
| `quality` | 质量分数，比如某个模态是否可靠。 |
| `attrs` | 公开属性，不能含 hidden/oracle/true_* 这类泄漏字段。 |

变量解释：

| 符号 | 含义 |
|---|---|
| `B` | batch size，一批里有多少样本。 |
| `N` | token 数量，比如一句话 token 数、图像候选区域数、音频片段数。 |
| `D` | 特征维度，比如 CLIP 是 512，CMU 文本是 300。 |
| `P` | 位置特征维度。 |

### 5.3 `QueryField`

```python
class QueryField:
    x: Any
    pos: Any
    query_type: Any
    mask: Any
```

通俗解释：

| 字段 | 含义 |
|---|---|
| `x` | 查询本身的特征。 |
| `pos` | 查询位置。 |
| `query_type` | 查询属于哪种关系/任务提示。 |
| `mask` | 哪些查询有效。 |

形状通常是：

```text
query.x      [B, Q, Dq]
target_y     [B, Q, Dy]
target_mask  [B, Q]
```

其中：

| 符号 | 含义 |
|---|---|
| `Q` | 每个样本里查询的数量。 |
| `Dq` | query 特征维度。 |
| `Dy` | 输出答案维度。 |

### 5.4 `SupervisionBank`

`SupervisionBank` 是监督信息仓库。

| 字段 | 含义 |
|---|---|
| `task_label` | 主任务标签，比如情绪分数、区域匹配标签。 |
| `alignment_pairs` | 跨模态对齐对，比如短语和区域的对应。 |
| `alignment_weights` | 对齐权重。 |
| `bbox_targets` | 边界框目标。 |
| `region_targets` | 区域目标。 |
| `timestamp_targets` | 时间戳目标。 |
| `modality_missing_mask` | 哪个模态缺失。 |
| `corruption_metadata` | 腐蚀/噪声相关元信息，只能作为监督/诊断，不能作为模型输入。 |
| `weak_labels` | 弱标签。 |
| `pseudo_label_source` | 伪标签来源。 |

### 5.5 `ProvenanceBank`

`ProvenanceBank` 记录数据来源，保证可追溯。

| 字段 | 含义 |
|---|---|
| `source_id` | 样本唯一 ID。 |
| `original_split` | 原始数据划分。 |
| `raw_ref` | 原始文件或原始记录引用。 |
| `license_tag` | 数据许可标签。 |
| `preprocessing_version` | 预处理版本。 |
| `feature_extractor_version` | 特征抽取器版本，例如 CLIP / RoBERTa / CMU SDK。 |
| `pseudo_label_version` | 伪标签版本。 |

### 5.6 防止“作弊”的输入检查

`typed_batch.py` 里有一组禁止字段：

```python
FORBIDDEN_MULTIMODAL_INPUT_KEYS
```

它会拒绝这些信息进入模型：

- `true_active_operator`
- `true_router`
- `true_adapter`
- `true_alignment`
- `hidden`
- `oracle`
- `corruption_strength`
- `mismatch_source_id`

---

## 6. 缓存格式：为什么要先 build cache

公开数据不能直接乱读原始文件。项目要求先转成统一缓存：

```text
data/multimodal_cache/<dataset_name>/v0.1/
```

代码位置：

```text
moat_ovha_torch/data/multimodal/cache_schema.py
scripts/multimodal/build_cache.py
scripts/multimodal/validate_cache.py
```

缓存里必须有：

| 文件/目录 | 作用 |
|---|---|
| `data_card.json` | 数据集说明，包含 dataset、modalities、tasks、leakage_controls。 |
| `splits.json` | train/val/test 划分。 |
| `samples.parquet` | 样本总表。 |
| `checksums.json` | 所有产物哈希，防止文件被悄悄改。 |
| `token_fields/manifest_<split>.json` | 每个 split 的 token 文件清单。 |
| `masks/` | mask 文件。 |
| `positions/` | 位置文件。 |
| `supervision/` | 标签、对齐、bbox、缺失模态等监督。 |
| `provenance/` | source ids、sample records、failed samples、feature versions。 |

为什么这么复杂：

1. 保证 OVHA 和所有 baseline 用同一套 frozen features。
2. 保证 train/val/test 不混。
3. 保证公开数据没有 hidden/oracle 泄漏。
4. 保证结果可以复现。
5. 后续写论文时能说清楚每个数字来自哪里。

---

## 7. 当前三个公开数据集的关系

### 7.1 RefCOCO 和 COCO 的关系

COCO 是图像数据集，提供图片和目标框。

RefCOCO 是建立在 COCO 图片上的“指代表达”数据集。

举例：

```text
COCO：有一张图片，里面有人、狗、车，还有对应 bbox。
RefCOCO：给一句话，比如 “the man in red shirt”，要求模型找到对应区域。
```

所以：

```text
COCO 提供图像和区域框
RefCOCO 提供文本短语和指向哪个区域的标注
```

项目里的 RefCOCO 任务是：

```text
phrase-region grounding
文本短语 -> 图像区域
```

当前特征：

```text
text   [142210, 1, 512]
region [142210, 1, 512]
```

512 来自 CLIP ViT-B/32。

### 7.2 CMU-MOSEI

CMU-MOSEI 是多模态情感/情绪数据集。

它包含：

- 文本
- 音频
- 视频/视觉
- sentiment 标签
- emotion 标签

当前不是重新用 RoBERTa / CLIP / Wav2Vec2 抽特征，而是使用 CMU MultimodalSDK 的官方 computational sequences：

```text
CMU_MOSEI_TimestampedWordVectors.csd  -> text
CMU_MOSEI_COVAREP.csd                 -> audio
CMU_MOSEI_VisualFacet42.csd           -> vision
CMU_MOSEI_Labels.csd                  -> labels
```

当前特征：

```text
text      [3225, 1, 300]
audio     [3225, 1, 74]
vision    [3225, 1, 35]
sentiment [3225, 1]
emotion   [3225, 6]
```

这里 `[3225, 1, 300]` 的意思是：

| 维度 | 含义 |
|---|---|
| `3225` | 样本数。 |
| `1` | 每个样本被 mean pooling 成一个时间步。 |
| `300` | 文本特征维度。 |

### 7.3 MELD

MELD 是对话情绪数据集。

当前已经完成正式特征抽取和缓存：

```text
sample_count = 13708
text   [13708, 16, 768]
audio  [13708, 32, 768]
visual [13708, 8, 768]
```

使用的抽取器：

| 模态 | 抽取器 |
|---|---|
| text | `FacebookAI/roberta-base` |
| audio | `facebook/wav2vec2-base-960h` |
| visual | `openai/clip-vit-base-patch32` |

MELD 当前缓存已通过验证，但当前 50k public main 长跑先集中在 RefCOCO 和 CMU-MOSEI。

---

## 8. 数据处理流水线

### 8.1 RefCOCO 流水线

RefCOCO 的处理顺序：

```text
下载/解压 RefCOCO + COCO2014
  -> build_refcoco_stage_records.py
  -> extract_refcoco_clip_features.py
  -> stage_refcoco_raw.py
  -> build_cache.py
  -> validate_cache.py
  -> accept_public_data.py
  -> run_public_main.py
  -> validate_public_main_artifacts.py
```

每一步的作用：

| 脚本 | 作用 |
|---|---|
| `build_refcoco_stage_records.py` | 把 RefCOCO refs 和 COCO annotations 转成项目内部 records/splits。 |
| `extract_refcoco_clip_features.py` | 用 CLIP 抽文本和图像区域特征。 |
| `stage_refcoco_raw.py` | 把抽好的 `.npy` 和 records 整理成 raw manifest 格式。 |
| `build_cache.py` | 把 raw manifest 写成正式 cache。 |
| `validate_cache.py` | 检查 cache 完整性。 |
| `accept_public_data.py` | 检查 raw/cache/controlled gate，并跑 public smoke。 |
| `run_public_main.py` | 正式多 seed、多模型训练评估。 |
| `validate_public_main_artifacts.py` | 检查正式产物是否覆盖 5 seeds 和所有 baseline。 |

### 8.2 CMU-MOSEI 流水线

CMU-MOSEI 的处理顺序：

```text
下载 CMU SDK .csd 文件
  -> inspect_cmu_sdk_sequences.py
  -> extract_cmu_sdk_stage_inputs.py
  -> stage_cmu_sentiment_raw.py
  -> build_cache.py
  -> validate_cache.py
  -> accept_public_data.py
  -> run_public_main.py
  -> validate_public_main_artifacts.py
```

每一步作用：

| 脚本 | 作用 |
|---|---|
| `inspect_cmu_sdk_sequences.py` | 检查 `.csd` 文件结构，找到 text/audio/vision/labels 序列。 |
| `extract_cmu_sdk_stage_inputs.py` | 从 CMU SDK sequences 读出特征，按 split 顺序写成 `.npy`。 |
| `stage_cmu_sentiment_raw.py` | 把 `.npy` 和标签整理成 raw manifest。 |
| `build_cache.py` | 写正式 cache。 |
| `validate_cache.py` | 检查 cache。 |
| `accept_public_data.py` | 验收 public 数据和 controlled entry。 |
| `run_public_main.py` | 正式训练评估。 |
| `validate_public_main_artifacts.py` | 验证正式结果产物。 |

### 8.3 为什么 CMU 不需要再用别的模型抽特征

RefCOCO 原始输入是图片和文本，所以需要 CLIP 抽 frozen features。

MELD 原始输入有文本、音频、视频，所以用了 RoBERTa、Wav2Vec2、CLIP 抽 frozen features。

CMU-MOSEI 下载的 `.csd` 本身就是官方计算好的 computational sequences。里面已经有文本、音频、视觉特征，所以这里不再用 RoBERTa/CLIP/Wav2Vec2 二次抽取。

---

## 9. 多模态 OVHA 模型怎么计算

核心模型代码：

```text
moat_ovha_torch/models/multimodal/ovha_multimodal.py
```

核心类：

```python
MultimodalOVHA
```

forward 流程可以写成：

```text
MultimodalEpisodeBatch
  -> batch.model_inputs() 检查无泄漏
  -> MultimodalEvidenceEncoder 提取证据
  -> MultimodalOperatorMemory 生成每个 candidate 的 memory
  -> RCEOReliabilityPrior 生成可靠性偏置
  -> MultimodalJointRouterAdapter 同时生成 router weights 和 adapter params
  -> 四个 primitive 分别预测 candidate value
  -> router weights 加权求和
  -> y_hat
```

### 9.1 `y_hat`

`y_hat` 是模型最终预测。

形状通常是：

```text
[B, Q, Dy]
```

意思是：

- B 个样本
- 每个样本 Q 个查询
- 每个查询输出 Dy 维答案

### 9.2 `candidate_values`

四个 primitive 各自给一个候选答案：

```text
TLEO value
SPO value
LRIO value
CATO value
```

堆起来叫：

```python
candidate_values
```

形状：

```text
[B, Q, 4, Dy]
```

中间的 `4` 就是四个 candidate。

### 9.3 `router_weights`

router 给四个 candidate 分配权重：

```text
[B, Q, 4]
```

例如某个 query 的权重可能是：

```text
TLEO 0.10
SPO  0.20
LRIO 0.30
CATO 0.40
```

最终预测：

```text
y_hat =
  0.10 * TLEO_value
+ 0.20 * SPO_value
+ 0.30 * LRIO_value
+ 0.40 * CATO_value
```

代码里是：

```python
y_hat = (router_weights.unsqueeze(-1) * candidate_values).sum(dim=-2)
```

### 9.4 `evidence`

代码位置：

```text
moat_ovha_torch/models/multimodal/evidence.py
```

`MultimodalEvidenceEncoder` 把公开输入转换成几类证据：

| 字段 | 通俗解释 |
|---|---|
| `query_features` | query 编码后的特征。 |
| `global_features` | 所有模态 pooled 后的全局特征。 |
| `local_features` | 局部相关 token 特征，主要给 TLEO。 |
| `prototype_features` | 语义原型特征，主要给 SPO。 |
| `low_rank_features` | 模态交互特征，主要给 LRIO。 |
| `alignment_features` | 对齐/搬运特征，主要给 CATO。 |
| `candidate_evidence_logits` | evidence 对四个 candidate 的直接倾向。 |
| `local_entropy` | 局部注意力分布的熵，越高表示越不确定。 |
| `alignment_entropy` | 对齐分布的熵，越高表示对齐越散。 |
| `field_features` | 每个模态投影后的 token 特征。 |
| `diagnostics` | 诊断信息。 |

### 9.5 `memory_bank`

代码位置：

```text
moat_ovha_torch/models/multimodal/memory.py
```

`MultimodalOperatorMemory` 从全局特征生成每个 candidate 自己的记忆槽：

```text
memory_bank["TLEO"]
memory_bank["SPO"]
memory_bank["LRIO"]
memory_bank["CATO"]
```

形状大致是：

```text
[B, memory_tokens, d_model]
```

变量解释：

| 变量 | 含义 |
|---|---|
| `memory_tokens` | 每个 candidate 有多少个记忆 token。比如 16 就是每类 primitive 有 16 个记忆槽。 |
| `d_model` | 模型内部隐藏维度。比如 256 表示内部所有证据、memory、router、adapter 都映射到 256 维。 |

### 9.6 `router`

代码位置：

```text
moat_ovha_torch/models/multimodal/router.py
```

router 的输入：

- candidate memory
- evidence logits
- reliability bias
- query features

router 的输出：

| 字段 | 含义 |
|---|---|
| `logits` | softmax 前的分数。 |
| `weights` | softmax 后的权重。 |
| `logit_parts` | 分解后的来源：memory/evidence/reliability。 |
| `diagnostics` | router entropy、每个 candidate 平均 load 等。 |

公式简化：

```text
router_logits =
  memory_logit
+ evidence_logit
+ reliability_logit

router_weights = softmax(router_logits)
```

### 9.7 `hyper_adapter`

代码位置：

```text
moat_ovha_torch/models/multimodal/hyper_adapter.py
```

它的作用是给每个 candidate 生成参数，而不是只让 primitive 用固定参数。

当前允许的参数：

| Candidate | 参数 |
|---|---|
| `TLEO` | `lengthscale`, `local_temperature`, `scale`, `bias` |
| `SPO` | `prototype_temperature`, `prototype_logits_shift`, `scale`, `bias` |
| `LRIO` | `rank_logits`, `interaction_temperature`, `scale`, `bias` |
| `CATO` | `alignment_temperature`, `transport_scale`, `scale`, `bias` |

通俗解释：

| 参数 | 含义 |
|---|---|
| `lengthscale` | 局部窗口/局部关系尺度。 |
| `local_temperature` | 局部选择分布的温度，影响“尖锐还是平滑”。 |
| `prototype_temperature` | 原型选择温度。 |
| `prototype_logits_shift` | 调整原型偏好。 |
| `rank_logits` | 低秩交互里选择 rank 的倾向。 |
| `interaction_temperature` | 交互强弱/平滑程度。 |
| `alignment_temperature` | 对齐分布温度。 |
| `transport_scale` | 跨模态搬运强度。 |
| `scale` | 输出缩放。 |
| `bias` | 输出偏置。 |

### 9.8 `RCEO reliability prior`

代码位置：

```text
moat_ovha_torch/models/multimodal/reliability_prior.py
```

它根据每个模态的 `quality` 或 `mask` 推断模态可靠性。

如果某个模态缺失或质量低，RCEO 会给 router 一个额外偏置，帮助模型少依赖坏模态。

重要诊断：

| 字段 | 含义 |
|---|---|
| `modality_reliability` | 每个模态的平均可靠性。 |
| `reliability_bias_norm` | 可靠性偏置强度。 |
| `corruption_response` | 对缺失/损坏的响应程度。 |

### 9.9 四个 primitive

代码位置：

```text
moat_ovha_torch/models/multimodal/primitives/
```

#### TLEO

文件：

```text
typed_local_evidence.py
```

输入：

- `evidence.local_features`
- `memory_slot`
- adapter 参数

作用：根据局部证据预测。

#### SPO

文件：

```text
semantic_prototype.py
```

输入：

- `evidence.prototype_features`
- `memory_slot`
- adapter 参数

作用：根据语义原型预测。

#### LRIO

文件：

```text
low_rank_interaction.py
```

输入：

- `evidence.low_rank_features`
- `memory_slot`
- adapter 参数

作用：根据低秩跨模态交互预测。

#### CATO

文件：

```text
alignment_transport.py
```

输入：

- `evidence.alignment_features`
- `memory_slot`
- adapter 参数

作用：根据跨模态对齐和信息搬运预测。

---

## 10. 训练脚本怎么跑

正式公开主线训练入口：

```text
scripts/multimodal/run_public_main.py
```

核心参数：

| 参数 | 含义 |
|---|---|
| `config` | 实验配置 JSON。 |
| `--cache-root` | 正式缓存根目录。 |
| `--controlled-report` | controlled go/no-go 报告。公开数据训练前必须通过它。 |
| `--artifact-root` | 输出目录。 |
| `--train-steps` | OVHA full 每个 seed 训练多少步。 |
| `--baseline-train-steps` | 每个 baseline 每个 seed 训练多少步。 |
| `--train-split` | 训练集。 |
| `--eval-split` | 评估集。 |
| `--device` | `cpu` 或 `cuda`。 |
| `--d-model` | 内部隐藏维度。 |
| `--memory-tokens` | 每个 candidate 的 memory token 数。 |
| `--learning-rate` | 学习率。 |
| `--progress-interval` | 每多少 step 打印一次进度。 |

### 10.1 它内部做了什么

`run_public_main.py` 的顺序：

1. 读取 `MultimodalExperimentConfig`。
2. 检查不能是 smoke config。
3. 检查 `train_steps` 和 `baseline_train_steps` 必须大于 0。
4. 检查 OVHA 和 baseline 使用同一套 frozen features。
5. 验证 cache 是否完整。
6. 验证 controlled report 是否允许进入 public main。
7. 对每个 seed：
   - 加载 train batch。
   - 加载 eval batch。
   - 初始化 `MultimodalOVHA`。
   - 训练 OVHA full。
   - 在 test/val 上评估 OVHA full。
   - 训练所有 baseline。
   - 评估所有 baseline。
   - 生成 diagnostics 和 robustness rows。
8. 写出：
   - `raw_metrics.jsonl`
   - `diagnostics.jsonl`
   - `robustness_rows.jsonl`
9. 最后打印一个 JSON summary。

### 10.2 为什么现在会实时显示训练进度

最近新增了进度打印：

```text
[public-main:start]
[public-main:seed:start]
[public-main:model:start]
[public-main:train]
[public-main:model:done]
[public-main:seed:done]
```

这些输出写到 `stderr`，所以你用：

```bash
2>&1 | tee outputs/multimodal/xxx_train.log
```

时，终端和日志文件都会看到。

### 10.3 为什么建议用 `python -m`

远端环境里可能存在第三方包也叫 `scripts`，会干扰：

```text
from scripts.multimodal.run_public_smoke import ...
```

所以更稳的运行方式是：

```bash
PYTHONPATH="$PWD" python -m scripts.multimodal.run_public_main ...
```

这样 Python 会优先使用当前仓库里的 `scripts/` 包。

---

## 11. 配置文件怎么理解

当前两个正式公开主线配置：

```text
configs/multimodal_refcoco_public_main.json
configs/multimodal_cmu_mosei_public_main.json
```

公共字段解释：

| 字段 | 含义 |
|---|---|
| `name` | 实验名。 |
| `dataset_name` | 数据集名。 |
| `task_type` | 任务类型。 |
| `cache_version` | 缓存版本，目前是 `v0.1`。 |
| `cache_root` | 缓存根目录。 |
| `output_dir` | 默认输出目录。 |
| `seeds` | 随机种子列表。正式 main 用 5 个 seed。 |
| `main_table_seed_policy` | 主表 seed 策略。 |
| `training_stages` | 训练阶段。公开数据是 `T0` 和 `T5`。 |
| `candidate_names` | 四个 candidate：TLEO/SPO/LRIO/CATO。 |
| `baseline_names` | 要对比的 baseline 列表。 |
| `eval_splits` | 评估 split，一般是 `val` 和 `test`。 |
| `eval_episode_count` | 评估 episode 数。 |
| `enforce_same_features_for_baselines` | 是否强制 baseline 和 OVHA 用同一套特征。必须 true。 |
| `fail_on_missing_cache_artifact` | cache 缺文件是否直接失败。必须 true。 |
| `allow_hidden_losses` | 公开数据是否允许 hidden loss。必须 false。 |
| `losses_by_stage` | 每个阶段允许哪些 loss。 |
| `adapter_params_by_candidate` | 每个 candidate 可以生成哪些 adapter 参数。 |

### 11.1 `training_stages`

公开数据只允许：

```text
T0: cache_validation
T5: public main training/evaluation
```

controlled 数据有更多阶段，比如 T1/T2/T3/T4，用于机制诊断。

### 11.2 `baseline_names`

RefCOCO 的内部对照分两类：最简单的 same-feature sanity probe，以及 OVHA 自身消融。这里不再把 MDETR、GLIP、GroundingDINO、TransVG、LAVT、SeqTR 这类外部论文模型写进 `baseline_names`，因为它们不是当前 runner 训练出的同特征线性头。

| 名称 | 通俗解释 |
|---|---|
| `text_only` | 只看文本特征的 sanity probe。 |
| `region_only` | 只看图像区域特征的 sanity probe。 |
| `concat_fusion` | 文本和区域特征简单拼接的 sanity probe。 |
| `cato_only` | 只保留 CATO。 |
| `ovha_no_cato` | 去掉 CATO 的 OVHA 消融。 |
| `ovha_no_rceo` | 去掉可靠性先验。 |
| `ovha_no_evidence_router` | 去掉 evidence router。 |

CMU-MOSEI 的内部对照同样只保留 sanity probe 和 OVHA 消融。TFN/LMF、MulT、MISA、MAG-BERT、Self-MM 属于外部 SOTA/reference/reproduction 表，不放进 `baseline_names`。

| 名称 | 通俗解释 |
|---|---|
| `text_only` | 只看文本特征的 sanity probe。 |
| `audio_only` | 只看音频特征的 sanity probe。 |
| `vision_only` | 只看视觉特征的 sanity probe。 |
| `concat_fusion` | 文本、音频、视觉简单拼接的 sanity probe。 |
| `ovha_no_lrio` | 去掉 LRIO。 |
| `ovha_no_spo` | 去掉 SPO。 |
| `ovha_no_rceo` | 去掉 RCEO。 |
| `ovha_no_evidence_router` | 去掉 evidence router。 |

当前代码里的内部 baseline 训练在 `run_public_main.py` 里仍然是同特征线性探针式训练，用来回答“OVHA 结构是否比冻结特征的简单读出更有用”。外部 SOTA 对照由 `configs/multimodal_external_sota_references.json` 和 `scripts/multimodal/build_external_sota_runbook.py` 单独管理，不能混进同特征 baseline policy。

---

## 12. 输出产物怎么理解

正式 public main 输出目录：

```text
outputs/multimodal/refcoco_main/
outputs/multimodal/cmu_mosei_main/
```

核心文件：

| 文件 | 作用 |
|---|---|
| `raw_metrics.jsonl` | 每个 seed、每个模型的原始指标行。 |
| `diagnostics.jsonl` | OVHA 诊断信息，比如 router、adapter、candidate 统计。 |
| `robustness_rows.jsonl` | 鲁棒性/缺失模态/扰动相关行。 |
| `*_train.log` | 训练终端日志。 |

### 12.1 `raw_metrics.jsonl`

每行通常包含：

| 字段 | 含义 |
|---|---|
| `artifact_type` | 产物类型，正式 main 应是 `public_main_raw_metric`。 |
| `evidence_scope` | 证据范围，正式 main 应是 `public_main_table`。 |
| `dataset` | 数据集。 |
| `task` | 任务。 |
| `model` | 模型名，比如 `ovha_full`。 |
| `split` | `val` 或 `test`。 |
| `seed` | 随机种子。 |
| `metric_name` | 指标名。 |
| `score` | 指标数值。 |
| `higher_is_better` | 分数越高是否越好。当前 heldout task loss 是越低越好。 |
| `parameter_count` | 参数量。 |
| `training_steps` | 训练步数。 |
| `public_metrics` | 任务相关公开指标。 |

### 12.2 `diagnostics.jsonl`

诊断文件用来回答：

- router 有没有只偏向一个 candidate？
- CATO 有没有发挥对齐作用？
- LRIO 的交互是否有效？
- RCEO 对缺失/腐蚀有没有反应？
- adapter 参数是否有非平凡变化？

### 12.3 `robustness_rows.jsonl`

鲁棒性行用来回答：

- 模态缺失时性能如何？
- 特征被扰动时模型如何变化？
- RCEO 是否真的帮助模型降低对坏模态的依赖？

---

## 13. 当前已经完成的工作

截至 2026-06-01，已经完成：

### 13.1 MELD

- 正式特征抽取完成。
- cache 构建完成。
- cache 验证通过。

结果：

```text
sample_count = 13708
text   [13708, 16, 768]
audio  [13708, 32, 768]
visual [13708, 8, 768]
```

### 13.2 RefCOCO

- RefCOCO + COCO 数据已处理。
- stage records 完成。
- CLIP 特征抽取完成。
- raw staging 完成。
- cache 构建和验证通过。
- public acceptance smoke 通过。
- 100-step dryrun 通过。
- 正式 50000-step 命令已准备，目标 GPU 3。

关键数字：

```text
sample_count = 142210
train = 113762
val   = 14246
test  = 14202
text   [142210, 1, 512]
region [142210, 1, 512]
```

100-step dryrun 验证：

```text
seed_count = 5
seeds = 201, 202, 203, 204, 205
raw_metric_row_count = 55
ok = true
```

55 的来源：

```text
5 seeds * 11 models = 55 rows
```

### 13.3 CMU-MOSEI

- CMU SDK `.csd` 文件下载完成。
- sequence inspection / extraction 问题已修复。
- 非有限值 nan/inf 已处理为 0 再做 temporal reduction。
- raw staging 完成。
- cache 构建和验证通过。
- public acceptance smoke 通过。
- 100-step dryrun 通过。
- 正式 50000-step 命令已准备，目标 GPU 4。

关键数字：

```text
sample_count = 3225
text      [3225, 1, 300]
audio     [3225, 1, 74]
vision    [3225, 1, 35]
sentiment [3225, 1]
emotion   [3225, 6]
```

100-step dryrun 验证：

```text
seed_count = 5
seeds = 301, 302, 303, 304, 305
raw_metric_row_count = 55
ok = true
```

### 13.4 代码修复记录

近期关键修复：

| 提交 | 内容 |
|---|---|
| `4ecd8a1` | 添加 MELD 正式 transformer 特征抽取。 |
| `f835382` | 添加 RefCOCO CLIP 特征抽取。 |
| `dfedfb1` | 支持 nested CMU SDK HDF5 groups。 |
| `68fd012` | 清理 CMU SDK non-finite 特征。 |
| `0cb7c14` | 扩展 CMU video folds 到 segment ids。 |
| `ef1924f` | 保留 CMU segment dialogue ids。 |
| `3657ac6` | 添加 `scripts/__init__.py` 和 `scripts/multimodal/__init__.py`，避免 import 被第三方 `scripts` 包遮蔽。 |
| `28327c4` | public entry 接受 trained controlled diagnostics。 |
| `0e6730d` | public main 训练增加实时进度打印。 |

---

## 14. 当前正式 50000-step 运行建议

远端建议用模块方式运行，避免 `scripts` import 冲突：

```bash
PYTHONPATH="$PWD" python -m scripts.multimodal.run_public_main ...
```

### 14.1 RefCOCO on GPU 3

```bash
cd /home/david/work/Operator-Valued-Hyper-Attention
source .venv/bin/activate
mkdir -p outputs/multimodal

CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD" \
python -m scripts.multimodal.run_public_main \
  configs/multimodal_refcoco_public_main.json \
  --cache-root data/multimodal_cache \
  --controlled-report outputs/multimodal/controlled_v1_smoke/seed_101/controlled_report.json \
  --artifact-root outputs/multimodal/refcoco_main \
  --train-steps 50000 \
  --baseline-train-steps 50000 \
  --train-split train \
  --eval-split test \
  --device cuda \
  --d-model 256 \
  --memory-tokens 16 \
  --learning-rate 3e-4 \
  --progress-interval 500 \
  2>&1 | tee outputs/multimodal/refcoco_main_train.log
```

### 14.2 CMU-MOSEI on GPU 4

```bash
cd /home/david/work/Operator-Valued-Hyper-Attention
source .venv/bin/activate
mkdir -p outputs/multimodal

CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD" \
python -m scripts.multimodal.run_public_main \
  configs/multimodal_cmu_mosei_public_main.json \
  --cache-root data/multimodal_cache \
  --controlled-report outputs/multimodal/controlled_v1_smoke/seed_101/controlled_report.json \
  --artifact-root outputs/multimodal/cmu_mosei_main \
  --train-steps 50000 \
  --baseline-train-steps 50000 \
  --train-split train \
  --eval-split test \
  --device cuda \
  --d-model 256 \
  --memory-tokens 16 \
  --learning-rate 3e-4 \
  --progress-interval 500 \
  2>&1 | tee outputs/multimodal/cmu_mosei_main_train.log
```

监控：

```bash
watch -n 1 'nvidia-smi -i 3,4'
```

---

## 15. 正式跑完后要做什么

### 15.1 验证 RefCOCO 产物

```bash
python scripts/multimodal/validate_public_main_artifacts.py \
  --config configs/multimodal_refcoco_public_main.json \
  --raw-metrics outputs/multimodal/refcoco_main/raw_metrics.jsonl \
  --diagnostics outputs/multimodal/refcoco_main/diagnostics.jsonl \
  --robustness-rows outputs/multimodal/refcoco_main/robustness_rows.jsonl \
  --split test
```

### 15.2 验证 CMU-MOSEI 产物

```bash
python scripts/multimodal/validate_public_main_artifacts.py \
  --config configs/multimodal_cmu_mosei_public_main.json \
  --raw-metrics outputs/multimodal/cmu_mosei_main/raw_metrics.jsonl \
  --diagnostics outputs/multimodal/cmu_mosei_main/diagnostics.jsonl \
  --robustness-rows outputs/multimodal/cmu_mosei_main/robustness_rows.jsonl \
  --split test
```

如果返回：

```json
{
  "ok": true
}
```

说明正式 main artifacts 在结构上可进入后续 top-conference gate bundling。

---

## 16. 常见术语表

| 术语 | 通俗解释 |
|---|---|
| `OVHA` | Operator-Valued Hyper-Attention，让 attention 输出/组合 operator，而不是只输出普通向量。 |
| `operator` | 从一种输入对象映射到另一种输出对象的规则。比如输入图像+短语，输出区域；输入文本/音频/视觉，输出情绪。 |
| `operator-valued attention` | attention 的 value 不再只是一个向量，而是一组可组合的 operator candidate。 |
| `primitive` | 基础算子组件，一种可解释的处理方式。 |
| `candidate` | 候选 primitive。当前是 TLEO/SPO/LRIO/CATO。 |
| `router` | 给不同 candidate 分配权重的模块。 |
| `routing weights` | router 输出的权重，表示每个 candidate 用多少。 |
| `logits` | softmax 前的分数。 |
| `softmax` | 把分数变成总和为 1 的概率/权重。 |
| `hyper-adapter` | 根据当前样本生成 primitive 参数的模块。 |
| `adapter params` | hyper-adapter 生成的参数，如 `scale`、`bias`、`lengthscale`。 |
| `memory` | 从输入上下文压缩出来的任务记忆。 |
| `memory_tokens` | memory 里 token 的个数。 |
| `d_model` | 模型内部隐藏维度。越大容量越高，计算也更重。 |
| `evidence` | 从公开输入里提取的证据，不是 hidden/oracle。 |
| `frozen features` | 预先抽好的固定特征，训练 OVHA 时不再更新抽取器。 |
| `same-feature baseline` | baseline 和 OVHA 使用同一套 frozen features，保证公平。 |
| `baseline` | 对照模型，用来比较 OVHA 是否真的有提升。 |
| `ablation` | 消融实验，去掉某个模块看性能是否下降。 |
| `controlled` | 人造受控数据，知道隐藏真值，用于机制诊断。 |
| `oracle` | 上帝视角信息，只能诊断，不能进入模型输入。 |
| `public main` | 公开数据正式主实验，不是 smoke，不是临时测试。 |
| `smoke` | 小步数快速冒烟测试，只验证流程能跑，不作为论文主结论。 |
| `dryrun100` | 100 step 的试跑，用来验证流程、GPU、产物结构。 |
| `cache` | 统一数据缓存格式。 |
| `raw_root` | 原始 staging 数据目录。 |
| `cache_root` | 正式 cache 根目录。 |
| `split` | 数据划分，如 train/val/test。 |
| `source_id` | 样本唯一 ID。 |
| `provenance` | 数据来源记录。 |
| `checksum` | 文件哈希，用来检查文件没有被改。 |
| `task_loss` | 主任务损失。 |
| `candidate_individual_loss` | 每个 candidate 自己预测的损失，用于诊断。 |
| `public_alignment_ce` | 公开 region-text 对齐交叉熵。 |
| `entropy` | 不确定性，分布越散 entropy 越高。 |
| `router_entropy` | router 权重是否分散。 |
| `alignment_entropy` | 跨模态对齐是否分散。 |
| `local_entropy` | 局部证据选择是否分散。 |
| `RCEO` | 可靠性/腐蚀证据模块。 |
| `modality` | 模态，如文本、图像区域、音频、视觉。 |
| `missing_modality_mask` | 哪些模态缺失的标记。 |
| `quality` | 模态质量分数。 |
| `bbox` | bounding box，图像目标框。 |
| `phrase-region grounding` | 文本短语定位图像区域。 |
| `sentiment` | 情感极性，比如正面/负面程度。 |
| `emotion` | 情绪类别，如 joy、anger 等。 |
| `temporal_policy=mean` | 时间序列特征用平均池化压成一个时间步。 |
| `non_finite` | NaN 或 Inf 这类非正常数值。 |
| `nan_to_num_zero` | 把 NaN/Inf 转成 0，防止训练崩。 |
| `CUDA_VISIBLE_DEVICES=3` | 只让当前进程看到物理 GPU 3。PyTorch 内部会把它当 `cuda:0`。 |
| `PYTHONUNBUFFERED=1` | 让 Python 输出不缓存，训练日志能实时显示。 |
| `PYTHONPATH="$PWD"` | 让 Python 优先 import 当前仓库代码。 |
| `tee` | 终端显示一份，同时写日志文件一份。 |

---

## 17. 最容易混淆的几点

### 17.1 `--device cuda` 为什么不是 `cuda:3`

当你设置：

```bash
CUDA_VISIBLE_DEVICES=3
```

进程只能看到物理 GPU 3。对 PyTorch 来说，这张可见卡就是 `cuda:0`。所以脚本里写：

```bash
--device cuda
```

是对的。

### 17.2 为什么 100-step 通过不等于论文结果

100-step dryrun 只是验证：

- 数据能读。
- 模型能跑。
- 所有 seed/model 产物齐全。
- 验证器通过。

它不能作为论文主结果。正式主结果需要 50000 step 或更合理的长跑配置。

### 17.3 为什么必须先过 controlled report

因为 public data 上没有 hidden truth，不能直接知道 router/adapter 是不是“误打误撞”。

controlled report 的作用是先证明：

- router 能响应已知关系。
- adapter 参数有意义。
- RCEO 对缺失/腐蚀有反应。
- 去掉关键模块会变差。

通过后才允许进入 public main。

### 17.4 为什么强调 same frozen features

如果 OVHA 用强特征，baseline 用弱特征，那么比较不公平。

所以配置里强制：

```json
"enforce_same_features_for_baselines": true
```

意思是所有模型都吃同一套 frozen features，差异主要来自融合/路由/adapter 机制。

---

## 18. 一张文字版总流程图

```text
原始数据
  ├─ RefCOCO + COCO 图片
  ├─ CMU-MOSEI .csd
  └─ MELD 文本/音频/视频

特征抽取 / SDK 特征读取
  ├─ RefCOCO: CLIP text + CLIP region
  ├─ CMU-MOSEI: CMU SDK computational sequences
  └─ MELD: RoBERTa + Wav2Vec2 + CLIP

staging raw manifest
  ├─ annotations / metadata
  ├─ features
  ├─ labels / supervision
  └─ splits

build_cache.py
  -> data/multimodal_cache/<dataset>/v0.1

validate_cache.py
  -> 确认 cache 完整、无泄漏、可复现

accept_public_data.py
  -> raw manifest OK
  -> cache OK
  -> controlled entry OK
  -> public smoke OK

run_public_main.py
  -> 5 seeds
  -> OVHA full
  -> 10 same-feature baselines
  -> raw_metrics / diagnostics / robustness

validate_public_main_artifacts.py
  -> 确认正式 public main 产物可进入后续报告/gate
```

---

## 19. 当前最重要的下一步

1. 用 `python -m scripts.multimodal.run_public_main` 方式启动 RefCOCO 50000-step，GPU 3。
2. 同时启动 CMU-MOSEI 50000-step，GPU 4。
3. 观察训练日志中是否出现：

```text
[public-main:start]
[public-main:train]
```

4. 跑完后分别执行 `validate_public_main_artifacts.py`。
5. 如果两个都 `ok: true`，再构建 public gate report 和 top-conference entry manifest。

---

## 20. 读代码的推荐顺序

如果之后有人要继续开发，建议按这个顺序读：

1. `docs/ovha_project_objective_zh.md`
2. `docs/data_protocol_multimodal.md`
3. `configs/multimodal_refcoco_public_main.json`
4. `configs/multimodal_cmu_mosei_public_main.json`
5. `moat_ovha_torch/data/multimodal/typed_batch.py`
6. `moat_ovha_torch/data/multimodal/cache_schema.py`
7. `moat_ovha_torch/models/multimodal/ovha_multimodal.py`
8. `moat_ovha_torch/models/multimodal/evidence.py`
9. `moat_ovha_torch/models/multimodal/memory.py`
10. `moat_ovha_torch/models/multimodal/router.py`
11. `moat_ovha_torch/models/multimodal/hyper_adapter.py`
12. `moat_ovha_torch/models/multimodal/primitives/`
13. `scripts/multimodal/run_public_main.py`
14. `scripts/multimodal/validate_public_main_artifacts.py`

这条线读下来，就能理解当前 public multimodal OVHA 主线。
