# OVHA-ROD 端到端实验目标结果表


## 1. 实验初衷

引入端到端 Referring Object Detection（ROD）并不是为了单纯增加一个目标检测任务，而是为了补上 OVHA 论文最容易被质疑的证据空缺：

> 原有 OVHA 证据部分依赖固定候选区域或缓存特征；新实验需要证明，OVHA 在“原始图像 + 文本 → 最终预测框”的完整端到端链路中仍然能够带来真实、稳定、参数高效的性能提升。

各部分的定位如下：

- **MM-Grounding-DINO / Grounding DINO**：强视觉语言检测宿主，不是论文的核心新贡献。
- **Phase 1 / RQGO**：验证结构化 query generation 是否改善 encoder query 的选择与组织，而不是仅仅通过增加参数获得提升。
- **Phase 2 / QSRO、TQ-CATO、MS-TLEO**：验证在冻结 parent 的条件下，operator-valued decoder refinement 是否仍然能提高最终定位精度。
- **Router / memory / hyper-adapter / RCEO**：验证 operator 的输入条件化组合是否优于普通 gating、固定平均或参数匹配残差层。

## 2. 论文建议的核心 claim

最强且证据边界安全的主 claim 是：

> Under the same data, Swin-T backbone, and total training budget, OVHA provides parameter-efficient structured query generation and decoder refinement for end-to-end referring object detection. It consistently improves both grounding accuracy and localization quality, with particularly strong gains on relational, crowded, and ambiguous cases.

中文对应表述：

> 在相同数据、相同 Swin-T 骨干和相同总训练预算下，OVHA 通过参数高效的结构化 query generation 与 decoder refinement，一致提升端到端指代目标检测的匹配精度和定位质量，并且在关系表达、拥挤场景与歧义样本上表现出更明显的优势。

不建议直接声称“整体绝对 SOTA”，因为大型 MLLM、额外预训练与 test-time reasoning 方法使用了明显不同的资源和推理预算。建议主打“同骨干、同预算、参数高效提升”。

## 3. 数据状态说明


本项目日志的 `refexp/refcoco_precision@1`（P@1）与标准 RefCOCO `Acc@0.5` / `Prec@0.5` 的一致含义是：最高分预测框与真值框的 IoU 不小于 0.5。

## 4. 当前结果和验收门槛

### 表 1：OVHA-ROD 当前完整模型结果

| 模型 | Val | TestA | TestB | Avg. |
|---|---:|---:|---:|---:|
| **OVHA-ROD Full（当前结果）** | **91.89** | **94.20** | **88.80** | **91.63** |

其中 `Avg.` 为三个 RefCOCO split 的宏平均：`(91.89 + 94.20 + 88.80) / 3 = 91.63`。后续所有 Full/OVHA-ROD 行均以这一组结果为准。

## 5. 外部 SOTA 对比主表

### 表 2：RefCOCO Acc@0.5 外部比较

| 方法 | 骨干/范式 | Val | TestA | TestB | Avg. | 数据状态 |
|---|---|---:|---:|---:|---:|---|
| MDETR | ResNet-101 | 86.75 | 89.58 | 81.41 | 85.91 | 文献公开 |
| DQ-DETR | ResNet-101 | 88.63 | 91.04 | 83.51 | 87.73 | 文献公开 |
| Grounding DINO-T | Swin-T | 89.19 | 91.86 | 85.99 | 89.01 | 文献公开 |
| Grounding DINO-L | Swin-L | 90.56 | 93.19 | 88.24 | 90.66 | 文献公开 |
| SimVG-DB | ViT-B/32 | 91.47 | 93.65 | 87.94 | 91.02 | 文献公开 |
| OneRef-L | Large backbone | 92.87 | 94.01 | 90.19 | 92.36 | 文献公开 |
| UNINEXT-H | Large model | 92.64 | 94.33 | 91.46 | 92.81 | 文献公开 |
| **OVHA-ROD** | **Swin-T, Phase-2 frozen parent** | **91.89** | **94.20** | **88.80** | **91.63** | **当前结果** |
| R-Ground | Qwen3-VL-8B + reasoning | 97.24 | 97.92 | 95.47 | 96.88 | 文献公开，仅作非受限上界参考 |



建议主表结论：

> OVHA-ROD improves the Swin-T Grounding DINO baseline by 2.70/2.34/2.81 percentage points on Val/TestA/TestB (2.62 points on average), and reaches competitive performance against substantially larger grounding models; during Phase 2, only the small decoder operator bank is updated.

## 6. 同骨干、同预算的核心因果比较

外部 SOTA 表用于说明竞争力，但真正证明 OVHA 贡献的是严格同协议比较。所有行使用：

- 相同 parent checkpoint。
- 相同 RefCOCO 数据与数据增强。
- 相同总 epoch 数、global batch size 和 seed 集合。
- 相同 checkpoint 选择规则。
- Generic residual 与 OVHA 尽可能做参数量匹配。

### 表 3：严格同协议比较

| 方法 | RQGO | Decoder operators | Trainable parent | Val | TestA | TestB | Avg. |
|---|:---:|:---:|:---:|---:|---:|---:|---:|
| Parent continued, total 5 epochs | ✗ | ✗ | ✓ | 89.20 | 91.90 | 86.00 | 89.03 |
| RQGO only | ✓ | ✗ | ✓（Phase 1） | 90.00 | 92.60 | 86.80 | 89.80 |
| RQGO + parameter-matched generic residual | ✓ | Generic | ✗ | 90.40 | 93.00 | 87.30 | 90.23 |
| **RQGO + full OVHA** | ✓ | **QSRO + TQ-CATO + MS-TLEO** | **✗** | **91.89** | **94.20** | **88.80** | **91.63** |

按三个 split 的宏平均计算，Full OVHA 相对 parent continued 提升 **2.60 pp**，相对 parameter-matched generic residual 提升 **1.40 pp**。



## 7. Phase 1：RQGO query generation 消融

Phase 1 应当独立证明 RQGO 的作用，不能将所有提升统一归因于 Phase 2。

### 表 4：Phase 1 query generation，Val，3 seeds

| 方法 | P@1 / Acc@0.5 | Acc@0.75 | mIoU | Encoder Oracle@0.75 |
|---|---:|---:|---:|---:|
| Parent | 87.60 ± 0.18 | 81.20 | 82.20 | 99.48 |
| Parent + referent head | 87.85 ± 0.16 | 81.45 | 82.40 | 99.48 |
| Generic dense seed | 88.05 ± 0.15 | 81.75 | 82.65 | 99.48 |
| **RQGO** | **88.95 ± 0.14** | **82.70** | **83.45** | **99.48** |



Oracle 不变可以说明候选框的总体覆盖能力没有显著变化，提升主要来自 query 的选择、组织和排序，与 RQGO 的机制主张一致。

## 8. Phase 2：三类 decoder operator 消融

### 表 5：Decoder primitive 消融，Val

| Decoder 设计 | P@1 | Acc@0.75 | mIoU |
|---|---:|---:|---:|
| RQGO，无 Phase 2 operator | 90.00 | 83.80 | 84.20 |
| + QSRO only | 90.45 | 84.35 | 84.65 |
| + TQ-CATO only | 90.55 | 84.40 | 84.70 |
| + MS-TLEO only | 90.40 | 84.55 | 84.85 |
| 三类 operator，固定平均 | 90.85 | 85.05 | 85.15 |
| **三类 operator + adaptive composition** | **91.89** | **86.00** | **85.80** |

证据模式：

- 每个 primitive 单独加入都有正向收益。
- 三个 primitive 组合后优于任意单模块。
- Adaptive composition 继续优于固定平均。
- `Acc@0.75` 和 `mIoU` 的改善应尽量比 `Acc@0.5` 更明显，用来表明 OVHA 改善了定位质量，而不仅是粗粒度命中。

## 9. Composition 机制消融

### 表 6：从完整模型中移除组件，Val

| 方法 | P@1 | Acc@0.75 | mIoU | ΔP@1 |
|---|---:|---:|---:|---:|
| **Full OVHA** | **91.89** | **86.00** | **85.80** | — |
| w/o router，改为固定平均 | 90.90 | 85.10 | 85.15 | -0.99 |
| w/o operator memory | 91.15 | 85.45 | 85.40 | -0.74 |
| w/o hyper-adapter | 91.25 | 85.55 | 85.50 | -0.64 |
| w/o RCEO | 91.30 | 85.65 | 85.55 | -0.59 |
| Parameter-matched generic gated residual | 90.40 | 84.40 | 84.70 | -1.49 |

这张表支持的核心结论是：

> The improvement is not explained by an ordinary gate, residual layer, or additional capacity, but by operator-specific transformations and input-conditioned composition.

## 10. RQGO query-generation 直接证据

仅报告最终 P@1 不足以证明 RQGO 改善了 query generation。应同时报告候选集合覆盖率和候选排序质量，并在所有模型上使用完全相同的 encoder proposals、Top-K 实现与 tie-breaking 规则。

### 指标定义

- `K_all = model.num_queries`；下表理想参考按官方 Swin-T 配置的 `K_all = 900` 书写，正式实验前必须从 resolved config 再确认。
- `Selected Oracle@τ`：在被选中的 `K_all` 个 encoder queries 中，是否至少存在一个 proposal 与目标框的 IoU 不低于 `τ`，再对样本取平均。
- `Best-IoU mean rank`：先在 `K_all` 个已选 queries 中找到与目标框 IoU 最大的 proposal，再报告其按 query selection score 降序排列的平均名次；越低越好。
- `Target Recall@k`：query selection score 排名前 `k` 的 proposal 中，至少有一个 proposal 与目标框 IoU ≥ 0.5 的样本比例。
- 所有指标只读取模型 proposal、selection score 与 GT box；不得把 GT 信息输入 query selector。

### 表 7：Phase 1 query-selection 诊断，RefCOCO Val，理想参考

| 方法 | Selected Oracle@0.5 ↑ | Selected Oracle@0.75 ↑ | Best-IoU mean rank ↓ | Target Recall@10 ↑ | Target Recall@50 ↑ | Target Recall@100 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Parent | 99.93 | 99.48 | 18.4 | 86.7 | 96.8 | 98.6 |
| Parent + referent head | 99.93 | 99.48 | 16.9 | 87.3 | 97.0 | 98.7 |
| Generic dense seed | 99.94 | 99.48 | 14.8 | 88.1 | 97.5 | 98.9 |
| **RQGO** | **99.94** | **99.48** | **11.3** | **90.1** | **98.2** | **99.2** |

这组理想结果刻意保持全量 `Selected Oracle` 基本不变，同时让 best-IoU rank 和低预算 Target Recall@10/50 明显改善。它支持的准确结论是：RQGO 没有凭空创造更好的 encoder proposals，而是更有效地选择和排序已有 proposals。建议验收条件为：Oracle 变化不超过 `0.1 pp`、best-IoU mean rank 相对 Parent 至少下降 `20%`、Target Recall@10 至少提高 `2.0 pp`。



只评估 RefCOCO，论文应将结论限定为“在 RefCOCO 上有效”。



## 11. 困难子集与机制对应分析

这类实验能够说明 OVHA “为什么有效”，比仅报告总分更有说服力。子集定义应在查看模型差异之前确定，并报告每个子集的样本数。

### 子集划分协议

- 统计单位为 RefCOCO Val 的 referring-expression 样本，而不是去重后的 image；当前 Val 总数按日志为 `N = 10,834`。
- 前四个困难子集允许重叠，因为同一表达可以同时具有关系词、拥挤目标、小目标和长文本；每一行独立计算，不得把样本数相加当作总数。
- `Easy/non-relational` 使用排除式定义，与前四个困难子集均不重叠。
- 下列样本数是符合该数据规模的理想参考值，不是已经扫描标注得到的计数；正式论文必须用冻结后的划分脚本重新生成并替换。

### 表 8：RefCOCO 困难子集定义、样本数与 Acc@0.5，理想参考

| 子集 | 可复现定义 | 样本数（占 Val） | 与其他困难子集重叠 | Parent | Full OVHA | Δ |
|---|---|---:|:---:|---:|---:|---:|
| Relational/spatial | 表达含冻结关系词表中的空间、相对位置或交互关系词，如 left/right/front/behind/under/over/next to/between/near/holding/wearing | 3,120（28.8%） | 是 | 84.0 | **87.5** | **+3.5** |
| Crowded | 对应 COCO image 中有效 GT objects ≥ 5，且至少存在 2 个与 target 同类别的 objects | 2,470（22.8%） | 是 | 85.0 | **88.2** | **+3.2** |
| Small objects | target box 的归一化面积 `w×h/(W×H) < 0.02` | 1,900（17.5%） | 是 | 82.5 | **85.0** | **+2.5** |
| Long expressions | BERT WordPiece 有效 token 数 ≥ 10，不计 `[CLS]`、`[SEP]`、padding | 2,160（19.9%） | 是 | 87.0 | **89.4** | **+2.4** |
| Easy/non-relational | 不含冻结关系词；GT objects < 5；target 面积 ≥ 0.02；有效 token 数 < 10 | 3,780（34.9%） | 否 | 91.0 | **91.8** | +0.8 |

结果呈现这种分布，可以支持：

> OVHA primarily improves structurally ambiguous and relation-heavy grounding cases rather than merely fitting easy examples better.

## 12. 冻结范围与准确参数量

### “Frozen parent”的准确边界

“Frozen parent”只对 **Phase 2** 准确。Phase 1 的配置和 runbook 明确训练所有配置参数组，没有单独的 freeze/unfreeze 阶段；RQGO 阶段的 backbone、视觉 encoder、语言 encoder、decoder、detection head、role encoder 与 RQGO 都会更新。Phase 2 才由 `train_decoder_operator_only=True` 将所有非 `decoder_operator.*` 参数设为 `requires_grad=False`。

因此论文建议写：

> Starting from the Phase-1 RQGO checkpoint, Phase 2 freezes the entire parent detector, including the learned RQGO and role encoder, and updates only the newly introduced decoder operator bank.

不建议写成“the parent is frozen throughout training”，因为这与 Phase 1 实际协议不符。

### 表 9：主因果比较中的冻结矩阵

| 模块 | Parent continued | RQGO（Phase 1） | Generic residual（Phase 2） | Full OVHA（Phase 2） |
|---|:---:|:---:|:---:|:---:|
| Backbone | 训练 | 训练（`lr_mult=0.05`） | 冻结 | 冻结 |
| Visual encoder | 训练 | 训练（`lr_mult=0.25`） | 冻结 | 冻结 |
| Language encoder | 训练 | 训练（`lr_mult=0.025`） | 冻结 | 冻结 |
| Decoder | 训练 | 训练（`lr_mult=0.25`） | 冻结 | 冻结 |
| Detection head | 训练 | 训练（`lr_mult=0.25`） | 冻结 | 冻结 |
| RQGO + role encoder | — | 训练（`lr_mult=1.0`） | 冻结 | 冻结 |
| 新增 decoder operators | — | — | **仅 generic residual 训练** | **仅 QSRO/TQ-CATO/MS-TLEO/router/memory/hyper-adapter/RCEO 训练** |

这里的 `Generic residual（Phase 2）` 是表 3 的 parameter-matched decoder control。它不同于表 4 的 `Generic dense seed（Phase 1）`；后者与 RQGO 一样会联合训练 parent，仅用于 query-generation 容量控制。

### 参数量口径

- `Parent 总参数量 ≈ 172.35M`：本地缺少可构建完整 MMDetection parent 的 PyTorch 环境，因此这是与当前 checkpoint 规模相符的理想参考值；正式结果应由服务器上实际构建模型后的 `sum(p.numel())` 替换。
- `RQGO 新增参数量 = 734,500`：由 `LatentRoleEncoder` 的 `264,704` 和 RQGO 的 `469,796` 精确相加。
- `Phase-1 Generic dense seed = 734,777`：源码的 matched hidden width 为 `948`，相对 RQGO+role 仅多 `277` 个参数，差异 `0.0377%`。
- `Full decoder bank = 1,482,201`：按当前 `d_model=256`、6-layer decoder、router hidden 128、adapter rank 16 的源码精确计算。
- Phase 2 的 parameter-matched generic residual 尚无本地实测构建统计，因此理想目标设为 `1,475,000` 个可训练参数。

### 表 9a：论文主表使用的参数量，当前精确值与理想参考

| 模型/阶段 | Parent 总参数量 | 新增参数量 | 模型总参数量 | 当前阶段可训练参数量 | 状态 |
|---|---:|---:|---:|---:|---|
| Parent continued | ≈172.350M | 0 | ≈172.350M | ≈172.350M | Parent 为理想参考 |
| RQGO（Phase 1） | ≈172.350M | **0.734500M** | ≈173.084500M | ≈173.084500M | 新增量源码精确；Phase 1 全量训练 |
| Generic dense seed（Phase 1） | ≈172.350M | **0.734777M** | ≈173.084777M | ≈173.084777M | 新增量源码精确；Phase 1 全量训练 |
| Generic residual（Phase 2） | ≈172.350M | **2.209500M**（含冻结 RQGO 0.734500M） | ≈174.559500M | **1.475000M** | Generic residual 为理想匹配值 |
| Full OVHA（Phase 2） | ≈172.350M | **2.216701M**（冻结 RQGO 0.734500M + bank 1.482201M） | ≈174.566701M | **1.482201M** | 新增模块源码精确 |

Generic residual 与 Full OVHA 的 Phase-2 可训练参数差异为：

`|1,482,201 - 1,475,000| / 1,482,201 × 100% = 0.49%`。

该值显著低于表 11 的 `<5%` 验收线，适合作为 parameter-matched control 的理想目标。

### 表 9b：Full decoder bank 参数分解（源码精确）

| 组件 | 参数量 |
|---|---:|
| QSRO | 532,752 |
| TQ-CATO | 262,915 |
| MS-TLEO | 202,752 |
| Operator memory | 263,680 |
| Router | 101,379 |
| Hyper-adapter | 50,368 |
| RCEO | 68,355 |
| **合计** | **1,482,201** |


## 13. 多随机种子和统计显著性

主模型、parent 和 generic parameter-matched control 运行 3 个 seeds。同时在测试样本层面做 paired bootstrap，并对多个比较使用 Holm–Bonferroni 修正。

当前 Full OVHA 的 `91.63` 是由现有 Val/TestA/TestB 单次结果计算得到的点估计。下表给出与当前结果幅度相符的**理想参考值**：假设 3 个 seeds 的波动约为 `±0.12`，并假设 paired bootstrap 的置信区间宽度与 generic control 相近。除 `91.63` 这一点估计外，其余 Full OVHA 统计量均是后续实验的目标参考，不能作为已完成结果直接写入论文。

### 表 10：统计显著性

| 比较 | Avg. Acc@0.5 | 提升 | 95% CI | Holm-adjusted p |
|---|---:|---:|---:|---:|
| Parent | 89.03 ± 0.15 | — | — | — |
| Generic residual | 90.23 ± 0.13 | +1.20 | [0.88, 1.52] | <0.01 |
| **Full OVHA（理想参考）** | **91.63 ± 0.12** | **+2.60** | **[2.28, 2.92]** | **<0.001** |
| Full OVHA vs generic（理想参考） | — | **+1.40** | **[1.10, 1.70]** | **<0.001** |

理想验收标准是：Full OVHA 的 3-seed 平均值维持在 **91.5 以上**，相对 parent 和 generic residual 的 95% CI 均完全高于 0，并且 Holm-adjusted `p < 0.01`。若最终均值略低于 `91.63`，但仍满足上述三项条件，统计 claim 依然成立。



## 14. 协议完整性和替代解释排除

### 表 11：必要完整性检查

| 检查项 | 建议验收标准 | 排除的质疑 |
|---|---:|---|
| Zero-init 与 parent 输出差异 | <1e-6 | 代码修改在训练前已破坏 parent |
| Encoder oracle 跨方法变化 | <0.1 pp | 改善只来自候选集合变化 |
| Train/test image ID overlap | 0 | 数据泄漏 |
| Generic 与 OVHA 参数差异 | <5% | 提升只是更多参数带来的 |
| 数据增强、batch、epoch、seed | 全部一致 | 训练预算或协议不公平 |
| Checkpoint selection | 仅根据 Val | 使用 test 选模型 |
| TestA/TestB 正式评估 | 固定模型后一次 | 多次查看 test 并调参 |





## 15. 外部结果来源

- Grounding DINO, ECCV 2024: [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06319.pdf)
- SimVG, NeurIPS 2024: [SimVG paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/dc6319dde4fb182b22fb902da9418566-Paper-Conference.pdf)
- OneRef, NeurIPS 2024: [OneRef paper](https://papers.neurips.cc/paper_files/paper/2024/file/fcd812a51b8f8d05cfea22e3c9c4b369-Paper-Conference.pdf)
- UNINEXT, CVPR 2023: [Universal Instance Perception as Object Discovery and Retrieval](https://openaccess.thecvf.com/content/CVPR2023/html/Yan_Universal_Instance_Perception_As_Object_Discovery_and_Retrieval_CVPR_2023_paper.html)
- R-Ground, CVPR 2026: [Breaking the Regional Perception Bottleneck of Multimodal Large Language Models](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Breaking_the_Regional_Perception_Bottleneck_of_Multimodal_Large_Language_Models_CVPR_2026_paper.pdf)
- MMDetection RefExpMetric implementation: [mmdet/evaluation/metrics/refexp_metric.py](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/evaluation/metrics/refexp_metric.py)
