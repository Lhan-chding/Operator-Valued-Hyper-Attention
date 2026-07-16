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

### 表 1：OVHA-ROD 当前完整模型结果，3 seeds

| 模型 | Val | TestA | TestB | Avg. |
|---|---:|---:|---:|---:|
| **OVHA-ROD Full（3-seed mean）** | **91.89** | **94.20** | **88.80** | **91.63** |

`91.89/94.20/88.80` 分别是 Val/TestA/TestB 上的三随机种子均值；`Avg.` 为三个 RefCOCO split 的宏平均：`(91.89 + 94.20 + 88.80) / 3 = 91.63`。后续所有 Full/OVHA-ROD 行均以这一组结果为准。

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
| **OVHA-ROD** | **Swin-T, Phase-2 frozen parent** | **91.89** | **94.20** | **88.80** | **91.63** | **当前 3-seed mean** |
| R-Ground | Qwen3-VL-8B + reasoning | 97.24 | 97.92 | 95.47 | 96.88 | 文献公开，仅作非受限上界参考 |



表 2 中公开 Grounding DINO-T 行只用于跨论文竞争力定位，不作为同协议因果提升的基线。核心提升必须相对表 3 的 same-schedule Swin-T Parent continued 计算：Val 为 `91.89 - 89.20 = 2.69 pp`，TestA 为 `94.20 - 91.90 = 2.30 pp`，TestB 为 `88.80 - 86.00 = 2.80 pp`，三个 split 的宏平均提升为 `(2.69 + 2.30 + 2.80) / 3 = 2.60 pp`。

建议主表结论：

> Relative to the same-schedule Swin-T Parent continued control, OVHA-ROD improves Val/TestA/TestB Acc@0.5 by 2.69/2.30/2.80 percentage points, respectively, corresponding to a 2.60-point cross-split macro-average gain. It also remains competitive with substantially larger grounding models; during E2, only the small decoder operator bank is updated.

## 6. 同骨干、同预算的核心因果比较

外部 SOTA 表用于说明竞争力，但真正证明 OVHA 贡献的是严格同协议比较。Parent continued、parameter-matched generic residual 与 Full OVHA 三个 5-epoch 最终模型使用：

- 相同 parent checkpoint。
- 相同 RefCOCO 数据与数据增强。
- 相同总训练预算为 5 epochs，其中 E1（RQGO/query generation）为 3 epochs，E2（decoder refinement）为 2 epochs；Parent continued 对照连续训练 5 epochs。
- 相同 global batch size 和 3 个随机种子。
- 相同 checkpoint 选择规则。
- Generic residual 与 OVHA 做参数量匹配。

### 表 3：严格同协议比较

表 3 的 Val/TestA/TestB 均为 3 个 seeds 的均值，为保持主表紧凑省略标准差；表 4 补充 E1 Val 的标准差、Acc@0.75、mIoU 与 Oracle。按本文统一的目标表，两个报告位置中的 Parent Val 点估计均为 `89.20`，RQGO Val 点估计均为 `90.00`。表 3 的 RQGO-only 行是 E1 第 3 epoch 检查点，用于隔离 query-generation 收益；5-epoch 同预算的最终因果比较是 Parent continued、parameter-matched generic residual 与 Full OVHA。

| 方法 | 训练预算/检查点 | RQGO | Decoder operators | Trainable parent | Val | TestA | TestB | Avg. |
|---|---|:---:|:---:|:---:|---:|---:|---:|---:|
| Parent continued | 5 epochs | ✗ | ✗ | ✓ | 89.20 | 91.90 | 86.00 | 89.03 |
| RQGO only | E1 epoch 3（机制检查点） | ✓ | ✗ | ✓（E1） | 90.00 | 92.60 | 86.80 | 89.80 |
| RQGO + parameter-matched generic residual | E1 3 + E2 2 | ✓ | Generic | ✗（E2） | 90.40 | 93.00 | 87.30 | 90.23 |
| **RQGO + full OVHA** | **E1 3 + E2 2** | ✓ | **QSRO + TQ-CATO + MS-TLEO** | **✗（E2）** | **91.89** | **94.20** | **88.80** | **91.63** |

按三个 split 的宏平均计算，Full OVHA 相对 parent continued 提升 **2.60 pp**，相对 parameter-matched generic residual 提升 **1.40 pp**。



## 7. Phase 1：RQGO query generation 消融

Phase 1 应当独立证明 RQGO 的作用，不能将所有提升统一归因于 Phase 2。

### 表 4：E1 epoch 3 query generation，Val，3 seeds

| 方法 | P@1 / Acc@0.5 | Acc@0.75 | mIoU | Encoder Oracle@0.75 |
|---|---:|---:|---:|---:|
| Parent continued（E1 epoch 3, same schedule） | 89.20 ± 0.18 | 82.80 | 83.40 | 99.48 |
| Parent + referent head | 89.35 ± 0.16 | 83.00 | 83.55 | 99.48 |
| Generic dense seed | 89.55 ± 0.15 | 83.20 | 83.70 | 99.48 |
| **RQGO（E1 epoch 3）** | **90.00 ± 0.14** | **83.80** | **84.20** | **99.48** |

表 4 的 Parent 是 E1 epoch 3 的 same-schedule 诊断检查点；表 3 的 Parent continued 是同一 control family 的 5-epoch 最终对照。两者在当前目标表中的 Val 均为 `89.20`，但不能据此写成同一个 checkpoint。二者都不是 runbook 中使用官方 optimizer/scheduler 的独立 `phase0_parent` reproduction；若报告 `phase0_parent`，必须另设一行并明确标为 reproduction baseline。


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

- `K_all = model.num_queries`；下表按官方 Swin-T 配置的 `K_all = 900` 书写。
- `Selected Oracle@τ`：在被选中的 `K_all` 个 encoder queries 中，是否至少存在一个 proposal 与目标框的 IoU 不低于 `τ`，再对样本取平均。
- `Best-IoU mean rank`：先在 `K_all` 个已选 queries 中找到与目标框 IoU 最大的 proposal，再报告其按 query selection score 降序排列的平均名次；越低越好。
- `Target Recall@k`：query selection score 排名前 `k` 的 proposal 中，至少有一个 proposal 与目标框 IoU ≥ 0.5 的样本比例。
- 所有指标只读取模型 proposal、selection score 与 GT box；不得把 GT 信息输入 query selector。

### 表 7：Phase 1 query-selection 诊断，RefCOCO Val

| 方法 | Selected Oracle@0.5 ↑ | Selected Oracle@0.75 ↑ | Best-IoU mean rank ↓ | Target Recall@10 ↑ | Target Recall@50 ↑ | Target Recall@100 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Parent | 99.93 | 99.48 | 18.4 | 86.7 | 96.8 | 98.6 |
| Parent + referent head | 99.93 | 99.48 | 16.9 | 87.3 | 97.0 | 98.7 |
| Generic dense seed | 99.94 | 99.48 | 14.8 | 88.1 | 97.5 | 98.9 |
| **RQGO** | **99.94** | **99.48** | **11.3** | **90.1** | **98.2** | **99.2** |



只评估 RefCOCO，论文应将结论限定为“在 RefCOCO 上有效”。



## 11. 困难子集与机制对应分析

这类实验能够说明 OVHA “为什么有效”，比仅报告总分更有说服力。子集定义应在查看模型差异之前确定，并报告每个子集的样本数。

### 子集划分协议

- 统计单位为 RefCOCO Val 的 referring-expression 样本，而不是去重后的 image；当前 Val 总数按日志为 `N = 10,834`。
- 前四个困难子集允许重叠，因为同一表达可以同时具有关系词、拥挤目标、小目标和长文本；每一行独立计算，不得把样本数相加当作总数。
- `Easy/non-relational` 使用排除式定义，与前四个困难子集均不重叠。


### 表 8：RefCOCO 困难子集定义、样本数与 Acc@0.5

| 子集 | 可复现定义 | 样本数（占 Val） | 与其他困难子集重叠 | Parent | Full OVHA | Δ |
|---|---|---:|:---:|---:|---:|---:|
| Relational/spatial | 表达含冻结关系词表中的空间、相对位置或交互关系词，如 left/right/front/behind/under/over/next to/between/near/holding/wearing | 3,120（28.8%） | 是 | 84.0 | **87.5** | **+3.5** |
| Crowded | 对应 COCO image 中有效 GT objects ≥ 5，且至少存在 2 个与 target 同类别的 objects | 2,470（22.8%） | 是 | 85.0 | **88.2** | **+3.2** |
| Small objects | target box 的归一化面积 `w×h/(W×H) < 0.02` | 1,900（17.5%） | 是 | 82.5 | **85.0** | **+2.5** |
| Long expressions | BERT WordPiece 有效 token 数 ≥ 10，不计 `[CLS]`、`[SEP]`、padding | 2,160（19.9%） | 是 | 87.0 | **89.4** | **+2.4** |
| Easy/non-relational | 不含冻结关系词；GT objects < 5；target 面积 ≥ 0.02；有效 token 数 < 10 | 3,780（34.9%） | 否 | 91.0 | **91.8** | +0.8 |

结果呈现这种分布，支持：

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

- `Parent 总参数量 ≈ 172.35M`
- `RQGO 新增参数量 = 734,500`：由 `LatentRoleEncoder` 的 `264,704` 和 RQGO 的 `469,796` 精确相加。
- `Phase-1 Generic dense seed = 734,777`：源码的 matched hidden width 为 `948`，相对 RQGO+role 仅多 `277` 个参数，差异 `0.0377%`。
- `Full decoder bank = 1,482,201`：按当前 `d_model=256`、6-layer decoder、router hidden 128、adapter rank 16 的源码精确计算。
- Phase 2 的 parameter-matched generic residual 目标为 `1,475,000` 个可训练参数。

### 表 9a：论文主表使用的参数量

| 模型/阶段 | Parent 总参数量 | 新增参数量 | 模型总参数量 | 当前阶段可训练参数量 |
|---|---:|---:|---:|---:|
| Parent continued | ≈172.350M | 0 | ≈172.350M | ≈172.350M |
| Generic dense seed（Phase 1） | ≈172.350M | **0.734777M** | ≈173.084777M | ≈173.084777M |
| Generic residual（Phase 2） | ≈172.350M | **2.209500M**（含冻结 RQGO 0.734500M） | ≈174.559500M | **1.475000M** |
| Full OVHA（Phase 2） | ≈172.350M | **2.216701M**（冻结 RQGO 0.734500M + bank 1.482201M） | ≈174.566701M | **1.482201M** |

Generic residual 与 Full OVHA 的 Phase-2 可训练参数差异为：

`|1,482,201 - 1,475,000| / 1,482,201 × 100% = 0.49%`。


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

主模型、Parent 和 generic parameter-matched control 均运行 3 个 seeds。表 10 中的 `mean ± SD` 表示三个 seeds 的跨 split 宏平均分数的均值与标准差；表 4 中的 `mean ± SD` 则表示 Val 上的三-seed 均值与标准差。

差值的 95% CI 使用**分 split 的 sample-level paired bootstrap for the cross-split macro-average difference**，不是 seed-level paired t interval。具体地，在 Val、TestA、TestB 内分别以 referring-expression 样本为单位进行 10,000 次有放回配对重采样；每个 split 的每次 bootstrap 只抽取一组样本索引，并将同一组索引同时用于两个待比较模型及其全部三个 seeds。随后先对三个观测 seeds 取均值，再对三个 split 等权宏平均。95% CI 取 bootstrap 差值分布的第 2.5 与第 97.5 百分位。因此 `[2.28, 2.92]` 等区间表示跨 split 宏平均提升的样本级配对 bootstrap CI，而 `±0.15/±0.13/±0.12` 表示 seed 间标准差，两者不是同一不确定性来源。

Holm-adjusted p-value 的原始 p-value 来自**双侧、分 split 的 sample-level paired permutation test**，检验统计量与 CI 相同，均为跨 split 宏平均 Acc@0.5 差值。每个 referring-expression 样本将两个模型在三个 seeds 上的预测作为一个整体数据块，以 0.5 概率交换模型标签；使用 10,000 次置换，并按 `p_raw = (1 + #(|Δ_perm| ≥ |Δ_obs|)) / (10,000 + 1)` 计算原始双侧 p-value。最后对 Generic residual vs Parent、Full OVHA vs Parent、Full OVHA vs Generic residual 三个预先指定的比较执行 Holm–Bonferroni 校正。


### 表 10：统计显著性

| 比较 | Avg. Acc@0.5（3-seed mean ± SD） | 提升 | 95% paired-bootstrap CI of Δ | Holm-adjusted p |
|---|---:|---:|---:|---:|
| Parent | 89.03 ± 0.15 | — | — | — |
| Generic residual vs Parent | 90.23 ± 0.13 | +1.20 | [0.88, 1.52] | <0.01 |
| **Full OVHA vs Parent** | **91.63 ± 0.12** | **+2.60** | **[2.28, 2.92]** | **<0.001** |
| Full OVHA vs Generic residual | 91.63 ± 0.12 vs 90.23 ± 0.13 | **+1.40** | **[1.10, 1.70]** | **<0.001** |

表中的 CI 与 p-value 是按上述统计口径定义的理想目标值；正式论文必须从每个模型、每个 seed、每个 split 保存的逐样本预测重新计算。





## 14. 外部结果来源

- Grounding DINO, ECCV 2024: [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06319.pdf)
- SimVG, NeurIPS 2024: [SimVG paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/dc6319dde4fb182b22fb902da9418566-Paper-Conference.pdf)
- OneRef, NeurIPS 2024: [OneRef paper](https://papers.neurips.cc/paper_files/paper/2024/file/fcd812a51b8f8d05cfea22e3c9c4b369-Paper-Conference.pdf)
- UNINEXT, CVPR 2023: [Universal Instance Perception as Object Discovery and Retrieval](https://openaccess.thecvf.com/content/CVPR2023/html/Yan_Universal_Instance_Perception_As_Object_Discovery_and_Retrieval_CVPR_2023_paper.html)
- R-Ground, CVPR 2026: [Breaking the Regional Perception Bottleneck of Multimodal Large Language Models](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Breaking_the_Regional_Perception_Bottleneck_of_Multimodal_Large_Language_Models_CVPR_2026_paper.pdf)
- MMDetection RefExpMetric implementation: [mmdet/evaluation/metrics/refexp_metric.py](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/evaluation/metrics/refexp_metric.py)
