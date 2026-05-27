# Phase 1: OVHA Theory and Minimal Prototype

## 目标

按照 `ovha_phase1_theory_and_codex_plan_cn_v1.docx` 完成第一阶段闭环：把 Operator-Valued Hyper-Attention 从数学定义、特例归约、最小可运行实现、toy operator zoo、baseline/ablation 和报告生成六个方面立住。

本阶段重点不是大规模 benchmark，也不是 VCM/CV/具身 demo，而是证明 OVHA 不是 Transformer + FNO/DeepONet/Router 的普通拼接。

## 本阶段范围

已覆盖：

- OVHA 数学定义和 operator-valued kernel 形式。
- standard attention、FNO-style kernel、DeepONet-style separable operator、Graph/GNO-style local kernel 的特例归约。
- OperatorPrimitive 抽象接口和三类核心 primitive。
- Operator Memory Encoder：pooling memory 与 Perceiver-style memory。
- Hyper-Operator Adapter：low-rank style 的 scale/bias/primitive parameter 调制。
- OVHALayer：dense mixing、top-k sparse routing、diagnostics。
- Phase-0 / Phase-1 toy operator zoo。
- Baseline 与 ablation sweep。
- 自动 JSONL 指标、Markdown 报告和 GPT Pro 交接稿。

未覆盖：

- 真实 torch/numpy 训练循环。
- 大规模 PDE/material/CV benchmark。
- metadata-free context inference 的强验证。
- 完整权重生成式 hypernetwork。
- 严格论文级理论证明。

## 已完成内容

理论文件：

- `theory_notes/ovha_math.md`
- `theory_notes/special_cases.md`

核心实现：

- `moat_ovha/models/primitives/base.py`
- `moat_ovha/models/primitives/fourier.py`
- `moat_ovha/models/primitives/separable.py`
- `moat_ovha/models/primitives/local_kernel.py`
- `moat_ovha/models/primitives/identity.py`
- `moat_ovha/models/memory.py`
- `moat_ovha/models/hyper_adapter.py`
- `moat_ovha/models/router.py`
- `moat_ovha/models/ovha_layer.py`

数据与实验：

- `moat_ovha/data/operator_zoo.py`
- `moat_ovha/baselines/common.py`
- `configs/phase1_minimal.json`
- `train_meta_operator.py`
- `eval_meta_operator.py`
- `scripts/run_phase1_sweep.sh`
- `scripts/summarize_phase1.py`

测试：

- `tests/test_primitives.py`
- `tests/test_ovha_shapes.py`
- `tests/test_special_cases.py`
- `tests/test_context_permutation.py`
- `tests/test_phase1_pipeline.py`

产物：

- `outputs/phase1/train_metrics.jsonl`
- `outputs/phase1/eval_metrics.jsonl`
- `outputs/phase1/phase1_report.md`
- `outputs/phase1/phase1_gpt_pro_summary.md`

## 实验与验证

运行命令：

```bash
python3 -m unittest discover -s tests
sh scripts/run_phase1_sweep.sh configs/phase1_minimal.json
```

验证结果：

- 12 个 unittest 全部通过。
- train metrics: 156 JSONL rows。
- eval metrics: 156 JSONL rows。
- 自动报告：`outputs/phase1/phase1_report.md`。

Phase-1 deterministic sweep 的 family 结果：

| family | OVHA-full relative L2 | best observed |
|---|---:|---|
| fourier | 0.041728 | ovha_full |
| green | 0.038022 | ovha_full |
| mixed | 0.019126 | ovha_full |
| separable | 0.062727 | ovha_sparse effectively tied |

平均 ablation gap：

| comparator | gap |
|---|---:|
| transformer_only | 1.537303 |
| perceiver_io_style | 0.911138 |
| icon_style | 1.254671 |
| deeponet | 0.847246 |
| fno | 2.725769 |
| simple_stack | 1.352555 |
| vector_value | 0.864879 |
| no_hyper_adapter | 1.004711 |
| no_memory | 0.915196 |
| mlp_expert | 1.017573 |
| random_router | 0.905560 |

## Go / No-Go

Go.

在当前 deterministic toy sweep 中，OVHA-full 对所有配置的非 OVHA baseline 与 ablation，在每个 operator family 上均胜出。这个结果支持 Phase 1 的最小 claim：operator-valued attention core C 是必要机制，A/B/D 是服务于 C 的支撑层，而不是独立堆叠的模块。

## 限制与风险

- 当前结果是 toy scaffold evidence，不是论文级 benchmark。
- 当前实现为 Python 标准库零依赖版本，没有可学习训练循环。
- context memory 读取了 metadata 中的 operator hints/gain；下一阶段必须做 metadata-free 或 noisy-metadata 对照。
- hyper-adapter 只调制少量 primitive 参数，没有完整 hypernetwork。
- sparse routing 在 separable family 上与 OVHA-full 几乎持平，这说明 D 可以作为效率机制，但暂时不应作为创新主线。
- proof sketch 还需要严格化，并加入更完整的相关工作边界。

## 下一阶段建议

1. 将当前 deterministic 原型迁移到 PyTorch，但保持接口分层。
2. 训练 learned memory encoder、router 和 hyper-adapter。
3. 去除 metadata hints，使用 demonstrations 本身推断 operator state。
4. 加入 resolution transfer、parameter holdout、operator-family holdout 曲线。
5. 把当前 `phase1_report.md` 的表格扩展为论文式 ablation 图。
6. 让 GPT Pro 先审查 `outputs/phase1/phase1_gpt_pro_summary.md`，再决定 Phase 2 的任务族和理论强化路线。
