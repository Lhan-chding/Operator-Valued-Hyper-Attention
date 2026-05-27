# OVHA Phase 1 汇总交接稿

## 目标来源

本阶段按照 `ovha_phase1_theory_and_codex_plan_cn_v1.docx` 执行，目标是先把 Operator-Valued Hyper-Attention（OVHA）的理论核心 C 立住，并用最小可运行原型验证它不是 Transformer + FNO/DeepONet/Router 的普通拼接。

核心定位：

- C = Operator-Valued Attention：理论核心，把 attention value 从有限维向量升级为 operator primitive。
- B = Hyper-Operator Adapter：实现机制，用 context memory 调制 primitive 参数。
- A = Operator Memory：上下文承载，把 demonstrations 压成 task-specific operator state。
- D = Router / MoE：工程扩展，用于 dense/top-k primitive routing，不作为主创新。

## 已完成产物

1. 理论文档
   - `theory_notes/ovha_math.md`
   - `theory_notes/special_cases.md`
   - 内容包括 meta-operator learning 设定、OVHA 离散/连续核定义、standard attention/FNO/DeepONet/Graph kernel 特例归约、5 个命题与 proof sketch。

2. 最小可运行原型
   - `moat_ovha/models/primitives/`
   - `moat_ovha/models/memory.py`
   - `moat_ovha/models/hyper_adapter.py`
   - `moat_ovha/models/router.py`
   - `moat_ovha/models/ovha_layer.py`
   - 支持 Fourier、separable、local kernel、identity value primitives；支持 pooling memory、Perceiver-style memory、low-rank hyper adapter、dense/top-k routing、diagnostics。

3. Phase-0 / Phase-1 operator zoo
   - `moat_ovha/data/operator_zoo.py`
   - 支持 fourier、green/local、separable、nonlinear、mixed operator families。
   - 支持 train/test split、parameter holdout、context-size sweep。

4. Baselines 与 ablations
   - Transformer-only value attention
   - Perceiver IO-style
   - ICON-style
   - DeepONet-only
   - FNO-only
   - simple stack
   - vector-value attention
   - no hyper adapter
   - no memory
   - MLP expert
   - random router

5. 自动化脚本与报告
   - `configs/phase1_minimal.json`
   - `train_meta_operator.py`
   - `eval_meta_operator.py`
   - `scripts/run_phase1_sweep.sh`
   - `scripts/summarize_phase1.py`
   - `outputs/phase1/train_metrics.jsonl`
   - `outputs/phase1/eval_metrics.jsonl`
   - `outputs/phase1/phase1_report.md`

## 验证结果

命令：

```bash
python3 -m unittest discover -s tests
sh scripts/run_phase1_sweep.sh configs/phase1_minimal.json
```

测试结果：

- `12` 个 unittest 全部通过。
- 覆盖 primitives shape/batch/finite-difference、OVHA dense/top-k、memory masking、context permutation invariance、special-case helpers、training/evaluation JSONL pipeline。

Phase-1 deterministic sweep 结果：

| family | OVHA-full relative L2 | best observed |
|---|---:|---|
| fourier | 0.041728 | ovha_full |
| green | 0.038022 | ovha_full |
| mixed | 0.019126 | ovha_full |
| separable | 0.062727 | ovha_sparse is effectively tied |

平均 ablation gap（comparator mean relL2 - OVHA mean relL2）：

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

当前 Go/No-Go：

Go: 在 deterministic toy sweep 中，OVHA-full 对所有配置的非 OVHA baseline 与 ablation，在每个 operator family 上均胜出。该结果支持 Phase-1 论点：C 不是装饰项，B/A/D 是服务于 operator-valued attention 的支撑机制。

## 重要限制

这些结果只能作为 Phase-1 scaffold 证据，不能直接视为论文级 benchmark：

- 当前实现为零外部依赖标准库版本，没有 torch/numpy 训练循环。
- operator zoo 是 deterministic toy family，不是大规模 PDE/material/CV benchmark。
- 当前 memory 使用了 context metadata 中的 operator hints/gain，用于验证机制闭环；下一阶段需要做 metadata-free 或 noisy-metadata 对照。
- hyper-adapter 只做 low-rank style 的 scale/bias/primitive parameter 调制，没有生成完整权重。
- proof sketch 需要进一步补齐严格假设、引用边界与可发表表述。

## 建议 GPT Pro 下一步分析问题

1. 理论上，OVHA 的 operator-valued kernel 定义是否足够区别于 OFormer/GNOT/ICON/HyperFNO/HyperDeepONet？
2. 命题 2 的 universal approximation proof sketch 还需要哪些精确定义：compactness、context identifiability、continuity、primitive closure？
3. 如何把当前 metadata-conditioned toy evidence 改成更强的 metadata-free context inference evidence？
4. Phase 2 应优先接入哪些真实任务：PDE operator learning、材料响应、dense-query field prediction，还是跨模态 toy VCM？
5. PyTorch 版本应如何设计，才能保持当前接口，同时加入可学习 memory、router、hyper-adapter 与 primitive 参数？
6. 论文 method section 应如何组织：先给 OVHA 核定义，再给 special cases，再给 A/B/D 支撑模块，最后给 ablation 逻辑。
