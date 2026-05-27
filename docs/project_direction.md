# OVHA Project Direction

## 一句话目标

构建 Operator-Valued Hyper-Attention（OVHA）：把 attention 的 value 从有限维向量扩展为可作用在函数、场或轨迹上的 operator primitive，并用 context-conditioned memory / hyper-adapter / router 支撑一个统一的 meta-operator learning 框架。

## 主线判断

C 是理论核心，不是普通模块：

- C = Operator-Valued Attention：把 attention 的 value 域升级为 operator-valued primitive。
- A = Operator Memory：从 context demonstrations 中形成 task-specific operator state。
- B = Hyper-Operator Adapter：由 memory/query 调制 primitive 的低秩参数、kernel 参数或 basis 系数。
- D = Operator-Primitive Router / MoE：用于规模化和稀疏 primitive 选择，不作为核心创新表述。

当前路线不应先做 VCM、CV demo 或具身 demo。Phase 1 的核心任务是先把数学对象、特例归约、最小原型和 ablation 证据立住。

## 文档约定

每个阶段必须维护两个层次的 Markdown 说明：

- `docs/project_direction.md`：总方向、当前状态、阶段索引、最新结论、下一步优先级。
- `docs/phases/phase_N.md`：单阶段目标、范围、产物、验证、Go/No-Go、限制、下一阶段建议。

每次完成一个阶段、改变路线、得到新的实验结论或做出关键技术决策时，都要同步更新这两个层次。新窗口优先读取本文件，再读取当前阶段文件。

## 阶段索引

| Phase | 状态 | 阶段文件 | 结论 |
|---|---|---|---|
| Phase 1 | Completed | `docs/phases/phase_1.md` | Go: deterministic toy sweep 支持 OVHA-full 优于配置内 baselines/ablations。 |
| Phase 1.5 | Local complete | `docs/phases/phase_1_5.md` | Metadata-free PyTorch OVHA 已实现；Mac CPU smoke provisional Go；GPU/A800 scripts 已准备。 |
| Phase 2 | Not started | 待创建 | 真实 PDE/material/dense-query benchmark，取决于 Phase 1.5 Go/No-Go。 |

## 当前仓库状态

当前已完成：

- Phase 1 理论笔记：`theory_notes/ovha_math.md`、`theory_notes/special_cases.md`。
- Phase 1 零依赖 Python 原型：`moat_ovha/`。
- Phase 1 toy operator zoo：`moat_ovha/data/operator_zoo.py`。
- Phase 1 baseline / ablation：`moat_ovha/baselines/`。
- Phase 1 自动化：`train_meta_operator.py`、`eval_meta_operator.py`、`scripts/run_phase1_sweep.sh`、`scripts/summarize_phase1.py`。
- Phase 1 输出：`outputs/phase1/`。
- Phase 1.5 PyTorch surface：`moat_ovha_torch/`。
- Phase 1.5 CPU smoke 输出：`outputs/phase1_5_cpu_smoke/`。

## 最新验证结论

最近一次验证命令：

```bash
python3 -m unittest discover -s tests
sh scripts/run_phase1_sweep.sh configs/phase1_minimal.json
```

结果：

- 12 个 unittest 全部通过。
- `outputs/phase1/train_metrics.jsonl` 和 `outputs/phase1/eval_metrics.jsonl` 各 156 行。
- `outputs/phase1/phase1_report.md` 给出 Go 结论。
- `outputs/phase1/phase1_gpt_pro_summary.md` 是交给 GPT Pro 或新窗口继续分析的压缩交接稿。

## 当前 Go / No-Go

Go: 在 deterministic toy sweep 中，OVHA-full 对所有配置的非 OVHA baseline 与 ablation，在每个 operator family 上均胜出。

这个结论只证明 Phase 1 scaffold 闭环有效，不等于论文级 benchmark。它支持“C 不是装饰项”的最小证据，但下一阶段必须降低 metadata 依赖，并接入可学习训练循环。

## Phase 1.5 优先级

1. 在 Ubuntu/A800 上运行 `scripts/run_phase1_5_gpu_small.sh`。
2. 若 GPU small 稳定，再运行 `scripts/run_phase1_5_gpu_main.sh`。
3. 对比 `ovha_full`、`simple_stack`、`transformer_only`、`no_memory`、`no_hyper_adapter`、`vector_value_only`。
4. 检查 context scaling、resolution transfer、confusable context 和 oracle upper-bound gap。
5. 用 GPU 结果决定是否进入 Phase 2 真实 benchmark。

## 给新窗口的读取顺序

1. `docs/project_direction.md`
2. `docs/phases/phase_1.md`
3. `docs/phases/phase_1_5.md`
4. `theory_notes/ovha_math.md`
5. `theory_notes/ovha_context_identifiability.md`
6. `outputs/phase1/phase1_gpt_pro_summary.md`
7. `README.md`
