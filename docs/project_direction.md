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
| Phase 1.5 | GPU provisional complete | `docs/phases/phase_1_5.md` | A800 seeds 31/32/33/34 均完成；OVHA-full 多 seed 优于 transformer/vector/simple-stack baselines。 |
| Phase 2 | Not started | 待创建 | 真实 PDE/material/dense-query benchmark；同时补强 component-necessity ablation。 |

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
- Phase 1.5 A800 多 seed 输出：`outputs/a800_phase1_5/`。

## 最新验证结论

最近一次本机 Phase 1 验证命令：

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

Phase 1 Go: 在 deterministic toy sweep 中，OVHA-full 对所有配置的非 OVHA baseline 与 ablation，在每个 operator family 上均胜出。

Phase 1.5 provisional Go: 在 A800 seeds 31/32/33/34 上，metadata-free `ovha_full` 均优于 `transformer_only`、`ovha_vector_value_only` 和 `simple_stack`。四 seed aggregate mean relL2 分别为 `ovha_full=1.324814`、`transformer_only=2.379144`、`ovha_vector_value_only=2.382044`、`simple_stack=2.480682`。

限制：`local_only`、`separable_only` 与若干 no-query/no-memory ablation 在 aggregate 上接近或略优于 `ovha_full`，所以当前结果支持 operator-valued attention 相对 vector baseline 的 metadata-free 比较，但还不构成每个组件必要性的强证明。

## Phase 2 优先级

1. 创建 `docs/phases/phase_2.md`，定义真实 PDE/material/dense-query benchmark。
2. 保留 metadata-free 边界：模型输入仍只允许 context/target/support/masks。
3. 设计更强 component-necessity ablation，明确 memory/router/hyper-adapter 的贡献。
4. 检查 context scaling、resolution transfer、confusable context 和 oracle upper-bound gap。
5. 如需继续 synthetic validation，加入更重 A800 配置和更难 local/nonlinear families。

## 给新窗口的读取顺序

1. `docs/project_direction.md`
2. `docs/phases/phase_1.md`
3. `docs/phases/phase_1_5.md`
4. `theory_notes/ovha_math.md`
5. `theory_notes/ovha_context_identifiability.md`
6. `outputs/phase1/phase1_gpt_pro_summary.md`
7. `outputs/a800_phase1_5/phase1_5_a800_multiseed_summary.md`
8. `README.md`
