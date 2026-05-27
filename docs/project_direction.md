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
| Phase 2 | Not started | 待创建 | 建议进入 PyTorch 可学习版本、metadata-free context inference 和更强 OOD/holdout。 |

## 当前仓库状态

当前已完成：

- Phase 1 理论笔记：`theory_notes/ovha_math.md`、`theory_notes/special_cases.md`。
- Phase 1 零依赖 Python 原型：`moat_ovha/`。
- Phase 1 toy operator zoo：`moat_ovha/data/operator_zoo.py`。
- Phase 1 baseline / ablation：`moat_ovha/baselines/`。
- Phase 1 自动化：`train_meta_operator.py`、`eval_meta_operator.py`、`scripts/run_phase1_sweep.sh`、`scripts/summarize_phase1.py`。
- Phase 1 输出：`outputs/phase1/`。

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

## Phase 2 优先级

1. PyTorch 化当前接口，保持 `memory -> router/hyper_adapter -> primitives -> OVHALayer` 的结构不变。
2. 去掉或扰动 context metadata hints，验证 memory 是否能从 demonstrations 本身识别 operator family / parameters。
3. 加入真实训练循环与 learned router/hyper-adapter，而不是 deterministic oracle-style 调制。
4. 扩展 resolution transfer、parameter holdout、operator-family holdout。
5. 选择一个更接近论文场景的任务族：PDE operator learning、材料响应或 dense-query field prediction。
6. 把 proof sketch 升级为可发表 method section 的严谨版本，明确 compactness、context identifiability、continuity 与 primitive closure 条件。

## 给新窗口的读取顺序

1. `docs/project_direction.md`
2. `docs/phases/phase_1.md`
3. `theory_notes/ovha_math.md`
4. `outputs/phase1/phase1_gpt_pro_summary.md`
5. `README.md`
