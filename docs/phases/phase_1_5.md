# Phase 1.5: Metadata-Free PyTorch OVHA

## 目标

把 Phase 1 从 metadata-conditioned deterministic scaffold 推进到 metadata-free episodic meta-operator learning，并新增 PyTorch 可学习实现。

主问题：

如果不给模型 family label、gain、hints、latent parameters 或 hidden mixture weights，OVHA 是否能只从 context observations 中形成 operator memory，并通过 operator-valued attention 组合 primitives？

## 本阶段范围

包含：

- metadata-free episode dataclass 与 leakage guard
- PyTorch context encoder、memory encoder、hyper-adapter、router、operator primitives、OVHA core
- baseline / ablation factory
- train/eval CLI
- CPU smoke config
- GPU/A800 scripts
- Phase 1.5 report summarizer
- context identifiability 与 metadata-free 边界理论文档

不包含：

- 在 Mac Air 上跑大 sweep
- PDE/CV/robotics benchmark
- A800 实际训练
- full hypernetwork 生成全部权重
- SSM/Mamba、SE(3)、复杂 mesh primitive

## 关键文件

- `moat_ovha_torch/`
- `configs/phase1_5_cpu_smoke.json`
- `configs/phase1_5_gpu_small.json`
- `configs/phase1_5_gpu_main.json`
- `configs/phase1_5_ablation_matrix.json`
- `train_torch_meta_operator.py`
- `eval_torch_meta_operator.py`
- `scripts/run_phase1_5_cpu_smoke.sh`
- `scripts/run_phase1_5_gpu_small.sh`
- `scripts/run_phase1_5_gpu_main.sh`
- `scripts/run_phase1_5_ablation_matrix.sh`
- `scripts/summarize_phase1_5.py`
- `theory_notes/ovha_context_identifiability.md`
- `theory_notes/ovha_phase1_5_boundaries.md`
- `docs/execution_policy_mac_vs_a800.md`

## 验证标准

Local Mac validation:

```bash
python3 -m unittest discover -s tests
bash scripts/run_phase1_5_cpu_smoke.sh
```

如果 torch 未安装：

- standard-library Phase 1 tests must still pass.
- torch-dependent tests should skip.
- CPU smoke scripts should write `environment.json` and `phase1_5_report.md` explaining the skip.

如果 torch 已安装：

- CPU smoke should run training/evaluation.
- outputs should include train/eval metrics, diagnostics, environment and report.

## Go / No-Go 标准

Phase 1.5 进入 Phase 2 的 Go 条件：

1. metadata leakage tests 全部通过。
2. CPU smoke 正常，GPU small 至少能稳定下降。
3. metadata-free OVHA-full 在 mixed/compositional split 上优于 best non-OVHA baseline。
4. no_memory、no_hyper_adapter、vector_value_only、random_router 明显退化。
5. context size 增加时 error slope 为负。
6. resolution transfer gap 可控。
7. router 不完全塌缩。
8. oracle metadata upper bound 单独报告，且 metadata-free 不大幅失败。
9. confusable_context 被识别为困难 split，而不是被错误宣传成成功。

## 当前本机验证结果

当前 Mac 环境一开始没有 `torch` / `numpy`。已按用户许可在项目 `.venv` 中安装 `torch==2.8.0` 与 `numpy==2.0.2`，并完成本机 CPU smoke。

验证命令：

```bash
.venv/bin/python -m unittest discover -s tests
PYTHON=.venv/bin/python bash scripts/run_phase1_5_cpu_smoke.sh
```

结果：

- 19 个 unittest 全部通过。
- torch optional forward/backward test 已真实执行。
- `train_metrics.jsonl`: 20 rows。
- `eval_metrics.jsonl`: 320 rows。
- `diagnostics.jsonl`: 320 rows。
- `phase1_5_report.md`: local smoke provisional Go。

注意：CPU smoke 只验证 pipeline 和最小训练可运行性，不替代 GPU small/main benchmark。

## 下一步

用户可在 Ubuntu/A800 上运行：

```bash
bash scripts/run_phase1_5_gpu_small.sh
bash scripts/run_phase1_5_gpu_main.sh
python3 scripts/summarize_phase1_5.py outputs/phase1_5_gpu_main
```
