# OVHA Phase 1.5 汇总交接稿

## 目标

Phase 1.5 按 `ovha_phase1_5_metadata_free_pytorch_plan_cn_v1.docx` 执行，目标是把 Phase 1 的 metadata-conditioned deterministic scaffold 推进到 metadata-free episodic meta-operator learning，并新增 PyTorch 可学习实现。

核心问题：

不给模型 family label、gain、operator hints、latent params 或 hidden mixture weights 时，OVHA 是否能从 context observations 中形成 operator memory，并通过 operator-valued attention 组合 primitives？

## 已完成产物

代码：

- `moat_ovha_torch/data/episodes.py`
- `moat_ovha_torch/data/operator_zoo_torch.py`
- `moat_ovha_torch/models/context_encoder.py`
- `moat_ovha_torch/models/memory.py`
- `moat_ovha_torch/models/hyper_adapter.py`
- `moat_ovha_torch/models/router.py`
- `moat_ovha_torch/models/primitives/`
- `moat_ovha_torch/models/ovha.py`
- `moat_ovha_torch/models/baselines.py`
- `moat_ovha_torch/train/`
- `moat_ovha_torch/eval/`

CLI / scripts：

- `train_torch_meta_operator.py`
- `eval_torch_meta_operator.py`
- `scripts/run_phase1_5_cpu_smoke.sh`
- `scripts/run_phase1_5_gpu_small.sh`
- `scripts/run_phase1_5_gpu_main.sh`
- `scripts/run_phase1_5_ablation_matrix.sh`
- `scripts/summarize_phase1_5.py`

配置：

- `configs/phase1_5_cpu_smoke.json`
- `configs/phase1_5_gpu_small.json`
- `configs/phase1_5_gpu_main.json`
- `configs/phase1_5_ablation_matrix.json`
- `requirements_torch.txt`

理论 / 文档：

- `theory_notes/ovha_context_identifiability.md`
- `theory_notes/ovha_phase1_5_boundaries.md`
- `docs/execution_policy_mac_vs_a800.md`
- `docs/phases/phase_1_5.md`
- `docs/project_direction.md`

## Metadata-Free 防线

模型输入只包含：

- `context_u`
- `context_q`
- `context_y`
- `target_u`
- `target_q`
- `support_grid`
- `context_mask`
- `target_mask`

Hidden metadata 单独保存在 `EpisodeHiddenInfo`，只能用于 data generation、offline diagnostics、leakage tests 和显式 oracle upper-bound baseline。

测试覆盖：

- public batch fields 不包含 forbidden metadata keys。
- hidden metadata 变化不改变 model input hash。
- forbidden key 进入 model input 时抛错。
- `oracle_metadata_upper_bound` 在报告中单独标记。

## 本机环境

本机已创建 `.venv` 并安装：

- torch 2.8.0
- numpy 2.0.2

`outputs/phase1_5_cpu_smoke/environment.json`：

- `torch_available: true`
- `numpy_available: true`
- `device: cpu`
- `machine: arm64`
- `cuda_available: false`
- `mps_available: false`

## 验证命令

```bash
.venv/bin/python -m unittest discover -s tests
PYTHON=.venv/bin/python bash scripts/run_phase1_5_cpu_smoke.sh
```

测试结果：

- 19 个 unittest 全部通过。
- torch optional forward/backward test 已真实执行。

CPU smoke 产物：

- `outputs/phase1_5_cpu_smoke/train_metrics.jsonl`：20 rows
- `outputs/phase1_5_cpu_smoke/eval_metrics.jsonl`：320 rows
- `outputs/phase1_5_cpu_smoke/diagnostics.jsonl`：320 rows
- `outputs/phase1_5_cpu_smoke/phase1_5_report.md`

## 当前本机结论

Local CPU smoke: Provisional Go.

报告结论：OVHA-full 在本机 smoke run 中优于 `simple_stack` 和 vector attention comparator。该结论只说明本地 Phase 1.5 pipeline 能真实训练、评估、诊断和报告；不能替代 GPU small/main 的稳定训练结论。

## 重要限制

- CPU smoke 只有 20 steps，不能作为论文级 evidence。
- 当前 eval 每个 baseline 是独立随机初始化评估，不是完整训练后的公平 benchmark。
- `oracle_metadata_upper_bound` 目前是显式 upper-bound baseline 名称，但仍需在 GPU run 中完善其真正 oracle usage。
- context scaling slope、family holdout gap、resolution transfer gap 已有字段/脚本框架，但需要 GPU small/main 产生更稳的统计。
- MPS 在当前环境不可用，Mac 只完成 CPU smoke。

## 建议 GPT Pro 下一步分析

1. 当前 PyTorch architecture 是否忠实保持 OVHA 公式，而不是 Transformer -> primitive stack？
2. metadata-free leakage guard 是否足够，是否还需要 AST/config 层面的防线？
3. GPU small 应如何调整，使 `ovha_full`、`no_memory`、`no_hyper_adapter`、`vector_value_only` 的 ablation gap 更可信？
4. `oracle_metadata_upper_bound` 应如何实现得更像真正 upper bound，同时不污染主模型输入？
5. context identifiability 的 theory note 是否足够支撑 confusable_context 的失败解释？
6. Phase 2 前是否需要先做 multi-seed GPU small，而不是直接进入 PDE/material benchmark？

## Ubuntu/A800 下一步命令

```bash
git pull
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_torch.txt
python3 -m unittest discover -s tests
bash scripts/run_phase1_5_gpu_small.sh
bash scripts/run_phase1_5_gpu_main.sh
python3 scripts/summarize_phase1_5.py outputs/phase1_5_gpu_main
```
