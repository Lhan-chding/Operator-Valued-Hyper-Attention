# OVHA Project Memory

## 当前状态（2026-05-27）

- **阶段**: Phase 1.6: Learned-Evaluation Integrity + Benchmark Protocol Foundation
- **状态**: 进行中，已完成 checkpoint-loaded evaluation 与 protocol foundation；A800 训练刚暴露并修复 `simple_stack` 梯度断图问题。
- **核心方向更正**: OVHA 的长期目标不是 PDE、FNO、DeepONet、CV 或物理场中的任一单点应用，而是一个面向多模态/多域的 operator-valued meta-transformer 路线。
- **关键边界**: PDEBench/FNO/Mechanical MNIST/CV/物理场只是验证场景、接口压力测试或论文证据来源，不是方法定义或应用边界。

## 长期技术路线

用户明确希望坚持以下理解，后续 session 不要被具体 benchmark 名称带偏：

1. **理论核心**: Operator-Valued Attention。把 attention value 从有限维向量升级为可组合、可路由、可调制的 operator primitive。
2. **实现主体**: Hyper-Operator Transformer。用 transformer 结构生成、调制或组合 operator-valued primitives，而不是只做普通 token mixing。
3. **上下文机制**: Operator Memory Transformer。模型从 context demonstrations 中形成 task/operator memory，用于识别当前任务对应的 operator 组合。
4. **多模态含义**: 多模态不是仅指 image/text/audio 标签，而是不同输入空间、query 空间、输出场/轨迹/密集预测之间都可通过统一的 operator-query episode 接口表达。
5. **实验路线**: synthetic controlled stress 用于验证机制；PDE/material/CV/trajectory 等真实任务用于外部有效性。任何单一数据域都不能反过来限定 OVHA 的定义。

## 当前实现状态

- `moat_ovha_torch/eval/evaluator.py` 默认要求加载 trained checkpoint。
- `moat_ovha_torch/train/train_many.py` 支持按 config 中的 `seeds` 和 `train_models` 训练。
- `moat_ovha_torch/data/component_stress_zoo.py` 提供 controlled analytical stress families。
- `moat_ovha_torch/data/benchmark_registry.py` 和 `moat_ovha_torch/data/public_benchmarks/` 只是 public benchmark 接入骨架。
- Phase 1.6 GPU scripts 默认使用 physical GPU 3，除非显式设置 `CUDA_VISIBLE_DEVICES`。

## 注意事项

- 不要把“DeepONet/FNO/PDEBench/CV/物理场”写成 OVHA 的限定目标；它们只是比较对象或应用样例。
- 不要把 Phase 1.6 synthetic stress 结果写成多模态能力证明；它只能证明机制是否可训练、是否 checkpoint-loaded、ablation 是否公平。
- 后续真正靠近多模态时，需要把 episode adapter 扩展到 image-to-field、dense-query、trajectory-query、sequence-to-field 等接口，并设计跨域 shared primitive/router/memory 的评估。

## 下一步计划

- **立即要做**: 在 A800 上 `git pull`，确认 `c18e8dc` 之后的修复和本 memory 更新都在分支上，然后用 GPU 3 继续 Phase 1.6 integrity/component 训练。
- **后续阶段**: 将 `Operator-Valued Attention + Hyper-Operator Transformer + Operator Memory Transformer` 写成更正式的 architecture note。
- **待解决问题**: public benchmark loader 仍只是骨架；真正多模态 benchmark 还需要单独设计数据接口和实验矩阵。
