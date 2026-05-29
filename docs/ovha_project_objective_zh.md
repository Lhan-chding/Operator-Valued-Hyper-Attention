# OVHA 项目总目标长期记忆

更新时间：2026-05-29

本文档记录 OVHA 项目的长期总目标。后续代码设计、实验设计、文档叙事和审稿回复都应以此为边界，避免把项目收缩成某一个 benchmark、某一种 neural operator baseline，或某一个物理仿真垂直应用。

## 必须长期保持的总目标

OVHA 的总目标不是做一个 PDEBench-only、PINN-only、FNO-only、DeepONet-only 或电磁仿真专用项目。PDEBench、DeepONet、FNO、PINN、物理场、电磁仿真、CV 等都只是阶段性验证场景或解释例子，不是研究边界。

核心目标是提出一种面向多模态/跨模态泛化的通用方法论：

- 以 **Operator-Valued Attention** 作为理论核心；
- 以 **Hyper-Operator Transformer** 作为实现框架；
- 以 **Operator Memory Transformer** 作为上下文记忆机制；
- 通过 **primitive routing**、**operator-specific memory**、**hyper-adapter 参数生成**、**可解释 operator decomposition**，实现结构化、可诊断、可组合的 operator learning。

后续任何代码设计、实验设计、文档叙事都必须围绕这个总目标展开，不能把项目收缩成 “few-shot neural operator” 或 “PDEBench 上的 ICON 变体”。

PDEBench / controlled-v2 当前只用于验证机制是否成立，最终目标是形成一种可迁移到连续场、图像、文本、音频、跨模态 token 等不同数据形态的通用 operator-valued attention 框架。

## 对后续工作的约束

1. **不要把验证场景写成研究边界。**
   PDEBench、controlled-v2、DeepONet/FNO/PINN 对比、电磁仿真、CV dense query 等，都只能被写作 evidence surfaces 或 staged validation，不应被写作 OVHA 的定义域。

2. **不要把方法降格成普通 few-shot neural operator。**
   Few-shot operator learning 可以是一个评估协议，但 OVHA 的核心卖点应保持为 operator-valued attention、operator memory、router/adapter/primitive decomposition 和跨模态可迁移框架。

3. **代码设计要保留跨模态接口。**
   `u/q/y` episode contract、public cache、field-to-episode adapter、memory/router/adapter 抽象都应尽量保持数据形态中立，避免写死为 PDE-only。

4. **实验设计要区分机制验证和外部验证。**
   Controlled-v2 用于检查 router、memory、adapter、primitive decomposition 是否成立；PDEBench 和后续真实数据用于外部有效性，不是最终应用边界。

5. **文档叙事要先定义方法，再列证据场景。**
   推荐叙事顺序是：Operator-Valued Attention 方法论 -> Hyper-Operator Transformer 实现 -> Operator Memory Transformer 上下文机制 -> staged evidence on controlled-v2 / PDEBench / future multimodal benchmarks。

## 一句话版本

OVHA 要做的是一种可诊断、可组合、可迁移的通用 operator-valued attention 框架；PDEBench 和 controlled-v2 只是当前阶段用来证明机制的实验地面，不是项目天花板。
