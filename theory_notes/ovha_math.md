# OVHA 数学定义与 Phase-1 Proof Sketch

## 1. Meta-Operator Learning 设定

设任务 `tau ~ P(T)`，每个任务对应未知算子：

`G_tau: X_tau -> Y_tau`. {#eq:task-operator}

推理时模型只观察少量 context demonstrations：

`D_tau = {(u_j, y_j, m_j)}_{j=1..n}, y_j = G_tau(u_j)`. {#eq:context}

给定 query `q` 与待预测输入函数 `u*`，目标是在无参数更新条件下预测：

`y_hat(q) = M_Theta(D_tau, u*, q) ≈ [G_tau(u*)](q)`. {#eq:meta-objective}

这里 `q` 不是固定 PDE 网格点；它可以是空间点、时间点、pixel/ray、材料状态、轨迹 waypoint 或动作槽位。Phase 1 因此只使用抽象的 context-to-query operator 设定，不把 VCM/CV/具身任务放进主线。

## 2. Operator-Valued Attention

标准 attention 的 value 是有限维向量：

`Attn(q; K,V) = Σ_i alpha_i(q,K) v_i`,  
`alpha_i = softmax(q^T k_i / sqrt(d))`. {#eq:standard-attention}

OVHA 将 attention value 从 vector-valued object 扩展为 operator primitive：

`OVHA(q; D_tau, u*) = Σ_r pi_r(q, M_D) [O_r(eta_r(M_D, q))(u*)](q)`. {#eq:ovha-discrete}

其中：

- `M_D = E_phi(D_tau)` 是 operator memory，承载当前任务的 context state。
- `O_r` 是结构化 operator primitive，例如 Fourier/spectral、separable basis、local graph kernel、implicit field。
- `eta_r(M_D, q)` 是 hyper-adapter 生成或调制的 primitive 参数。
- `pi_r(q, M_D)` 是 query-conditioned primitive routing 权重。

连续核形式为：

`y(q) = ∫ K_Theta(q,s;M_D) u*(s) ds`. {#eq:kernel-operator}

`K_Theta(q,s;M_D) = Σ_r pi_r(q,M_D) K_r(q,s;eta_r(M_D,q))`. {#eq:ovha-kernel}

这个定义的核心不是把 FNO/DeepONet/Transformer 串起来，而是把 attention 的 value 域升级为 operator-valued kernel component。

## 3. A/B/C/D 边界

`C: Operator-Valued Attention` 是理论核心，即 `{#eq:ovha-discrete}` 与 `{#eq:ovha-kernel}`。

`A: Operator Memory` 只负责从 `D_tau` 产生 `M_D`。Phase 1 提供 permutation-aware pooling 和 Perceiver-style latent cross-attention 两个版本。

`B: Hyper-Operator Adapter` 只调制 primitive 的低秩参数、scale/bias、basis/kernel coefficient。Phase 1 不生成完整大权重。

`D: Router / MoE` 是扩展机制。Phase 1 支持 dense mixing 与 top-k sparse routing，但不把 routing 本身当作主创新。

## 4. 命题 1：包含性

**命题。** OVHA 包含 standard attention、FNO-like Fourier kernel、DeepONet-like separable operator、Graph/GNO-style local kernel 作为特例。

**Proof sketch.**

1. Standard attention：令 primitive 只有 `O_id`，其核为离散 delta value kernel，`[O_id(u*)](q)=Σ_i alpha_i(q,k_i)v_i`，且 `pi_id=1`，则 `{#eq:ovha-discrete}` 退化为 `{#eq:standard-attention}`。
2. FNO-like layer：令 `K_r(q,s)=Σ_m a_m exp(i m(q-s))`，并在规则网格上用 FFT 或有限 Fourier basis 计算，则 `{#eq:kernel-operator}` 是 spectral convolution primitive。
3. DeepONet-like layer：令 `K_r(q,s)=Σ_l phi_l(q) psi_l(s)`，branch 计算 `∫ psi_l(s)u(s)ds`，trunk 计算 `phi_l(q)`，则得到 separable low-rank operator。
4. Graph/GNO-style layer：令 `K_r(q,s)` 只在邻域或图边 `E` 上非零，并由几何边特征调制，则得到 sparse local operator kernel。

## 5. 命题 2：Operator Universal Approximation

**命题。** 在 compact function/operator family 上，如果 primitive 集合中至少一个成员是 universal operator approximator，且 memory encoder、router、hyper-adapter 具有足够容量，则 OVHA 可逼近连续的 context-conditioned meta-operator。

**Proof sketch.**

设目标 meta-operator 为 `F(D_tau,u*,q)`，且关于 sufficient context statistic `S(D_tau)` 连续。由 DeepONet/MIONet 类 operator approximation，可构造 primitive `O_r` 逼近 `u* -> y(q)`。由普通 universal approximation，可令 `E_phi(D_tau)` 逼近 `S(D_tau)`，令 `eta_r` 与 `pi_r` 逼近所需的连续参数映射和混合权重。连续函数在加权组合下闭包，因此 `{#eq:ovha-discrete}` 可逼近 `F`。该命题依赖 context identifiability，不是无条件保证。

## 6. 命题 3：Context Identifiability

**命题。** 若 context 在任务分布上能把算子区分到 `epsilon` 级别，即

`d(G_tau, G_tau') <= C d(E(D_tau), E(D_tau')) + epsilon`, {#eq:context-identifiability}

则存在 memory encoder 近似 sufficient task statistic。

**Proof sketch / 实验路径。** 该条件应作为 Phase 1 假设和实验变量，而不是强行证明。实验上通过 context-size sweep、context noise、ambiguous context 对比，观察 error 是否随 context 数量下降，以及 memory-free ablation 是否显著变差。

## 7. 命题 4：Discretization Consistency

**命题。** 若每个 `K_r(q,s;eta_r)` 对 `s` 可积且对 `q` 连续，离散 samples 用 quadrature 近似 `{#eq:kernel-operator}`，则采样分辨率增加时 OVHA 输出收敛到连续 operator 形式。

**Proof sketch.** 对固定 `q`，`K_r(q,s)u(s)` 可积，标准 quadrature convergence 给出 `Σ_i w_i K_r(q,s_i)u(s_i) -> ∫ K_r(q,s)u(s)ds`。有限个 primitive 的加权和保持收敛；若 `pi_r` 与 `eta_r` 对 query/memory 连续且有界，收敛可传递到 `{#eq:ovha-kernel}`。

## 8. 命题 5：结构化 Primitive 的样本效率假设

**命题。** 对可分解为 spectral + separable + local components 的任务族，OVHA 相比纯 vector-value attention、simple stack 或单 primitive baseline 应表现出更低误差、更强 parameter holdout 和更可解释 primitive specialization。

**验证路径。** Phase 1 不把该命题当作已证明定理，而是作为可证伪假设。对应验收指标是 ablation gap、context scaling、parameter holdout、operator-family holdout、primitive usage entropy 与 usage-family correlation。

## 9. Phase-1 最小实现映射

- `moat_ovha.models.primitives`: `FourierPrimitive`、`SeparablePrimitive`、`LocalKernelPrimitive`、`IdentityValuePrimitive`。
- `moat_ovha.models.memory`: pooling memory 与 Perceiver-style latent memory。
- `moat_ovha.models.hyper_adapter`: low-rank scale/bias adapter。
- `moat_ovha.models.router`: dense/top-k router 与 entropy diagnostic。
- `moat_ovha.models.ovha_layer`: `{#eq:ovha-discrete}` 的可运行版本。
- `moat_ovha.data.operator_zoo`: Fourier、Green/local、separable、nonlinear、mixed operator families。
- `moat_ovha.baselines`: Transformer-only、Perceiver IO-style、ICON-style、DeepONet-only、FNO-only、simple stack 与 ablations。

Phase 1 的目标是建立“定义清楚 + 特例归约 + 最小实验不可替代性”的闭环，而不是追求大 benchmark。
