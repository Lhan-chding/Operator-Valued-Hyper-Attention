# OVHA 特例归约

## Standard Attention

设 primitive 集合只有 `O_id`，并令 `pi_id(q,M_D)=1`。若 `O_id` 对离散 value samples 执行：

`[O_id(u*)](q)=Σ_i softmax(l(q,k_i)) v_i`,

则 OVHA 退化为普通 attention。代码中对应 `IdentityValuePrimitive` 与 `standard_attention`。

## FNO-Style Spectral Kernel

令：

`K(q,s)=Σ_m a_m cos(2πm(q-s)) + b_m sin(2πm(q-s))`.

规则网格上这可由 FFT 高效计算；Phase 1 的零依赖实现使用有限 Fourier basis 直接求和。代码中对应 `FourierPrimitive` 与 `fourier_kernel`。

## DeepONet-Style Separable Kernel

令：

`K(q,s)=Σ_l phi_l(q) psi_l(s)`.

于是：

`y(q)=Σ_l phi_l(q) ∫ psi_l(s)u(s)ds`.

这正是 branch/trunk 分解的 operator view。代码中对应 `SeparablePrimitive` 与 `deep_operator_kernel`。

## Graph / GNO-Style Local Kernel

令：

`K(q,s)=0 if (q,s) not in E`,  
`K(q,s)=rho(q,s,e_qs) if (q,s) in E`.

其中 `E` 可以来自 kNN、mesh adjacency、接触图或几何邻域。Phase 1 以 radial local kernel 近似该设定，对应 `LocalKernelPrimitive` 与 `graph_local_kernel`。

## Simple Stack 不是 OVHA

`simple_stack` baseline 先分别运行多个 primitive，再平均输出。它没有 query/context-conditioned `pi_r(q,M_D)`，也没有 `eta_r(M_D,q)`。因此它只能证明“多个算子模块拼起来”的上限，而不能替代 operator-valued attention。
