# CMU-MOSEI 公共多模态实验客观审查材料

本文档用于提交给外部审查者进行严格代码与实验协议审查。本文只陈述已经观察到的代码状态、运行条件和实验结果，不把结果解释为论文结论。

## 使用方式

建议把本文档和项目源码压缩包一起提交给审查者。请审查者优先阅读本文档中的异常现象，再对照源码检查训练目标、评估指标、router/candidate 实现、baseline 协议和 artifact 写出路径。

源码压缩包应只包含仓库跟踪的源码、配置、测试和文档，不包含：

- `data/`
- `outputs/`
- `.venv/`
- `gpt_pro_exports/`
- 其他运行缓存或大体积实验产物

## 审查目标

请审查本仓库中 CMU-MOSEI 公共多模态实验路径是否存在关键实现或协议问题，重点包括：

- 数据加载、标签、split、特征与监督目标是否对齐。
- `run_public_main.py` 的训练、评估、指标写出是否一致。
- OVHA full 与内部消融模型是否真正训练了对应结构，而不是同一个线性 probe。
- `task_loss`、`candidate_individual_loss`、MAE、heldout loss、robustness score 等指标是否被正确计算和解释。
- Router 权重、candidate loss、candidate 输出尺度是否存在异常。
- 当前实验是否存在过拟合、路由塌缩、候选分支输出爆炸、指标错配或训练/评估 split 混用。

## 当前代码状态

本地分支：

- `phase1_5_metadata_free_torch`

相关近期提交：

- `cbd7513 fix: train public ovha ablation variants`
  - 目标：让 `ovha_no_*` / `cato_only` 等内部消融走真实 OVHA 结构变体，不再走线性 baseline。
- `aabe43c fix: report effective ovha ablation router loads`
  - 目标：修复消融后的有效 router 权重没有写回 `router_load_by_candidate` 的诊断问题。
- `4c3394e feat: weight public multimodal loss components`
  - 目标：支持通过 `loss_metadata.<loss>.weight` 配置 public loss 组件权重。

关键文件：

- `scripts/multimodal/run_public_main.py`
- `scripts/multimodal/run_public_smoke.py`
- `moat_ovha_torch/models/multimodal/ovha_multimodal.py`
- `moat_ovha_torch/models/multimodal/router.py`
- `moat_ovha_torch/models/multimodal/joint_router_adapter.py`
- `moat_ovha_torch/train/multimodal_protocol.py`
- `configs/multimodal_cmu_mosei_public_main.json`
- `tests/test_multimodal_mainline_contracts.py`
- `tests/test_multimodal_training_protocol.py`
- `tests/test_multimodal_experiment_protocol.py`

## 数据和任务

数据集：`cmu_mosei`

任务类型：`sentiment_emotion`

当前公共主实验配置路径：

- `configs/multimodal_cmu_mosei_public_main.json`

缓存路径：

- `data/multimodal_cache/cmu_mosei/v0.1`

已观察到的数据规模与标签分布：

```text
train shape (2249, 1, 1)
train mean 0.30293336510658264
train std 0.7449261546134949
train min -2.266666889190674
train max 2.6666667461395264
train zero_loss 0.646436870098114

test shape (676, 1, 1)
test mean 0.283248633146286
test std 0.7539870738983154
test min -2.3333332538604736
test max 2.5
test zero_loss 0.6478853225708008

train_mean_on_test_loss 0.5680429935455322
train_mean_on_test_mae 0.5858057737350464
```

这些数字来自远端服务器 `/home/david/work/Operator-Valued-Hyper-Attention` 的当前 cache。

## 当前模型/对照设置

CMU-MOSEI public main 配置中的 same-feature baselines 包括：

```text
text_only
audio_only
vision_only
concat_fusion
ovha_no_lrio
ovha_no_spo
ovha_no_rceo
ovha_no_evidence_router
```

当前协议中：

- `text_only`、`audio_only`、`vision_only`、`concat_fusion` 是 same-feature sanity probe，参数量很小。
- `ovha_no_lrio`、`ovha_no_spo`、`ovha_no_rceo`、`ovha_no_evidence_router` 应为内部 OVHA 结构消融。
- `ovha_full` 是完整 OVHA。

在新版本中观察到参数量：

```text
text_only params=301
audio_only params=75
vision_only params=36
concat_fusion params=410
ovha_full params=386613
ovha_no_lrio params=386613
ovha_no_spo params=386613
ovha_no_rceo params=386093
ovha_no_evidence_router params=386613
```

## 已运行实验 1：旧/新消融混合阶段的 2 seed 6000 steps 诊断

在运行此实验之前，还有一次旧 public main 50k steps 运行。该运行发生在内部消融和 baseline 命名/协议修正之前，因此不应作为正式结果，只用于说明异常最早出现的形态：

```text
old cmu_mosei_main 50k, 5 seeds, test:
  linear/probe baseline family mean approximately 0.3566 heldout loss
  ovha_full mean approximately 0.6651 heldout loss
  ovha_full training loss reached very low values near the end
```

该旧结果显示：训练 loss 很低，但 test heldout loss 明显差。后续实验的目的不是美化该结果，而是确认问题是否来自旧 baseline 协议、内部消融实现、candidate loss 权重或更底层的模型/评估问题。

另有一次 2 seed 3000 steps 诊断，发生在真实内部消融接入之后：

```text
cmu_mosei_probe_new_ablation_2seed_3000, test:
  ovha_no_lrio                     mean=0.370238 n=2
  text_only                        mean=0.383979 n=2
  concat_fusion                    mean=0.415630 n=2
  ovha_no_evidence_router          mean=0.430329 n=2
  ovha_full                        mean=0.431527 n=2
  ovha_no_rceo                     mean=0.441691 n=2
  ovha_no_spo                      mean=0.455530 n=2
  audio_only                       mean=0.525442 n=2
  vision_only                      mean=0.525766 n=2
```

该 3000 steps 诊断中，`ovha_no_lrio` 曾优于 same-feature probes，但 `ovha_full` 仍没有稳定优于简单 probe。随后 6000 steps 诊断中，text/concat 反而明显优于 OVHA full 和多数消融。

路径：

- `outputs/multimodal/cmu_mosei_probe_new_ablation_2seed_6000/raw_metrics.jsonl`
- `outputs/multimodal/cmu_mosei_probe_new_ablation_2seed_6000/robustness_rows.jsonl`
- `outputs/multimodal/cmu_mosei_probe_new_ablation_2seed_6000_train.log`

设置：

```text
seeds: 301, 302
train_steps: 6000
baseline_train_steps: 6000
d_model: 128
memory_tokens: 8
learning_rate: 1e-4
eval_split: test
device: cuda
```

聚合结果，`score` 为 `heldout_task_loss`，越低越好：

```text
text_only                        mean=0.347137 n=2 params=301
concat_fusion                    mean=0.349067 n=2 params=410
ovha_no_lrio                     mean=0.458605 n=2 params=386613
audio_only                       mean=0.521026 n=2 params=75
vision_only                      mean=0.523385 n=2 params=36
ovha_full                        mean=0.529495 n=2 params=386613
ovha_no_evidence_router          mean=0.543013 n=2 params=386613
ovha_no_rceo                     mean=0.553165 n=2 params=386093
ovha_no_spo                      mean=0.557854 n=2 params=386613
```

每 seed 对比：

```text
seed 301:
  ovha_full=0.5422738194465637
  ovha_no_lrio=0.4626452326774597
  concat_fusion=0.353342205286026
  text_only=0.3477528989315033
  best=text_only

seed 302:
  ovha_full=0.5167160630226135
  ovha_no_lrio=0.45456501841545105
  concat_fusion=0.34479188919067383
  text_only=0.34652209281921387
  best=concat_fusion
```

训练末尾 loss 摘要：

```text
seed 301:
  text_only train_loss_final=0.295268714427948
  audio_only train_loss_final=0.5012677907943726
  vision_only train_loss_final=0.48056232929229736
  concat_fusion train_loss_final=0.3007565140724182
  ovha_no_lrio train_loss_final=0.14655058085918427
  ovha_no_spo train_loss_final=0.15750817954540253
  ovha_no_rceo train_loss_final=0.13748769462108612
  ovha_no_evidence_router train_loss_final=0.14668317139148712

seed 302:
  text_only train_loss_final=0.2923172116279602
  audio_only train_loss_final=0.5105862617492676
  vision_only train_loss_final=0.48137685656547546
  concat_fusion train_loss_final=0.287581205368042
  ovha_no_lrio train_loss_final=0.15916521847248077
  ovha_no_spo train_loss_final=0.1334303915500641
  ovha_no_rceo train_loss_final=0.15406924486160278
  ovha_no_evidence_router train_loss_final=0.15034069120883942
```

观察到的现象：

- OVHA 内部消融训练 loss 明显低于 text/concat 线性 probe。
- OVHA full 和大部分 OVHA 消融的 test heldout loss 明显高于 text/concat。
- `ovha_no_lrio` 明显优于 `ovha_full`，但仍劣于 text/concat。

## 已运行实验 2：candidate_individual_loss 权重为 0 的 1 seed 3000 steps 诊断

路径：

- `outputs/multimodal/cmu_mosei_probe_candw0_1seed_3000/raw_metrics.jsonl`
- `outputs/multimodal/cmu_mosei_probe_candw0_1seed_3000/robustness_rows.jsonl`
- `outputs/multimodal/cmu_mosei_probe_candw0_1seed_3000_train.log`

设置：

```text
seed: 301
train_steps: 3000
baseline_train_steps: 3000
d_model: 128
memory_tokens: 8
learning_rate: 1e-4
eval_split: val
device: cuda
loss_metadata:
  task_loss.weight = 1.0
  candidate_individual_loss.weight = 0.0
models:
  ovha_full
  text_only
  concat_fusion
  ovha_no_lrio
```

Raw metrics，`heldout_loss` 越低越好：

```text
ovha_full:
  heldout_loss=0.5335292816162109
  mae=0.53013014793396
  params=386613
  router clean:
    CATO=0.34274561533446235
    LRIO=0.5285818320827044
    SPO=0.07813771435696525
    TLEO=0.05053483822586798

text_only:
  heldout_loss=0.3281627595424652
  mae=0.4421261250972748
  params=301

concat_fusion:
  heldout_loss=0.37303876876831055
  mae=0.4554082751274109
  params=410

ovha_no_lrio:
  heldout_loss=0.37431105971336365
  mae=0.4714677929878235
  params=386613
  router clean:
    CATO=0.10578136969939894
    LRIO=0.0
    SPO=0.19614863541833039
    TLEO=0.6980699948822706
```

Seed report：

```text
seed 301:
  ovha_parameter_l2_delta=20.603116989135742
  text_only train_loss_final=0.3506251275539398
  concat_fusion train_loss_final=0.38271021842956543
  ovha_no_lrio train_loss_final=0.14195042848587036
```

Clean candidate/router rows from robustness artifact:

```text
model=ovha_full
robust_score=0.652090580850262
router:
  CATO=0.34274561533446235
  LRIO=0.5285818320827044
  SPO=0.07813771435696525
  TLEO=0.05053483822586798
candidate_loss:
  CATO=7.0574235916137695
  LRIO=4.3905839920043945
  SPO=109.87661743164062
  TLEO=545.6832275390625

model=ovha_no_lrio
robust_score=0.7276373081131773
router:
  CATO=0.10578136969939894
  LRIO=0.0
  SPO=0.19614863541833039
  TLEO=0.6980699948822706
candidate_loss:
  CATO=333.5632019042969
  LRIO=1.5699927806854248
  SPO=180.05499267578125
  TLEO=2.496526002883911
```

观察到的现象：

- 将 `candidate_individual_loss.weight` 降到 0 后，`ovha_full` 在 val 上仍明显差于 `text_only` 和 `concat_fusion`。
- `ovha_no_lrio` 接近 `concat_fusion`，但仍差于 `text_only`。
- `ovha_full` 的 router 在 clean val 上给 LRIO 较高权重。
- `candidate_loss` 在不同模型之间分布差异很大，部分 candidate loss 达到非常大的数值。

## 已知修复和待确认点

已修复：

- 内部消融不再全部走线性 probe。
- 消融模型的有效 router 权重已写回 `router_load_by_candidate`。
- public loss 组件支持通过 `loss_metadata.<loss>.weight` 配置权重。

这些修复只说明部分协议/诊断问题已经被改动，并不说明当前模型效果已经可信。修复后最新 probe 仍显示 `ovha_full` 在 clean CMU-MOSEI heldout loss 上明显差于 `text_only` / `concat_fusion`。

仍需审查：

- `candidate_loss` 是否应该跨不同模型直接比较。
- `robust_score` 与 `heldout_task_loss` 的定义是否容易混淆。
- `candidate_individual_loss.weight=0` 时，`candidate_loss` 的诊断值是否仍有解释意义。
- `candidate_values` 的输出尺度是否可能失控。
- `router_load_by_candidate` 是否应基于有效权重还是 raw learned router，两者是否都应在 artifact 中写出。
- `run_public_main.py` 是否只在最终 step 评估，是否缺少 validation early stopping 或 best checkpoint。
- `MultimodalOVHA` 在 CMU-MOSEI 这种低维回归标签任务上是否存在结构不匹配。
- 当前 CMU-MOSEI 的 text feature 是否已经包含足够强的语义信息，导致 text-only 线性头成为强 baseline。

## 请求外部审查者重点回答

请严格审查并回答：

1. 是否存在训练目标与评估指标不一致的问题。
2. 是否存在 split、label、feature 对齐问题。
3. 是否存在 baseline 与 OVHA 使用的输入特征不一致或不公平问题。
4. 是否存在 OVHA candidate 输出尺度过大或 router 被错误候选吸引的问题。
5. 是否存在 public main runner 只记录最终模型、缺少 validation selection 导致的协议问题。
6. 是否存在代码中把 robustness score、heldout loss、MAE 混用的问题。
7. 对当前结果，最可能的实现 bug 排查优先级是什么。
8. 在不美化结果的前提下，当前结果是否足以说明该 CMU-MOSEI 路径存在关键问题。

## 不能作为正式结论的内容

以下运行均为 probe/诊断运行，不应作为论文主表结果：

- 2 seed 6000 steps 运行。
- 1 seed 3000 steps 运行。
- 使用 wrapper 绕过正式 5 seed public main 校验的运行。
- 使用 `candidate_individual_loss.weight=0.0` 的临时配置运行。

正式主表至少需要：

- 5 seed。
- 固定公开配置。
- 使用 validation 选择超参或 checkpoint。
- test split 只用于最终一次报告。
- 与外部 SOTA/reference reproduction 分开报告。
