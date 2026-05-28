# OVHA Phase 1.7 GPU3 Results for GPT Pro

Date: 2026-05-28

Branch: `phase1_5_metadata_free_torch`

Relevant pushed commits:

- `e17315d test: add phase1.7 expressivity reproducers`
- `af40989 fix: implement phase1.7 controlled v2 gates`

Machine/run context:

- Host: `qtech800`
- OS: Ubuntu 22.04.3 LTS
- Device: CUDA / GPU 3
- Python env: `.venv`
- Main output roots:
  - `outputs/phase1_7/controlled_v2_sanity_gpu3_parallel/`
  - `outputs/phase1_7/g1_single_iid_spectral/`
  - `outputs/phase1_7/g1_single_iid_local/`
  - `outputs/phase1_7/g1_single_iid_separable/`
  - `outputs/phase1_7/g2_g3_component_iid/`

## Code/Protocol Changes Already Applied

This run used the patched Phase 1.7 protocol:

- `model_aligned` iid generator was constrained to the short-term adapter sanity domain:
  - `gain in [0.9, 1.1]`
  - `local lengthscale in [0.08, 0.20]`
- Separable model primitive basis was corrected to match generator basis:
  - fourth basis is `cos(2*pi*s) * cos(2*pi*q)`
- Hidden diagnostics now preserve `true_operator_tensors` for non-public oracle analysis.
- Added model-primitive true-param oracle:
  - `model_primitive_true_param_relative_l2`
  - `model_router_true_param_relative_l2`
- ContextTokenEncoder now includes primitive-aligned candidate features:
  - spectral candidates
  - local fixed-lengthscale candidates
  - separable branch/trunk candidates
  - residuals against observed `context_y`
- Training supports router auxiliary CE:
  - `router_auxiliary_loss_weight = 0.05`
- Summary now reports split-wise rows and `matched_single`, not only family means.

Local validation before GPU handoff:

- `.venv/bin/python -m unittest tests.test_phase1_7_controlled_v2_protocol`
- `.venv/bin/python -m unittest discover tests`
- All passed locally before GPU runs.

## High-Level Result

Current conclusion:

```text
Controlled-v2 is still No-Go for larger/public benchmark experiments.
The failure is no longer a generator/primitive expressivity bug and is not primarily router choosing the wrong primitive.
The narrowed failure is that OVHA-full's shared memory/context/hyper-adapter path does not consistently match the corresponding single-primitive specialist, especially spectral and separable.
```

Do not start PDEBench/FNO/Mechanical/OpenFWI or larger public benchmark runs yet.

## Mixed 5k Sanity Result

Output:

```text
outputs/phase1_7/controlled_v2_sanity_gpu3_parallel/phase1_6_summary.json
outputs/phase1_7/controlled_v2_sanity_gpu3_parallel/phase1_6_report.md
```

Go/No-Go:

```text
Scientific No-Go: checkpoint path is wired, but OVHA-full does not beat controlled-stress baselines/ablations; diagnose model/config before running the larger main experiment.
```

Controlled stress summary:

| family | split | ovha_full | matched_single | best_single | vector_big | no_memory | no_router | no_adapter | conclusion |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| context_identifiable_mixture | iid | 0.319561 |  | 0.469395 | 0.569054 | 0.709580 | 0.335274 | 0.612273 | provisional component signal |
| query_piecewise_router | iid | 0.450427 |  | 0.692758 | 0.913292 | 0.729630 | 0.597394 | 0.680270 | provisional component signal |
| single_primitive_local | iid | 0.173271 | 0.150689 | 0.150689 | 0.509049 | 0.482056 | 0.146589 | 0.484784 | negative component signal |
| single_primitive_separable | iid | 0.509044 | 0.514382 | 0.514382 | 3.102343 | 0.951960 | 0.433677 | 0.934243 | mixed; ablation signal missing |
| single_primitive_spectral | iid | 0.504026 | 0.558796 | 0.558796 | 1.397950 | 0.967154 | 0.517069 | 0.894557 | provisional component signal |

Interpretation:

- Mixed 5k shows real positive signal on router and context/memory families.
- The scientific No-Go is triggered by single-primitive degeneration and ablation semantics, especially local/separable in the mixed setting.
- Mixed 5k alone is not enough to identify whether this is expressivity, router, adapter, or training-budget interference, so G1/G2/G3 runs were inspected.

## G1 Single-Primitive Local

Output:

```text
outputs/phase1_7/g1_single_iid_local/phase1_6_report.md
```

Summary:

| model | eval_relL2 |
|---|---:|
| ovha_full | 0.063441 |
| local_only / matched_single | 0.064251 |
| ovha_no_query_router | 0.063510 |
| ovha_no_hyper_adapter | 0.476825 |

Diagnostics:

| model | primitive_load |
|---|---|
| ovha_full | local=0.999997, separable=0.000001, spectral=0.000002 |
| ovha_no_query_router | local=0.999996, separable=0.000002, spectral=0.000002 |
| local_only | local=1.000000 |

Conclusion:

```text
G1 local passes.
Full OVHA can safely collapse to local when trained only on the local family.
Hyper-adapter is necessary, because no_hyper_adapter is much worse.
```

The mixed sanity local failure is therefore not a local primitive formula bug and not a local router-collapse bug.

## G1 Single-Primitive Spectral

Output:

```text
outputs/phase1_7/g1_single_iid_spectral/phase1_6_report.md
```

Summary:

| model | eval_relL2 |
|---|---:|
| ovha_full | 0.710492 |
| spectral_only / matched_single | 0.424686 |
| ovha_no_query_router | 0.707596 |
| ovha_no_hyper_adapter | 0.894245 |

Diagnostics:

| model | primitive_load |
|---|---|
| ovha_full | spectral=0.999999, local=0.000001, separable=0.000001 |
| ovha_no_query_router | spectral=0.999997, local=0.000001, separable=0.000001 |
| spectral_only | spectral=1.000000 |

Conclusion:

```text
G1 spectral fails.
Router collapse is correct, but OVHA-full still underperforms the spectral specialist by a large margin.
```

This points away from router selection and toward shared representation / hyper-adapter quality.

## G1 Single-Primitive Separable

Output:

```text
outputs/phase1_7/g1_single_iid_separable/phase1_6_report.md
```

Summary:

| model | eval_relL2 |
|---|---:|
| ovha_full | 0.504667 |
| separable_only / matched_single | 0.463253 |
| ovha_no_query_router | 0.454290 |
| ovha_no_hyper_adapter | 0.940825 |

Diagnostics:

| model | primitive_load |
|---|---|
| ovha_full | separable=0.999994, local=0.000004, spectral=0.000002 |
| ovha_no_query_router | separable=0.999998, local=0.000001, spectral=0.000001 |
| separable_only | separable=1.000000 |

Conclusion:

```text
G1 separable fails.
Router collapse is correct, but full remains worse than separable-only and no_query_router.
```

Since separable basis was already corrected and the router is one-hot, the remaining issue is likely adapter/shared-memory training quality, not primitive/generator mismatch.

## G2/G3 Router and Context/Memory IID

Output:

```text
outputs/phase1_7/g2_g3_component_iid/phase1_6_report.md
```

Controlled stress summary:

| family | split | ovha_full | vector_big | no_memory | no_router | conclusion |
|---|---|---:|---:|---:|---:|---|
| context_identifiable_mixture | iid | 0.324738 | 0.557459 | 0.705001 | 0.304199 | mixed; ablation signal missing |
| query_piecewise_router | iid | 0.461791 | 0.895895 | 0.705142 | 0.593079 | provisional component signal |

Diagnostics:

| model | family | entropy | memory_norm | primitive_load |
|---|---|---:|---:|---|
| ovha_full | context_identifiable_mixture | 1.021727 | 20.934123 | local=0.341354, separable=0.314921, spectral=0.343725 |
| ovha_no_query_router | context_identifiable_mixture | 1.015987 | 8.362325 | local=0.337297, separable=0.335738, spectral=0.326965 |
| ovha_full | query_piecewise_router | 0.353655 | 8.419676 | local=0.343276, separable=0.289908, spectral=0.366816 |
| ovha_no_query_router | query_piecewise_router | 1.094649 | 3.865872 | local=0.314017, separable=0.342231, spectral=0.343752 |

Conclusion:

```text
G2 query-piecewise router has a clear positive signal.
G3 context-identifiable mixture is mixed because no_query_router is slightly better than full.
```

This suggests that query-conditioned router helps for query-piecewise routing, but the full stack still has optimization/interference issues in context mixture.

## Oracle/Expressivity Notes

The added true-param oracle diagnostics indicate that matching primitives can reproduce `model_aligned` targets with near-zero true-param oracle error.

Important observed pattern:

```text
primitive_load is already essentially one-hot for single-primitive G1 tasks:
- local:      full local load      ~= 0.999997
- spectral:   full spectral load   ~= 0.999999
- separable:  full separable load  ~= 0.999994
```

Therefore the next failure is not:

- fixed-batch sampling
- checkpoint wiring
- separable basis mismatch
- local lengthscale/gain range mismatch
- router choosing the wrong primitive

The narrowed issue is:

```text
OVHA-full's shared memory/context/hyper-adapter path does not consistently match the corresponding single-specialist adapter path.
```

## Current Diagnosis

The current evidence supports the following diagnosis:

1. Controlled-v2 expressivity is mostly fixed.
2. Router collapse is working on single-primitive tasks.
3. Full OVHA has real signal on query routing and context/memory tasks.
4. Full OVHA still underperforms specialists on spectral and separable even when the router selects the correct primitive.
5. The likely bottleneck is hyper-adapter/shared-memory training interference, not primitive formula mismatch.

## Recommended Next Patch

Do not run public benchmarks or larger main experiments yet.

Recommended code changes for the next iteration:

1. Add adapter-level diagnostics:
   - predicted scale/bias mean/std by primitive and family
   - spectral mode logits entropy
   - spectral mode logits CE/KL against true logits on controlled-v2
   - separable rank logits entropy
   - separable rank logits CE/KL against true logits on controlled-v2
   - learned-adapter vs true-param oracle gap by primitive

2. Add controlled-v2 adapter supervision:
   - for synthetic controlled-v2 only, use hidden true params to supervise relevant adapter outputs
   - start with spectral mode logits and separable rank logits
   - keep this out of public benchmark model inputs

3. Add a single-primitive specialist-collapse gate:
   - on `single_primitive_spectral`, full should match `spectral_only`
   - on `single_primitive_local`, full should match `local_only`
   - on `single_primitive_separable`, full should match `separable_only`
   - gate should check both relative L2 and adapter oracle gap

4. Consider staged training:
   - Stage A: train primitive adapters with oracle router / strong router CE
   - Stage B: unfreeze normal router
   - Stage C: mixed family training

5. Optional after diagnostics:
   - increase `router_auxiliary_loss_weight` only if router CE/load is still poor
   - current single-primitive loads show router is already almost perfectly one-hot, so stronger router CE alone is unlikely to fix spectral/separable.

## Request for GPT Pro

Please review the above results and advise the next minimal patch.

The key question is:

```text
Given one-hot router collapse and near-zero true-param expressivity oracle, why does OVHA-full underperform spectral_only and separable_only on isolated G1 single-primitive tasks?
```

Most likely next action seems to be adapter-level supervision/diagnostics or staged adapter training, rather than more GPU scale.

