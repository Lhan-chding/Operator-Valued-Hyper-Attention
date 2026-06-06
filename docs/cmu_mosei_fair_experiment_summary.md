# CMU-MOSEI Fair Experiment Summary

Last updated: 2026-06-06

This note collects the current CMU-MOSEI comparison rows that were recomputed with the project standard metric script. Values are raw metric values, not percentages. MAE and MSE are lower-is-better; all other metrics are higher-is-better.

## Protocol Boundary

The main comparison rows below are test-split, 5-seed summaries unless a section explicitly marks a row as a pilot. The metric protocol is the project MOSEI standard recompute path, using the same metric definitions as `scripts/multimodal/recompute_mosei_standard_metrics.py`.

- `acc2_excl0` / `f1_excl0`: binary sentiment after excluding exact-zero labels.
- `acc2_nonneg` / `f1_nonneg`: binary sentiment with non-negative labels treated as positive.
- `acc5` / `acc7`: clipped rounded class accuracy on the MOSEI sentiment scale.
- `pearson_correlation`: Pearson correlation over regression predictions and truths.

## Main Comparison

| Model | Evidence label | MSE ↓ | MAE ↓ | Corr ↑ | Acc7 ↑ | Acc5 ↑ | Acc2 excl0 ↑ | F1 excl0 ↑ | Acc2 nonneg ↑ | F1 nonneg ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OVHA/TANSO | latest project result, `configs/multimodal_cmu_mosei_tanso_primary_main.json` | 0.671856 | 0.6152 | 0.6785 | 0.4931 | 0.5021 | 0.8045 | 0.8004 | - | - |
| MULT | external MULT reproduction, standard metric recompute | - | 0.6029 ± 0.0070 | 0.6306 ± 0.0071 | 0.5139 | 0.5281 | 0.7967 | 0.7926 | 0.7922 | 0.7936 |
| Self-MM fixed-order | same OVHA cached features, non-BERT, official Self-MM repo, `M` head, standard metric recompute | 0.717077 ± 0.006987 | 0.639256 ± 0.004770 | 0.647211 ± 0.003942 | 0.479494 ± 0.004788 | 0.485929 ± 0.005763 | 0.806269 ± 0.002850 | 0.803905 ± 0.003100 | 0.784127 ± 0.004982 | 0.787582 ± 0.003843 |
| Self-MM official fixed-order | official Self-MM `unaligned_50.pkl` split/features, non-BERT, fixed valid/test order | - | 0.570594 ± 0.003959 | 0.724363 ± 0.004160 | 0.517321 ± 0.002730 | 0.528654 ± 0.003278 | 0.843368 ± 0.004046 | 0.842280 ± 0.003832 | 0.814853 ± 0.011680 | 0.818514 ± 0.009758 |

## Self-MM Fixed-Order Details

The Self-MM fixed-order row is not the official Self-MM BERT/raw-text setting. It is a same-cache external reproduction designed to compare the Self-MM architecture against this project's cached CMU-MOSEI inputs:

- Same split sizes: train 16,327; validation 1,871; test 4,662.
- Same cached feature tensors: text `(N, 64, 300)`, audio `(N, 128, 74)`, vision `(N, 128, 35)`.
- Same task labels as `data/multimodal_cache/cmu_mosei/v0.1/supervision/task_labels_{train,val,test}.npy`.
- BERT disabled; raw text is not used.
- The multimodal `M` head is used for final test prediction artifacts; auxiliary `T`, `A`, and `V` heads are not used for the main comparison row.
- The official Self-MM loader was patched so only the train split shuffles: `shuffle=(ds == 'train')`. The original external reproduction used `shuffle=True` for train, valid, and test, which made saved test prediction/truth arrays order-incompatible with the project cache.

Self-MM prediction artifacts were converted from object `.npy` outputs to numeric `M`-head `.npy` outputs before recomputation, then evaluated with the project standard script. The 5 seeds were `1111`, `1112`, `1113`, `1114`, and `1115`.

Fixed-order verification against the project cache passed for all five truth files:

```text
selfmm_mosei_seed1111_test_truths.npy shape (4662,) allclose True corr 1.0 mismatch 0
selfmm_mosei_seed1112_test_truths.npy shape (4662,) allclose True corr 1.0 mismatch 0
selfmm_mosei_seed1113_test_truths.npy shape (4662,) allclose True corr 1.0 mismatch 0
selfmm_mosei_seed1114_test_truths.npy shape (4662,) allclose True corr 1.0 mismatch 0
selfmm_mosei_seed1115_test_truths.npy shape (4662,) allclose True corr 1.0 mismatch 0
```

## Evidence Artifacts

Remote result paths on `qtech800`:

- OVHA/TANSO: `~/work/Operator-Valued-Hyper-Attention/outputs/multimodal/cmu_mosei_tanso_primary_gpu4_seed_301_rerun1` and `~/work/Operator-Valued-Hyper-Attention/outputs/multimodal/cmu_mosei_tanso_primary_gpu4_seed_{302..305}_rerun2`
- MULT: `~/work/Operator-Valued-Hyper-Attention/outputs/multimodal/external_mult_recompute/mult_mosei_standard_metrics.jsonl`
- Self-MM fixed-order summary: `~/work/Operator-Valued-Hyper-Attention/outputs/multimodal/external_selfmm_fixed_order_recompute_20260606_020955/selfmm_mosei_standard_metrics.jsonl`
- Self-MM fixed-order numeric `M` artifacts: `/home/david/work/external_repros/self_mm_cmu_mosei/results/results/normals/predictions_m_numeric_fixed_order_20260606_020955`

Self-MM fixed-order per-seed unified recompute values:

| Seed | MSE ↓ | MAE ↓ | Corr ↑ | Acc7 ↑ | Acc5 ↑ | Acc2 excl0 ↑ | F1 excl0 ↑ | Acc2 nonneg ↑ | F1 nonneg ↑ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1111 | 0.716397 | 0.640839 | 0.647470 | 0.474474 | 0.479622 | 0.808084 | 0.805959 | 0.785929 | 0.789522 |
| 1112 | 0.718538 | 0.636354 | 0.645006 | 0.487344 | 0.493994 | 0.802310 | 0.801291 | 0.774775 | 0.780195 |
| 1113 | 0.716505 | 0.640923 | 0.649018 | 0.476834 | 0.481553 | 0.810283 | 0.807473 | 0.788503 | 0.791319 |
| 1114 | 0.705992 | 0.632011 | 0.653174 | 0.482625 | 0.491634 | 0.806709 | 0.805541 | 0.783569 | 0.788221 |
| 1115 | 0.727954 | 0.646154 | 0.641386 | 0.476190 | 0.482840 | 0.803959 | 0.799259 | 0.787859 | 0.788653 |

## Reading

The current fair comparison separates same-cache external reproductions from official-feature reproductions.

- MULT remains strongest on MAE, Acc5, Acc7, and the nonnegative binary metrics.
- OVHA/TANSO beats fixed-order Self-MM on MSE, MAE, Pearson correlation, Acc5, and Acc7.
- Fixed-order Self-MM is slightly ahead of OVHA/TANSO on exclude-zero binary Acc2/F1, but the margin is small.
- The prior Self-MM object/numeric artifacts with shuffled test truths should not be used for order-sensitive comparison. The fixed-order row above supersedes that same-cache Self-MM comparison.

This supports a fair same-cache comparison claim: under the shared CMU-MOSEI cache and shared metric recompute protocol, OVHA/TANSO is stronger than Self-MM fixed-order on the main regression and multiclass sentiment metrics, while Self-MM is marginally stronger on exclude-zero binary classification. The next separate experiment is an official Self-MM BERT/raw-text feature run, which must be reported as a different evidence row rather than merged into the same-cache fixed-order row.

## Official Self-MM Split/Feature Baseline

This row records the separate official Self-MM split/feature run. It must not be merged with the same-cache comparison above because the official Self-MM test split has 4,659 samples, while the project cache comparison uses 4,662 test samples.

Remote official feature source:

- `/home/david/work/external_repros/self_mm_cmu_mosei/data/MOSEI/Processed/unaligned_50.pkl`

Remote numeric artifacts:

- `/home/david/work/external_repros/self_mm_cmu_mosei/results/results/normals/predictions_numeric_fixed_order_official`

Remote unified metric records:

- `/home/david/work/Operator-Valued-Hyper-Attention/outputs/multimodal/cmu_mosei_official_split/self_mm_fixed_order/metrics.jsonl`
- `/home/david/work/Operator-Valued-Hyper-Attention/outputs/multimodal/cmu_mosei_official_split/self_mm_fixed_order/summary.csv`
- `/home/david/work/Operator-Valued-Hyper-Attention/outputs/multimodal/cmu_mosei_official_split/self_mm_fixed_order/summary.json`

The official split truth arrays were verified after object-to-numeric conversion:

```text
selfmm_mosei_seed1111_test_truths.npy shape (4659,) allclose True mismatch_n 0
selfmm_mosei_seed1112_test_truths.npy shape (4659,) allclose True mismatch_n 0
selfmm_mosei_seed1113_test_truths.npy shape (4659,) allclose True mismatch_n 0
selfmm_mosei_seed1114_test_truths.npy shape (4659,) allclose True mismatch_n 0
selfmm_mosei_seed1115_test_truths.npy shape (4659,) allclose True mismatch_n 0
```

Official Self-MM fixed-order 5-seed summary:

| Metric | Mean | Std |
|---|---:|---:|
| MAE | 0.570594 | 0.003959 |
| Corr | 0.724363 | 0.004160 |
| Acc7 | 0.517321 | 0.002730 |
| Acc5 | 0.528654 | 0.003278 |
| Acc2 excl0 | 0.843368 | 0.004046 |
| F1 excl0 | 0.842280 | 0.003832 |
| Acc2 nonneg | 0.814853 | 0.011680 |
| F1 nonneg | 0.818514 | 0.009758 |

For the next OVHA official-split run, write the OVHA records under a sibling remote directory, for example:

```text
/home/david/work/Operator-Valued-Hyper-Attention/outputs/multimodal/cmu_mosei_official_split/ovha_fixed_order/
```

Then compare against the Self-MM official-split summary above, not against the 4,662-sample same-cache row.

## OVHA Official Self-MM Split Pilot

This section records the quick sanity run of the project model on the official Self-MM `unaligned_50.pkl` split/features. It is not a final 5-seed result. Treat it as a pilot for deciding whether to launch the full 50k-step, 5-seed run.

Run setup:

- Date: 2026-06-06
- Remote repo: `/home/david/work/Operator-Valued-Hyper-Attention`
- Config: `configs/multimodal_cmu_mosei_tanso_primary_selfmm_official.json`
- Cache version: `v0.1_selfmm_official`
- Artifact root: `outputs/multimodal/cmu_mosei_official_split/ovha_tanso_primary_pilot_seed301`
- Raw metrics: `outputs/multimodal/cmu_mosei_official_split/ovha_tanso_primary_pilot_seed301/raw_metrics.jsonl`
- Train split: `train`; selection split: `val`; eval split: `test`
- Seed: `301`
- Train steps: `3500`
- Best OVHA checkpoint: step `800`
- Best validation task loss: `0.412021`

Pilot test metrics versus the Self-MM official fixed-order 5-seed mean:

| Model | MAE ↓ | Δ MAE vs Self-MM ↑ | Corr ↑ | Δ Corr | Acc7 ↑ | Δ Acc7 | Acc5 ↑ | Δ Acc5 | Acc2 excl0 ↑ | Δ Acc2 excl0 | F1 excl0 ↑ | Δ F1 excl0 | Acc2 nonneg ↑ | Δ Acc2 nonneg | F1 nonneg ↑ | Δ F1 nonneg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ovha_tanso_primary | 0.578809 | -0.008215 | 0.722255 | -0.002109 | 0.511912 | -0.005409 | 0.523503 | -0.005151 | 0.843148 | -0.000220 | 0.842017 | -0.000263 | 0.813479 | -0.001374 | 0.817197 | -0.001317 |
| ovha_lrio_tanso | 0.562505 | +0.008089 | 0.738477 | +0.014114 | 0.525220 | +0.007899 | 0.537669 | +0.009015 | 0.847826 | +0.004458 | 0.846369 | +0.004089 | 0.814338 | -0.000515 | 0.817862 | -0.000652 |
| ovha_all_candidates_exploratory | 0.568420 | +0.002174 | 0.730396 | +0.006033 | 0.516849 | -0.000472 | 0.530371 | +0.001717 | 0.835993 | -0.007375 | 0.834439 | -0.007841 | 0.805967 | -0.008886 | 0.809603 | -0.008911 |
| ovha_no_rceo | 0.562654 | +0.007940 | 0.736323 | +0.011959 | 0.517064 | -0.000258 | 0.530371 | +0.001717 | 0.839296 | -0.004073 | 0.837926 | -0.004353 | 0.811548 | -0.003305 | 0.815019 | -0.003496 |
| ovha_with_evidence_router | 0.553932 | +0.016662 | 0.745212 | +0.020849 | 0.531445 | +0.014123 | 0.545396 | +0.016742 | 0.845625 | +0.002256 | 0.843995 | +0.001715 | 0.819918 | +0.005065 | 0.822679 | +0.004164 |

Pilot reading:

- `ovha_with_evidence_router` is the strongest 3500-step pilot row. It beats the Self-MM official 5-seed mean on every listed metric.
- `ovha_lrio_tanso` is also strong on MAE, correlation, Acc7, Acc5, and exclude-zero binary metrics, but is slightly below Self-MM on the nonnegative binary metrics.
- `ovha_tanso_primary` underperforms Self-MM at 3500 steps, so the evidence-router variant is the better candidate to prioritize for the next official-split run.
- Because this is a single seed and only 3500 steps, the result is a launch signal, not a final paper-table comparison. The next comparable run should use the same official split/features with 50k steps and seeds `301` to `305`.
