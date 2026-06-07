# CMU-MOSEI Fair Experiment Summary

Last updated: 2026-06-07

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
- Because this is a single seed and only 3500 steps, the result is a launch signal, not a final paper-table comparison. It is superseded by the 5-seed 4000-step official-split run below.

## OVHA Official Self-MM Split 5-Seed 4000-Step Run

This section records the 5-seed official-split run launched from the same official Self-MM split/features as the pilot, capped at 4000 optimizer steps per seed.

Run setup:

- Date: 2026-06-07
- Remote repo: `/home/david/work/Operator-Valued-Hyper-Attention`
- Config: `configs/multimodal_cmu_mosei_tanso_primary_selfmm_official.json`
- Artifact root: `outputs/multimodal/cmu_mosei_official_split/ovha_tanso_primary_5seed_4000`
- Raw metrics: `outputs/multimodal/cmu_mosei_official_split/ovha_tanso_primary_5seed_4000/raw_metrics.jsonl`
- Summary CSV: `outputs/multimodal/cmu_mosei_official_split/ovha_tanso_primary_5seed_4000/summary_vs_selfmm_5seed_4000.csv`
- Train split: `train`; selection split: `val`; eval split: `test`
- Seeds: `301`, `302`, `303`, `304`, `305`
- Train steps: `4000`

5-seed test metrics versus the Self-MM official fixed-order 5-seed mean:

| Model | MAE ↓ | Δ MAE vs Self-MM ↑ | Corr ↑ | Δ Corr | Acc7 ↑ | Δ Acc7 | Acc5 ↑ | Δ Acc5 | Acc2 excl0 ↑ | Δ Acc2 excl0 | F1 excl0 ↑ | Δ F1 excl0 | Acc2 nonneg ↑ | Δ Acc2 nonneg | F1 nonneg ↑ | Δ F1 nonneg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ovha_tanso_primary | 0.562936 ± 0.003907 | +0.007657 | 0.740043 ± 0.002928 | +0.015680 | 0.520369 ± 0.003489 | +0.003048 | 0.533419 ± 0.004085 | +0.004765 | 0.842653 ± 0.002090 | -0.000715 | 0.841717 ± 0.002345 | -0.000563 | 0.809101 ± 0.006159 | -0.005752 | 0.813377 ± 0.005070 | -0.005137 |
| ovha_with_evidence_router | 0.561342 ± 0.004430 | +0.009252 | 0.741131 ± 0.004057 | +0.016768 | 0.523417 ± 0.006274 | +0.006096 | 0.536510 ± 0.006669 | +0.007856 | 0.843919 ± 0.002144 | +0.000550 | 0.843007 ± 0.001976 | +0.000727 | 0.810346 ± 0.007118 | -0.004507 | 0.814595 ± 0.005761 | -0.003919 |
| spo_only | 0.572216 ± 0.005137 | -0.001622 | 0.725727 ± 0.004534 | +0.001364 | 0.515347 ± 0.004362 | -0.001975 | 0.526980 ± 0.004793 | -0.001674 | 0.840341 ± 0.004680 | -0.003027 | 0.839518 ± 0.004559 | -0.002762 | 0.805538 ± 0.009900 | -0.009315 | 0.810139 ± 0.008738 | -0.008375 |
| text_only | 0.634432 ± 0.001238 | -0.063838 | 0.679477 ± 0.001987 | -0.044886 | 0.462245 ± 0.002992 | -0.055076 | 0.467912 ± 0.003001 | -0.060743 | 0.822179 ± 0.002426 | -0.021189 | 0.823332 ± 0.002387 | -0.018948 | 0.778107 ± 0.003502 | -0.036746 | 0.786222 ± 0.003233 | -0.032293 |
| vision_only | 0.819725 ± 0.001169 | -0.249131 | 0.194937 ± 0.003755 | -0.529426 | 0.415840 ± 0.001346 | -0.101481 | 0.415840 ± 0.001346 | -0.112814 | 0.635883 ± 0.004161 | -0.207485 | 0.568694 ± 0.003056 | -0.273586 | 0.691865 ± 0.005105 | -0.122988 | 0.640105 ± 0.002323 | -0.178410 |

5-seed 4000-step reading:

- `ovha_with_evidence_router` is the strongest overall candidate from this run. It improves over Self-MM official fixed-order on MAE, Pearson correlation, Acc7, Acc5, and the exclude-zero binary metrics.
- The evidence-router row does not yet beat Self-MM on the nonnegative binary metrics: Acc2 nonneg is lower by `0.004507`, and F1 nonneg is lower by `0.003919`.
- `ovha_tanso_primary` also improves regression, correlation, Acc7, and Acc5, but trails Self-MM on all binary metrics.
- `spo_only` is roughly tied on correlation but below Self-MM on MAE and classification metrics, so the gain is not coming from the SPO path alone.
- `text_only` and `vision_only` are sanity baselines and are clearly below the multimodal rows.
- This supports a fair official-split claim that OVHA improves the main regression and multiclass sentiment metrics over the fixed-order Self-MM non-BERT baseline at 4000 steps, but it should not be phrased as a complete win across all MOSEI metrics.

## EMOE Fair Precomputed-BERT-Text Run

This section records an external EMOE training run on the same official Self-MM MOSEI pkl and the same precomputed 768-dimensional BERT text features used by the fair feature-matched comparison. This is the appropriate row for a fair training-metric comparison under the shared pkl feature protocol.

Run setup:

- Date: 2026-06-07
- Remote repo: `/home/david/work/external_repros/emoe_cmu_mosei`
- Upstream code target: `https://github.com/fuyyyyy/EMOE`
- Data source: `/home/david/work/external_repros/self_mm_cmu_mosei/data/MOSEI/Processed/unaligned_50.pkl`
- Result root: `result/emoe_mosei_5seed`
- Metrics: `result/emoe_mosei_5seed/metrics.jsonl`
- Log: `result/emoe_mosei_5seed/train_precomputed_bert_text.log`
- Prediction artifacts: `result/emoe_mosei_5seed/predictions_numeric/`
- Runtime environment: `/home/david/work/external_repros/mult_cmu_mosei/.venv-mult`
- Runtime versions observed: PyTorch `2.2.2+cu121`; Transformers `4.30.2`
- Seeds: `1111`, `1112`, `1113`, `1114`, `1115`
- Test samples: `4659`

The official pkl text fields were inspected before the run:

```text
train text      (16326, 50, 768) float32
train text_bert (16326, 3, 50) int64
valid text      (1871, 50, 768) float32
valid text_bert (1871, 3, 50) int64
test text       (4659, 50, 768) float32
test text_bert  (4659, 3, 50) int64
```

Feature protocol:

- `use_bert=false` and `use_finetune=false`, so the run reads `data[split]["text"]` directly.
- `feature_dims[0]=768` to match the precomputed BERT-text feature dimension.
- The pkl's `text` field already contains downloaded/precomputed 768-dimensional BERT text features, so the run does not regenerate text features online.
- The DataLoader was patched to shuffle only train, not valid/test, to keep saved test predictions order-stable.
- A local run script, `run_mosei_emoe_5seed.py`, was used because upstream `train.py` defaults to MOSI and does not save numeric prediction/truth artifacts.

Per-seed test metrics:

| Seed | MAE ↓ | Corr ↑ | Acc7 ↑ | Acc5 ↑ | Acc2 excl0 ↑ | F1 excl0 ↑ | Acc2 nonneg ↑ | F1 nonneg ↑ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1111 | 0.588121 | 0.710483 | 0.500966 | 0.514917 | 0.832691 | 0.832537 | 0.794376 | 0.800282 |
| 1112 | 0.581926 | 0.724158 | 0.512342 | 0.525435 | 0.813704 | 0.815840 | 0.748873 | 0.759815 |
| 1113 | 0.577864 | 0.717914 | 0.512771 | 0.528654 | 0.839571 | 0.837215 | 0.811977 | 0.814445 |
| 1114 | 0.579724 | 0.719961 | 0.511912 | 0.529298 | 0.843974 | 0.843938 | 0.800601 | 0.806583 |
| 1115 | 0.590436 | 0.711857 | 0.499893 | 0.513200 | 0.829939 | 0.827957 | 0.805108 | 0.808112 |

5-seed summary:

| Metric | Mean | Std |
|---|---:|---:|
| MAE | 0.583614 | 0.005429 |
| Corr | 0.716875 | 0.005694 |
| Acc7 | 0.507577 | 0.006543 |
| Acc5 | 0.522301 | 0.007689 |
| Acc2 excl0 | 0.831976 | 0.011621 |
| F1 excl0 | 0.831497 | 0.010562 |
| Acc2 nonneg | 0.792187 | 0.025051 |
| F1 nonneg | 0.797847 | 0.021849 |

Fair precomputed-BERT-text comparison table:

| Model | Role | MAE ↓ | Corr ↑ | Acc7 ↑ | Acc5 ↑ | Acc2 excl0 ↑ | F1 excl0 ↑ | Acc2 nonneg ↑ | F1 nonneg ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Self-MM official fixed-order | external baseline | 0.570594 ± 0.003959 | 0.724363 ± 0.004160 | 0.517321 ± 0.002730 | 0.528654 ± 0.003278 | 0.843368 ± 0.004046 | 0.842280 ± 0.003832 | 0.814853 ± 0.011680 | 0.818514 ± 0.009758 |
| EMOE fair precomputed-BERT-text | external SOTA architecture under shared feature protocol | 0.583614 ± 0.005429 | 0.716875 ± 0.005694 | 0.507577 ± 0.006543 | 0.522301 ± 0.007689 | 0.831976 ± 0.011621 | 0.831497 ± 0.010562 | 0.792187 ± 0.025051 | 0.797847 ± 0.021849 |
| OVHA/TANSO primary | ours | 0.562936 ± 0.003907 | 0.740043 ± 0.002928 | 0.520369 ± 0.003489 | 0.533419 ± 0.004085 | 0.842653 ± 0.002090 | 0.841717 ± 0.002345 | 0.809101 ± 0.006159 | 0.813377 ± 0.005070 |
| OVHA with evidence router | ours, strongest row | 0.561342 ± 0.004430 | 0.741131 ± 0.004057 | 0.523417 ± 0.006274 | 0.536510 ± 0.006669 | 0.843919 ± 0.002144 | 0.843007 ± 0.001976 | 0.810346 ± 0.007118 | 0.814595 ± 0.005761 |

Reading of the fair precomputed-BERT-text comparison:

- `OVHA with evidence router` is the best row on MAE, Pearson correlation, Acc7, Acc5, Acc2 excl0, and F1 excl0.
- `Self-MM official fixed-order` remains strongest on the nonnegative binary metrics.
- `EMOE fair precomputed-BERT-text` is below both OVHA rows and the Self-MM row under this shared feature protocol.

EMOE fair-run reading:

- This EMOE fair precomputed-BERT-text run is below both the Self-MM official fixed-order baseline and the OVHA 5-seed 4000-step rows on the main metrics.
- This is the fair comparison row to use when comparing models trained on the same official pkl, same 4,659-sample test split, same precomputed BERT text features, and stable test ordering.
- Paper-reported SOTA numbers that use different text-encoding or fine-tuning protocols should be kept in a separate reported-results table rather than mixed into this fair feature-matched table.
