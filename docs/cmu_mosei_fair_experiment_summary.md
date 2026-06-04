# CMU-MOSEI Fair Experiment Summary

Last updated: 2026-06-04

This note collects the current CMU-MOSEI comparison rows that were recomputed with the project standard metric script. Values are raw metric values, not percentages. MAE is lower-is-better; all other metrics are higher-is-better.

## Protocol Boundary

All rows below are test-split, 5-seed summaries. The metric protocol is the project MOSEI standard recompute path, using the same metric definitions as `scripts/multimodal/recompute_mosei_standard_metrics.py`.

- `acc2_excl0` / `f1_excl0`: binary sentiment after excluding exact-zero labels.
- `acc2_nonneg` / `f1_nonneg`: binary sentiment with non-negative labels treated as positive.
- `acc5` / `acc7`: clipped rounded class accuracy on the MOSEI sentiment scale.
- `pearson_correlation`: Pearson correlation over regression predictions and truths.

## Main Comparison

| Model | Evidence label | MAE ↓ | Corr ↑ | Acc7 ↑ | Acc5 ↑ | Acc2 excl0 ↑ | F1 excl0 ↑ | Acc2 nonneg ↑ | F1 nonneg ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OVHA/TANSO | latest project result, `configs/multimodal_cmu_mosei_tanso_public_main.json` | 0.6372 ± 0.0051 | 0.6603 ± 0.0062 | 0.4765 | 0.4837 | 0.8045 | 0.8023 | 0.7821 | 0.7858 |
| MULT | external MULT reproduction, standard metric recompute | 0.6029 ± 0.0070 | 0.6306 ± 0.0071 | 0.5139 | 0.5281 | 0.7967 | 0.7926 | 0.7922 | 0.7936 |
| Self-MM adapted | same OVHA cached features, non-BERT, masked text pooling, standard metric recompute | 0.6403 ± 0.0048 | 0.6439 ± 0.0052 | 0.4789 ± 0.0032 | 0.4851 ± 0.0041 | 0.8048 ± 0.0031 | 0.8024 ± 0.0037 | 0.7839 ± 0.0071 | 0.7872 ± 0.0057 |

## Self-MM Adaptation Details

The Self-MM row is not the official Self-MM BERT/raw-text setting. It is an adapted fairness row designed to compare against this project's cached CMU-MOSEI inputs:

- Same split sizes: train 16,327; validation 1,871; test 4,662.
- Same cached feature tensors: text `(N, 64, 300)`, audio `(N, 128, 74)`, vision `(N, 128, 35)`.
- Same task labels as `data/multimodal_cache/cmu_mosei/v0.1/supervision/task_labels_{train,val,test}.npy`.
- BERT disabled; raw text is not used.
- Text sequence features are reduced with masked pooling for the adapted Self-MM text branch.
- The multimodal `M` head is used for final test prediction artifacts; auxiliary `T`, `A`, and `V` heads are not used for the main comparison row.

Self-MM prediction artifacts were converted from object `.npy` outputs to numeric `.npy` outputs before recomputation, then evaluated with the project standard script. The 5 seeds were `1111`, `1112`, `1113`, `1114`, and `1115`.

## Evidence Artifacts

Remote result paths on `qtech800`:

- OVHA/TANSO: `~/work/Operator-Valued-Hyper-Attention/outputs/multimodal/cmu_mosei_tanso_main/raw_metrics.jsonl`
- MULT: `~/work/Operator-Valued-Hyper-Attention/outputs/multimodal/external_mult_recompute/mult_mosei_standard_metrics.jsonl`
- Self-MM adapted: `~/work/Operator-Valued-Hyper-Attention/outputs/multimodal/external_selfmm_adapted_recompute/selfmm_mosei_standard_metrics.jsonl`

Self-MM adapted per-seed unified recompute values:

| Seed | MAE ↓ | Corr ↑ | Acc7 ↑ | Acc5 ↑ | Acc2 excl0 ↑ | F1 excl0 ↑ | Acc2 nonneg ↑ | F1 nonneg ↑ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1111 | 0.6444 | 0.6401 | 0.4828 | 0.4891 | 0.8070 | 0.8036 | 0.7949 | 0.7964 |
| 1112 | 0.6448 | 0.6376 | 0.4749 | 0.4792 | 0.8007 | 0.7976 | 0.7825 | 0.7852 |
| 1113 | 0.6406 | 0.6443 | 0.4775 | 0.4837 | 0.8062 | 0.8031 | 0.7887 | 0.7910 |
| 1114 | 0.6314 | 0.6529 | 0.4826 | 0.4903 | 0.8086 | 0.8082 | 0.7758 | 0.7820 |
| 1115 | 0.6404 | 0.6445 | 0.4768 | 0.4831 | 0.8018 | 0.7994 | 0.7776 | 0.7815 |

## Reading

The current fair comparison does not show a single model dominating every metric.

- MULT remains strongest on MAE, Acc5, Acc7, and the nonnegative binary metrics.
- OVHA/TANSO is strongest on Pearson correlation.
- Self-MM adapted is essentially tied with OVHA/TANSO on exclude-zero binary classification and close on Acc5/Acc7, while trailing OVHA/TANSO on correlation and trailing MULT on MAE.

This supports a fair-comparison claim: the project result is competitive under the shared MOSEI metric recompute protocol, but the strongest baseline depends on which metric family is prioritized.
