# Final Experiment

Values are reported on the CMU-MOSEI test split. MAE and Corr are raw metric values; accuracy and F1 are shown as percentages in `excl. 0 / nonneg` form. `†` marks literature-reported baselines. Bold marks the best value in each column.

| Model | MAE ↓ | Corr ↑ | Acc-7 ↑ | Acc-2 ↑ | F1 ↑ |
|---|---:|---:|---:|---:|---:|
| TFN† | 0.573 | 0.714 | 51.60 | 78.50/81.89 | 78.96/81.74 |
| LMF† | 0.576 | 0.717 | 51.59 | 80.54/83.48 | 80.94/83.36 |
| MulT† | 0.559 | 0.733 | 52.84 | 81.15/84.63 | 81.56/84.52 |
| MISA† | 0.555 | 0.756 | 52.20 | 83.60/85.50 | 83.80/85.30 |
| Self-MM | 0.534 | 0.755 | 52.17 | 82.89/85.00 | 82.78/84.57 |
| FDMER† | 0.536 | 0.741 | **54.10** | -/86.10 | -/85.80 |
| TETFN | 0.551 | 0.748 | - | 84.12/85.18 | 84.18/85.27 |
| ALMT | 0.609 | **0.776** | - | 84.66/84.49 | **85.13**/85.16 |
| ConFEDE | 0.539 | 0.769 | 52.35 | 81.13/85.69 | 81.68/85.67 |
| SFTTR | 0.531 | **0.776** | 53.53 | 81.69/**86.16** | 82.25/**86.18** |
| FeaDA | 0.548 | 0.771 | 53.49 | 84.25/85.47 | 84.22/85.16 |
| OVHA / TANSOBase no-RCEO | **0.527** | 0.770 | 53.32 | **84.79**/85.12 | 84.94/83.55 |

## OVHA Variant CMU-Ablation

Internal OVHA ablation rows on the same CMU-MOSEI test split. Acc-5 is omitted. Bold marks the best value in each metric column.

| OVHA Variant | MAE ↓ | Corr ↑ | Acc-7 ↑ | Acc-2 ↑ | F1 ↑ |
|---|---:|---:|---:|---:|---:|
| OVHA no-RCEO / TANSOBase official | **0.527** | **0.770** | **53.32** | 84.79 | **84.94** |
| OVHA LRIO + TANSO | 0.5380 | 0.7422 | 52.70 | **84.80** | 84.70 |
| OVHA all-candidates exploratory | 0.5666 | 0.7332 | 52.07 | 84.22 | 84.11 |
| LRIO-only | 0.5707 | 0.7310 | 51.75 | 84.37 | 84.32|
| SPO-only | 0.5850 | 0.7168 | 50.62 | 83.08 | 83.15 |

## TANSO Component Ablation

5-seed summaries on the CMU-MOSEI test split. Acc-5 is omitted. Acc-2 uses `excl. 0 / nonneg` form. Bold marks the best value in each metric column.

| Model | MAE ↓ | Corr ↑ | Acc-7 ↑ | Acc-2 ↑ |
|---|---:|---:|---:|---:|
| raw_tanso_mlp | 0.6626 | 0.6093 | 50.64 | **83.76**/79.45 |
| ovha_tanso_no_source_gate | 0.5949 | 0.7108 | 52.08 | 79.63/**80.77** |
| ovha_tanso_no_operator_memory | 0.5927 | 0.7005 | 52.06 | 81.51/79.71 |
| ovha_tanso_no_gate_aux | **0.5842** | **0.7197** | **52.09** | 80.33/79.38 |

## OVHA RefCOCO Ablation (fixed 32 candidates + frozen CLIP features)
| Test | model | Acc@0.5 ↑ | R@1 | mIoU ↑ |
|---|---|---:|---:|---:|
| val | ovha_refcoco_prso_sro_tleo_cato_prim | 0.8101 | 0.8065 | 0.8398 |
| val | clip_geometry_mlp | 0.7959 | 0.7826 | 0.8150 |
| val | ovha_no_evidence_router | 0.7887 | 0.7746 | 0.8089 |
| val | ovha_no_rceo | 0.7872 | 0.7738 | 0.8080 |
| val | ovha_no_cato | 0.7859 | 0.7713 | 0.8054 |
| val | lightweight_transvg_style_reranker | 0.7652 | 0.7521 | 0.7859 |
| val | box_aware_cross_attention_reranker | 0.7639 | 0.7511 | 0.7849 |
| val | cato_only | 0.5096 | 0.4966 | 0.5423 |
| val | box_prior | 0.3220 | 0.3041 | 0.3721 |
| val | prso_clip_similarity | 0.3149 | 0.3046 | 0.3517 |
| val | train_slot_prior | 0.1709 | 0.1567 | 0.2266 |
| val | index_prior_only | 0.1709 | 0.1567 | 0.2266 |
| val | concat_fusion | 0.1708 | 0.1580 | 0.2185 |
| val | text_only | 0.1673 | 0.1519 | 0.2219 |
| val | random_valid | 0.1631 | 0.1541 | 0.1965 |
| val | region_only | 0.1623 | 0.1529 | 0.2116 |
| testA | ovha_refcoco_prso_sro_tleo_cato_prim | 0.8248 | 0.8170 | 0.8458 |
| testA | clip_geometry_mlp | 0.8073 | 0.7967 | 0.8265 |
| testA | ovha_no_rceo | 0.7928 | 0.7850 | 0.8151 |
| testA | ovha_no_evidence_router | 0.7893 | 0.7812 | 0.8116 |
| testA | ovha_no_cato | 0.7845 | 0.7762 | 0.8056 |
| testA | box_aware_cross_attention_reranker | 0.7656 | 0.7556 | 0.7885 |
| testA | lightweight_transvg_style_reranker | 0.7634 | 0.7535 | 0.7868 |
| testA | cato_only | 0.5632 | 0.5517 | 0.5937 |
| testA | box_prior | 0.3500 | 0.3390 | 0.3944 |
| testA | prso_clip_similarity | 0.2821 | 0.2728 | 0.3178 |
| testA | concat_fusion | 0.1389 | 0.1290 | 0.1815 |
| testA | train_slot_prior | 0.1379 | 0.1273 | 0.1923 |
| testA | index_prior_only | 0.1379 | 0.1273 | 0.1923 |
| testA | region_only | 0.1326 | 0.1253 | 0.1741 |
| testA | random_valid | 0.1266 | 0.1202 | 0.1609 |
| testA | text_only | 0.1264 | 0.1181 | 0.1807 |
| testB | ovha_refcoco_prso_sro_tleo_cato_prim | 0.7984 | 0.7809 | 0.8103 |
| testB | clip_geometry_mlp | 0.7837 | 0.7615 | 0.8015 |
| testB | ovha_no_rceo | 0.7790 | 0.7604 | 0.7888 |
| testB | ovha_no_evidence_router | 0.7715 | 0.7500 | 0.7913 |
| testB | ovha_no_cato | 0.7702 | 0.7486 | 0.7888 |
| testB | box_aware_cross_attention_reranker | 0.7689 | 0.7483 | 0.7880 |
| testB | lightweight_transvg_style_reranker | 0.7680 | 0.7466 | 0.7871 |
| testB | cato_only | 0.4406 | 0.4243 | 0.4771 |
| testB | prso_clip_similarity | 0.3415 | 0.3262 | 0.3779 |
| testB | box_prior | 0.2956 | 0.2785 | 0.3554 |
| testB | train_slot_prior | 0.2222 | 0.2029 | 0.2752 |
| testB | index_prior_only | 0.2222 | 0.2029 | 0.2752 |
| testB | random_valid | 0.2184 | 0.2045 | 0.2561 |
| testB | concat_fusion | 0.2167 | 0.1998 | 0.2711 |
| testB | text_only | 0.2139 | 0.1973 | 0.2756 |
| testB | region_only | 0.2112 | 0.1971 | 0.2685 |

## RefCOCO Fixed-Candidate Paired Significance

Paired bootstrap confidence intervals and paired permutation tests are reported for testA Acc@0.5. Positive deltas indicate that OVHA improves over the corresponding strong baseline.

| baseline | delta | 95% CI | paired p |
|---|---:|---:|---:|
| clip_geometry_mlp | 0.0175 | [0.0074, 0.0234] | 0.0032 |
| box_aware_cross_attention_reranker | 0.0592 | [0.0261, 0.0696] | 0.0015 |
| lightweight_transvg_style_reranker | 0.0614 | [0.0060, 0.0718] | 0.012 |

These results indicate statistically significant Acc@0.5 improvements over all three strong same-candidate reranker baselines on testA.


## RefCOCO External Same-Candidate Baselines

External baselines are projected onto the same fixed 32 RefCOCO candidates and scored with the same target-slot metrics.

| Test | model | protocol | Acc@0.5 ↑ | R@1 | mIoU ↑ |
|---|---|---|---:|---:|---:|
| val | CLIP crop similarity | crop each candidate, rank by CLIP text-image cosine | 0.3582 | 0.3473 | 0.3937 |
| testA | CLIP crop similarity | crop each candidate, rank by CLIP text-image cosine | 0.3534 | 0.3435 | 0.3920 |
| testB | CLIP crop similarity | crop each candidate, rank by CLIP text-image cosine | 0.3784 | 0.3662 | 0.4163 |
| val | GroundingDINO-SwinT | open-box predictions projected to candidates by score x IoU | 0.5486 | 0.5365 | 0.5771 |
| testA | GroundingDINO-SwinT | open-box predictions projected to candidates by score x IoU | 0.6173 | 0.6113 | 0.6388 |
| testB | GroundingDINO-SwinT | open-box predictions projected to candidates by score x IoU | 0.4758 | 0.4630 | 0.5143 |

## CLIP Crop Baseline Sanity Check

This checks that the CLIP crop baseline is not failing because of invalid crops or broken candidate targets.

| Test | samples | target valid | mean valid K | crop failures | hit@1 | hit@5 | target rank <=2 | target rank <=3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 10834 | 10834/10834 | 10.60 | 0 | 0.3473 | 0.8341 | 0.5711 | 0.6989 |
| testA | 5657 | 5657/5657 | 12.22 | 0 | 0.3435 | 0.8045 | 0.5300 | 0.6597 |
| testB | 5095 | 5095/5095 | 7.92 | 0 | 0.3662 | 0.8836 | 0.6202 | 0.7590 |

## RefCOCO Candidate-Slot Leakage Audit

To rule out candidate-position leakage, we audited the RefCOCO candidate construction by grouping samples according to the number of valid candidate regions \(K\) and measuring the empirical distribution of the ground-truth target slot within each \(K\)-bucket. We report slot-balance deviations only for buckets with at least \(20K\) samples; smaller buckets are retained in the cache-level audit but are not interpreted as reliable distributional evidence.

Across all splits, the ground-truth target was unmasked in 100.0% of samples, meaning that the target region was always present among the valid candidate regions. The number of valid candidates varied from 2 to 32, confirming the variable-\(K\) candidate setting. Among reliable \(K\)-buckets, the worst absolute deviation from the ideal uniform target-slot distribution was 1.67 percentage points on train, 4.21 percentage points on val, 6.92 percentage points on testA, and 4.74 percentage points on testB. The \(K=2\) sanity buckets were also close to balanced; for example, testB had target-slot counts \([300, 293]\), corresponding to only a 0.59 percentage-point deviation from the ideal 50/50 split.

These audits indicate that the candidate construction does not introduce a systematic target-position shortcut. We additionally include `random_valid`, `train_slot_prior`, and `index_prior_only` as sanity baselines in the controlled comparison table to verify that candidate order alone is insufficient to solve the task.

| split | samples | target unmasked | valid K min/max | mean K | reliable buckets | worst reliable dev | K=2 sanity |
|---|---:|---:|---:|---:|---:|---:|---|
| train | 120624 | 100.0% | 2/32 | 10.34 | 28 | K=28, 1.67pp | [3721, 3802] dev=0.54pp |
| val | 10834 | 100.0% | 2/32 | 10.60 | 13 | K=6, 4.21pp | [289, 304] dev=1.26pp |
| testA | 5657 | 100.0% | 2/32 | 12.22 | 10 | K=3, 6.92pp | [10, 11] dev=2.38pp |
| testB | 5095 | 100.0% | 2/32 | 7.92 | 9 | K=9, 4.74pp | [300, 293] dev=0.59pp |

## GroundingDINO Proposal Oracle

This table reports the upper bound of the proposal-conditioned setting. A reranker cannot exceed oracle Acc@0.5 when no proposal reaches IoU >= 0.5.

| Test | proposal oracle Acc@0.5 | oracle best IoU | empty proposals | OVHA Acc@0.5 / oracle |
|---|---:|---:|---:|---:|
| val | 0.6268 | 0.6135 | 362/10834 (3.34%) | 0.9515 |
| testA | 0.6700 | 0.6455 | 212/5657 (3.75%) | 0.9693 |
| testB | 0.5884 | 0.5797 | 181/5095 (3.55%) | 0.9444 |

## OVHA RefCOCO Proposal-Conditioned

Proposal-conditioned evaluation: rerankers score candidates conditioned on proposal features rather than fixed 32-candidate slots with frozen CLIP features alone.

| Test | model | Acc@0.5 ↑ | mIoU ↑ |
|---|---|---:|---:|
| val | ovha_admitted | 0.5964 | 0.5872 |
| val | lightweight_transvg_style_reranker | 0.5873 | 0.5806 |
| val | clip_geometry_mlp | 0.5812 | 0.5749 |
| val | box_aware_cross_attention_reranker | 0.5785 | 0.5714 |
| val | candidate_mlp_reranker | 0.5733 | 0.5708 |
| testA | ovha_admitted | 0.6494 | 0.6252 |
| testA | lightweight_transvg_style_reranker | 0.6405 | 0.6200 |
| testA | clip_geometry_mlp | 0.6341 | 0.6150 |
| testA | box_aware_cross_attention_reranker | 0.6312 | 0.6146 |
| testA | candidate_mlp_reranker | 0.6284 | 0.6086 |
| testB | ovha_admitted | 0.5557 | 0.5491 |
| testB | lightweight_transvg_style_reranker | 0.5531 | 0.5487 |
| testB | clip_geometry_mlp | 0.5430 | 0.5402 |
| testB | box_aware_cross_attention_reranker | 0.5423 | 0.5388 |
| testB | candidate_mlp_reranker | 0.5372 | 0.5343 |

Since proposal quality upper-bounds this setting, Acc@0.5 and mIoU should be interpreted together with the proposal oracle. OVHA reaches 95.15%, 96.93%, and 94.44% of the oracle Acc@0.5 on val, testA, and testB respectively.
