# Phase-1 OVHA Report

## Aggregate Metrics

| model | family | mean relative L2 | mean MSE | mean entropy |
|---|---:|---:|---:|---:|
| deeponet | fourier | 1.269627 | 0.039808 | 0.000000 |
| deeponet | green | 0.884862 | 0.192894 | 0.000000 |
| deeponet | mixed | 0.787073 | 0.026507 | 0.000000 |
| deeponet | separable | 0.609025 | 0.008919 | 0.000000 |
| fno | fourier | 2.464875 | 0.102597 | 0.000000 |
| fno | green | 0.965386 | 0.119721 | 0.000000 |
| fno | mixed | 1.357229 | 0.047779 | 0.000000 |
| fno | separable | 6.277191 | 0.100816 | 0.000000 |
| icon_style | fourier | 2.109865 | 0.076113 | 0.000000 |
| icon_style | green | 0.554525 | 0.079081 | 0.000000 |
| icon_style | mixed | 0.833252 | 0.035354 | 0.000000 |
| icon_style | separable | 1.682646 | 0.041985 | 0.000000 |
| mlp_expert | fourier | 1.608540 | 0.052546 | 0.000000 |
| mlp_expert | green | 0.732725 | 0.137429 | 0.000000 |
| mlp_expert | mixed | 0.906779 | 0.039877 | 0.000000 |
| mlp_expert | separable | 0.983852 | 0.020925 | 0.000000 |
| no_hyper_adapter | fourier | 2.427410 | 0.096853 | 0.000030 |
| no_hyper_adapter | green | 0.446951 | 0.039380 | 0.000030 |
| no_hyper_adapter | mixed | 0.687439 | 0.016886 | 1.080528 |
| no_hyper_adapter | separable | 0.618648 | 0.009326 | 0.000030 |
| no_memory | fourier | 1.526251 | 0.048355 | 1.096450 |
| no_memory | green | 0.777894 | 0.139914 | 1.096450 |
| no_memory | mixed | 0.663398 | 0.015639 | 1.096450 |
| no_memory | separable | 0.854843 | 0.023687 | 1.096450 |
| ovha_full | fourier | 0.041728 | 0.000051 | 0.000030 |
| ovha_full | green | 0.038022 | 0.000364 | 0.000030 |
| ovha_full | mixed | 0.019126 | 0.000013 | 1.080528 |
| ovha_full | separable | 0.062727 | 0.000019 | 0.000030 |
| ovha_sparse | fourier | 0.041728 | 0.000051 | 0.000015 |
| ovha_sparse | green | 0.038023 | 0.000364 | 0.000015 |
| ovha_sparse | mixed | 0.472384 | 0.013533 | 0.690923 |
| ovha_sparse | separable | 0.062726 | 0.000019 | 0.000015 |
| perceiver_io_style | fourier | 1.239414 | 0.039161 | 0.000000 |
| perceiver_io_style | green | 0.820151 | 0.172459 | 0.000000 |
| perceiver_io_style | mixed | 0.892294 | 0.037726 | 0.000000 |
| perceiver_io_style | separable | 0.854295 | 0.018306 | 0.000000 |
| random_router | fourier | 1.474608 | 0.040496 | 0.897807 |
| random_router | green | 0.568469 | 0.066568 | 0.897807 |
| random_router | mixed | 0.592100 | 0.021444 | 0.897807 |
| random_router | separable | 1.148664 | 0.035487 | 0.897807 |
| simple_stack | fourier | 1.724158 | 0.053938 | 0.000000 |
| simple_stack | green | 0.661267 | 0.112243 | 0.000000 |
| simple_stack | mixed | 0.745719 | 0.020570 | 0.000000 |
| simple_stack | separable | 2.440679 | 0.029423 | 0.000000 |
| transformer_only | fourier | 2.569347 | 0.106039 | 0.000000 |
| transformer_only | green | 0.421574 | 0.046397 | 0.000000 |
| transformer_only | mixed | 0.958372 | 0.051894 | 0.000000 |
| transformer_only | separable | 2.361522 | 0.072833 | 0.000000 |
| vector_value | fourier | 1.211490 | 0.037674 | 0.000000 |
| vector_value | green | 0.831109 | 0.180249 | 0.000000 |
| vector_value | mixed | 0.815488 | 0.030205 | 0.000000 |
| vector_value | separable | 0.763032 | 0.017647 | 0.000000 |

## Ablation Gap

| comparator | OVHA mean relL2 | comparator mean relL2 | gap |
|---|---:|---:|---:|
| transformer_only | 0.040401 | 1.577704 | 1.537303 |
| perceiver_io_style | 0.040401 | 0.951539 | 0.911138 |
| icon_style | 0.040401 | 1.295072 | 1.254671 |
| deeponet | 0.040401 | 0.887647 | 0.847246 |
| fno | 0.040401 | 2.766170 | 2.725769 |
| simple_stack | 0.040401 | 1.392956 | 1.352555 |
| vector_value | 0.040401 | 0.905280 | 0.864879 |
| no_hyper_adapter | 0.040401 | 1.045112 | 1.004711 |
| no_memory | 0.040401 | 0.955596 | 0.915196 |
| mlp_expert | 0.040401 | 1.057974 | 1.017573 |
| random_router | 0.040401 | 0.945960 | 0.905560 |

## Family Conclusions

- fourier: OVHA-full relL2=0.041728; best observed=ovha_full (0.041728).
- green: OVHA-full relL2=0.038022; best observed=ovha_full (0.038022).
- mixed: OVHA-full relL2=0.019126; best observed=ovha_full (0.019126).
- separable: OVHA-full relL2=0.062727; best observed=ovha_sparse (0.062726).

## Go / No-Go

Go: OVHA-full beats all configured non-OVHA baselines and ablations for every evaluated operator family in this deterministic sweep.

Interpretation: the sweep supports the Phase-1 claim that C is not just decoration, because vector-value attention, simple stacking, no-memory, no-hyper, MLP-expert, and random-router variants all lose the configured ablation comparison.

## Required Phase-2 Questions

- Expand beyond deterministic toy operators with learned training loops once torch/numpy are available.
- Stress-test primitive specialization under metadata-free contexts.
- Add higher-resolution transfer and operator-family holdout curves.
