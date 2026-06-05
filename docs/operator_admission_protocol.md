# Operator Admission Protocol

Date: 2026-06-05

OVHA should not claim that adding more operators monotonically improves empirical performance. The supported claim is controlled operator admission: a new operator is promoted to the primary bank only when it explains residuals, does not interfere with the current bank, and passes a pre-registered statistical gate.

## Residual Utility

Let the current bank output be `f_K`, the target be `y`, and the residual be:

```text
r = y - f_K
```

For a candidate residual operator `g`, the perturbed predictor is:

```text
f_epsilon = f_K + epsilon * g
```

The MSE expansion is:

```text
||y - f_epsilon||^2 = ||r - epsilon g||^2
                    = ||r||^2 - 2 epsilon <r, g> + epsilon^2 ||g||^2
```

If `<r, g> > 0`, then a small positive `epsilon` can reduce the objective. This is the residual utility gate.

## Six Gates

| Gate | Requirement | Failure state |
|---|---|---|
| Type gate | candidate output belongs to the task output space or a legal residual space | rejected |
| Residual utility gate | residual alignment is positive on validation data | rejected |
| Non-interference gate | adding the candidate does not worsen the old bank beyond tolerance | diagnostic only |
| Dataset relevance gate | the dataset contains the relation type the operator encodes | use a different dataset |
| Diagnostic gate | gate/source/adapter diagnostics match the claimed mechanism | no mechanism claim |
| Statistical gate | five-seed paired bootstrap or permutation test supports the gain | pending statistical gate |

## CLI

Use `scripts/multimodal/operator_admission_eval.py` on validation per-sample predictions:

```bash
.venv/bin/python scripts/multimodal/operator_admission_eval.py \
  --predictions-jsonl outputs/multimodal/cmu_mosei_tanso_primary_main/per_sample_predictions.jsonl \
  --base-model ovha_tanso_primary \
  --candidate-model ovha_lrio_residual \
  --split val \
  --output outputs/multimodal/cmu_mosei_tanso_primary_main/operator_admission_lrio.json
```

The output fields required for paper audit are:

```text
residual_alignment
optimal_alpha
optimal_alpha_distribution
non_interference_delta
paired_bootstrap_p
admission_decision
```

## Reporting Boundary

Allowed wording:

```text
OVHA supports controlled operator admission: new operators are added only if they explain residuals, pass non-interference tests, and improve validation performance under a pre-registered protocol.
```

Forbidden wording until proven by the gates:

```text
Adding operators monotonically improves performance.
```
