# Phase 1.5 Boundaries

## Main Claim

Phase 1.5 tests whether OVHA can form an operator memory from context observations without oracle metadata. The main evidence is metadata-free context inference, not performance with family labels or hidden gains.

## What Counts as Model Input

Allowed:

- context input/output observations
- target input samples
- query coordinates
- support grid
- masks

Forbidden:

- operator family
- operator id
- gain
- hidden mixture weights
- latent primitive parameters
- primitive usage labels
- oracle hints

Hidden fields may be used only for data generation, offline diagnostics, leakage tests and explicitly labeled oracle upper-bound baselines.

## Special-Case Containment

FNO-like spectral operators, DeepONet-like separable basis operators, graph/local kernels and vector attention are contained as instantiations of the operator-valued primitive interface. They are not the method name and not independent modules stacked after a Transformer.

## Baseline Interpretation

`simple_stack` is a baseline for ordinary module composition. It lacks query/context-conditioned operator-valued aggregation.

`oracle_metadata_upper_bound` is an upper bound. It must be reported separately and cannot be used as OVHA-full evidence.

## Execution Boundary

Mac Air local work is limited to code, unit tests, CPU smoke and report generation. Large sweeps, CUDA runs, PDE/CV/robotics benchmarks and A800 experiments are not run locally.
