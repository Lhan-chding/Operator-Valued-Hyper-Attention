# OVHA-ROD Post-RQGO Operator Blueprint

## Objective

Use the time occupied by the Phase-1 RefCOCO RQGO experiment to prepare the
later decoder-operator implementation without changing the currently audited
scientific comparison. The formal Phase-1 configuration remains RQGO-only.

This blueprint has two independent tracks:

1. benchmark safe single-A800 throughput changes before adopting them; and
2. implement decoder-operator primitives behind a disabled-by-default boundary.

## Non-negotiable boundaries

- The current RQGO gate remains the prerequisite for decoder integration.
- Post-RQGO modules must not be imported by the formal RefCOCO configs yet.
- A new module may propose non-zero residuals, but its zero-initialized gate
  must make the fused contribution exactly zero.
- No ground-truth boxes, candidate crops, or annotations may enter inference.
- A speed experiment must use its own work directory and run identity.
- FP16 and BF16 are excluded by observed runtime failures in the locked stack.
- Batch 16 is excluded from the formal run because its measured sample
  throughput is lower than batch 8.

## Track A: single-A800 throughput benchmark

The current run is compute-bound: data time is about 0.05 seconds while a
training step is several seconds. The current diagnostics hook also performs a
host synchronization for every seed-operator parameter gradient. First replace
that observational path with on-device accumulation and one synchronization
per iteration. This preserves the existing per-step non-finite fail-fast check.
The inherited Grounding DINO base additionally enables
Swin checkpointing (`backbone.with_cp=True`) and checkpoints all six encoder
layers (`encoder.num_cp=6`). Benchmark the following cells after the low-sync
diagnostics change:

| Cell | TF32 matmul | Backbone checkpoint | Encoder checkpoint | Purpose |
|---|---:|---:|---:|---|
| A0 | off | on | 6 | exact current control |
| A1 | on | on | 6 | isolate Tensor Core matmul gain |
| A2 | off | off | 0 | isolate recomputation cost |
| A3 | on | off | 0 | candidate fast formal profile |

Run every cell for the same 100 post-warmup iterations with batch 8, the same
seed, data order, image pipeline, optimizer, and accumulation. Record:

- median and p90 seconds per iteration after 10 warm-up iterations;
- samples per second and estimated five-epoch wall time;
- peak allocated and reserved CUDA memory;
- finite loss/gradient status and skipped-step count;
- relative loss drift from A0 over matched batches.

Adopt A3 only if it is finite, fits with at least 10 GiB reserved headroom, and
improves median throughput by at least 15 percent. Otherwise adopt the fastest
cell that passes the same conditions. A throughput profile is part of the run
identity and cannot resume an exact-A0 checkpoint under a different profile.

Changes such as fewer epochs, lower image resolution, fewer queries, frozen
backbone/language model, or a data subset are pilot protocols, not equivalent
replacements for the five-epoch main experiment.

## Track B: operator implementation order

### Phase B0: shared immutable contract

Create a frozen result object with three residual channels:

- `query_delta`: `[B, Q, D]` query-feature residual;
- `box_delta`: `[B, Q, 4]` box-trajectory residual in inverse-sigmoid space;
- `score_delta`: `[B, Q]` referent-score residual;
- `gate_logits`: `[B, Q, 3]` per-channel gate logits;
- `valid`: boolean `[B, Q]` query mask;
- scalar diagnostics.

Validation must reject wrong ranks, shapes, dtypes, or non-finite values. A
fusion module combines a tuple of results without mutating inputs. Fusion uses
`tanh(gate_logits)` so zero gate logits preserve parent query, box, and score
tensors exactly. Each operator owns a zero-initialized gate-logit head; this
keeps the no-op boundary explicit and lets the gate head receive gradients.

### Phase B1: Q-SRO

Purpose: relation-versus-distractor contrast for each decoder query.

Inputs: query features, normalized boxes, relation role vector, and valid query
mask. The primitive computes relation-conditioned pairwise evidence and emits
only a structured result. It must be permutation equivariant over queries,
mask padding exactly, stay finite for one valid query, and begin with an exact
zero fused contribution through its gate logits.

### Phase B2: TQ-CATO

Purpose: competitive transport from valid language tokens to decoder queries.

Inputs: query features, text features, valid query mask, and valid text mask.
The primitive produces a masked transport matrix and structured residual. Each
valid token row must sum to one over valid queries, invalid rows/columns must be
zero, and all-masked samples must fail fast. The first implementation uses a
stable masked softmax contract; balanced/Sinkhorn transport is a later ablation
and must not be assumed without a written mathematical specification.

### Phase B3: MS-TLEO

Purpose: compare multi-scale interior, boundary, and context evidence around
the current query box.

Inputs: a tuple of feature maps, normalized boxes, and query-valid mask. Use
pure PyTorch `grid_sample` in the isolated implementation so CPU tests remain
possible. Interior, boundary, and context sampling layouts require an explicit
coordinate convention test before CUDA integration.

The isolated primitive uses nine interior samples, eight perimeter samples,
and eight context-ring samples per query and level. Normalized image-edge
coordinates are mapped with `2 * coordinate - 1` and
`align_corners=False`; out-of-bounds locations use zero padding. Valid boxes
must have positive width and height, while masked boxes may be degenerate and
remain exact zeros. Level descriptors use an equal arithmetic mean. The
sampler does not materialize query-pair tensors or repeat feature maps: with
25 fixed points it costs `O(L * B * D * Q * 25)` sampling time and
`O(B * D * Q * 25)` peak sampling workspace, followed by
`O(B * Q * D^2)` residual heads.

### Phase B4: router and memory

Only after B1-B3 contracts are stable, add the router, operator memory,
hyper-adapter, and RCEO reliability prior. Router output must be query- and
layer-conditioned, masked, normalized, diagnosable, and initialized so the
complete operator bank remains an exact parent-model no-op.

### Phase B5: decoder integration gate

Integrate only after all of the following hold:

1. the RQGO gate is accepted;
2. the parent/RQGO formal result is frozen and reproducible;
3. every primitive passes shape, masking, finite-backward, equivariance, and
   zero-initialization tests;
4. the pinned MMDetection decoder call graph is reviewed layer by layer;
5. parent equivalence is demonstrated with every operator gate at zero.

## TDD sequence

1. Add failing tests for the shared result and structured fusion.
2. Commit and push the RED state.
3. Implement only the minimum shared contract/fusion code.
4. Run focused tests, the full unit suite, and coverage on the new modules.
5. Review for shape safety, masking, finite numerics, and no integration leak.
6. Commit and push GREEN.
7. Repeat RED/GREEN separately for Q-SRO, TQ-CATO, and MS-TLEO.

## Verification commands

```bash
PYTHONPYCACHEPREFIX=/tmp/ovha_operator_pycache \
PYTHONPATH="$PWD/projects/ovha_rod" \
.venv/bin/python -m unittest \
  projects.ovha_rod.tests.unit.test_decoder_operator_contracts -v

PYTHONPYCACHEPREFIX=/tmp/ovha_operator_pycache \
PYTHONPATH="$PWD/projects/ovha_rod" \
.venv/bin/python -m unittest discover \
  -s projects/ovha_rod/tests/unit -v
```

Coverage must be at least 80 percent for every newly introduced operator file;
the implementation is not allowed to reduce the existing unit-suite pass rate.

## Rollback

Post-RQGO files are isolated and unused by formal configs. Rollback consists of
reverting their commits; no checkpoint or run directory is compatible with a
later integration commit unless its run identity records that exact revision.
