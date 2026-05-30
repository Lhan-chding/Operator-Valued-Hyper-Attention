# Multimodal OVHA Data Protocol

This protocol defines the top-conference mainline for OVHA after PDEBench is frozen as architecture feasibility evidence.

## Scientific Boundary

PDEBench is not the main paper claim. The main claim must be validated on multimodal typed-token relation-operator tasks with controlled truth, public data, robustness stress, diagnostics, and multi-seed statistics.

## Batch Contract

Every sample is represented as:

- typed token fields: text, vision, audio, video, region, or dataset-specific public fields
- query field
- task-space target `target_y: [B,Q,Dy]`
- supervision bank
- provenance bank
- optional controlled `hidden` truth, never model input

The model input path may consume public token fields, query, masks, and explicit public task controls. It must not consume `true_active_operator`, `true_router_weights`, `true_adapter_params`, corruption strength used only for reporting, or dataset-specific hidden metadata.

## Candidate Bank

Version 1 has exactly four candidate operators:

- `TLEO`: typed local evidence
- `SPO`: semantic prototype
- `LRIO`: low-rank interaction
- `CATO`: cross-modal alignment transport

Every candidate returns `CandidateOutput.value: [B,Q,Dy]`. Only these four outputs may enter `torch.stack(candidate_values, dim=-2)`.

`RCEO` is a support module only. It may produce router prior bias, adapter conditioning features, and diagnostics. It must not appear in the candidate stack.

## Cache Requirements

Every formal cache must include:

- `data_card.json`
- `splits.json`
- `checksums.json`
- `samples.parquet` or an equivalent indexed manifest
- per-split token field shards
- `token_fields/manifest_<split>.json` mapping every data-card modality to `x`, `pos`, and `mask` shard paths
- per-split masks and positions
- supervision shards
- provenance/source id files
- `provenance/sample_records_<split>.jsonl` with one row per retained sample
- `provenance/failed_samples_<split>.jsonl`, even when empty
- frozen feature extractor versions
- pseudo label versions when weak labels exist

The cache must preserve `source_id`, split provenance, license tag, feature extractor version, pseudo-label provenance, failed sample manifests, and checksum records.

Checksum manifest keys must be existing relative artifact paths that remain inside the cache root.

Each sample-record manifest row must be a JSON object with `source_id`, `split`, `raw_ref`, and `license_tag`; the row set must match the split's source-id file.

Each non-empty failed-sample manifest row must be a JSON object with `source_id`, `split`, and `reason`. Failed sample source IDs must not also appear in the retained split source-id file. Empty files are valid only when the split has no failed download, decode, feature, or validation sample.

## Leakage Controls

- No dataset hidden metadata as model input.
- No test split labels for training pseudo-label generation.
- Same frozen features for OVHA and baselines.
- No formal run without `data_card.json`, provenance, and checksums.
- No silent drop of failed downloads or invalid samples.
- Weak or pseudo labels must be reported as weak or pseudo labels.
- Controlled metadata such as `corruption_strength` and `true_active_operator` is diagnostics/reporting only.

## Execution Gates

1. Stackability and candidate-name contract tests pass.
2. Controlled-Multimodal single-family candidate gates pass.
3. Controlled oracle matrix isolates router, adapter, candidate, memory, and reliability failures.
4. Region-text public validates CATO/TLEO with same-feature baselines.
5. Sentiment/emotion public validates LRIO/SPO/RCEO with same-feature baselines.
6. Missing, low-quality, and mismatch robustness shows lower degradation and coherent reliability/router shifts.
