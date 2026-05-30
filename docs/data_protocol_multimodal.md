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

`task_type`, `split`, and `source_dataset` must be non-empty strings, and every `ProvenanceBank.original_split` entry in a batch must match the batch `split`.

`TokenField` names, `modality`, and public `attrs` keys must be non-empty strings and must not contain controlled or hidden metadata identifiers such as `true_active_operator`, `corruption_strength`, or `mismatch_source_id`.

If `SupervisionBank.weak_labels` is present, matching `weak_label_confidence` and `pseudo_label_source` entries are required for every weak-label key. Weak-label tensors must align to the batch/query axes `[B,Q,...]`, and each confidence tensor must match its weak-label shape. Pseudo-label sources must be non-empty strings so weak or pseudo supervision cannot be silently treated as ground truth.

If public alignment supervision is present, `alignment_pairs` must align to `[B,Q,2]` and `alignment_weights` must align to `[B,Q]` so CATO public alignment losses cannot silently train on mismatched query-region labels.

If missing-modality supervision is present, `modality_missing_mask` must align to `[B,M]`, where `M` is the number of typed token fields in the batch. If corruption metadata is present, it must be a non-empty map from public metadata names to tensor-like values with a leading batch dimension `B`; optional second dimensions may only encode a singleton sample value, query axis `Q`, or modality axis `M`. Controlled corruption measurements may be kept in `SupervisionBank` for reporting and robustness supervision, but they must remain excluded from `model_inputs`.

## Candidate Bank

Version 1 has exactly four candidate operators:

- `TLEO`: typed local evidence
- `SPO`: semantic prototype
- `LRIO`: low-rank interaction
- `CATO`: cross-modal alignment transport

Every candidate returns `CandidateOutput.value: [B,Q,Dy]`. Only these four outputs may enter `torch.stack(candidate_values, dim=-2)`.

`RCEO` is a support module only. It may produce router prior bias, adapter conditioning features, and diagnostics. It must not appear in the candidate stack.

Controlled v1 experiments must include the router decomposition ablations `no_evidence_router`, `no_reliability_prior`, `memory_only_router`, and `evidence_only_router` so the reported router gain can be attributed to memory, evidence, and reliability terms rather than an undiagnosed mixture shortcut.

Controlled reports must include the full learned/true router-adapter oracle matrix, per-candidate `TLEO_oracle_gap`, `SPO_oracle_gap`, `LRIO_oracle_gap`, and `CATO_oracle_gap`, plus a positive `rceo_prior_effect` for the reliability-corruption family. These fields are required before any public multimodal entry because they localize whether failures come from candidate expressivity, router choice, adapter parameters, memory, or reliability prior.

Oracle smoke output is report-shaped but must be labeled `oracle_smoke_only`; it is a schema and expressivity sanity check, not a substitute for trained controlled go/no-go rows with memory, adapter, router, and reliability ablation deltas.

Public go/no-go gates must reject full-vs-baseline claims without at least three seeds for both models, at least three paired common seeds, and paired permutation plus bootstrap interval evidence.

Public statistics summaries must also include top-conference reporting metadata, not only means: per-model `std`, `ci95`, raw per-seed scores, parameter counts, training steps, frozen feature versions, hardware, wall-clock summary, and a per-seed table. Reports with only best seed, single run, or mean-only aggregate evidence must fail public gates.

Public statistics summaries must include every same-feature baseline from the Step 16 defense table for the task, not just the single comparison baseline used in the main delta. Region-text reports must include text-only, region-only, concat fusion, cross-attention transformer, modality expert MoE, CLIP-style retrieval, CATO-only, `ovha_no_cato`, `ovha_no_rceo`, and `ovha_no_evidence_router`. Sentiment/emotion reports must include concat fusion, TFN/LMF, MulT-style transformer, MISA-style shared/private, modality expert MoE, quality-aware fusion, `ovha_no_lrio`, `ovha_no_spo`, `ovha_no_rceo`, and `ovha_no_evidence_router`.

Robustness summaries must include required ablation degradation for `ovha_no_rceo` and `ovha_no_evidence_router`; both ablations must drop more than `ovha_full` under corruption or missing-modality stress.

Robustness summaries must prove Step 6 stress-family coverage: missing text, missing vision, missing audio, image quality corruption, audio quality corruption, text noise, and hard-negative mismatch. Temporal-shift coverage is required only when the dataset has temporal alignment. These stress descriptors must come from reporting metadata such as `corruption_type`, `missing_modalities`, `mismatch_source_id`, or `temporal_shift_sec`, never from model-input fields.

Region-text public gates must include CATO top alignment accuracy, grounding accuracy improvement paired with lower CATO alignment entropy, and RCEO-supported router-load shift under corrupted or missing visual-region settings.

Sentiment/emotion public gates must include clean-setting LRIO rank entropy, SPO prototype entropy, SPO top-prototype differentiation across emotion classes, and RCEO reliability calibration with expected calibration error evidence. Router-load or ablation deltas alone are not enough to claim LRIO/SPO/RCEO validity.

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

Every `provenance/source_ids_<split>.txt` line must already be a non-empty normalized source ID. Leading or trailing whitespace and blank manifest rows are invalid because they can hide split leakage or silent provenance drops.

`checksums.json` must be a non-empty object whose keys are existing relative file artifact paths that remain inside the cache root.

For region-text or phrase-region grounding tasks, formal cache validation requires per-split `alignment_pairs`, `bbox_targets`, and `region_targets` supervision shards. When `RCEO` is declared in `operator_supervision`, formal cache validation also requires a per-split `corruption` supervision shard, even if the shard is an explicit empty/no-corruption artifact. These required supervision artifacts must be covered by `checksums.json`.

For sentiment/emotion tasks, formal cache validation requires every `sample_records_<split>.jsonl` row to preserve `utterance_id`, `dialogue_id`, `transcript_source`, `missing_modality_mask_ref`, and `corruption_metadata_ref`. If `data_card.json.metadata_availability.speaker_id` is true, every row must also preserve `speaker_id`. The referenced missing-modality and corruption artifacts must be split-local cache artifacts, and per-split `missing_modality_mask_<split>.npy` must be present and checksum-covered.

`provenance/pseudo_label_versions.json` must list pseudo-label source splits in `generated_from_splits`. Sources must be auditable split names from `splits.json`, the current validation splits, or the upstream `train` split; `test` and unknown split names are invalid.

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
