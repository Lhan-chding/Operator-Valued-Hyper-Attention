# External SOTA Reference Runbook

external SOTA entries are separate reference/reproduction evidence; they must not be mixed into same-feature public-main baseline coverage

## cmu_mosei

| Model | Evidence | Status | Source | Repo |
|---|---|---|---|---|
| LMF | external_reference | legacy_repo_available | [paper/project](https://arxiv.org/abs/1806.00064) | [repo](https://github.com/Justin1904/Low-rank-Multimodal-Fusion) |
| MAG-BERT | external_reference | manual_dependency_audit_required | [paper/project](https://arxiv.org/abs/1908.05787) | [repo](https://github.com/WasifurRahman/BERT_multimodal_transformer) |
| MISA | external_reproduction | official_repo_available | [paper/project](https://arxiv.org/abs/2005.03545) | [repo](https://github.com/declare-lab/MISA) |
| MulT | external_reproduction | official_repo_available | [paper/project](https://arxiv.org/abs/1906.00295) | [repo](https://github.com/yaohungt/Multimodal-Transformer) |
| Self-MM | external_reproduction | official_repo_available | [paper/project](https://arxiv.org/abs/2102.04830) | [repo](https://github.com/thuiar/Self-MM) |
| TFN | external_reference | legacy_repo_available | [paper/project](https://arxiv.org/abs/1707.07250) | [repo](https://github.com/Justin1904/TensorFusionNetworks) |

## refcoco

| Model | Evidence | Status | Source | Repo |
|---|---|---|---|---|
| GLIP | external_reproduction | official_repo_available_manual_weights | [paper/project](https://arxiv.org/abs/2112.03857) | [repo](https://github.com/microsoft/GLIP) |
| GroundingDINO | external_reproduction | official_repo_available_manual_weights | [paper/project](https://arxiv.org/abs/2303.05499) | [repo](https://github.com/IDEA-Research/GroundingDINO) |
| GroundingDINO-1.5 | external_reproduction | api_or_official_checkpoint_required | [paper/project](https://arxiv.org/abs/2405.10300) | [repo](https://github.com/IDEA-Research/Grounding-DINO-1.5-API) |
| LAVT | external_reference | related_task_reference | [paper/project](https://arxiv.org/abs/2112.02244) | [repo](https://github.com/yz93/LAVT-RIS) |
| MDETR | external_reference | official_repo_available | [paper/project](https://arxiv.org/abs/2104.12763) | [repo](https://github.com/ashkamath/mdetr) |
| SeqTR | external_reference | official_repo_available | [paper/project](https://arxiv.org/abs/2203.16265) | [repo](https://github.com/sean-zhuh/SeqTR) |
| TransVG | external_reference | official_repo_available | [paper/project](https://arxiv.org/abs/2104.08541) | [repo](https://github.com/djiajunustc/TransVG) |

## RefCOCO Comparison Tracks

### A_external_open_box_reference

- Input: raw image plus referring expression
- Output: free-form predicted boxes
- Scope: external detector/reference table only
- Required metrics: Acc@0.5, mean IoU, split-specific val/testA/testB where available

### B_same_candidate_scorer

- Input: OVHA candidate boxes plus external detector predictions
- Output: candidate score_i=max_j IoU(candidate_i, box_j) * score_j
- Scope: same-candidate table
- Required metrics: R@1, R@5, Acc@0.5, mean IoU, MRR

### C_proposal_generator_plus_reranker

- Input: shared GroundingDINO top-K proposals for every reranker
- Output: proposal upper bound plus reranker metrics
- Scope: proposal-conditioned reranker table
- Required metrics: proposal_recall@K first, then reranker R@1/R@5/Acc@0.5/mean IoU/MRR

## RefCOCO External Alignment Experiment Design

- Schema: refcoco-external-alignment-experiments-v0.1
- Objective: Upgrade the current fixed-candidate RefCOCO evidence into a reviewable external-alignment experiment suite without mixing fixed-candidate and open-box claims.
- Fixed-candidate feature policy: frozen_clip_same_candidate
- Main seeds: 201, 202, 203, 204, 205

### Comparison Tracks

| Track | Scope | Key constraint |
|---|---|---|
| A_external_open_box_reference | separate_external_reference_table | no fixed-candidate win/loss claims |
| B_groundingdino_same_candidate_scorer | same_candidate_table_with_feature_source_note | max_j IoU(candidate_i, predicted_box_j) * score_j |
| C_groundingdino_proposals_plus_reranker | proposal_conditioned_reranker_table | proposal_oracle_recall_at_k, oracle_best_iou, empty_proposal_rate, positive_candidate_rate |

### Strong Same-Candidate Baselines

- box_aware_cross_attention_reranker: Controls for the possibility that the previous cross-attention reranker was too weak or geometry-blind.
- clip_geometry_mlp: Tests whether OVHA gains are only explained by adding bbox geometry.
- lightweight_transvg_style_reranker: Provides a stronger same-candidate grounding-transformer-style reranker without claiming a TransVG reproduction.
- groundingdino_same_candidate_scorer: Aligns GroundingDINO with the fixed-candidate evaluator while marking the feature source difference.

### Operator Subset Diagnostics

- spatial_subset: target SRO; expression contains spatial terms such as left, right, top, bottom, above, below, near, next to, behind, in front of, between
- attribute_subset: target AMO; expression contains color, size, or appearance words such as red, blue, small, large, striped, wearing, hat, shirt
- dense_distractor_subset: target TLEO; same-category candidates >= 2, high candidate overlap or close center distance, or close CLIP top-k scores
- long_relational_subset: target CATO; token count above threshold or expression contains multiple noun phrases or relation words

### Required Artifacts

- Per-sample fields: source_id, split, expression, candidate_boxes, candidate_mask, target_index, selected_index, selected_iou, operator_contributions, baseline_scores, model_name, seed, checkpoint, metric_source

### Final Tables

- table_1_cmu_mosei_main: existing sentiment main table
- table_2_refcoco_fixed_candidate_mechanism: same fixed candidates and frozen CLIP features, plus explicitly marked GroundingDINO same-candidate scorer
- table_3_refcoco_operator_subset_diagnostics: operator responsibility subsets for SRO, TLEO, CATO, and attribute error profile
- table_4_groundingdino_proposal_reranking: shared GroundingDINO proposal pool with oracle upper bound reported first
- table_5_external_open_box_reference: reported or separately reproduced open-box grounding references; no fixed-candidate win/loss claims
- table_6_cross_task_operator_admission: CMU-MOSEI, RefCOCO fixed-candidate, and GroundingDINO proposal admitted banks

## Rules

- These entries are external references or external reproductions.
- They are not trained by `run_public_main.py`.
- They must not be listed in `baseline_names` for same-feature public-main configs.
- Any reproduced result must record checkpoint, commit, environment, split, metric, and data license.
