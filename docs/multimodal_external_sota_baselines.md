# Multimodal External SOTA Baseline Policy

This project now separates two evidence types:

1. Internal OVHA evidence: `run_public_main.py` trains the configured main model, simple same-feature sanity probes, and OVHA ablations on the same frozen cache.
2. External SOTA evidence: published references or separate official-repo reproductions are tracked outside the same-feature baseline registry.

## Internal Baselines

RefCOCO internal rows:

- `random_valid`
- `train_slot_prior`
- `box_prior`
- `prso_clip_similarity`
- `candidate_mlp_reranker`
- `cross_attention_reranker`
- `index_prior_only`
- `text_only`
- `region_only`
- `concat_fusion`
- `cato_only`
- `ovha_no_cato`
- `ovha_no_rceo`
- `ovha_no_evidence_router`

CMU-MOSEI internal rows:

- `text_only`
- `audio_only`
- `vision_only`
- `concat_fusion`
- `ovha_spo_lrio`
- `ovha_lrio_tanso`
- `ovha_all_candidates_exploratory`
- `ovha_no_spo`
- `ovha_no_rceo`
- `ovha_no_evidence_router`

These rows are not claims that the project reproduced MDETR, GLIP, GroundingDINO, TFN, MulT, MISA, MAG-BERT, or Self-MM. They are mechanism-isolation comparisons over the same cached features.

## RefCOCO External Comparison Tracks

RefCOCO/GroundingDINO-style results must be split into three tracks:

- Track A, external open-box reference: raw image + expression in, model outputs free boxes. Report separately as external detector/reference evidence.
- Track B, same-candidate scorer: external detector boxes are mapped onto the exact OVHA candidate boxes with `max IoU(candidate_i, predicted_box_j) * score_j`, then evaluated with the same candidate metrics.
- Track C, proposal generator plus OVHA reranker: GroundingDINO supplies shared top-K proposals, proposal recall@K is reported first, then rerankers are compared on that shared proposal set.

Do not put Track A rows in `baseline_names`. Only Track B/C outputs may be imported into a candidate-table comparison, and only with the exact candidate/proposal manifest, checkpoint, commit, and metric source recorded.

## External SOTA References

The external list lives in `configs/multimodal_external_sota_references.json`.
The RefCOCO external-alignment experiment design lives in `configs/multimodal_refcoco_external_alignment_experiments.json`.

Generate a reviewable runbook with:

```bash
python scripts/multimodal/build_external_sota_runbook.py \
  --references configs/multimodal_external_sota_references.json \
  --refcoco-experiment-plan configs/multimodal_refcoco_external_alignment_experiments.json \
  --output-dir outputs/multimodal/external_sota_runbook
```

This creates:

- `outputs/multimodal/external_sota_runbook/external_sota_runbook.json`
- `outputs/multimodal/external_sota_runbook/external_sota_runbook.md`

The command intentionally does not download weights or run external repositories. Before importing an external result, record the source URL, official repo, checkpoint, data split, metric definition, software commit, and hardware.

## RefCOCO Experiment Execution Order

Use the structured experiment plan to run the remaining RefCOCO/COCO grounding work in this order:

1. `P0_fixed_candidate_table`: rerun fixed 32-candidate RefCOCO rows with 5 seeds, val/testA/testB split tables, R@1/R@5/MRR/Acc@0.5/Acc@0.7/mIoU/NLL, per-sample predictions, and paired statistics.
2. `P1_strong_same_candidate_baselines`: add `box_aware_cross_attention_reranker`, `clip_geometry_mlp`, `lightweight_transvg_style_reranker`, and `groundingdino_same_candidate_scorer`.
3. `P2_operator_subset_diagnostics`: report spatial/SRO, dense-distractor/TLEO, long-relational/CATO, and attribute error-profile subsets, plus qualitative hard cases.
4. `P3_groundingdino_proposal_pipeline`: generate GroundingDINO top-K proposals, extract proposal CLIP features, compute proposal upper bounds, then rerank the shared proposal pool.
5. `P4_external_reference_table`: keep GroundingDINO, MDETR, GLIP, TransVG/LAVT/MAttNet as external open-box references or separately documented reproductions.
6. `P5_cross_task_admission`: summarize the admitted operator bank across CMU-MOSEI, RefCOCO fixed-candidate, and GroundingDINO-proposal settings.

The final writeup should keep six tables separate: CMU-MOSEI main, RefCOCO fixed-candidate mechanism, RefCOCO operator diagnostics, GroundingDINO proposal reranking, external open-box reference, and cross-task operator admission. Track A open-box rows must never be used for fixed-candidate win/loss claims.

## Source Links

RefCOCO / region-text grounding:

- MDETR: https://arxiv.org/abs/2104.12763 and https://github.com/ashkamath/mdetr
- GLIP: https://arxiv.org/abs/2112.03857 and https://github.com/microsoft/GLIP
- GroundingDINO: https://arxiv.org/abs/2303.05499 and https://github.com/IDEA-Research/GroundingDINO
- GroundingDINO-1.5: https://arxiv.org/abs/2405.10300 and https://github.com/IDEA-Research/Grounding-DINO-1.5-API
- TransVG: https://arxiv.org/abs/2104.08541 and https://github.com/djiajunustc/TransVG
- LAVT: https://arxiv.org/abs/2112.02244 and https://github.com/yz93/LAVT-RIS
- SeqTR: https://arxiv.org/abs/2203.16265 and https://github.com/sean-zhuh/SeqTR

CMU-MOSEI sentiment/emotion:

- TFN: https://arxiv.org/abs/1707.07250 and https://github.com/Justin1904/TensorFusionNetworks
- LMF: https://arxiv.org/abs/1806.00064 and https://github.com/Justin1904/Low-rank-Multimodal-Fusion
- MulT: https://arxiv.org/abs/1906.00295 and https://github.com/yaohungt/Multimodal-Transformer
- MISA: https://arxiv.org/abs/2005.03545 and https://github.com/declare-lab/MISA
- MAG-BERT: https://arxiv.org/abs/1908.05787 and https://github.com/WasifurRahman/BERT_multimodal_transformer
- Self-MM: https://arxiv.org/abs/2102.04830 and https://github.com/thuiar/Self-MM
