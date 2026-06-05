# Multimodal External SOTA Baseline Policy

This project now separates two evidence types:

1. Internal OVHA evidence: `run_public_main.py` trains the configured main model, simple same-feature sanity probes, and OVHA ablations on the same frozen cache.
2. External SOTA evidence: published references or separate official-repo reproductions are tracked outside the same-feature baseline registry.

## Internal Baselines

RefCOCO internal rows:

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

## External SOTA References

The external list lives in `configs/multimodal_external_sota_references.json`.

Generate a reviewable runbook with:

```bash
python scripts/multimodal/build_external_sota_runbook.py \
  --references configs/multimodal_external_sota_references.json \
  --output-dir outputs/multimodal/external_sota_runbook
```

This creates:

- `outputs/multimodal/external_sota_runbook/external_sota_runbook.json`
- `outputs/multimodal/external_sota_runbook/external_sota_runbook.md`

The command intentionally does not download weights or run external repositories. Before importing an external result, record the source URL, official repo, checkpoint, data split, metric definition, software commit, and hardware.

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
