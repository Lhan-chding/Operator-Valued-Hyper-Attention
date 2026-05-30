# PDEBench Architecture Feasibility Note

Date: 2026-05-30

This PDEBench run is used only as architecture feasibility evidence. It is not used as the main top-conference benchmark claim. The main rigorous evaluation will be conducted on multimodal typed-token relation-operator tasks.

The current PDEBench evidence supports a narrow statement: the existing `evidence -> memory -> joint router-adapter -> primitive mixture` path has a positive signal outside synthetic controlled-v2. It does not establish that OVHA is the strongest neural-operator benchmark model, and it does not prove the multimodal typed-token relation-operator framework.

Archived local record:

- Config: `outputs/pdebench_architecture_feasibility_20260530/config.json`
- Summary: `outputs/pdebench_architecture_feasibility_20260530/summary.json`
- Report: `outputs/pdebench_architecture_feasibility_20260530/report.md`
- Git commit: `outputs/pdebench_architecture_feasibility_20260530/git_commit.txt`
- Minimal environment: `outputs/pdebench_architecture_feasibility_20260530/environment_minimal.json`

The archived numbers are from the selected Advection/Darcy public slice reported in `gpt_pro_exports/ovha_advection_darcy_full_key_code_20260530/reports/RESULTS_ADVECTION_DARCY_OVERNIGHT.md` and `gpt_pro_exports/ovha_advection_darcy_gated_adapter_20260530/reports/RESULTS_SELECTED_PUBLIC_BENCHMARK.md`. Raw phase-1.7 metrics are not present under the tracked `outputs/` tree in this checkout; no backfilled large PDE run is claimed.

Required framing:

- PDEBench is architecture feasibility and cross-domain sanity evidence.
- It is not the main top-conference benchmark claim.
- Multimodal typed-token relation-operator experiments carry the top-conference scientific claim.
