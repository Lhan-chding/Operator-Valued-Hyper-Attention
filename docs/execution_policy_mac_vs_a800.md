# Execution Policy: Mac Air vs Ubuntu/A800

## Mac Air / Codex Local

Allowed:

```bash
python3 -m unittest discover -s tests
python3 train_torch_meta_operator.py --config configs/phase1_5_cpu_smoke.json --device cpu
python3 eval_torch_meta_operator.py --config configs/phase1_5_cpu_smoke.json --device cpu
python3 scripts/summarize_phase1_5.py outputs/phase1_5_cpu_smoke
PYTHON=.venv/bin/python bash scripts/run_phase1_6_cpu_integrity_smoke.sh
```

CPU smoke constraints:

- `batch_size <= 4`
- `support_points <= 32`
- `query_points <= 32`
- `steps <= 50`
- CPU only

If torch is not installed, torch-dependent tests skip and scripts write a clear report:

`Torch is not installed; torch implementation tests skipped.`

## Ubuntu / A800

Run after pulling the branch:

```bash
git pull
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_torch.txt
python3 -m unittest discover -s tests
bash scripts/run_phase1_5_gpu_small.sh
bash scripts/run_phase1_5_gpu_main.sh
python3 scripts/summarize_phase1_5.py outputs/phase1_5_gpu_main
```

The GPU scripts are intentionally not run on Mac Air.

Phase 1.6 A800 handoff:

```bash
git pull
export CUDA_VISIBLE_DEVICES=3
bash scripts/run_phase1_6_gpu_integrity_short.sh
bash scripts/run_phase1_6_gpu_component_main.sh
bash scripts/run_phase1_6_gpu_public_pilot.sh
python scripts/summarize_phase1_6.py --root outputs/phase1_6
```

Do not run Phase 1.6 GPU scripts on Mac Air. They are configured for long multi-model or public benchmark workloads. The Phase 1.6 GPU scripts default to physical GPU 3 if `CUDA_VISIBLE_DEVICES` is unset, and respect an explicit `CUDA_VISIBLE_DEVICES` value if the host allocation changes.

## Reporting Rule

Reports must state:

- whether torch was installed
- whether CPU smoke actually ran
- whether GPU scripts were only prepared
- device, seed, config hash, parameter count, training steps and wall time
- whether every eval row is checkpoint-loaded
- whether results are CPU smoke, controlled stress, public benchmark, oracle diagnostic or untrained diagnostic
