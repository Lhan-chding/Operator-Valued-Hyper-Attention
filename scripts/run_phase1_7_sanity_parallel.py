#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SanityJob:
    seed: int
    model: str
    gpu: str
    slot: int
    output_dir: Path
    config_path: Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1.7 sanity shards in parallel by seed/model.")
    parser.add_argument("--config", default="configs/phase1_7_controlled_v2_sanity.json")
    parser.add_argument("--output-root", default="outputs/phase1_7/controlled_v2_sanity_parallel")
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "3"))
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--python", default=os.environ.get("PYTHON", sys.executable))
    parser.add_argument("--skip-summary", action="store_true")
    args = parser.parse_args()

    base_config_path = Path(args.config)
    output_root = Path(args.output_root)
    base_config = json.loads(base_config_path.read_text())
    gpus = _parse_gpus(args.gpus)
    if not gpus:
        raise SystemExit("No GPUs specified. Use --gpus 3 or --gpus 0,1,2,3.")
    if args.jobs_per_gpu < 1:
        raise SystemExit("--jobs-per-gpu must be >= 1")

    output_root.mkdir(parents=True, exist_ok=True)
    jobs = _build_jobs(base_config, output_root, gpus, args.jobs_per_gpu)
    print(
        f"[parallel:start] jobs={len(jobs)} gpus={','.join(gpus)} jobs_per_gpu={args.jobs_per_gpu} "
        f"output_root={output_root}",
        flush=True,
    )
    _run_jobs(jobs, args.python)
    if not args.skip_summary:
        _run_summary(args.python, output_root)
    print("[parallel:done]", flush=True)


def _parse_gpus(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _build_jobs(base_config: dict[str, Any], output_root: Path, gpus: list[str], jobs_per_gpu: int) -> list[SanityJob]:
    seeds = list(base_config.get("seeds") or [base_config["seed"]])
    models = list(base_config["train_models"])
    config_dir = output_root / "_parallel_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    slots = [(gpu, slot) for gpu in gpus for slot in range(jobs_per_gpu)]
    jobs: list[SanityJob] = []
    for index, (seed, model) in enumerate((seed, model) for seed in seeds for model in models):
        gpu, slot = slots[index % len(slots)]
        safe_model = _safe_name(model)
        output_dir = output_root / f"seed_{seed}_{safe_model}"
        config = dict(base_config)
        config["seed"] = seed
        config["seeds"] = [seed]
        config["train_models"] = [model]
        config["eval_models"] = [model]
        config["output_dir"] = str(output_dir)
        config_path = config_dir / f"seed_{seed}_{safe_model}.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        jobs.append(SanityJob(seed=seed, model=model, gpu=gpu, slot=slot, output_dir=output_dir, config_path=config_path))
    return jobs


def _run_jobs(jobs: list[SanityJob], python_bin: str) -> None:
    work_queue: queue.Queue[SanityJob] = queue.Queue()
    for job in jobs:
        work_queue.put(job)
    failures: list[tuple[SanityJob, int]] = []
    lock = threading.Lock()

    def worker(worker_id: int) -> None:
        while True:
            try:
                job = work_queue.get_nowait()
            except queue.Empty:
                return
            try:
                status = _run_one_job(job, python_bin)
                if status != 0:
                    with lock:
                        failures.append((job, status))
            finally:
                work_queue.task_done()

    workers = []
    worker_count = len({(job.gpu, job.slot) for job in jobs})
    for worker_id in range(worker_count):
        thread = threading.Thread(target=worker, args=(worker_id,), daemon=True)
        thread.start()
        workers.append(thread)
    for thread in workers:
        thread.join()
    if failures:
        for job, status in failures:
            print(f"[parallel:failed] seed={job.seed} model={job.model} gpu={job.gpu} status={status}", flush=True)
        raise SystemExit(1)


def _run_one_job(job: SanityJob, python_bin: str) -> int:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = job.gpu
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("OVHA_PROGRESS_INTERVAL", "100")
    env.setdefault("OVHA_EVAL_PROGRESS_INTERVAL", "256")
    log_path = job.output_dir / "parallel_job.log"
    commands = [
        [python_bin, "-u", "train_torch_meta_operator.py", "--config", str(job.config_path), "--device", "cuda", "--output-dir", str(job.output_dir)],
        [python_bin, "-u", "eval_torch_meta_operator.py", "--config", str(job.config_path), "--device", "cuda", "--output-dir", str(job.output_dir)],
    ]
    print(f"[job:start] seed={job.seed} model={job.model} gpu={job.gpu} output={job.output_dir}", flush=True)
    start = time.time()
    with log_path.open("w") as log_file:
        for command in commands:
            log_file.write("+ " + " ".join(command) + "\n")
            log_file.flush()
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            assert process.stdout is not None
            for line in process.stdout:
                tagged = f"[gpu={job.gpu} seed={job.seed} model={job.model}] {line}"
                print(tagged, end="", flush=True)
                log_file.write(tagged)
                log_file.flush()
            status = process.wait()
            if status != 0:
                print(f"[job:failed] seed={job.seed} model={job.model} gpu={job.gpu} status={status}", flush=True)
                return status
    elapsed = time.time() - start
    print(f"[job:done] seed={job.seed} model={job.model} gpu={job.gpu} elapsed={elapsed:.1f}s", flush=True)
    return 0


def _run_summary(python_bin: str, output_root: Path) -> None:
    command = [python_bin, "-u", "scripts/summarize_phase1_6.py", "--root", str(output_root)]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


if __name__ == "__main__":
    main()
