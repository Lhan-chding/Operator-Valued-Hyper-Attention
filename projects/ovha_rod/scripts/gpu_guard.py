#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from ovha_rod.runtime_contracts import (
    require_selected_gpus_idle, require_visible_device_ids)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Require an explicit set of currently idle physical GPUs")
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()
    device_ids = require_visible_device_ids(
        os.environ.get("CUDA_VISIBLE_DEVICES"), args.expected_count)
    require_selected_gpus_idle(device_ids)
    print("locked idle GPUs:", ",".join(str(index) for index in device_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
