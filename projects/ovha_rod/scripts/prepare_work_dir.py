#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from ovha_rod.runtime_contracts import prepare_fresh_private_work_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create or validate a fresh private OVHA run directory")
    parser.add_argument("work_dir", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    print(prepare_fresh_private_work_dir(args.work_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
