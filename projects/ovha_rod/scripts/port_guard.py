#!/usr/bin/env python3
from __future__ import annotations

import argparse
import socket
from typing import Callable


def require_port_available(
        port: int, socket_factory: Callable = socket.socket) -> None:
    if not 1024 <= port <= 65535:
        raise ValueError("master port must be in [1024, 65535]")
    try:
        with socket_factory(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", port))
    except OSError as exc:
        raise RuntimeError(
            f"localhost master port {port} is unavailable") from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Require an available localhost distributed-training port")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    require_port_available(args.port)
    print(f"available localhost master port: {args.port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
