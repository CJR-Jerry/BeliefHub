#!/usr/bin/env python3
"""Run train_match_cn6.py while measuring end-to-end CUDA overhead."""

from __future__ import annotations

import argparse
import json
import os
import resource
import runpy
import sys
import time
import traceback
from pathlib import Path

import torch


def parse_episode_count(arguments: list[str]) -> int | None:
    for index, value in enumerate(arguments):
        if value == "--episodes" and index + 1 < len(arguments):
            return int(arguments[index + 1])
        if value.startswith("--episodes="):
            return int(value.split("=", 1)[1])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--method", required=True, choices=("match", "db_all"))
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    train_args = list(args.train_args)
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]

    episodes = parse_episode_count(train_args)
    if episodes is None:
        raise ValueError("The wrapped command must specify --episodes")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    started_at = time.time()
    start = time.perf_counter()
    status = "succeeded"
    error_text = None

    original_argv = sys.argv[:]
    sys.argv = ["train_match_cn6.py", *train_args]
    try:
        runpy.run_path("train_match_cn6.py", run_name="__main__")
    except BaseException:
        status = "failed"
        error_text = traceback.format_exc()
    finally:
        sys.argv = original_argv

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    report = {
        "status": status,
        "method": args.method,
        "training_seed": args.training_seed,
        "episodes": episodes,
        "wall_time_seconds": elapsed,
        "seconds_per_1k_episodes": elapsed / episodes * 1000.0,
        "episodes_per_second": episodes / elapsed,
        "peak_cuda_allocated_mb": peak_allocated / 1024**2,
        "peak_cuda_reserved_mb": peak_reserved / 1024**2,
        "peak_process_rss_mb": max_rss_kb / 1024.0,
        "gpu_name": torch.cuda.get_device_name(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "started_unix_time": started_at,
        "train_arguments": train_args,
        "error": error_text,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    if status != "succeeded":
        raise RuntimeError(f"Wrapped training failed; see {report_path}")


if __name__ == "__main__":
    main()
