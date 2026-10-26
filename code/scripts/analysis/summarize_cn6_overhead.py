#!/usr/bin/env python3
"""Summarize MATCH and DB-All overhead benchmark reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


TRAIN_METRICS = (
    "seconds_per_1k_episodes",
    "episodes_per_second",
    "peak_cuda_allocated_mb",
    "peak_cuda_reserved_mb",
)
INFERENCE_METRICS = (
    "mean_end_to_end_step_latency_ms",
    "p95_episode_latency_ms",
    "peak_inference_cuda_allocated_mb",
    "execution_parameter_count",
    "online_training_parameter_elements",
)


def load_reports(directory: Path) -> list[dict]:
    reports = []
    for path in sorted(directory.glob("*.json")):
        with path.open("r", encoding="utf-8") as stream:
            report = json.load(stream)
        if report.get("status", "succeeded") != "succeeded":
            raise RuntimeError(f"Failed report: {path}")
        report["source_file"] = str(path)
        reports.append(report)
    return reports


def aggregate(reports: list[dict], metrics: tuple[str, ...], phase: str) -> list[dict]:
    grouped = defaultdict(list)
    for report in reports:
        grouped[report["method"]].append(report)

    rows = []
    for metric in metrics:
        method_values = {}
        for method in ("match", "db_all"):
            values = np.asarray([float(row[metric]) for row in grouped[method]], dtype=float)
            if len(values) == 0 or not np.all(np.isfinite(values)):
                raise RuntimeError(f"Missing/non-finite {phase} metric: {method} {metric}")
            method_values[method] = values

        match_mean = float(method_values["match"].mean())
        db_mean = float(method_values["db_all"].mean())
        rows.append(
            {
                "phase": phase,
                "metric": metric,
                "match_mean": match_mean,
                "match_std": float(method_values["match"].std(ddof=1)) if len(method_values["match"]) > 1 else 0.0,
                "db_all_mean": db_mean,
                "db_all_std": float(method_values["db_all"].std(ddof=1)) if len(method_values["db_all"]) > 1 else 0.0,
                "db_all_overhead_percent": (db_mean / match_mean - 1.0) * 100.0,
                "n_match": len(method_values["match"]),
                "n_db_all": len(method_values["db_all"]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/cn6_overhead")
    parser.add_argument("--output-prefix", default="outputs/cn6_overhead/cn6_overhead_summary")
    args = parser.parse_args()

    root = Path(args.root)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    training = load_reports(root / "training_reports")
    inference = load_reports(root / "inference_reports")
    if len(training) < 2:
        raise RuntimeError(f"Expected both methods in training reports, found {len(training)} files")
    if len(inference) < 2:
        raise RuntimeError(f"Expected both methods in inference reports, found {len(inference)} files")

    rows = aggregate(training, TRAIN_METRICS, "training")
    rows += aggregate(inference, INFERENCE_METRICS, "inference")
    write_csv(output_prefix.with_suffix(".csv"), rows)

    labels = {
        "seconds_per_1k_episodes": "Training seconds / 1k episodes",
        "episodes_per_second": "Training episodes / second",
        "peak_cuda_allocated_mb": "Peak training CUDA allocated (MB)",
        "peak_cuda_reserved_mb": "Peak training CUDA reserved (MB)",
        "mean_end_to_end_step_latency_ms": "End-to-end inference latency / step (ms)",
        "p95_episode_latency_ms": "P95 episode latency (ms)",
        "peak_inference_cuda_allocated_mb": "Peak inference CUDA allocated (MB)",
        "execution_parameter_count": "Execution parameter count",
        "online_training_parameter_elements": "Online training parameter count",
    }
    lines = [
        "# CN6 overhead benchmark",
        "",
        "| Metric | MATCH | DB-All | DB-All difference |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {labels[row['metric']]} | {row['match_mean']:.4f} | "
            f"{row['db_all_mean']:.4f} | {row['db_all_overhead_percent']:+.2f}% |"
        )
    lines += [
        "",
        "Execution-time communication overhead: 0 additional messages and 0 additional bytes,",
        "provided that Local Belief remains entirely agent-local as implemented.",
        "",
        "Positive percentages mean DB-All uses more time/memory/parameters. For throughput,",
        "a negative percentage means DB-All is slower. Combine these measurements with",
        "episodes-to-threshold to compute wall-clock time-to-target.",
    ]
    output_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for row in rows:
        if not all(math.isfinite(float(row[key])) for key in ("match_mean", "db_all_mean", "db_all_overhead_percent")):
            raise RuntimeError(f"Non-finite summary row: {row}")
    print(output_prefix.with_suffix(".md"))


if __name__ == "__main__":
    main()

