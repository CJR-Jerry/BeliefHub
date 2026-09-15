#!/usr/bin/env python3
"""Aggregate CN6 loss-sweep evaluations and create publication-ready figures."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy import stats
except ImportError:  # pragma: no cover - fallback for minimal environments
    stats = None


INPUT_DIR = Path("tmp/cn6_loss_sweep_input/results/cn6_loss_sweep")
PDF_DIR = Path("output/pdf")
CHART_DIR = Path("output/charts")
DATA_DIR = Path("output/data")

EXPECTED_METHODS = ("match", "db_all")
EXPECTED_SEEDS = tuple(range(5))
EXPECTED_LOSSES = tuple(range(0, 101, 10))
FILE_RE = re.compile(r"^(match|db_all)_seed([0-4])_loss(\d{3})\.json$")

LABELS = {"match": "MATCH", "db_all": "DB-All"}
COLORS = {"match": "#0072B2", "db_all": "#D55E00"}
MARKERS = {"match": "o", "db_all": "s"}
LINESTYLES = {"match": "-", "db_all": "--"}


def ci95(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return 0.0
    sem = values.std(ddof=1) / math.sqrt(n)
    critical = float(stats.t.ppf(0.975, n - 1)) if stats else 2.776445
    return critical * sem


def paired_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    if stats is None:
        return float("nan")
    return float(stats.ttest_rel(a, b).pvalue)


def load_and_validate() -> list[dict]:
    files = sorted(INPUT_DIR.glob("*.json"))
    if len(files) != 110:
        raise RuntimeError(f"Expected 110 JSON files, found {len(files)}")

    rows: list[dict] = []
    seen: set[tuple[str, int, int]] = set()
    checkpoint_episodes: dict[tuple[str, int], set[int]] = defaultdict(set)

    for path in files:
        match = FILE_RE.match(path.name)
        if not match:
            raise RuntimeError(f"Unexpected filename: {path.name}")
        method, seed_text, loss_text = match.groups()
        seed = int(seed_text)
        loss_pct = int(loss_text)
        key = (method, seed, loss_pct)
        if key in seen:
            raise RuntimeError(f"Duplicate result: {key}")
        seen.add(key)

        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)

        expected_method = "MATCH" if method == "match" else "MATCH+LocalBelief"
        if payload["method"] != expected_method:
            raise RuntimeError(f"Method mismatch in {path.name}")
        if payload["test_episodes"] != 200:
            raise RuntimeError(f"Expected 200 test episodes in {path.name}")
        if payload["test_seed"] != 20000:
            raise RuntimeError(f"Unexpected test seed in {path.name}")
        if not math.isclose(payload["packet_loss"] * 100.0, loss_pct, abs_tol=1e-6):
            raise RuntimeError(f"Packet-loss mismatch in {path.name}")
        if not math.isclose(float(payload["sight_range"]), 0.8, abs_tol=1e-8):
            raise RuntimeError(f"Sight-range mismatch in {path.name}")

        numeric_fields = (
            "mean_return",
            "mean_pol",
            "full_occupancy_rate",
            "mean_packet_success_rate",
        )
        if not all(math.isfinite(float(payload[field])) for field in numeric_fields):
            raise RuntimeError(f"Non-finite value in {path.name}")

        checkpoint_episodes[(method, seed)].add(int(payload["checkpoint_episode"]))
        rows.append(
            {
                "method": method,
                "seed": seed,
                "packet_loss_pct": loss_pct,
                "checkpoint_episode": int(payload["checkpoint_episode"]),
                "mean_pol": float(payload["mean_pol"]),
                "mean_return": float(payload["mean_return"]),
                "full_occupancy_rate": float(payload["full_occupancy_rate"]),
                "mean_packet_success_rate": float(payload["mean_packet_success_rate"]),
            }
        )

    expected = {
        (method, seed, loss)
        for method in EXPECTED_METHODS
        for seed in EXPECTED_SEEDS
        for loss in EXPECTED_LOSSES
    }
    if seen != expected:
        raise RuntimeError(f"Missing combinations: {sorted(expected - seen)}")
    inconsistent = {key: values for key, values in checkpoint_episodes.items() if len(values) != 1}
    if inconsistent:
        raise RuntimeError(f"Checkpoint changed across loss rates: {inconsistent}")
    return sorted(rows, key=lambda row: (row["packet_loss_pct"], row["method"], row["seed"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_group: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_group[(row["method"], row["packet_loss_pct"])].append(row)

    summary: list[dict] = []
    comparisons: list[dict] = []
    for loss in EXPECTED_LOSSES:
        method_arrays: dict[str, dict[str, np.ndarray]] = {}
        for method in EXPECTED_METHODS:
            group = sorted(by_group[(method, loss)], key=lambda row: row["seed"])
            if [row["seed"] for row in group] != list(EXPECTED_SEEDS):
                raise RuntimeError(f"Seed mismatch for {method} at {loss}%")
            arrays = {
                field: np.array([row[field] for row in group], dtype=float)
                for field in (
                    "mean_pol",
                    "mean_return",
                    "full_occupancy_rate",
                    "mean_packet_success_rate",
                )
            }
            method_arrays[method] = arrays
            summary.append(
                {
                    "method": LABELS[method],
                    "packet_loss_pct": loss,
                    "n_training_seeds": len(group),
                    "mean_pol": arrays["mean_pol"].mean(),
                    "ci95_pol": ci95(arrays["mean_pol"]),
                    "mean_return": arrays["mean_return"].mean(),
                    "ci95_return": ci95(arrays["mean_return"]),
                    "mean_full_occupancy_rate": arrays["full_occupancy_rate"].mean(),
                    "mean_packet_success_rate": arrays["mean_packet_success_rate"].mean(),
                }
            )

        match_values = method_arrays["match"]
        db_values = method_arrays["db_all"]
        pol_delta = db_values["mean_pol"] - match_values["mean_pol"]
        return_delta = db_values["mean_return"] - match_values["mean_return"]
        comparisons.append(
            {
                "packet_loss_pct": loss,
                "match_mean_pol": match_values["mean_pol"].mean(),
                "db_all_mean_pol": db_values["mean_pol"].mean(),
                "db_minus_match_pol": pol_delta.mean(),
                "ci95_paired_pol_delta": ci95(pol_delta),
                "paired_t_pvalue_pol": paired_pvalue(db_values["mean_pol"], match_values["mean_pol"]),
                "match_mean_return": match_values["mean_return"].mean(),
                "db_all_mean_return": db_values["mean_return"].mean(),
                "db_minus_match_return": return_delta.mean(),
                "ci95_paired_return_delta": ci95(return_delta),
                "paired_t_pvalue_return": paired_pvalue(db_values["mean_return"], match_values["mean_return"]),
            }
        )
    return summary, comparisons


def series(summary: list[dict], method_label: str, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = sorted(
        (row for row in summary if row["method"] == method_label),
        key=lambda row: row["packet_loss_pct"],
    )
    x = np.array([row["packet_loss_pct"] for row in selected], dtype=float)
    y = np.array([row[f"mean_{metric}"] for row in selected], dtype=float)
    ci = np.array([row[f"ci95_{metric}"] for row in selected], dtype=float)
    return x, y, ci


def style_axes(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
    axis.set_axisbelow(True)
    axis.set_xticks(EXPECTED_LOSSES)
    axis.set_xlim(-2, 102)
    axis.set_xlabel("Test packet loss rate (%)")


def add_metric(axis: plt.Axes, summary: list[dict], metric: str) -> None:
    ci_key = metric
    for method in EXPECTED_METHODS:
        label = LABELS[method]
        x, y, ci = series(summary, label, ci_key)
        axis.fill_between(x, y - ci, y + ci, color=COLORS[method], alpha=0.14, linewidth=0)
        axis.plot(
            x,
            y,
            label=label,
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            marker=MARKERS[method],
            markersize=4.2,
            markerfacecolor="white",
            markeredgewidth=1.1,
            linewidth=1.55,
        )
    style_axes(axis)
    if metric == "pol":
        axis.set_ylabel("Proportion of occupied landmarks (POL)")
        axis.set_ylim(0.0, 1.0)
    elif metric == "return":
        axis.set_ylabel("Mean episode return")
    axis.legend(frameon=False, loc="best", handlelength=2.7)


def save_figure(figure: plt.Figure, stem: str) -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(PDF_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    figure.savefig(CHART_DIR / f"{stem}.png", dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)


def create_figures(summary: list[dict]) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 8.2,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axis = plt.subplots(figsize=(3.48, 2.52), constrained_layout=True)
    add_metric(axis, summary, "pol")
    save_figure(figure, "cn6_loss_sweep_pol")

    figure, axis = plt.subplots(figsize=(3.48, 2.52), constrained_layout=True)
    add_metric(axis, summary, "return")
    save_figure(figure, "cn6_loss_sweep_return")

    figure, axes = plt.subplots(1, 2, figsize=(7.16, 2.52), constrained_layout=True)
    add_metric(axes[0], summary, "pol")
    add_metric(axes[1], summary, "return")
    axes[0].text(0.02, 0.96, "(a)", transform=axes[0].transAxes, va="top", fontweight="bold")
    axes[1].text(0.02, 0.96, "(b)", transform=axes[1].transAxes, va="top", fontweight="bold")
    save_figure(figure, "cn6_loss_sweep_combined")


def robustness_by_seed(rows: list[dict]) -> tuple[list[dict], dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["seed"])].append(row)

    seed_rows: list[dict] = []
    for method in EXPECTED_METHODS:
        for seed in EXPECTED_SEEDS:
            ordered = sorted(grouped[(method, seed)], key=lambda row: row["packet_loss_pct"])
            x = np.array([row["packet_loss_pct"] / 100.0 for row in ordered], dtype=float)
            y = np.array([row["mean_pol"] for row in ordered], dtype=float)
            seed_rows.append(
                {
                    "method": LABELS[method],
                    "seed": seed,
                    "pol_auc": float(np.trapezoid(y, x)),
                    "pol_change_0_to_100": float(y[-1] - y[0]),
                    "pol_linear_slope_per_full_loss_range": float(np.polyfit(x, y, 1)[0]),
                }
            )

    indexed = {(row["method"], row["seed"]): row for row in seed_rows}
    metrics = {}
    for metric in ("pol_auc", "pol_change_0_to_100", "pol_linear_slope_per_full_loss_range"):
        match_values = np.array([indexed[("MATCH", seed)][metric] for seed in EXPECTED_SEEDS])
        db_values = np.array([indexed[("DB-All", seed)][metric] for seed in EXPECTED_SEEDS])
        delta = db_values - match_values
        metrics[metric] = {
            "match_mean": float(match_values.mean()),
            "db_mean": float(db_values.mean()),
            "delta_mean": float(delta.mean()),
            "delta_ci95": ci95(delta),
            "paired_pvalue": paired_pvalue(db_values, match_values),
        }
    return seed_rows, metrics


def write_report(rows: list[dict], summary: list[dict], comparisons: list[dict]) -> None:
    by_method = defaultdict(list)
    for row in summary:
        by_method[row["method"]].append(row)
    auc = {}
    for method, method_rows in by_method.items():
        ordered = sorted(method_rows, key=lambda row: row["packet_loss_pct"])
        x = np.array([row["packet_loss_pct"] / 100.0 for row in ordered])
        y = np.array([row["mean_pol"] for row in ordered])
        auc[method] = float(np.trapezoid(y, x))

    deltas = np.array([row["db_minus_match_pol"] for row in comparisons])
    best = max(comparisons, key=lambda row: row["db_minus_match_pol"])
    worst = min(comparisons, key=lambda row: row["db_minus_match_pol"])
    seed_robustness, robustness = robustness_by_seed(rows)
    write_csv(DATA_DIR / "cn6_loss_sweep_robustness_by_seed.csv", seed_robustness)

    auc_result = robustness["pol_auc"]
    endpoint_result = robustness["pol_change_0_to_100"]
    slope_result = robustness["pol_linear_slope_per_full_loss_range"]

    report = [
        "# CN6 packet-loss sweep summary",
        "",
        "All models were trained at 30% packet loss and evaluated at 0%-100% packet loss.",
        "Each point is the mean across five training seeds; shaded bands are 95% t-confidence intervals across those seeds.",
        "Each seed-level point averages 200 evaluation episodes with the same evaluation seed schedule.",
        "",
        "## Aggregate results",
        "",
        f"- MATCH normalized POL AUC: {auc['MATCH']:.4f}",
        f"- DB-All normalized POL AUC: {auc['DB-All']:.4f}",
        f"- DB-All minus MATCH AUC: {auc['DB-All'] - auc['MATCH']:+.4f}",
        f"- Mean DB-All minus MATCH POL across all loss rates: {deltas.mean():+.4f}",
        f"- Largest POL advantage: {best['db_minus_match_pol']:+.4f} at {best['packet_loss_pct']}% packet loss",
        f"- Smallest POL advantage: {worst['db_minus_match_pol']:+.4f} at {worst['packet_loss_pct']}% packet loss",
        f"- Paired seed-level AUC difference: {auc_result['delta_mean']:+.4f} ± {auc_result['delta_ci95']:.4f} (p={auc_result['paired_pvalue']:.4g})",
        f"- MATCH POL change from 0% to 100% loss: {endpoint_result['match_mean']:+.4f}",
        f"- DB-All POL change from 0% to 100% loss: {endpoint_result['db_mean']:+.4f}",
        f"- Difference in endpoint change (DB-All minus MATCH): {endpoint_result['delta_mean']:+.4f} ± {endpoint_result['delta_ci95']:.4f} (p={endpoint_result['paired_pvalue']:.4g})",
        f"- Difference in fitted loss-response slope: {slope_result['delta_mean']:+.4f} ± {slope_result['delta_ci95']:.4f} (p={slope_result['paired_pvalue']:.4g})",
        "",
        "## Per-loss POL comparison",
        "",
        "| Loss | MATCH | DB-All | Delta | 95% CI of paired delta | Paired t-test p |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        report.append(
            f"| {row['packet_loss_pct']}% | {row['match_mean_pol']:.4f} | "
            f"{row['db_all_mean_pol']:.4f} | {row['db_minus_match_pol']:+.4f} | "
            f"±{row['ci95_paired_pol_delta']:.4f} | {row['paired_t_pvalue_pol']:.4g} |"
        )
    (DATA_DIR / "cn6_loss_sweep_analysis.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_and_validate()
    summary, comparisons = aggregate(rows)
    write_csv(DATA_DIR / "cn6_loss_sweep_per_seed.csv", rows)
    write_csv(DATA_DIR / "cn6_loss_sweep_summary.csv", summary)
    write_csv(DATA_DIR / "cn6_loss_sweep_paired_comparison.csv", comparisons)
    write_report(rows, summary, comparisons)
    create_figures(summary)
    print(f"Validated {len(rows)} evaluations.")
    print(f"Wrote {len(summary)} aggregate rows and {len(comparisons)} paired comparisons.")


if __name__ == "__main__":
    main()
