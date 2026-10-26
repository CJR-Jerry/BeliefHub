#!/usr/bin/env python3
"""Plot the CN6 robustness gap: BeliefHub-Invisible minus BeliefHub-All."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


ROOT = Path(__file__).resolve().parent
DATA_ROOT = (
    ROOT
    / "extracted/dense/outputs/cn6_missing_information_sweep_500ep"
)
SIGHT_CODES = [f"{value:02d}" for value in range(2, 13)]
LOSS_CODES = [f"{value:02d}" for value in range(0, 100, 10)] + ["99"]
SIGHT_LABELS = [f"{value / 10:.1f}" for value in range(2, 13)]
LOSS_LABELS = [f"{value / 100:.2f}" for value in range(0, 100, 10)] + ["0.99"]
SEEDS = range(5)


def mean_pol(condition: Path, method: str) -> float:
    values = []
    for seed in SEEDS:
        summary = condition / f"{method}_seed{seed}" / "evaluation_summary.json"
        with summary.open("r", encoding="utf-8") as handle:
            values.append(float(json.load(handle)["mean_pol"]))
    return float(np.mean(values))


def load_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_grid = np.empty((len(LOSS_CODES), len(SIGHT_CODES)))
    invisible_grid = np.empty_like(all_grid)
    for row, loss in enumerate(LOSS_CODES):
        for col, sight in enumerate(SIGHT_CODES):
            condition = DATA_ROOT / f"sr{sight}_pl{loss}"
            all_grid[row, col] = mean_pol(condition, "db_all")
            invisible_grid[row, col] = mean_pol(condition, "db_invisible")
    return all_grid, invisible_grid, invisible_grid - all_grid


def save_csv(all_grid: np.ndarray, invisible_grid: np.ndarray, gap: np.ndarray) -> None:
    output = ROOT / "fig06_invisible_minus_all_heatmap_values.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sight_range", "packet_loss", "beliefhub_all_pol",
             "beliefhub_invisible_pol", "delta_pol_invisible_minus_all"]
        )
        for row, loss in enumerate(LOSS_LABELS):
            for col, sight in enumerate(SIGHT_LABELS):
                writer.writerow(
                    [sight, loss, f"{all_grid[row, col]:.8f}",
                     f"{invisible_grid[row, col]:.8f}", f"{gap[row, col]:.8f}"]
                )


def plot(gap: np.ndarray) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.4,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # A compact, paper-ready aspect ratio; the matrix remains legible when
    # included at roughly 0.44\textwidth in a double-column figure.
    fig, ax = plt.subplots(figsize=(4.8, 2.55), constrained_layout=True)
    lower = float(np.floor(gap.min() * 1000) / 1000)
    upper = float(np.ceil(gap.max() * 1000) / 1000)
    delta_blues = LinearSegmentedColormap.from_list(
        "delta_blues",
        ["#c96b62", "#fffaf7", "#ffffff", "#d8e8f3", "#86b5d2", "#2166a5", "#08306b"],
        N=256,
    )
    image = ax.imshow(
        gap,
        origin="lower",
        aspect="auto",
        cmap=delta_blues,
        norm=TwoSlopeNorm(vmin=lower, vcenter=0.0, vmax=upper),
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(SIGHT_LABELS)), labels=SIGHT_LABELS)
    ax.set_yticks(np.arange(len(LOSS_LABELS)), labels=LOSS_LABELS)
    ax.set_xlabel("Sight range", labelpad=2)
    ax.set_ylabel("Packet-loss rate", labelpad=2)

    # Subtle cell boundaries improve grid reading without adding annotations.
    ax.set_xticks(np.arange(-0.5, len(SIGHT_LABELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(LOSS_LABELS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.32, alpha=0.72)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=2.2, width=0.65, pad=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)

    colorbar = fig.colorbar(image, ax=ax, pad=0.025, fraction=0.045)
    colorbar.set_label(r"$\Delta$POL (Invisible $-$ All)", labelpad=3)
    colorbar.ax.tick_params(length=2.0, width=0.6, pad=1.5)
    colorbar.outline.set_linewidth(0.7)

    for suffix in ("png", "pdf"):
        fig.savefig(
            ROOT / f"fig06_invisible_minus_all_heatmap.{suffix}",
            dpi=600 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.02,
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    all_grid, invisible_grid, gap = load_grid()
    save_csv(all_grid, invisible_grid, gap)
    plot(gap)
    print(
        f"conditions={gap.size}, min={gap.min():.6f}, "
        f"mean={gap.mean():.6f}, max={gap.max():.6f}"
    )


if __name__ == "__main__":
    main()

