#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "workdata/robustness/dense/cn6_dense_grid_pol_summary.csv"
OUT = ROOT / "beliefhub_framed_curves"

COLORS = ["#E66101", "#009E73"]
LABELS = ["BeliefHub-All", "BeliefHub-Invisible"]
COLS = ["delta_all_vs_match", "delta_invisible_vs_match"]


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.2,
        "axes.labelsize": 7.8,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.75,
    })

    df = pd.read_csv(DATA)
    if len(df) != 121:
        raise ValueError(f"Expected 121 robustness conditions, found {len(df)}")
    values = [100.0 * df[col].to_numpy(float) for col in COLS]

    fig, ax = plt.subplots(figsize=(2.45, 1.82))
    positions = np.array([1.0, 2.0])

    violins = ax.violinplot(
        values, positions=positions, widths=0.72,
        showmeans=False, showmedians=False, showextrema=False,
        bw_method=0.25,
    )
    for body, color in zip(violins["bodies"], COLORS):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.16)
        body.set_linewidth(0.8)

    bp = ax.boxplot(
        values, positions=positions, widths=0.22, patch_artist=True,
        showfliers=False, whis=(5, 95),
        medianprops={"color": "#202020", "linewidth": 1.0},
        whiskerprops={"color": "#333333", "linewidth": 0.7},
        capprops={"color": "#333333", "linewidth": 0.7},
        boxprops={"edgecolor": "#333333", "linewidth": 0.7},
    )
    for box, color in zip(bp["boxes"], COLORS):
        box.set_facecolor(color)
        box.set_alpha(0.48)

    rng = np.random.default_rng(2027)
    for pos, vals, color in zip(positions, values, COLORS):
        jitter = rng.uniform(-0.24, 0.24, size=len(vals))
        ax.scatter(
            pos + jitter, vals, s=4.5, facecolors="white", edgecolors=color,
            linewidths=0.38, alpha=0.68, zorder=3,
        )
        mean = float(np.mean(vals))
        ax.scatter(pos, mean, marker="D", s=18, color=color,
                   edgecolor="white", linewidth=0.45, zorder=5)
        ax.text(
            pos, 14.55, f"121/121 > 0\nmean +{mean:.2f} pp",
            ha="center", va="top", fontsize=5.8, color="#222222",
        )

    ax.axhline(0, color="#555555", linewidth=0.75, linestyle="--", zorder=1)
    ax.set_xlim(0.48, 2.52)
    ax.set_ylim(0, 15.0)
    ax.set_xticks(positions, LABELS)
    ax.set_ylabel("POL gain over MATCH (pp)", labelpad=1.5)
    ax.set_yticks(np.arange(0, 15.1, 3))
    ax.grid(axis="y", color="#D7DCE2", linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(
        axis="both", direction="in", top=True, right=True,
        width=0.7, length=2.4, pad=1.6,
    )
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#222222")
        ax.spines[side].set_linewidth(0.75)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.19, right=0.955, bottom=0.22, top=0.975)
    fig.savefig(OUT / "fig_cn6_121condition_gain_distribution.pdf",
                format="pdf", facecolor="white")
    fig.savefig(OUT / "fig_cn6_121condition_gain_distribution.png",
                dpi=400, facecolor="white")
    plt.close(fig)

    summary = OUT / "fig_cn6_121condition_gain_distribution_stats.txt"
    lines = []
    for label, vals in zip(LABELS, values):
        lines.append(
            f"{label}: mean={np.mean(vals):.4f} pp, "
            f"median={np.median(vals):.4f} pp, min={np.min(vals):.4f} pp, "
            f"max={np.max(vals):.4f} pp, positive={(vals > 0).sum()}/{len(vals)}"
        )
    inv_better = int(np.sum(values[1] > values[0]))
    lines.append(f"BeliefHub-Invisible > BeliefHub-All: {inv_better}/121 conditions")
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

