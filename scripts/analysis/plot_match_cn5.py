from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path("/workspace/scratch/e420e907051d")
LOG_ROOT = ROOT / "work" / "full_logs3" / "outputs"
OUT = ROOT / "MATCH_CN5_350k_figures"
OUT.mkdir(parents=True, exist_ok=True)

MAX_EPISODE = 350_000
N_SEEDS = 5
BIN_WIDTH = 2_000
T_95_DF4 = 2.776

COLORS = {
    "blue": "#3568A8",
    "orange": "#E07A3F",
    "green": "#3E8E7E",
    "red": "#C74E4E",
    "purple": "#7B61A8",
    "gray": "#6B7280",
}
SEED_COLORS = ["#8FB3D9", "#E8A77D", "#7DB6A8", "#D98D8D", "#A99AC7"]
EPISODE_FORMATTER = FuncFormatter(
    lambda value, _: "0" if value == 0 else f"{int(value / 1000)}k"
)

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_seed(seed: int) -> pd.DataFrame:
    original = pd.read_csv(
        LOG_ROOT / f"match_cn5_10m_seed{seed}" / "training_log.csv"
    )
    resumed = pd.read_csv(
        LOG_ROOT / f"match_cn5_10m_resumed_seed{seed}" / "training_log.csv"
    )
    original["segment_priority"] = 0
    resumed["segment_priority"] = 1
    merged = pd.concat([original, resumed], ignore_index=True)
    merged = (
        merged.sort_values(["episode", "segment_priority"])
        .drop_duplicates("episode", keep="last")
        .query("episode <= @MAX_EPISODE")
        .sort_values("episode")
        .drop(columns="segment_priority")
        .reset_index(drop=True)
    )
    merged["seed"] = seed
    return merged


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=320)
    plt.close(fig)


def add_steps_axis(ax: plt.Axes) -> None:
    sec = ax.secondary_xaxis(
        "top", functions=(lambda e: e * 25 / 1e6, lambda m: m * 1e6 / 25)
    )
    sec.set_xlabel("Environment steps (million)")
    sec.tick_params(labelsize=7)


def format_episode_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(EPISODE_FORMATTER)


def ci_frame(wide: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    mean = wide.mean(axis=1)
    sd = wide.std(axis=1, ddof=1)
    n = wide.notna().sum(axis=1)
    half = T_95_DF4 * sd / np.sqrt(n)
    return mean, mean - half, mean + half


def plot_mean_ci(
    ax: plt.Axes,
    wide: pd.DataFrame,
    color: str,
    label: str,
    show_seeds: bool = False,
) -> None:
    if show_seeds:
        for i, col in enumerate(wide.columns):
            ax.plot(
                wide.index,
                wide[col],
                lw=0.75,
                alpha=0.48,
                color=SEED_COLORS[i],
                label=f"Seed {col}",
            )
    mean, lo, hi = ci_frame(wide)
    ax.fill_between(mean.index, lo, hi, color=color, alpha=0.18, linewidth=0)
    ax.plot(mean.index, mean, color=color, lw=2.0, label=label)


def eval_wide(frames: list[pd.DataFrame], metric: str) -> pd.DataFrame:
    parts = []
    for seed, df in enumerate(frames):
        x = df.loc[df[metric].notna(), ["episode", metric]].copy()
        x["seed"] = seed
        parts.append(x)
    long = pd.concat(parts, ignore_index=True)
    return long.pivot(index="episode", columns="seed", values=metric).sort_index()


def binned_wide(
    frames: list[pd.DataFrame], metric: str, bin_width: int = BIN_WIDTH
) -> pd.DataFrame:
    parts = []
    for seed, df in enumerate(frames):
        x = df.loc[df[metric].notna(), ["episode", metric]].copy()
        x["episode_bin"] = ((x["episode"] - 1) // bin_width + 1) * bin_width
        agg = x.groupby("episode_bin", as_index=False)[metric].mean()
        agg["seed"] = seed
        parts.append(agg)
    long = pd.concat(parts, ignore_index=True)
    return long.pivot(index="episode_bin", columns="seed", values=metric).sort_index()


def final_window_stats(wide: pd.DataFrame, start: int = 330_000) -> pd.DataFrame:
    window = wide.loc[(wide.index >= start) & (wide.index <= MAX_EPISODE)]
    rows = []
    for seed in wide.columns:
        values = window[seed].dropna()
        rows.append(
            {
                "seed": int(seed),
                "window_start_episode": start,
                "window_end_episode": MAX_EPISODE,
                "evaluation_points": int(len(values)),
                "mean_evaluation_pol": float(values.mean()),
                "std_evaluation_pol": float(values.std(ddof=1)),
                "best_evaluation_pol": float(values.max()),
                "last_evaluation_pol": float(values.iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


frames = [load_seed(seed) for seed in range(N_SEEDS)]

# 1. Evaluation performance: the primary result figure.
eval_pol = eval_wide(frames, "evaluation_pol")
eval_return = eval_wide(frames, "evaluation_return")

fig, axes = plt.subplots(2, 1, figsize=(7.1, 5.8), sharex=True)
plot_mean_ci(
    axes[0], eval_pol, COLORS["blue"], "5-seed mean ± 95% CI", show_seeds=True
)
axes[0].set_ylabel("Evaluation POL")
axes[0].set_ylim(0, 1.04)
axes[0].set_title("(a) Evaluation landmark occupancy")
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles[-1:] + handles[:5], labels[-1:] + labels[:5], ncol=3)

plot_mean_ci(
    axes[1], eval_return, COLORS["orange"], "5-seed mean ± 95% CI"
)
axes[1].set_ylabel("Evaluation return")
axes[1].set_xlabel("Episode")
axes[1].set_title("(b) Evaluation episodic return")
axes[1].set_xlim(0, MAX_EPISODE)
format_episode_axis(axes[1])
axes[1].legend(loc="lower right")
add_steps_axis(axes[0])
fig.suptitle("MATCH on CN5: evaluation performance up to 350k episodes", y=1.02)
fig.tight_layout()
save_figure(fig, "fig1_evaluation_performance")

# 2. Training dynamics: per-episode metrics aggregated in 2k-episode bins.
train_pol = binned_wide(frames, "train_pol")
train_return = binned_wide(frames, "train_return")

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
plot_mean_ci(axes[0], train_pol, COLORS["green"], "Mean ± 95% CI")
axes[0].set(
    title="(a) Training POL", xlabel="Episode", ylabel="Training POL", xlim=(0, MAX_EPISODE), ylim=(0, 1.04)
)
axes[0].legend(loc="lower right")
format_episode_axis(axes[0])
plot_mean_ci(axes[1], train_return, COLORS["orange"], "Mean ± 95% CI")
axes[1].set(
    title="(b) Training episodic return",
    xlabel="Episode",
    ylabel="Training return",
    xlim=(0, MAX_EPISODE),
)
axes[1].legend(loc="lower right")
format_episode_axis(axes[1])
fig.suptitle("Training dynamics (2k-episode bin means)", y=1.04)
fig.tight_layout()
save_figure(fig, "fig2_training_dynamics")

# 3. Learning losses: use a log scale because early and late magnitudes differ.
loss_metrics = [
    ("td_loss", "TD loss", COLORS["blue"]),
    ("state_loss", "State reconstruction loss", COLORS["green"]),
    ("position_loss", "Position loss", COLORS["purple"]),
    ("attribute_loss", "Attribute loss", COLORS["red"]),
]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
for ax, (metric, title, color) in zip(axes.flat, loss_metrics):
    wide = binned_wide(frames, metric)
    plot_mean_ci(ax, wide, color, "Mean ± 95% CI")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.set_xlim(0, MAX_EPISODE)
    ax.set_ylabel("Loss (log scale)")
    ax.legend(loc="upper right")
    format_episode_axis(ax)
for ax in axes[-1]:
    ax.set_xlabel("Episode")
fig.suptitle("Optimization and reconstruction losses (2k-episode bin means)", y=1.01)
fig.tight_layout()
save_figure(fig, "fig3_loss_convergence")

# 4. Exploration and communication channel diagnostics.
epsilon = binned_wide(frames, "epsilon")
delivery = binned_wide(frames, "packet_success_rate")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
plot_mean_ci(axes[0], epsilon, COLORS["purple"], "Mean ± 95% CI")
axes[0].set(
    title="(a) Exploration schedule",
    xlabel="Episode",
    ylabel=r"$\epsilon$",
    xlim=(0, MAX_EPISODE),
    ylim=(0, 1.03),
)
axes[0].legend()
format_episode_axis(axes[0])
plot_mean_ci(axes[1], delivery, COLORS["green"], "Observed mean ± 95% CI")
axes[1].axhline(0.70, color=COLORS["gray"], ls="--", lw=1.2, label="Expected success = 0.70")
axes[1].set(
    title="(b) Packet delivery process",
    xlabel="Episode",
    ylabel="Packet success rate",
    xlim=(0, MAX_EPISODE),
    ylim=(0.64, 0.76),
)
axes[1].legend(loc="lower right")
format_episode_axis(axes[1])
fig.suptitle("Exploration decay and 30% packet-loss setting", y=1.04)
fig.tight_layout()
save_figure(fig, "fig4_exploration_and_communication")

# 5. Seed stability: milestone heatmap plus final-window summary.
milestones = np.arange(50_000, MAX_EPISODE + 1, 50_000)
heat = pd.DataFrame(index=range(N_SEEDS), columns=milestones, dtype=float)
for seed in range(N_SEEDS):
    s = eval_pol[seed].dropna()
    for m in milestones:
        nearest = s.index[np.argmin(np.abs(s.index.to_numpy() - m))]
        heat.loc[seed, m] = s.loc[nearest]

final_stats = final_window_stats(eval_pol)
fig, axes = plt.subplots(
    1, 2, figsize=(7.2, 3.25), gridspec_kw={"width_ratios": [1.65, 1.0]}
)
im = axes[0].imshow(heat.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="YlGnBu")
axes[0].set_xticks(range(len(milestones)), [f"{m//1000}k" for m in milestones])
axes[0].set_yticks(range(N_SEEDS), [f"Seed {s}" for s in range(N_SEEDS)])
axes[0].set_xlabel("Episode milestone")
axes[0].set_title("(a) Evaluation POL across seeds")
for i in range(N_SEEDS):
    for j in range(len(milestones)):
        v = heat.iloc[i, j]
        axes[0].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                     color="white" if v > 0.58 else "black")
cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
cb.set_label("Evaluation POL")

means = final_stats["mean_evaluation_pol"].to_numpy()
sds = final_stats["std_evaluation_pol"].to_numpy()
axes[1].bar(range(N_SEEDS), means, yerr=sds, color=SEED_COLORS, capsize=3)
axes[1].set_xticks(range(N_SEEDS), [f"S{s}" for s in range(N_SEEDS)])
axes[1].set_ylim(0, 1.04)
axes[1].set_ylabel("Evaluation POL")
axes[1].set_title("(b) Mean over 330k–350k")
for i, v in enumerate(means):
    axes[1].text(i, min(v + sds[i] + 0.035, 1.01), f"{v:.2f}", ha="center", fontsize=7)
fig.suptitle("Cross-seed stability and late-training performance", y=1.02)
fig.tight_layout()
save_figure(fig, "fig5_seed_stability")

# Export a compact numerical companion table.
mean_final = means.mean()
sd_final = means.std(ddof=1)
ci_final = T_95_DF4 * sd_final / math.sqrt(N_SEEDS)
overall = pd.DataFrame(
    [
        {
            "metric": "cross_seed_mean_of_330k_350k_mean_evaluation_pol",
            "value": mean_final,
        },
        {"metric": "cross_seed_sd", "value": sd_final},
        {"metric": "cross_seed_95ci_half_width", "value": ci_final},
        {"metric": "lower_95ci", "value": mean_final - ci_final},
        {"metric": "upper_95ci", "value": mean_final + ci_final},
    ]
)
final_stats.to_csv(OUT / "final_window_seed_statistics.csv", index=False)
overall.to_csv(OUT / "overall_summary_statistics.csv", index=False)

merged_rows = []
for seed, df in enumerate(frames):
    merged_rows.append(
        {
            "seed": seed,
            "first_episode": int(df.episode.min()),
            "last_episode_used": int(df.episode.max()),
            "rows_after_merge": int(len(df)),
            "evaluation_points": int(df.evaluation_pol.notna().sum()),
        }
    )
pd.DataFrame(merged_rows).to_csv(OUT / "merged_log_coverage.csv", index=False)

manifest = {
    "maximum_episode_used": MAX_EPISODE,
    "steps_per_episode": 25,
    "maximum_environment_steps": MAX_EPISODE * 25,
    "seed_count": N_SEEDS,
    "training_curve_bin_width_episodes": BIN_WIDTH,
    "evaluation_interval_episodes": 2000,
    "confidence_interval": "two-sided 95% Student-t CI across 5 seeds (df=4)",
    "merge_policy": "resumed segment wins on duplicate episode numbers",
    "late_window": "330000-350000 inclusive",
}
(OUT / "plot_methodology.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)

print(OUT)
print(final_stats.to_string(index=False))
print(overall.to_string(index=False))
