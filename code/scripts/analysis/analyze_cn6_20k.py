from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "analysis_input" / "outputs"
OUT = ROOT / "cn6_20k_analysis"
FIG = OUT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

METHODS = {
    "MATCH": {
        0: INPUT / "match_cn6_sr08_5k_seed0" / "training_log.csv",
        1: INPUT / "match_cn6_sr08_20k_seed1" / "training_log.csv",
        2: INPUT / "match_cn6_sr08_20k_seed2" / "training_log.csv",
    },
    "DB-MATCH-All": {
        0: INPUT / "db_match_cn6_sr08_5k_seed0" / "training_log.csv",
        1: INPUT / "db_match_cn6_all_sr08_20k_seed1" / "training_log.csv",
        2: INPUT / "db_match_cn6_all_sr08_20k_seed2" / "training_log.csv",
    },
    "DB-MATCH-Invisible": {
        0: INPUT / "db_match_cn6_invisible_sr08_20k_seed0" / "training_log.csv",
        1: INPUT / "db_match_cn6_invisible_sr08_20k_seed1" / "training_log.csv",
        2: INPUT / "db_match_cn6_invisible_sr08_20k_seed2" / "training_log.csv",
    },
}

COLORS = {
    "MATCH": "#3B6FB6",
    "DB-MATCH-All": "#D95F4A",
    "DB-MATCH-Invisible": "#2A9D74",
}

T95_DF2 = 4.3026527297


def finite_eval(df: pd.DataFrame) -> pd.DataFrame:
    result = df.loc[
        np.isfinite(df["evaluation_pol"].to_numpy(dtype=float)),
        [
            "episode",
            "evaluation_pol",
            "evaluation_return",
            "evaluation_delivery",
        ],
    ].copy()
    result["episode"] = result["episode"].astype(int)
    return result.sort_values("episode").reset_index(drop=True)


def mean_sd_ci(values: list[float]) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    ci = float(T95_DF2 * sd / math.sqrt(len(x)))
    return mean, sd, ci


frames: dict[str, dict[int, pd.DataFrame]] = {}
evals: dict[str, dict[int, pd.DataFrame]] = {}
seed_rows: list[dict[str, float | int | str]] = []

for method, paths in METHODS.items():
    frames[method] = {}
    evals[method] = {}
    for seed, path in paths.items():
        df = pd.read_csv(path)
        if len(df) != 20000 or int(df["episode"].iloc[-1]) != 20000:
            raise ValueError(f"Incomplete run: {path}")
        ev = finite_eval(df)
        if len(ev) != 200 or ev["episode"].tolist() != list(range(100, 20001, 100)):
            raise ValueError(f"Unexpected evaluation grid: {path}")
        late = ev.loc[ev["episode"] >= 18100]
        frames[method][seed] = df
        evals[method][seed] = ev
        seed_rows.append(
            {
                "method": method,
                "seed": seed,
                "final_eval_pol": float(ev["evaluation_pol"].iloc[-1]),
                "last20_eval_pol_mean": float(late["evaluation_pol"].mean()),
                "last20_eval_pol_sd_over_time": float(late["evaluation_pol"].std(ddof=1)),
                "best_eval_pol": float(ev["evaluation_pol"].max()),
                "normalized_pol_auc": float(
                    np.trapezoid(
                        ev["evaluation_pol"].to_numpy(),
                        ev["episode"].to_numpy(),
                    )
                    / (ev["episode"].iloc[-1] - ev["episode"].iloc[0])
                ),
                "last20_eval_return_mean": float(late["evaluation_return"].mean()),
                "final_eval_return": float(ev["evaluation_return"].iloc[-1]),
                "last20_eval_delivery_mean": float(late["evaluation_delivery"].mean()),
            }
        )

seed_df = pd.DataFrame(seed_rows)
seed_df.to_csv(OUT / "per_seed_metrics.csv", index=False)

group_rows: list[dict[str, float | str]] = []
for method in METHODS:
    sub = seed_df.loc[seed_df["method"] == method]
    row: dict[str, float | str] = {"method": method}
    for metric in [
        "final_eval_pol",
        "last20_eval_pol_mean",
        "best_eval_pol",
        "normalized_pol_auc",
        "last20_eval_return_mean",
        "last20_eval_delivery_mean",
    ]:
        mean, sd, ci = mean_sd_ci(sub[metric].astype(float).tolist())
        row[f"{metric}_mean"] = mean
        row[f"{metric}_sd"] = sd
        row[f"{metric}_ci95_halfwidth"] = ci
    group_rows.append(row)

group_df = pd.DataFrame(group_rows)
group_df.to_csv(OUT / "aggregate_metrics.csv", index=False)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.grid": True,
        "grid.alpha": 0.22,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Cross-seed evaluation POL curves. A five-evaluation rolling mean is applied
# independently to each seed before aggregation. Shading is a t-based 95% CI
# across the three seeds, clipped to the valid POL interval.
fig, ax = plt.subplots(figsize=(7.2, 4.5))
curve_rows: list[dict[str, float | int | str]] = []
for method in METHODS:
    x = evals[method][0]["episode"].to_numpy()
    ys = []
    for seed in range(3):
        y = (
            evals[method][seed]["evaluation_pol"]
            .rolling(5, min_periods=1)
            .mean()
            .to_numpy()
        )
        ys.append(y)
    arr = np.vstack(ys)
    mean = arr.mean(axis=0)
    sd = arr.std(axis=0, ddof=1)
    ci = T95_DF2 * sd / math.sqrt(3)
    ax.plot(x, mean, label=method, color=COLORS[method], linewidth=2.1)
    ax.fill_between(
        x,
        np.clip(mean - ci, 0, 1),
        np.clip(mean + ci, 0, 1),
        color=COLORS[method],
        alpha=0.12,
        linewidth=0,
    )
    for episode, m, half in zip(x, mean, ci):
        curve_rows.append(
            {
                "method": method,
                "episode": int(episode),
                "smoothed_mean_pol": float(m),
                "ci95_halfwidth": float(half),
            }
        )
ax.set_title("CN6 evaluation POL during training")
ax.set_xlabel("Training episode")
ax.set_ylabel("Evaluation POL")
ax.set_xlim(100, 20000)
ax.set_ylim(0, 0.4)
ax.legend(frameon=False, ncol=1, loc="upper left")
ax.text(
    0.99,
    0.02,
    "5-evaluation rolling mean; shaded 95% CI across 3 seeds",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=8,
    color="#555555",
)
save_figure(fig, "evaluation_pol_curve")
pd.DataFrame(curve_rows).to_csv(OUT / "evaluation_pol_curve_data.csv", index=False)

# Late-training POL bar chart.
fig, ax = plt.subplots(figsize=(7.0, 4.3))
names = list(METHODS)
means = [
    float(
        group_df.loc[
            group_df["method"] == m, "last20_eval_pol_mean_mean"
        ].iloc[0]
    )
    for m in names
]
sds = [
    float(
        group_df.loc[
            group_df["method"] == m, "last20_eval_pol_mean_sd"
        ].iloc[0]
    )
    for m in names
]
bars = ax.bar(
    names,
    means,
    yerr=sds,
    capsize=5,
    color=[COLORS[m] for m in names],
    alpha=0.9,
)
for bar, value in zip(bars, means):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.008,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )
ax.set_title("Late-training evaluation POL")
ax.set_ylabel("Mean POL over episodes 18.1K-20K")
ax.set_ylim(0, max(np.asarray(means) + np.asarray(sds)) * 1.28)
ax.text(
    0.99,
    0.02,
    "Bars: mean across 3 seeds; error bars: between-seed SD",
    transform=ax.transAxes,
    ha="right",
    fontsize=8,
    color="#555555",
)
save_figure(fig, "late_training_pol")

# Normalized AUC comparison.
fig, ax = plt.subplots(figsize=(7.0, 4.3))
auc_means = [
    float(
        group_df.loc[
            group_df["method"] == m, "normalized_pol_auc_mean"
        ].iloc[0]
    )
    for m in names
]
auc_sds = [
    float(
        group_df.loc[
            group_df["method"] == m, "normalized_pol_auc_sd"
        ].iloc[0]
    )
    for m in names
]
bars = ax.bar(
    names,
    auc_means,
    yerr=auc_sds,
    capsize=5,
    color=[COLORS[m] for m in names],
    alpha=0.9,
)
for bar, value in zip(bars, auc_means):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.005,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )
ax.set_title("Sample efficiency: normalized evaluation POL AUC")
ax.set_ylabel("Normalized AUC (100-20K episodes)")
ax.set_ylim(0, max(np.asarray(auc_means) + np.asarray(auc_sds)) * 1.28)
ax.text(
    0.99,
    0.02,
    "Higher is better; error bars show between-seed SD",
    transform=ax.transAxes,
    ha="right",
    fontsize=8,
    color="#555555",
)
save_figure(fig, "normalized_pol_auc")

# Per-seed late POL exposes run-to-run instability.
fig, ax = plt.subplots(figsize=(7.2, 4.5))
x = np.arange(3)
width = 0.24
for idx, method in enumerate(names):
    sub = (
        seed_df.loc[seed_df["method"] == method]
        .sort_values("seed")["last20_eval_pol_mean"]
        .to_numpy()
    )
    ax.bar(
        x + (idx - 1) * width,
        sub,
        width,
        label=method,
        color=COLORS[method],
        alpha=0.9,
    )
ax.set_xticks(x, ["Seed 0", "Seed 1", "Seed 2"])
ax.set_ylabel("Mean evaluation POL, last 20 evaluations")
ax.set_title("Seed-level stability")
ax.legend(frameon=False, ncol=3, loc="upper center")
ax.set_ylim(0, max(seed_df["last20_eval_pol_mean"]) * 1.25)
save_figure(fig, "seed_stability")

# Belief diagnostics for the two DB variants, using last 2K episodes.
belief_rows: list[dict[str, float | int | str]] = []
for method in ["DB-MATCH-All", "DB-MATCH-Invisible"]:
    for seed in range(3):
        df = frames[method][seed]
        late = df.loc[df["episode"] >= 18001]
        belief_rows.append(
            {
                "method": method,
                "seed": seed,
                "belief_rmse": float(late["belief_rmse"].mean()),
                "belief_gate": float(late["belief_gate"].mean()),
                "belief_variance": float(late["belief_variance"].mean()),
                "belief_supervised_fraction": float(
                    late["belief_supervised_fraction"].mean()
                ),
            }
        )
belief_df = pd.DataFrame(belief_rows)
belief_df.to_csv(OUT / "belief_metrics_last2k.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
for ax, metric, ylabel in [
    (axes[0], "belief_rmse", "Belief RMSE"),
    (axes[1], "belief_gate", "Mean belief gate"),
]:
    for idx, method in enumerate(["DB-MATCH-All", "DB-MATCH-Invisible"]):
        values = (
            belief_df.loc[belief_df["method"] == method]
            .sort_values("seed")[metric]
            .to_numpy()
        )
        ax.bar(
            np.arange(3) + (idx - 0.5) * 0.34,
            values,
            width=0.34,
            label=method,
            color=COLORS[method],
            alpha=0.9,
        )
    ax.set_xticks(np.arange(3), ["Seed 0", "Seed 1", "Seed 2"])
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " over last 2K episodes")
axes[0].legend(frameon=False, loc="upper left")
fig.tight_layout()
save_figure(fig, "belief_diagnostics")


def fmt_pm(mean: float, sd: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} +/- {sd:.{digits}f}"


summary_table: list[list[str]] = [
    [
        "Method",
        "Late POL",
        "Final POL",
        "Norm. AUC",
        "Late return",
    ]
]
for _, row in group_df.iterrows():
    summary_table.append(
        [
            str(row["method"]),
            fmt_pm(
                float(row["last20_eval_pol_mean_mean"]),
                float(row["last20_eval_pol_mean_sd"]),
            ),
            fmt_pm(
                float(row["final_eval_pol_mean"]),
                float(row["final_eval_pol_sd"]),
            ),
            fmt_pm(
                float(row["normalized_pol_auc_mean"]),
                float(row["normalized_pol_auc_sd"]),
            ),
            fmt_pm(
                float(row["last20_eval_return_mean_mean"]),
                float(row["last20_eval_return_mean_sd"]),
                2,
            ),
        ]
    )

match_late = float(
    group_df.loc[
        group_df["method"] == "MATCH", "last20_eval_pol_mean_mean"
    ].iloc[0]
)
all_late = float(
    group_df.loc[
        group_df["method"] == "DB-MATCH-All", "last20_eval_pol_mean_mean"
    ].iloc[0]
)
inv_late = float(
    group_df.loc[
        group_df["method"] == "DB-MATCH-Invisible",
        "last20_eval_pol_mean_mean",
    ].iloc[0]
)
all_gain = all_late - match_late
inv_gain = inv_late - match_late

latex_lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{CN6 results at sight range $0.8$ and packet loss $0.3$ (mean $\pm$ standard deviation over three seeds). Late metrics average the final 20 evaluations, i.e., episodes 18.1K--20K.}",
    r"\label{tab:cn6_20k_comparison}",
    r"\begin{tabular}{lcccc}",
    r"\toprule",
    r"Method & Late POL $\uparrow$ & Final POL $\uparrow$ & POL-AUC $\uparrow$ & Late return $\uparrow$ \\",
    r"\midrule",
]
for _, row in group_df.iterrows():
    latex_lines.append(
        "{} & ${:.3f} \\pm {:.3f}$ & ${:.3f} \\pm {:.3f}$ & "
        "${:.3f} \\pm {:.3f}$ & ${:.2f} \\pm {:.2f}$ \\\\".format(
            str(row["method"]).replace("-", r"\mbox{-}"),
            float(row["last20_eval_pol_mean_mean"]),
            float(row["last20_eval_pol_mean_sd"]),
            float(row["final_eval_pol_mean"]),
            float(row["final_eval_pol_sd"]),
            float(row["normalized_pol_auc_mean"]),
            float(row["normalized_pol_auc_sd"]),
            float(row["last20_eval_return_mean_mean"]),
            float(row["last20_eval_return_mean_sd"]),
        )
    )
latex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(OUT / "cn6_20k_table.tex").write_text(
    "\n".join(latex_lines) + "\n", encoding="utf-8"
)

md = [
    "# CN6 20K comparison",
    "",
    "Configuration: 6 agents/6 landmarks, sight range 0.8, packet loss 30%, "
    "20,000 training episodes, evaluation every 100 episodes with 20 episodes "
    "per evaluation, three training seeds per method.",
    "",
    "## Main result",
    "",
    f"DB-MATCH-All is the strongest and most stable method. Its late-training "
    f"POL is {all_late:.3f}, an absolute improvement of {all_gain:.3f} over "
    f"MATCH ({match_late:.3f}). DB-MATCH-Invisible reaches {inv_late:.3f}, "
    f"an absolute improvement of {inv_gain:.3f} over MATCH.",
    "",
    "The conclusion should remain provisional until all nine final checkpoints "
    "are independently evaluated on the same larger test-seed set. The training "
    "log evaluates only 20 episodes at each checkpoint.",
    "",
    "## Aggregate metrics",
    "",
    "| Method | Late POL | Final POL | Normalized POL-AUC | Late return |",
    "|---|---:|---:|---:|---:|",
]
for row in summary_table[1:]:
    md.append("| " + " | ".join(row) + " |")
md += [
    "",
    "Late POL and late return average the final 20 evaluations (episodes "
    "18.1K-20K). Aggregate entries are mean +/- between-seed standard deviation.",
    "",
    "## Interpretation",
    "",
    "- DB-MATCH-All has both the highest late POL and the highest normalized "
    "POL-AUC, so its gain is not limited to a single final checkpoint.",
    "- MATCH is highly seed-sensitive because seed 0 collapses after the 5K "
    "resume, while seeds 1 and 2 remain materially stronger.",
    "- DB-MATCH-Invisible improves over MATCH on average, but its seed 2 late "
    "performance is weaker, so it is less stable than DB-MATCH-All.",
    "- Packet delivery is essentially controlled and equal across methods "
    "(about 0.70), so the performance gap is not explained by easier "
    "communication realizations.",
    "- Negative belief_loss values are not automatically an error if the "
    "implemented objective is a Gaussian negative log-likelihood with learned "
    "variance; RMSE, gate, and downstream POL are safer diagnostics.",
    "",
    "## Reporting recommendation",
    "",
    "Use late-training POL as the main training-log result, normalized POL-AUC "
    "as the sample-efficiency result, and per-seed plots to disclose stability. "
    "Do not use the best single evaluation as the primary number. For the final "
    "paper result, run 200-500 independent evaluation episodes for every final "
    "checkpoint using identical evaluation seeds.",
]
(OUT / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(18 * mm, 12 * mm, "CN6 20K comparison")
    canvas.drawRightString(192 * mm, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="CenteredTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#183B56"),
        spaceAfter=10,
    )
)
styles.add(
    ParagraphStyle(
        name="SmallNote",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#555555"),
    )
)
doc = SimpleDocTemplate(
    str(OUT / "cn6_20k_comparison_report.pdf"),
    pagesize=A4,
    rightMargin=18 * mm,
    leftMargin=18 * mm,
    topMargin=16 * mm,
    bottomMargin=18 * mm,
)
story = [
    Paragraph("CN6 20K Method Comparison", styles["CenteredTitle"]),
    Paragraph(
        "MATCH vs. DB-MATCH-All vs. DB-MATCH-Invisible",
        ParagraphStyle(
            "Subtitle",
            parent=styles["Heading2"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"),
            spaceAfter=12,
        ),
    ),
    Paragraph(
        "Setup: 6 agents / 6 landmarks, sight range 0.8, packet loss 30%, "
        "20K training episodes, 3 seeds per method.",
        styles["BodyText"],
    ),
    Spacer(1, 8),
    Paragraph("Main finding", styles["Heading2"]),
    Paragraph(
        f"<b>DB-MATCH-All is the strongest and most stable variant.</b> "
        f"Its late-training POL is {all_late:.3f}, compared with "
        f"{match_late:.3f} for MATCH and {inv_late:.3f} for "
        "DB-MATCH-Invisible. The gain is also visible in normalized POL-AUC, "
        "so it is not driven by one lucky final evaluation.",
        styles["BodyText"],
    ),
    Spacer(1, 10),
]

table = Table(summary_table, colWidths=[42 * mm, 31 * mm, 31 * mm, 31 * mm, 31 * mm])
table.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#183B56")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.3),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F8")]),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8C4CC")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]
    )
)
story += [
    table,
    Spacer(1, 5),
    Paragraph(
        "Late metrics average the final 20 evaluations (episodes 18.1K-20K). "
        "Values are mean +/- between-seed standard deviation.",
        styles["SmallNote"],
    ),
    Spacer(1, 10),
    Image(str(FIG / "evaluation_pol_curve.png"), width=170 * mm, height=106 * mm),
    PageBreak(),
    Paragraph("Performance and stability", styles["Heading1"]),
    Image(str(FIG / "late_training_pol.png"), width=160 * mm, height=98 * mm),
    Spacer(1, 5),
    Image(str(FIG / "seed_stability.png"), width=160 * mm, height=100 * mm),
    PageBreak(),
    Paragraph("Sample efficiency and belief diagnostics", styles["Heading1"]),
    Image(str(FIG / "normalized_pol_auc.png"), width=160 * mm, height=98 * mm),
    Spacer(1, 4),
    Image(str(FIG / "belief_diagnostics.png"), width=170 * mm, height=76 * mm),
    PageBreak(),
    Paragraph("Interpretation and next step", styles["Heading1"]),
    Paragraph(
        "<b>1. DB-MATCH-All wins the training-log comparison.</b> It has the "
        "highest late POL and normalized AUC, and all three seeds remain useful "
        "near 20K episodes.",
        styles["BodyText"],
    ),
    Spacer(1, 8),
    Paragraph(
        "<b>2. MATCH has a reproducibility problem.</b> Seed 0 collapses after "
        "the 5K resume, while seeds 1 and 2 are materially stronger. The mean "
        "therefore understates the healthier MATCH runs but correctly exposes "
        "run-to-run risk.",
        styles["BodyText"],
    ),
    Spacer(1, 8),
    Paragraph(
        "<b>3. Invisible-only supervision is not the best current design.</b> "
        "It improves on MATCH on average, but seed 2 remains weak. All-entity "
        "supervision is more consistent in this experiment.",
        styles["BodyText"],
    ),
    Spacer(1, 8),
    Paragraph(
        "<b>4. Final-paper evaluation is still required.</b> Each logged "
        "evaluation contains only 20 episodes. Evaluate all nine final "
        "checkpoints on the same 200-500 test seeds, then report mean POL, "
        "return, and full-occupancy rate.",
        styles["BodyText"],
    ),
    Spacer(1, 12),
    Paragraph(
        "Caution: with only three training seeds, t-based 95% confidence "
        "intervals are wide. The report therefore emphasizes between-seed "
        "standard deviation and seed-level values instead of claiming formal "
        "statistical significance.",
        styles["SmallNote"],
    ),
]
doc.build(story, onFirstPage=footer, onLaterPages=footer)

print(group_df.to_string(index=False))
print(f"\nWrote analysis to {OUT}")

