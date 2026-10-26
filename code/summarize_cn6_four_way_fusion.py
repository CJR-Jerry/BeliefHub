#!/usr/bin/env python3

import argparse
import csv
import json
import statistics
from pathlib import Path


METHODS = (
    ("db_all", "BeliefHub-All"),
    ("nofusion_all", "NoFusion-All"),
    ("db_invisible", "BeliefHub-Invisible"),
    ("nofusion_invisible", "NoFusion-Invisible"),
)


def mean_and_sample_std(values):
    return statistics.fmean(values), statistics.stdev(values)


def load_method(input_dir, key, expected_episodes, expected_test_seed):
    records = []
    for train_seed in range(5):
        path = input_dir / f"{key}_seed{train_seed}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation result: {path}")

        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)

        if int(record.get("test_episodes", -1)) != expected_episodes:
            raise ValueError(
                f"{path}: expected test_episodes={expected_episodes}, "
                f"got {record.get('test_episodes')}"
            )
        if int(record.get("test_seed", -1)) != expected_test_seed:
            raise ValueError(
                f"{path}: expected test_seed={expected_test_seed}, "
                f"got {record.get('test_seed')}"
            )
        if int(record.get("checkpoint_episode", -1)) != 100000:
            raise ValueError(
                f"{path}: expected checkpoint_episode=100000, "
                f"got {record.get('checkpoint_episode')}"
            )
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/cn6_four_way_fusion_eval_500ep"),
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--test-seed", type=int, default=20270001)
    args = parser.parse_args()

    args.input_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for key, label in METHODS:
        records = load_method(
            args.input_dir,
            key,
            args.episodes,
            args.test_seed,
        )
        pol_values = [float(item["mean_pol"]) for item in records]
        return_values = [float(item["mean_return"]) for item in records]
        occupancy_values = [
            float(item["full_occupancy_rate"]) for item in records
        ]

        pol_mean, pol_std = mean_and_sample_std(pol_values)
        return_mean, return_std = mean_and_sample_std(return_values)
        occupancy_mean, occupancy_std = mean_and_sample_std(occupancy_values)

        summary.append(
            {
                "key": key,
                "variant": label,
                "training_seeds": 5,
                "evaluation_episodes_per_seed": args.episodes,
                "mean_pol": pol_mean,
                "std_pol_across_seeds": pol_std,
                "mean_return": return_mean,
                "std_return_across_seeds": return_std,
                "mean_full_occupancy_rate": occupancy_mean,
                "std_full_occupancy_rate_across_seeds": occupancy_std,
                "per_seed_mean_pol": pol_values,
            }
        )

    by_key = {row["key"]: row for row in summary}
    fusion_effects = {
        "all": (
            by_key["db_all"]["mean_pol"]
            - by_key["nofusion_all"]["mean_pol"]
        ),
        "invisible": (
            by_key["db_invisible"]["mean_pol"]
            - by_key["nofusion_invisible"]["mean_pol"]
        ),
    }

    csv_path = args.input_dir / "cn6_four_way_fusion_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "variant",
            "mean_pol",
            "std_pol_across_seeds",
            "mean_return",
            "std_return_across_seeds",
            "mean_full_occupancy_rate",
            "std_full_occupancy_rate_across_seeds",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow({field: row[field] for field in fields})

    json_path = args.input_dir / "cn6_four_way_fusion_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "protocol": {
                    "checkpoint": "final at episode 100000",
                    "training_seeds": [0, 1, 2, 3, 4],
                    "evaluation_episodes_per_seed": args.episodes,
                    "shared_evaluation_seed": args.test_seed,
                    "packet_loss": 0.3,
                    "sight_range": 0.8,
                },
                "methods": summary,
                "fusion_effect_mean_pol": fusion_effects,
            },
            handle,
            indent=2,
        )

    tex_path = args.input_dir / "cn6_four_way_fusion_table.tex"
    latex_labels = {
        "db_all": r"\methodall",
        "nofusion_all": r"\quad NoFusion-All",
        "db_invisible": r"\methodinv",
        "nofusion_invisible": r"\quad NoFusion-Invisible",
    }
    lines = [
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Variant & CN6 POL $\uparrow$ \\",
        r"\midrule",
    ]
    for row in summary:
        label = latex_labels[row["key"]]
        value = row["mean_pol"]
        std = row["std_pol_across_seeds"]
        lines.append(f"{label} & ${value:.4f}\\pm{std:.4f}$ \\\\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("CN6 four-way fusion ablation (mean ± sample std across 5 seeds)")
    for row in summary:
        print(
            f"{row['variant']:<27} "
            f"POL={row['mean_pol']:.4f} ± "
            f"{row['std_pol_across_seeds']:.4f}"
        )
    print(f"Fusion gain under All:       {fusion_effects['all']:+.4f}")
    print(f"Fusion gain under Invisible: {fusion_effects['invisible']:+.4f}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {tex_path}")


if __name__ == "__main__":
    main()

