# Reproducibility notes

## Scope

This update adds retained CN5/CN6 baseline entry points, the grid evaluator,
the original CUDA inference benchmark, and compact verification records.
It fixes the CN5 environment agent count to five. It does not bundle weights,
large training logs, or all archived episode CSVs.

The submitted paper's result table remains a transcription of the submission.
The independent CPU results below are separate verification results.

## Training parameters

Use the explicit MATCH/BeliefHub commands in README.md. The recovered checkpoint
arguments confirm hub reconstruction weight 0.01 for CN5 and 0.1 for CN6.
The full BeliefHub local belief loss weight is 0.01 in both settings.

All 50 archived main checkpoints record episode 100000. Seven CN6 checkpoints
record resumed training; their paths and arguments are retained in
`reproduction_results/checkpoint_manifest.json`. Fresh uninterrupted training
has not been demonstrated to recover the same trajectories or statistics.

QMIX and MASIA training examples (repeat seeds 0 through 4):

```bash
python train_qmix_cn5.py --episodes 100000 --seed 0 --sight-range 0.8 \
  --output-dir outputs/qmix_cn5_100k_seed0
python train_masia_cn5.py --episodes 100000 --seed 0 --sight-range 0.8 \
  --packet-loss 0.3 --output-dir outputs/masia_cn5_100k_seed0
python train_qmix_cn6.py --episodes 100000 --seed 0 --sight-range 0.8 \
  --output-dir outputs/qmix_cn6_100k_seed0
python train_masia_cn6.py --episodes 100000 --seed 0 --sight-range 0.8 \
  --packet-loss 0.3 --output-dir outputs/masia_cn6_100k_seed0
```

QMIX has no communication channel; do not add a packet-loss argument to it.
Use a fresh output directory for each run and change its seed suffix together
with `--seed`.

## Unified checkpoint evaluation

Run from the repository root. These commands require an archived final
checkpoint at the specified path, or a new checkpoint produced by training.

```bash
python evaluate_cn5_checkpoint.py --method qmix \
  --checkpoint outputs/qmix_cn5_100k_seed0/checkpoint_final.pt \
  --episodes 500 --seed 20270001 --device cpu \
  --output-dir outputs/eval_cn5_qmix_seed0

python evaluate_cn6_dense_grid_checkpoint.py --method db_invisible \
  --checkpoint outputs/db_match_cn6_invisible_100k_seed0/checkpoint_final.pt \
  --episodes 500 --seed 20270001 --sight-range 0.8 --packet-loss 0.3 \
  --device cpu --output-dir outputs/eval_cn6_invisible_seed0
```

Methods: `qmix`, `masia`, `match`, `db_all`, `db_invisible`.
The default checkpoint episode guard is 100000. For a short pipeline check,
set `--expected-checkpoint-episode` to the short checkpoint's actual episode.
JSON within-checkpoint standard deviations describe episode variation.
Paper table standard deviations describe variation across five training seeds.

## Main CN6 protocol

The retained CUDA benchmark performs 20 warm-up episodes at seeds 40000-40019,
followed by 200 test episodes at seeds 40020-40219, for each checkpoint:

```bash
PYTHONPATH=. python scripts/analysis/benchmark_cn6_inference.py \
  --method db_invisible \
  --checkpoint outputs/db_match_cn6_invisible_100k_seed0/checkpoint_final.pt \
  --warmup-episodes 20 --episodes 200 --seed 40000 \
  --output outputs/cn6_benchmark_invisible_seed0.json
```

This benchmark requires CUDA. Latency results cannot be reproduced on CPU.
For the independent CPU return/POL check, the same execution networks, warm-up
and test seeds, sight range 0.8, loss 0.3, and greedy actions were used, without
the CUDA timing calls. Every episode resets its recurrent states.

## Dense sight-loss grid

The original grid uses sights 0.2 through 1.2 in increments of 0.1; losses
0.0 through 0.9 in increments of 0.1, plus 0.99; three MATCH-family methods;
five training seeds; 500 test episodes per checkpoint and condition; base
evaluation seed 20270001.

The shell launcher expects final checkpoints in these directories:

```text
outputs/match_cn6_100k_seed{0,1,2,3,4}/
outputs/db_match_cn6_all_100k_seed{0,1,2,3,4}/
outputs/db_match_cn6_invisible_100k_seed{0,1,2,3,4}/
```

Once checkpoints are available, select the CUDA device IDs to use:

```bash
GPU_IDS_STR="0" bash run_cn6_dense_grid_500ep.sh
python plot_cn6_dense_grid.py
```

This launches 1815 evaluations totaling 907500 episodes. It is a full
evaluation sweep, not a short smoke test. The default launcher supports
one through seven GPU IDs and resumes completed conditions.

## Verification on 2026-10-26

- All 50 final checkpoints load strictly into their execution networks.
- CN5 last-ten-evaluation means and sample standard deviations agree with
  the manuscript at its displayed precision, for all five methods.
- All 1815 grid CSVs contain 500 episodes and agree with their JSON summary
  POL and return means. The supplied conditions cover all 121 pairs and
  five seeds per method.
- Mean/maximum grid POL gains over MATCH: All 0.05330468/0.09460000;
  Invisible 0.07973774/0.14133333. Both beat MATCH in all 121 conditions;
  Invisible beats All in 117 conditions.
- CN5/CN6 MATCH and BeliefHub short pipelines passed after the CN5 fix.
  The four QMIX/MASIA short training-and-evaluation pipelines also passed.
  Five belief tests and the standalone MATCH training pipeline passed.

Independent CPU evaluation of all 25 main CN6 checkpoints, each with 20
warm-up and 200 test episodes, produced:

| Method | Mean POL | Sample SD | Mean return | Sample SD |
| --- | ---: | ---: | ---: | ---: |
| QMIX | 0.379500 | 0.036451 | -27.995990 | 0.835051 |
| MASIA | 0.452000 | 0.012728 | -26.934609 | 0.732864 |
| MATCH | 0.758167 | 0.074527 | -20.398891 | 1.652797 |
| BeliefHub-All | 0.807833 | 0.048195 | -20.197376 | 0.890005 |
| BeliefHub-Invisible | 0.841500 | 0.026467 | -19.643638 | 0.594471 |

Environment: Python 3.12, PyTorch 2.5.1+cpu, NumPy 1.26.4,
Gymnasium 0.29.1, PettingZoo 1.24.3, pygame 2.6.1; one torch/BLAS thread
per evaluation process. Per-seed results are in `reproduction_results/`.

These CPU values are close to, but not identical to, the submitted CUDA
values. In a separate 20-episode comparison against archived CUDA grid
episodes, 18 POL values and 16 returns matched exactly; seeds and packet
counts matched for all 20 episodes. The precise cause of the remaining
numerical/action differences was not isolated.

This verification supports archived-checkpoint evaluation and archived
result aggregation. It does not establish statistical reproduction by
training all methods from scratch. Fusion-ablation rows and the original
GPU latency figures were not independently reproduced in this check.
