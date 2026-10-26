# BeliefHub

Implementation of **BeliefHub: Visibility-Aware Belief Reconstruction for
Communication-Robust Multi-Agent Reinforcement Learning**.

BeliefHub augments the MATCH/QMIX communication backbone with local recurrent
entity beliefs. Each observer-entity pair maintains a GRU state, predicts a
Gaussian entity state, and contributes to an uncertainty-weighted local belief
message. A scalar reliability-aware gate fuses this message with the
personalized hub response.

## Paper results

The following values are transcribed from the submitted manuscript, not from a
new reproduction run of this repository.

| Method | CN5 POL | CN6 POL | CN6 return |
| --- | ---: | ---: | ---: |
| QMIX | 0.385 ± 0.044 | 0.379 ± 0.036 | -27.996 ± 0.835 |
| MASIA | 0.368 ± 0.018 | 0.452 ± 0.013 | -26.935 ± 0.733 |
| MATCH | 0.633 ± 0.044 | 0.759 ± 0.075 | -20.403 ± 1.65 |
| BeliefHub-All | 0.775 ± 0.032 | 0.808 ± 0.048 | -20.204 ± 0.891 |
| BeliefHub-Invisible | 0.785 ± 0.042 | 0.841 ± 0.028 | -19.640 ± 0.596 |

Values are mean ± standard deviation across five training seeds. CN5 uses the
last ten training-time evaluations; main CN6 results use 20 warm-up episodes
and 200 test episodes per final checkpoint. Fusion ablations use endpoints of
seven-evaluation smoothed training curves. These summaries have different
evaluation protocols and should not be treated as interchangeable.

Invisible improves POL over MATCH by **15.2 percentage points on CN5** and
**8.2 percentage points on CN6** (approximately 24.0% and 10.8% relative gains).

## Included methods

- MATCH
- BeliefHub-All and BeliefHub-Invisible
- NoFusion-All and NoFusion-Invisible
- NoHistory and `lambda_B = 0` ablations
- QMIX and MASIA model/learner components used by the comparison pipeline

## Environment

The paper uses MPE Cooperative Navigation with 25 steps per episode.

| Setting | Agents | Landmarks | Observation | Global state | Actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| CN5 | 5 | 5 | 30 | 150 | 5 |
| CN6 | 6 | 6 | 36 | 216 | 5 |

The principal robustness setting uses sight range `0.8` and independent uplink
packet-loss rate `0.3`. Loss affects observation uploads only; queries and hub
responses are treated as reliable. This is not a complete transport simulation.

POL is the fraction of landmarks occupied at the end of an episode, with an
occupancy distance threshold of `0.3`. Return is undiscounted episode team reward.
True entity targets and the QMIX mixer are used during centralized training;
entity predictions and belief recurrent states remain local during execution.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
# Install PyTorch separately before the remaining dependencies.
# Example matching the experimental CUDA 12.1 build:
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

The experiments were run with PyTorch 2.5.1+cu121, Gymnasium 0.29.1, and
PettingZoo 1.24.3.
Use a Python version supported by these pinned packages (for example Python 3.10).
For a CPU environment, install an appropriate CPU build of PyTorch instead.
The training/evaluation entry points select CUDA when available and otherwise CPU.
`requirements.txt` intentionally leaves PyTorch to the user; `setup.sh` is for
an existing CUDA environment and asserts that CUDA is available.

## Quick pipeline check

Run from the repository root after installing dependencies:

```bash
python train_match_cn6.py --episodes 2 --seed 0 \
  --batch-size 2 --buffer-capacity 2 --warmup-episodes 2 \
  --eval-interval 2 --eval-episodes 1 \
  --sight-range 0.8 --packet-loss 0.3 \
  --use-local-belief --belief-supervision invisible \
  --output-dir outputs/smoke_cn6_invisible

python evaluate_match_cn6.py \
  --checkpoint outputs/smoke_cn6_invisible/checkpoint_final.pt \
  --episodes 2 --output outputs/smoke_cn6_invisible/evaluation.json
```

This checks the training/checkpoint/evaluation path; it does not reproduce paper
performance. Model and pipeline checks are in `tests/`. For example, run the standalone
pipeline check with `python -m tests.test_match_training_pipeline`.

## Training

### CN6 training

CN6 uses hub reconstruction weight `0.1`. Set it explicitly in each command.

MATCH:

```bash
python train_match_cn6.py --episodes 100000 --seed 0 \
  --sight-range 0.8 --packet-loss 0.3 \
  --reconstruction-weight 0.1 \
  --output-dir outputs/cn6_match_seed0
```

BeliefHub-All:

```bash
python train_match_cn6.py --episodes 100000 --seed 0 \
  --sight-range 0.8 --packet-loss 0.3 \
  --reconstruction-weight 0.1 \
  --use-local-belief --belief-supervision all \
  --belief-hidden-dim 32 --belief-loss-weight 0.01 \
  --output-dir outputs/cn6_all_seed0
```

BeliefHub-Invisible:

```bash
python train_match_cn6.py --episodes 100000 --seed 0 \
  --sight-range 0.8 --packet-loss 0.3 \
  --reconstruction-weight 0.1 \
  --use-local-belief --belief-supervision invisible \
  --belief-hidden-dim 32 --belief-loss-weight 0.01 \
  --output-dir outputs/cn6_invisible_seed0
```

### CN5 training

CN5 uses hub reconstruction weight `0.01`, whereas CN6 uses `0.1`, as recorded
in the retained training launch scripts and logs. Set this parameter explicitly
when running CN5. The hub reconstruction weight (`--reconstruction-weight`)
and local belief loss weight (`--belief-loss-weight`) are separate parameters.

MATCH:

```bash
python train_match_cn5.py --episodes 100000 --seed 0 \
  --sight-range 0.8 --packet-loss 0.3 \
  --reconstruction-weight 0.01 \
  --output-dir outputs/cn5_match_seed0
```

BeliefHub-All:

```bash
python train_match_cn5.py --episodes 100000 --seed 0 \
  --sight-range 0.8 --packet-loss 0.3 \
  --reconstruction-weight 0.01 \
  --use-local-belief --belief-supervision all \
  --belief-hidden-dim 32 --belief-loss-weight 0.01 \
  --output-dir outputs/cn5_all_seed0
```

BeliefHub-Invisible:

```bash
python train_match_cn5.py --episodes 100000 --seed 0 \
  --sight-range 0.8 --packet-loss 0.3 \
  --reconstruction-weight 0.01 \
  --use-local-belief --belief-supervision invisible \
  --belief-hidden-dim 32 --belief-loss-weight 0.01 \
  --output-dir outputs/cn5_invisible_seed0
```

### Seeds, outputs, and shared settings

Repeat each configuration with training seeds `0`, `1`, `2`, `3`, and `4`.
For each run, change both `--seed` and the output directory suffix.
Always set `--output-dir`: the default BeliefHub directory does not distinguish
All, Invisible, or the ablations, so reusing it can overwrite checkpoints and
mix training logs. Use a fresh output directory for each new run.

Each run writes `training_log.csv`, `checkpoint_latest.pt`,
`checkpoint_best.pt`, and `checkpoint_final.pt`. Use the final checkpoint when
following the main CN6 protocol; best-checkpoint evaluation is a different
selection rule. Training can resume with `--resume PATH`; keep the architecture
and experimental flags consistent with the original run.

Common training settings include learning rate `5e-4`, discount `0.99`,
and batch size `32`. The hub reconstruction weight is `0.01` for CN5 and `0.1`
for CN6; the local belief loss weight is `0.01` for the full BeliefHub variants.
The entry-point default sight range is unlimited and default training length
is 5,000 episodes; explicitly pass `--sight-range 0.8` and `--episodes 100000`
for the paper setting.

## Ablations

Append `--disable-belief-fusion` for NoFusion, append
`--disable-belief-history` for NoHistory, or set
`--belief-loss-weight 0` for the no-supervision ablation.

All ablation flags must be combined with `--use-local-belief` and the intended
`--belief-supervision all` or `invisible` setting. Assign each run a separate
output directory.

- NoFusion retains belief training/supervision but uses the original hub message
  for action selection.
- NoHistory resets only the local belief recurrent state at every step; MATCH
  agent history and hub history remain enabled.
- `lambda_B = 0` removes the auxiliary belief loss; it does not remove the
  belief branch or its reinforcement-learning gradient path.

NoHistory and zero-weight supervision are additional implementation options;
they are not rows in the submitted manuscript's fusion-ablation table.

## Evaluation

```bash
python evaluate_match_cn6.py \
  --checkpoint outputs/cn6_invisible_seed0/checkpoint_final.pt \
  --episodes 500 \
  --output outputs/cn6_invisible_seed0/evaluation_500.json
```

Use `evaluate_match_cn5.py` for CN5. The evaluator restores belief architecture,
fusion/history flags, and sight range from checkpoint arguments. It uses the
saved packet-loss rate unless `--packet-loss` overrides it.

The command above is a standalone 500-episode evaluation. The current evaluator
does not expose a separate 20-episode warm-up option, so this command alone does
not implement the manuscript's exact main CN6 evaluation protocol.

For an execution-time loss sweep on a CN6 checkpoint:

```bash
for loss in 0.00 0.30 0.50 0.70 0.90 0.99; do
  python evaluate_match_cn6.py \
    --checkpoint outputs/cn6_invisible_seed0/checkpoint_final.pt \
    --episodes 500 --seed 20000 --packet-loss "$loss" \
    --output "outputs/cn6_invisible_seed0/eval_loss_${loss}.json"
done
```

Use a unique JSON output for each condition; otherwise the default
`evaluation.json` is overwritten. JSON `std_pol` and `pol_ci95` describe
episodes for one checkpoint, not variability across training seeds. To compare
with five-seed paper tables, aggregate per-checkpoint means across five
independently trained seeds.

The current evaluator reads sight range from the checkpoint and has no
`--sight-range` override. The manuscript's full 11 × 11 sight–loss grid therefore
requires an additional evaluation entry point or extension; the loss sweep above
does not reproduce that grid.

## Implementation map

| Paper component | Implementation |
| --- | --- |
| Local entity-belief model | [belief_state.py](src/models/belief_state.py) |
| Agent utilities and belief fusion | [match_agent.py](src/models/match_agent.py) |
| Spatiotemporal information hub | [match_information_hub.py](src/models/match_information_hub.py) |
| TD, hub, and belief learning | [match_learner.py](src/learners/match_learner.py) |
| Visibility and entity observations | [entity_observation.py](src/envs/entity_observation.py) |
| Episode collection and uplink channel | [match_episode_runner.py](src/runners/match_episode_runner.py) |
| QMIX mixer | [qmix_mixer.py](src/models/qmix_mixer.py) |

## Repository structure

```text
src/envs/       Cooperative Navigation and entity adapters
src/models/     MATCH, BeliefHub, QMIX, and MASIA network modules
src/learners/   TD, hub reconstruction, and belief objectives
src/runners/    Episode collection and communication simulation
src/data/       Episode replay buffers
src/losses/     Reconstruction losses
tests/          Model and pipeline tests
scripts/        Retained experiment and analysis utilities
```

## Reproduction scope and available artifacts

- CN5/CN6 training and evaluation entry points are included for MATCH and the
  BeliefHub variants.
- QMIX and MASIA model/learner components are included; standalone baseline
  training entry points are not included in this snapshot.
- Checkpoints, original per-seed logs, and raw sight–loss grid results are not
  distributed in this snapshot. The result table above records manuscript
  values rather than independently verified outputs of this snapshot.
- Retained analysis utilities may contain project-specific input paths and
  assumptions; inspect them before use. They are not a single-command
  reproduction pipeline.
- SMAC-related source files and CN8 exploratory utilities are outside the
  submitted paper's CN5/CN6 evidence.

## Status

This package consolidates the complete retained training implementation used
throughout the project. Checkpoints and large experiment outputs are excluded.
Run a final end-to-end reproduction pass before creating a public archival tag.

