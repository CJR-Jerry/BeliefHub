# BeliefHub

Implementation of **BeliefHub: Visibility-Aware Belief Reconstruction for
Communication-Robust Multi-Agent Reinforcement Learning**.

BeliefHub augments the MATCH/QMIX communication backbone with local recurrent
entity beliefs. Each observer-entity pair maintains a GRU state, predicts a
Gaussian entity state, and contributes to an uncertainty-weighted local belief
message. A scalar reliability-aware gate fuses this message with the
personalized hub response.

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
packet-loss rate `0.3`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The experiments were run with PyTorch 2.5.1+cu121, Gymnasium 0.29.1, and
PettingZoo 1.24.3.

## Training

MATCH on CN6:

```bash
python train_match_cn6.py --episodes 100000 --seed 0 \
  --sight-range 0.8 --packet-loss 0.3
```

BeliefHub-All on CN6:

```bash
python train_match_cn6.py --episodes 100000 --seed 0 \
  --sight-range 0.8 --packet-loss 0.3 \
  --use-local-belief --belief-supervision all \
  --belief-hidden-dim 32 --belief-loss-weight 0.01
```

BeliefHub-Invisible on CN6:

```bash
python train_match_cn6.py --episodes 100000 --seed 0 \
  --sight-range 0.8 --packet-loss 0.3 \
  --use-local-belief --belief-supervision invisible \
  --belief-hidden-dim 32 --belief-loss-weight 0.01
```

Use `train_match_cn5.py` with the same flags for CN5.

## Ablations

Append `--disable-belief-fusion` for NoFusion, append
`--disable-belief-history` for NoHistory, or set
`--belief-loss-weight 0` for the no-supervision ablation.

## Evaluation

```bash
python evaluate_match_cn6.py \
  --checkpoint outputs/EXPERIMENT/checkpoint_final.pt \
  --episodes 500
```

## Repository structure

```text
src/envs/       Cooperative Navigation and entity adapters
src/models/     MATCH, BeliefHub, QMIX, and MASIA network modules
src/learners/   TD, hub reconstruction, and belief objectives
src/runners/    Episode collection and communication simulation
src/data/       Episode replay buffers
src/losses/     Reconstruction losses
tests/          Model and pipeline tests
scripts/        Multi-seed, evaluation, analysis, and plotting utilities
```

## Status

This package consolidates the complete retained training implementation used
throughout the project. Checkpoints and large experiment outputs are excluded.
Run a final end-to-end reproduction pass before creating a public archival tag.
