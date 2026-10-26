import argparse
import csv
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch

from src.data.episode_replay_buffer import EpisodeReplayBuffer
from src.envs.limited_sight_cooperative_navigation import LimitedSightCooperativeNavigation
from src.learners.qmix_learner import QMIXLearner
from src.models.agent_rnn import AgentRNN
from src.models.qmix_mixer import QMixer
from src.runners.episode_runner import EpisodeRunner


def linear_epsilon(
    episode,
    start=1.0,
    end=0.05,
    decay_episodes=3000,
):
    progress = min(episode / decay_episodes, 1.0)
    return start + progress * (end - start)


def evaluate(runner, n_episodes):
    returns = []
    pols = []

    for episode in range(n_episodes):
        _, statistics = runner.run(
            epsilon=0.0,
            seed=100000 + episode,
        )

        returns.append(statistics["episode_return"])
        pols.append(statistics["pol"])

    return float(np.mean(returns)), float(np.mean(pols))


def save_checkpoint(
    path,
    episode,
    agent_network,
    mixer,
    optimizer,
    arguments,
):
    torch.save(
        {
            "episode": episode,
            "agent_network": agent_network.state_dict(),
            "mixer": mixer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "arguments": vars(arguments),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-capacity", type=int, default=1000)
    parser.add_argument("--warmup-episodes", type=int, default=32)
    parser.add_argument("--target-update", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--epsilon-decay", type=int, default=3000)
    parser.add_argument("--output-dir", type=str, default="outputs/qmix_cn5")
    parser.add_argument("--sight-range", type=float, default=0.8)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = LimitedSightCooperativeNavigation(
        n_agents=6,
        max_cycles=25,
        occupancy_radius=0.3,
        sight_range=args.sight_range,
    )

    env.reset(seed=args.seed)

    first_agent = env.possible_agents[0]
    observation_dim = env.observation_space(first_agent).shape[0]
    n_actions = env.action_space(first_agent).n
    state_dim = env.env.state().size

    agent_network = AgentRNN(
        observation_dim=observation_dim,
        n_actions=n_actions,
        n_agents=6,
        hidden_dim=64,
    ).to(device)

    mixer = QMixer(
        n_agents=6,
        state_dim=state_dim,
        mixing_hidden_dim=32,
        hypernet_hidden_dim=64,
    ).to(device)

    learner = QMIXLearner(
        agent_network=agent_network,
        mixer=mixer,
        learning_rate=5e-4,
        gamma=0.99,
        grad_clip=10.0,
    )

    runner = EpisodeRunner(
        env=env,
        agent_network=agent_network,
        device=device,
    )

    replay_buffer = EpisodeReplayBuffer(
        capacity=args.buffer_capacity,
        max_steps=25,
        n_agents=6,
        observation_dim=observation_dim,
        state_dim=state_dim,
    )

    csv_path = output_dir / "training_log.csv"
    checkpoint_path = output_dir / "checkpoint_latest.pt"
    best_checkpoint_path = output_dir / "checkpoint_best.pt"


    recent_returns = deque(maxlen=100)
    recent_pols = deque(maxlen=100)

    best_evaluation_pol = -1.0
    latest_loss = float("nan")

    print("=" * 60)
    print("QMIX Cooperative Navigation Training")
    print("=" * 60)
    print("Device:", device)
    print("Agents:", 6)
    print("Observation dimension:", observation_dim)
    print("State dimension:", state_dim)
    print("Actions:", n_actions)
    print("Episodes:", args.episodes)
    print("Sight range:", args.sight_range)
    print("Output:", output_dir)
    print("=" * 60)

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "episode",
                "epsilon",
                "train_return",
                "train_pol",
                "loss",
                "evaluation_return",
                "evaluation_pol",
            ]
        )

        for episode in range(1, args.episodes + 1):
            epsilon = linear_epsilon(
                episode=episode,
                start=1.0,
                end=0.05,
                decay_episodes=args.epsilon_decay,
            )

            episode_data, statistics = runner.run(
                epsilon=epsilon,
                seed=args.seed + episode,
            )

            replay_buffer.insert(episode_data)

            recent_returns.append(
                statistics["episode_return"]
            )
            recent_pols.append(statistics["pol"])

            if (
                len(replay_buffer) >= args.warmup_episodes
                and len(replay_buffer) >= args.batch_size
            ):
                batch = replay_buffer.sample(
                    batch_size=args.batch_size,
                    device=device,
                )

                training_statistics = learner.train_batch(batch)
                latest_loss = training_statistics["loss"]

            if episode % args.target_update == 0:
                learner.update_targets()

            evaluation_return = float("nan")
            evaluation_pol = float("nan")

            if episode % args.eval_interval == 0:
                evaluation_return, evaluation_pol = evaluate(
                    runner,
                    args.eval_episodes,
                )

                save_checkpoint(
                    checkpoint_path,
                    episode,
                    agent_network,
                    mixer,
                    learner.optimizer,
                    args,
                )

                if evaluation_pol > best_evaluation_pol:
                    best_evaluation_pol = evaluation_pol

                    save_checkpoint(
                        best_checkpoint_path,
                        episode,
                        agent_network,
                        mixer,
                        learner.optimizer,
                        args,
                    )

                print(
                    f"episode={episode:5d} "
                    f"epsilon={epsilon:.3f} "
                    f"recent_return={np.mean(recent_returns):.3f} "
                    f"recent_POL={np.mean(recent_pols):.3f} "
                    f"eval_POL={evaluation_pol:.3f} "
                    f"loss={latest_loss:.4f}"
                )

            writer.writerow(
                [
                    episode,
                    epsilon,
                    statistics["episode_return"],
                    statistics["pol"],
                    latest_loss,
                    evaluation_return,
                    evaluation_pol,
                ]
            )

            csv_file.flush()

    save_checkpoint(
        output_dir / "checkpoint_final.pt",
        args.episodes,
        agent_network,
        mixer,
        learner.optimizer,
        args,
    )

    env.close()

    print("=" * 60)
    print("Training completed.")
    print("Best evaluation POL:", best_evaluation_pol)
    print("Results saved to:", output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()