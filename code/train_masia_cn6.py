import argparse
import csv
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch

from src.data.masia_episode_replay_buffer import MASIAEpisodeReplayBuffer
from src.envs.limited_sight_cooperative_navigation import LimitedSightCooperativeNavigation
from src.learners.masia_learner import MASIALearner
from src.models.masia_agent import MASIAAgent
from src.models.masia_latent_model import MASIALatentModel
from src.models.qmix_mixer import QMixer
from src.runners.masia_episode_runner import MASIAEpisodeRunner


def linear_epsilon(episode, start=1.0, end=0.05, decay_episodes=3000):
    progress = min(episode / decay_episodes, 1.0)
    return start + progress * (end - start)


def evaluate(runner, n_episodes):
    returns, pols = [], []
    for episode in range(n_episodes):
        _, statistics = runner.run(epsilon=0.0, seed=100000 + episode)
        returns.append(statistics["episode_return"])
        pols.append(statistics["pol"])
    return float(np.mean(returns)), float(np.mean(pols))


def save_checkpoint(path, episode, agent, mixer, latent_model, optimizer, args):
    torch.save(
        {
            "episode": episode,
            "agent_network": agent.state_dict(),
            "mixer": mixer.state_dict(),
            "latent_model": latent_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "arguments": vars(args),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-capacity", type=int, default=1000)
    parser.add_argument("--warmup-episodes", type=int, default=32)
    parser.add_argument("--target-update", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--epsilon-decay", type=int, default=3000)
    parser.add_argument("--sight-range", type=float, default=0.8)
    parser.add_argument("--packet-loss", type=float, default=0.3)
    parser.add_argument("--repr-coef", type=float, default=1.0)
    parser.add_argument("--spr-coef", type=float, default=1.0)
    parser.add_argument("--reward-prediction-coef", type=float, default=1.0)
    parser.add_argument("--prediction-length", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="outputs/masia_cn6_100k_seed0")
    args = parser.parse_args()

    if not 0.0 <= args.packet_loss <= 1.0:
        raise ValueError("packet-loss must be between 0 and 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    # Prefer the public state method, while retaining compatibility with the
    # repository's current wrapper.
    state_dim = env.env.state().size
    agent = MASIAAgent(
        observation_dim=observation_dim,
        state_dim=state_dim,
        n_actions=n_actions,
        n_agents=6,
    ).to(device)
    mixer = QMixer(
        n_agents=6,
        state_dim=state_dim,
        mixing_hidden_dim=32,
        hypernet_hidden_dim=64,
    ).to(device)
    latent_model = MASIALatentModel(
        latent_dim=agent.latent_dim,
        n_actions=n_actions,
        n_agents=6,
    ).to(device)
    learner = MASIALearner(
        agent_network=agent,
        mixer=mixer,
        latent_model=latent_model,
        repr_coef=args.repr_coef,
        spr_coef=args.spr_coef,
        reward_prediction_coef=args.reward_prediction_coef,
        prediction_length=args.prediction_length,
    )
    runner = MASIAEpisodeRunner(
        env=env,
        agent_network=agent,
        device=device,
        packet_loss=args.packet_loss,
    )
    replay_buffer = MASIAEpisodeReplayBuffer(
        capacity=args.buffer_capacity,
        max_steps=25,
        n_agents=6,
        observation_dim=observation_dim,
        state_dim=state_dim,
    )

    csv_path = output_dir / "training_log.csv"
    latest_checkpoint = output_dir / "checkpoint_latest.pt"
    best_checkpoint = output_dir / "checkpoint_best.pt"
    recent_returns = deque(maxlen=100)
    recent_pols = deque(maxlen=100)
    best_evaluation_pol = -1.0
    latest = {name: float("nan") for name in (
        "loss", "td_loss", "representation_loss", "reconstruction_loss",
        "spr_loss", "reward_prediction_loss"
    )}

    print("=" * 70)
    print("MASIA Cooperative Navigation Training")
    print("Device:", device)
    print("Agents/Landmarks: 6/6")
    print("Observation dimension:", observation_dim)
    print("State dimension:", state_dim)
    print("Actions:", n_actions)
    print("Episodes:", args.episodes)
    print("Sight range:", args.sight_range)
    print("Packet loss:", f"{args.packet_loss:.1%}")
    print("Output:", output_dir)
    print("=" * 70)

    fieldnames = [
        "episode", "epsilon", "train_return", "train_pol", "loss",
        "td_loss", "representation_loss", "reconstruction_loss", "spr_loss",
        "reward_prediction_loss", "gradient_norm", "evaluation_return", "evaluation_pol"
    ]
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for episode in range(1, args.episodes + 1):
            epsilon = linear_epsilon(episode, decay_episodes=args.epsilon_decay)
            episode_data, statistics = runner.run(
                epsilon=epsilon, seed=args.seed + episode
            )
            replay_buffer.insert(episode_data)
            recent_returns.append(statistics["episode_return"])
            recent_pols.append(statistics["pol"])
            if len(replay_buffer) >= max(args.warmup_episodes, args.batch_size):
                batch = replay_buffer.sample(args.batch_size, device)
                latest = learner.train_batch(batch)
            if episode % args.target_update == 0:
                learner.update_targets()

            evaluation_return = float("nan")
            evaluation_pol = float("nan")
            if episode % args.eval_interval == 0:
                evaluation_return, evaluation_pol = evaluate(
                    runner, args.eval_episodes
                )
                save_checkpoint(
                    latest_checkpoint, episode, agent, mixer, latent_model,
                    learner.optimizer, args
                )
                if evaluation_pol > best_evaluation_pol:
                    best_evaluation_pol = evaluation_pol
                    save_checkpoint(
                        best_checkpoint, episode, agent, mixer, latent_model,
                        learner.optimizer, args
                    )
                print(
                    f"episode={episode:6d} epsilon={epsilon:.3f} "
                    f"recent_return={np.mean(recent_returns):.3f} "
                    f"recent_POL={np.mean(recent_pols):.3f} "
                    f"eval_POL={evaluation_pol:.3f} "
                    f"td={latest['td_loss']:.4f} "
                    f"repr={latest['representation_loss']:.4f}"
                )

            row = {
                "episode": episode,
                "epsilon": epsilon,
                "train_return": statistics["episode_return"],
                "train_pol": statistics["pol"],
                "evaluation_return": evaluation_return,
                "evaluation_pol": evaluation_pol,
            }
            row.update({name: latest[name] for name in latest})
            writer.writerow(row)
            csv_file.flush()

    save_checkpoint(
        output_dir / "checkpoint_final.pt", args.episodes, agent, mixer,
        latent_model, learner.optimizer, args
    )
    env.close()
    print("=" * 70)
    print("Training completed.")
    print("Best evaluation POL:", best_evaluation_pol)
    print("Results saved to:", output_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()

