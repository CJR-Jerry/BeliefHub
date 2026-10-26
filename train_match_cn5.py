import argparse
import csv
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch

from src.data.match_episode_replay_buffer import (
    MATCHEpisodeReplayBuffer,
)
from src.envs.cooperative_navigation import (
    CooperativeNavigation,
)
from src.envs.entity_observation import (
    MPEEntityObservationAdapter,
)
from src.learners.match_learner import (
    MATCHLearner,
)
from src.models.match_agent import MATCHAgent
from src.models.match_information_hub import (
    MATCHInformationHub,
)
from src.models.qmix_mixer import QMixer
from src.runners.match_episode_runner import (
    MATCHEpisodeRunner,
)


def linear_epsilon(
    episode,
    start=1.0,
    end=0.05,
    decay_episodes=3000,
):
    progress = min(
        episode / decay_episodes,
        1.0,
    )

    return start + progress * (
        end - start
    )


def evaluate(
    runner,
    match_agent,
    information_hub,
    n_episodes,
    seed_start=100000,
):
    previous_agent_mode = (
        match_agent.training
    )

    previous_hub_mode = (
        information_hub.training
    )

    match_agent.eval()
    information_hub.eval()

    returns = []
    pols = []
    delivery_rates = []

    for episode in range(n_episodes):
        _, statistics = runner.run(
            epsilon=0.0,
            seed=seed_start + episode,
        )

        returns.append(
            statistics["episode_return"]
        )

        pols.append(
            statistics["pol"]
        )

        delivery_rates.append(
            statistics[
                "packet_success_rate"
            ]
        )

    if previous_agent_mode:
        match_agent.train()

    if previous_hub_mode:
        information_hub.train()

    return (
        float(np.mean(returns)),
        float(np.mean(pols)),
        float(np.mean(delivery_rates)),
    )


def save_checkpoint(
    path,
    episode,
    match_agent,
    information_hub,
    mixer,
    learner,
    best_evaluation_pol,
    arguments,
):
    torch.save(
        {
            "episode": episode,

            "match_agent": (
                match_agent.state_dict()
            ),

            "information_hub": (
                information_hub.state_dict()
            ),

            "mixer": mixer.state_dict(),

            "target_match_agent": (
                learner
                .target_match_agent
                .state_dict()
            ),

            "target_information_hub": (
                learner
                .target_information_hub
                .state_dict()
            ),

            "target_mixer": (
                learner
                .target_mixer
                .state_dict()
            ),

            "optimizer": (
                learner.optimizer.state_dict()
            ),

            "best_evaluation_pol": (
                best_evaluation_pol
            ),

            "arguments": vars(arguments),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from a checkpoint",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5000,
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--buffer-capacity",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--warmup-episodes",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--target-update",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--epsilon-decay",
        type=int,
        default=3000,
    )

    parser.add_argument(
        "--packet-loss",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--sight-range",
        type=float,
        default=-1.0,
        help=(
            "Negative means unlimited sight, "
            "matching the standard CN setting."
        ),
    )

    parser.add_argument(
        "--reconstruction-weight",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--use-local-belief",
        action="store_true",
        help="Enable recurrent agent-level probabilistic belief.",
    )

    parser.add_argument(
        "--belief-hidden-dim",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--belief-loss-weight",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--belief-supervision",
        choices=["all", "invisible"],
        default="all",
        help=(
            "'all' is required for unlimited sight; 'invisible' isolates "
            "missing-entity prediction in limited-sight experiments."
        ),
    )

    parser.add_argument(
        "--disable-belief-fusion",
        action="store_true",
        help=(
            "Train and supervise Local Belief, but force action selection "
            "to use the original Hub message only (DB-NoFusion)."
        ),
    )

    parser.add_argument(
        "--disable-belief-history",
        action="store_true",
        help=(
            "Reset only the Local Belief recurrent state at every step; "
            "the original MATCH agent and Hub histories remain enabled "
            "(DB-NoBeliefHistory)."
        ),
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    if not 0.0 <= args.packet_loss <= 1.0:
        raise ValueError(
            "packet-loss must be between 0 and 1."
        )

    if args.batch_size > args.buffer_capacity:
        raise ValueError(
            "batch-size cannot exceed "
            "buffer-capacity."
        )
    if args.belief_loss_weight < 0.0:
        raise ValueError(
            "belief-loss-weight must be non-negative."
        )
    if (
        args.disable_belief_fusion
        or args.disable_belief_history
    ) and not args.use_local_belief:
        raise ValueError(
            "Belief ablation flags require --use-local-belief."
        )

    if args.output_dir is None:
        method_name = (
            "db_match_local_belief_cn5"
            if args.use_local_belief
            else "match_cn5"
        )
        args.output_dir = f"outputs/{method_name}_seed{args.seed}"

    sight_range = (
        None
        if args.sight_range < 0.0
        else args.sight_range
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.seed
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    env = CooperativeNavigation(
        n_agents=5,
        max_cycles=25,
        occupancy_radius=0.3,
    )

    env.reset(seed=args.seed)

    first_agent = env.possible_agents[0]

    observation_dim = (
        env.observation_space(
            first_agent
        ).shape[0]
    )

    n_actions = (
        env.action_space(
            first_agent
        ).n
    )

    state_dim = env.env.state().size

    n_agents = 5
    n_landmarks = 5
    n_entities = n_agents + n_landmarks
    entity_observation_dim = 8
    entity_state_dim = 8
    max_steps = 25

    entity_adapter = (
        MPEEntityObservationAdapter(
            n_agents=n_agents,
            n_landmarks=n_landmarks,
            sight_range=sight_range,
        )
    )

    information_hub = (
        MATCHInformationHub(
            n_entities=n_entities,
            observation_dim=(
                entity_observation_dim
            ),
            embedding_dim=32,
            state_dim=entity_state_dim,
            history_length=4,
        ).to(device)
    )

    match_agent = MATCHAgent(
        n_agents=n_agents,
        n_entities=n_entities,
        entity_observation_dim=(
            entity_observation_dim
        ),
        hidden_dim=32,
        query_dim=32,
        reconstructed_entity_dim=32,
        n_actions=n_actions,
        use_local_belief=args.use_local_belief,
        belief_hidden_dim=args.belief_hidden_dim,
        belief_state_dim=entity_state_dim,
        fuse_local_belief=(
            not args.disable_belief_fusion
        ),
        use_belief_history=(
            not args.disable_belief_history
        ),
    ).to(device)


    mixer = QMixer(
        n_agents=n_agents,
        state_dim=state_dim,
        mixing_hidden_dim=32,
        hypernet_hidden_dim=64,
    ).to(device)

    learner = MATCHLearner(
        match_agent=match_agent,
        information_hub=information_hub,
        mixer=mixer,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        reconstruction_weight=(
            args.reconstruction_weight
        ),
        belief_loss_weight=args.belief_loss_weight,
        belief_supervision=args.belief_supervision,
        grad_clip=args.grad_clip,
    )

    runner = MATCHEpisodeRunner(
        env=env,
        match_agent=match_agent,
        information_hub=information_hub,
        entity_adapter=entity_adapter,
        device=device,
        packet_loss=args.packet_loss,
    )

    replay_buffer = (
        MATCHEpisodeReplayBuffer(
            capacity=args.buffer_capacity,
            max_steps=max_steps,
            n_agents=n_agents,
            observation_dim=observation_dim,
            state_dim=state_dim,
            n_entities=n_entities,
            entity_observation_dim=(
                entity_observation_dim
            ),
            entity_state_dim=(
                entity_state_dim
            ),
        )
    )

    csv_path = (
        output_dir / "training_log.csv"
    )

    checkpoint_path = (
        output_dir
        / "checkpoint_latest.pt"
    )

    best_checkpoint_path = (
        output_dir
        / "checkpoint_best.pt"
    )

    recent_returns = deque(
        maxlen=100
    )

    recent_pols = deque(
        maxlen=100
    )

    recent_delivery = deque(
        maxlen=100
    )

    best_evaluation_pol = -1.0
    start_episode = 1

    if args.resume is not None:
        resume_path = Path(args.resume)

        if not resume_path.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_path}"
            )

        print(f"Loading checkpoint: {resume_path}")

        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )
        checkpoint_uses_belief = bool(
            checkpoint.get("arguments", {}).get(
                "use_local_belief",
                False,
            )
        )
        if checkpoint_uses_belief != args.use_local_belief:
            raise ValueError(
                "Checkpoint architecture mismatch: pass "
                "--use-local-belief exactly when the checkpoint was "
                "trained with Local Belief."
            )
        saved_arguments = checkpoint.get("arguments", {})
        for flag_name in (
            "disable_belief_fusion",
            "disable_belief_history",
        ):
            saved_value = bool(saved_arguments.get(flag_name, False))
            current_value = bool(getattr(args, flag_name))
            if saved_value != current_value:
                option_name = "--" + flag_name.replace("_", "-")
                raise ValueError(
                    "Checkpoint ablation mismatch for "
                    f"{option_name}: checkpoint={saved_value}, "
                    f"current={current_value}."
                )

        required_keys = {
            "episode",
            "match_agent",
            "information_hub",
            "mixer",
            "target_match_agent",
            "target_information_hub",
            "target_mixer",
            "optimizer",
        }
        missing_keys = sorted(required_keys.difference(checkpoint))
        if missing_keys:
            raise KeyError(
                "Resume checkpoint is missing required keys: "
                + ", ".join(missing_keys)
            )

        match_agent.load_state_dict(
            checkpoint["match_agent"]
        )
        information_hub.load_state_dict(
            checkpoint["information_hub"]
        )
        mixer.load_state_dict(
            checkpoint["mixer"]
        )

        learner.target_match_agent.load_state_dict(
            checkpoint["target_match_agent"]
        )
        learner.target_information_hub.load_state_dict(
            checkpoint["target_information_hub"]
        )
        learner.target_mixer.load_state_dict(
            checkpoint["target_mixer"]
        )
        learner.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        completed_episode = int(
            checkpoint["episode"]
        )
        start_episode = completed_episode + 1
        best_evaluation_pol = float(
            checkpoint.get(
                "best_evaluation_pol",
                -1.0,
            )
        )

        print(
            f"Checkpoint episode: {completed_episode}"
        )
        print(
            f"Resume from episode: {start_episode}"
        )
        print(
            "Previous best evaluation POL: "
            f"{best_evaluation_pol:.4f}"
        )

        if start_episode > args.episodes:
            raise ValueError(
                f"Checkpoint is already at episode {completed_episode}, "
                f"but --episodes is {args.episodes}."
            )

    latest_training = {
        "loss": float("nan"),
        "td_loss": float("nan"),
        "state_loss": float("nan"),
        "position_loss": float("nan"),
        "attribute_loss": float("nan"),
        "belief_loss": float("nan"),
        "belief_rmse": float("nan"),
        "belief_variance": float("nan"),
        "belief_gate": float("nan"),
        "belief_supervised_fraction": float("nan"),
        "gradient_norm": float("nan"),
        "mean_total_q": float("nan"),
        "mean_target_q": float("nan"),
    }

    print("=" * 70)
    print(
        "MATCH Cooperative Navigation Training"
    )
    print("=" * 70)
    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(f"Agents/Landmarks: {n_agents}/{n_landmarks}")
    print(
        "Observation dimension:",
        observation_dim,
    )
    print("State dimension:", state_dim)
    print("Entity count:", n_entities)
    print(
        "Entity feature dimension:",
        entity_observation_dim,
    )
    print("Actions:", n_actions)
    print("Episodes:", args.episodes)
    print("Batch size:", args.batch_size)
    print(
        "Packet loss:",
        f"{args.packet_loss:.1%}",
    )
    print(
        "Sight range:",
        (
            "unlimited"
            if sight_range is None
            else sight_range
        ),
    )
    print(
        "Reconstruction weight:",
        args.reconstruction_weight,
    )
    print("Local belief:", args.use_local_belief)
    if args.use_local_belief:
        print("Belief hidden dimension:", args.belief_hidden_dim)
        print("Belief loss weight:", args.belief_loss_weight)
        print("Belief supervision:", args.belief_supervision)
        print(
            "Belief fusion:",
            not args.disable_belief_fusion,
        )
        print(
            "Belief recurrent history:",
            not args.disable_belief_history,
        )
    print("Output:", output_dir)
    print("=" * 70)

    csv_fields = [
        "episode",
        "epsilon",
        "train_return",
        "train_pol",
        "packet_success_rate",
        "loss",
        "td_loss",
        "state_loss",
        "position_loss",
        "attribute_loss",
        "belief_loss",
        "belief_rmse",
        "belief_variance",
        "belief_gate",
        "belief_supervised_fraction",
        "gradient_norm",
        "mean_total_q",
        "mean_target_q",
        "evaluation_return",
        "evaluation_pol",
        "evaluation_delivery",
    ]


    csv_already_exists = (
        csv_path.exists()
        and csv_path.stat().st_size > 0
    )

    with csv_path.open(
        "a",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=csv_fields,
        )

        if not csv_already_exists:
            writer.writeheader()

        for episode in range(
            start_episode,
            args.episodes + 1,
        ):
            epsilon = linear_epsilon(
                episode=episode,
                start=1.0,
                end=0.05,
                decay_episodes=(
                    args.epsilon_decay
                ),
            )

            (
                episode_data,
                statistics,
            ) = runner.run(
                epsilon=epsilon,
                seed=args.seed + episode,
            )

            replay_buffer.insert(
                episode_data
            )

            recent_returns.append(
                statistics[
                    "episode_return"
                ]
            )

            recent_pols.append(
                statistics["pol"]
            )

            recent_delivery.append(
                statistics[
                    "packet_success_rate"
                ]
            )

            if (
                len(replay_buffer)
                >= args.warmup_episodes
                and len(replay_buffer)
                >= args.batch_size
            ):
                batch = replay_buffer.sample(
                    batch_size=(
                        args.batch_size
                    ),
                    device=device,
                )

                latest_training = (
                    learner.train_batch(
                        batch
                    )
                )

            if (
                episode
                % args.target_update
                == 0
            ):
                learner.update_targets()

            evaluation_return = float("nan")
            evaluation_pol = float("nan")
            evaluation_delivery = float("nan")

            if (
                episode
                % args.eval_interval
                == 0
            ):
                (
                    evaluation_return,
                    evaluation_pol,
                    evaluation_delivery,
                ) = evaluate(
                    runner=runner,
                    match_agent=match_agent,
                    information_hub=(
                        information_hub
                    ),
                    n_episodes=(
                        args.eval_episodes
                    ),
                )

                save_checkpoint(
                    path=checkpoint_path,
                    episode=episode,
                    match_agent=match_agent,
                    information_hub=(
                        information_hub
                    ),
                    mixer=mixer,
                    learner=learner,
                    best_evaluation_pol=(
                        best_evaluation_pol
                    ),
                    arguments=args,
                )

                if (
                    evaluation_pol
                    > best_evaluation_pol
                ):
                    best_evaluation_pol = (
                        evaluation_pol
                    )

                    save_checkpoint(
                        path=(
                            best_checkpoint_path
                        ),
                        episode=episode,
                        match_agent=match_agent,
                        information_hub=(
                            information_hub
                        ),
                        mixer=mixer,
                        learner=learner,
                        best_evaluation_pol=(
                            best_evaluation_pol
                        ),
                        arguments=args,
                    )


                print(
                    f"episode={episode:5d} "
                    f"epsilon={epsilon:.3f} "
                    f"recent_return="
                    f"{np.mean(recent_returns):.3f} "
                    f"recent_POL="
                    f"{np.mean(recent_pols):.3f} "
                    f"delivery="
                    f"{np.mean(recent_delivery):.3f} "
                    f"eval_POL="
                    f"{evaluation_pol:.3f} "
                    f"td_loss="
                    f"{latest_training['td_loss']:.4f} "
                    f"state_loss="
                    f"{latest_training['state_loss']:.4f} "
                    f"belief_loss="
                    f"{latest_training['belief_loss']:.4f} "
                    f"belief_rmse="
                    f"{latest_training['belief_rmse']:.4f}"
                )

            row = {
                "episode": episode,
                "epsilon": epsilon,
                "train_return": (
                    statistics[
                        "episode_return"
                    ]
                ),
                "train_pol": (
                    statistics["pol"]
                ),
                "packet_success_rate": (
                    statistics[
                        "packet_success_rate"
                    ]
                ),
                "loss": (
                    latest_training["loss"]
                ),
                "td_loss": (
                    latest_training[
                        "td_loss"
                    ]
                ),
                "state_loss": (
                    latest_training[
                        "state_loss"
                    ]
                ),
                "position_loss": (
                    latest_training[
                        "position_loss"
                    ]
                ),
                "attribute_loss": (
                    latest_training[
                        "attribute_loss"
                    ]
                ),
                "belief_loss": latest_training["belief_loss"],
                "belief_rmse": latest_training["belief_rmse"],
                "belief_variance": latest_training[
                    "belief_variance"
                ],
                "belief_gate": latest_training["belief_gate"],
                "belief_supervised_fraction": latest_training[
                    "belief_supervised_fraction"
                ],
                "gradient_norm": (
                    latest_training[
                        "gradient_norm"
                    ]
                ),
                "mean_total_q": (
                    latest_training[
                        "mean_total_q"
                    ]
                ),
                "mean_target_q": (
                    latest_training[
                        "mean_target_q"
                    ]
                ),
                "evaluation_return": (
                    evaluation_return
                ),
                "evaluation_pol": (
                    evaluation_pol
                ),
                "evaluation_delivery": (
                    evaluation_delivery
                ),
            }

            writer.writerow(row)
            csv_file.flush()

    save_checkpoint(
        path=(
            output_dir
            / "checkpoint_final.pt"
        ),
        episode=args.episodes,
        match_agent=match_agent,
        information_hub=(
            information_hub
        ),
        mixer=mixer,
        learner=learner,
        best_evaluation_pol=(
            best_evaluation_pol
        ),
        arguments=args,
    )

    env.close()

    print("=" * 70)
    print("Training completed.")
    print(
        "Best evaluation POL:",
        best_evaluation_pol,
    )
    print(
        "Results saved to:",
        output_dir,
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
