import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.envs.cooperative_navigation import (
    CooperativeNavigation,
)
from src.envs.entity_observation import (
    MPEEntityObservationAdapter,
)
from src.models.match_agent import MATCHAgent
from src.models.match_information_hub import (
    MATCHInformationHub,
)
from src.runners.match_episode_runner import (
    MATCHEpisodeRunner,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "outputs/match_cn5_seed0/"
            "checkpoint_best.pt"
        ),
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--packet-loss",
        type=float,
        default=None,
        help=(
            "If omitted, use the packet-loss rate "
            "stored in the checkpoint."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    training_arguments = checkpoint.get(
        "arguments",
        {},
    )

    training_packet_loss = (
        training_arguments.get(
            "packet_loss",
            0.3,
        )
    )

    packet_loss = (
        training_packet_loss
        if args.packet_loss is None
        else args.packet_loss
    )

    if not 0.0 <= packet_loss <= 1.0:
        raise ValueError(
            "packet-loss must be between 0 and 1."
        )

    saved_sight_range = (
        training_arguments.get(
            "sight_range",
            -1.0,
        )
    )

    sight_range = (
        None
        if saved_sight_range < 0.0
        else saved_sight_range
    )

    n_agents = 5
    n_landmarks = 5
    n_entities = n_agents + n_landmarks
    max_steps = 25

    env = CooperativeNavigation(
        n_agents=n_agents,
        max_cycles=max_steps,
        occupancy_radius=0.3,
    )

    env.reset(seed=args.seed)

    first_agent = env.possible_agents[0]

    n_actions = env.action_space(
        first_agent
    ).n

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
            observation_dim=8,
            embedding_dim=32,
            state_dim=8,
            history_length=4,
        ).to(device)
    )

    match_agent = MATCHAgent(
        n_agents=n_agents,
        n_entities=n_entities,
        entity_observation_dim=8,
        hidden_dim=32,
        query_dim=32,
        reconstructed_entity_dim=32,
        n_actions=n_actions,
        use_local_belief=bool(
            training_arguments.get(
                "use_local_belief",
                False,
            )
        ),
        belief_hidden_dim=int(
            training_arguments.get(
                "belief_hidden_dim",
                32,
            )
        ),
        belief_state_dim=8,
        fuse_local_belief=(
            not bool(
                training_arguments.get(
                    "disable_belief_fusion",
                    False,
                )
            )
        ),
        use_belief_history=(
            not bool(
                training_arguments.get(
                    "disable_belief_history",
                    False,
                )
            )
        ),
    ).to(device)

    match_agent.load_state_dict(
        checkpoint["match_agent"]
    )

    information_hub.load_state_dict(
        checkpoint["information_hub"]
    )

    match_agent.eval()
    information_hub.eval()

    runner = MATCHEpisodeRunner(
        env=env,
        match_agent=match_agent,
        information_hub=information_hub,
        entity_adapter=entity_adapter,
        device=device,
        packet_loss=packet_loss,
    )

    returns = []
    pols = []
    delivery_rates = []

    for episode in range(
        args.episodes
    ):
        _, statistics = runner.run(
            epsilon=0.0,
            seed=args.seed + episode,
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

    returns = np.asarray(
        returns,
        dtype=np.float32,
    )

    pols = np.asarray(
        pols,
        dtype=np.float32,
    )


    delivery_rates = np.asarray(
        delivery_rates,
        dtype=np.float32,
    )

    mean_pol = float(pols.mean())

    std_pol = float(
        pols.std(ddof=1)
        if args.episodes > 1
        else 0.0
    )

    pol_ci95 = float(
        1.96
        * std_pol
        / np.sqrt(args.episodes)
    )

    mean_return = float(
        returns.mean()
    )

    std_return = float(
        returns.std(ddof=1)
        if args.episodes > 1
        else 0.0
    )

    full_occupancy_rate = float(
        np.mean(pols >= 0.999)
    )

    rounded_pols = np.round(
        pols,
        1,
    )

    unique_values, counts = np.unique(
        rounded_pols,
        return_counts=True,
    )

    distribution = {
        str(float(value)): int(count)
        for value, count in zip(
            unique_values,
            counts,
        )
    }

    results = {
        "method": (
            "MATCH+LocalBelief"
            if match_agent.use_local_belief
            else "MATCH"
        ),
        "checkpoint": args.checkpoint,
        "checkpoint_episode": (
            checkpoint["episode"]
        ),
        "test_episodes": args.episodes,
        "test_seed": args.seed,
        "packet_loss": packet_loss,
        "sight_range": (
            "unlimited"
            if sight_range is None
            else sight_range
        ),
        "mean_return": mean_return,
        "std_return": std_return,
        "mean_pol": mean_pol,
        "std_pol": std_pol,
        "pol_ci95": pol_ci95,
        "full_occupancy_rate": (
            full_occupancy_rate
        ),
        "mean_packet_success_rate": float(
            delivery_rates.mean()
        ),
        "pol_distribution": distribution,
    }

    print("=" * 60)
    print(f"{results['method']} Formal Evaluation")
    print("=" * 60)
    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(
        "Checkpoint episode:",
        checkpoint["episode"],
    )
    print(
        "Test episodes:",
        args.episodes,
    )
    print(
        "Packet loss:",
        f"{packet_loss:.1%}",
    )
    print(
        f"Mean return: "
        f"{mean_return:.3f}"
    )
    print(
        f"Mean POL: "
        f"{mean_pol:.3f}"
    )
    print(
        f"POL 95% CI: "
        f"±{pol_ci95:.3f}"
    )
    print(
        f"Full occupancy rate: "
        f"{full_occupancy_rate:.1%}"
    )
    print(
        "Mean packet success rate:",
        f"{delivery_rates.mean():.3f}",
    )
    print(
        "POL distribution:",
        distribution,
    )
    print("=" * 60)

    if args.output is None:
        output_path = (
            Path(args.checkpoint).parent
            / "evaluation.json"
        )
    else:
        output_path = Path(
            args.output
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w"
    ) as output_file:
        json.dump(
            results,
            output_file,
            indent=2,
        )

    print("Saved to:", output_path)

    env.close()


if __name__ == "__main__":
    main()
