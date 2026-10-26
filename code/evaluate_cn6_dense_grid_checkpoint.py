#!/usr/bin/env python3
"""Unified independent evaluator for five CN6 methods.

Supported methods:
    qmix, match, masia, db_all, db_invisible

The evaluator intentionally loads only execution-time networks. Mixers,
optimizers, and MASIA's auxiliary latent model are not used for decentralized
greedy execution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.envs.cooperative_navigation import CooperativeNavigation
from src.envs.entity_observation import MPEEntityObservationAdapter
from src.envs.limited_sight_cooperative_navigation import (
    LimitedSightCooperativeNavigation,
)
from src.models.agent_rnn import AgentRNN
from src.models.masia_agent import MASIAAgent
from src.models.match_agent import MATCHAgent
from src.models.match_information_hub import MATCHInformationHub
from src.runners.episode_runner import EpisodeRunner
from src.runners.masia_episode_runner import MASIAEpisodeRunner
from src.runners.match_episode_runner import MATCHEpisodeRunner


METHOD_LABELS = {
    "qmix": "QMIX",
    "match": "MATCH",
    "masia": "MASIA",
    "db_all": "DB-All",
    "db_invisible": "DB-Invisible",
}

N_AGENTS = 6
N_LANDMARKS = 6
MAX_STEPS = 25
OCCUPANCY_RADIUS = 0.3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one final CN6 checkpoint with a unified protocol."
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=tuple(METHOD_LABELS),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--packet-loss",
        type=float,
        default=None,
        help=(
            "Override the checkpoint packet-loss rate for communication "
            "methods. Omit for the main evaluation. Not applicable to QMIX."
        ),
    )
    parser.add_argument(
        "--sight-range",
        type=float,
        default=None,
        help="Override checkpoint sight range. Omit for the main evaluation.",
    )
    parser.add_argument(
        "--expected-checkpoint-episode",
        type=int,
        default=100000,
        help="Set to 0 to disable the checkpoint-episode guard.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.progress_interval <= 0:
        parser.error("--progress-interval must be positive")
    if args.packet_loss is not None and not 0.0 <= args.packet_loss <= 1.0:
        parser.error("--packet-loss must be in [0, 1]")
    if args.sight_range is not None and args.sight_range < 0.0:
        parser.error("--sight-range must be non-negative")
    return args


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def as_argument_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"Unsupported checkpoint arguments type: {type(value)!r}")


def atomic_json_dump(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def normal_ci95_half_width(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(1.96 * sample_std(values) / math.sqrt(len(values)))


def validate_checkpoint(
    method: str,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    expected_episode: int,
) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary")
    if "episode" not in checkpoint:
        raise KeyError("Checkpoint is missing 'episode'")

    checkpoint_episode = int(checkpoint["episode"])
    if expected_episode > 0 and checkpoint_episode != expected_episode:
        raise ValueError(
            f"Expected episode {expected_episode}, got {checkpoint_episode}: "
            f"{checkpoint_path}"
        )

    training_arguments = as_argument_dict(checkpoint.get("arguments", {}))
    configured_episodes = training_arguments.get("episodes")
    if configured_episodes is not None and int(configured_episodes) != checkpoint_episode:
        raise ValueError(
            "Checkpoint episode and saved training arguments disagree: "
            f"{checkpoint_episode} vs {configured_episodes}"
        )

    required_keys = {
        "qmix": {"agent_network"},
        "masia": {"agent_network", "latent_model"},
        "match": {"match_agent", "information_hub"},
        "db_all": {"match_agent", "information_hub"},
        "db_invisible": {"match_agent", "information_hub"},
    }[method]
    missing = sorted(required_keys.difference(checkpoint))
    if missing:
        raise KeyError(f"Checkpoint is missing required keys: {missing}")

    if method == "qmix" and "latent_model" in checkpoint:
        raise ValueError("MASIA-like checkpoint supplied for method=qmix")

    if method in {"match", "db_all", "db_invisible"}:
        use_local_belief = bool(training_arguments.get("use_local_belief", False))
        supervision = str(training_arguments.get("belief_supervision", "all"))
        if method == "match" and use_local_belief:
            raise ValueError("Local-belief checkpoint supplied for method=match")
        if method in {"db_all", "db_invisible"} and not use_local_belief:
            raise ValueError(f"Non-belief checkpoint supplied for method={method}")
        if method == "db_all" and supervision != "all":
            raise ValueError(f"Expected belief_supervision=all, got {supervision!r}")
        if method == "db_invisible" and supervision != "invisible":
            raise ValueError(
                f"Expected belief_supervision=invisible, got {supervision!r}"
            )

    return training_arguments


def resolve_sight_range(
    training_arguments: dict[str, Any],
    override: float | None,
) -> float:
    saved = float(training_arguments.get("sight_range", 0.8))
    sight_range = saved if override is None else float(override)
    if sight_range < 0.0:
        raise ValueError("Unlimited sight is not allowed in this CN6 protocol")
    return sight_range


def resolve_packet_loss(
    method: str,
    training_arguments: dict[str, Any],
    override: float | None,
) -> float | None:
    if method == "qmix":
        if override is not None:
            raise ValueError("QMIX has no explicit communication channel")
        return None
    saved = float(training_arguments.get("packet_loss", 0.3))
    packet_loss = saved if override is None else float(override)
    if not 0.0 <= packet_loss <= 1.0:
        raise ValueError("packet loss must be in [0, 1]")
    return packet_loss


def build_qmix_runner(
    checkpoint: dict[str, Any],
    device: torch.device,
    sight_range: float,
) -> tuple[LimitedSightCooperativeNavigation, EpisodeRunner]:
    env = LimitedSightCooperativeNavigation(
        n_agents=N_AGENTS,
        max_cycles=MAX_STEPS,
        occupancy_radius=OCCUPANCY_RADIUS,
        sight_range=sight_range,
    )
    env.reset(seed=0)
    first_agent = env.possible_agents[0]
    observation_dim = int(env.observation_space(first_agent).shape[0])
    n_actions = int(env.action_space(first_agent).n)

    state_dict = checkpoint["agent_network"]
    hidden_dim = int(state_dict["input_layer.weight"].shape[0])
    agent = AgentRNN(
        observation_dim=observation_dim,
        n_actions=n_actions,
        n_agents=N_AGENTS,
        hidden_dim=hidden_dim,
    ).to(device)
    agent.load_state_dict(state_dict, strict=True)
    agent.eval()
    runner = EpisodeRunner(env=env, agent_network=agent, device=device)
    return env, runner


def build_masia_runner(
    checkpoint: dict[str, Any],
    device: torch.device,
    sight_range: float,
    packet_loss: float,
) -> tuple[LimitedSightCooperativeNavigation, MASIAEpisodeRunner]:
    env = LimitedSightCooperativeNavigation(
        n_agents=N_AGENTS,
        max_cycles=MAX_STEPS,
        occupancy_radius=OCCUPANCY_RADIUS,
        sight_range=sight_range,
    )
    env.reset(seed=0)
    first_agent = env.possible_agents[0]
    observation_dim = int(env.observation_space(first_agent).shape[0])
    n_actions = int(env.action_space(first_agent).n)
    state_dim = int(env.env.state().size)

    agent = MASIAAgent(
        observation_dim=observation_dim,
        state_dim=state_dim,
        n_actions=n_actions,
        n_agents=N_AGENTS,
    ).to(device)
    agent.load_state_dict(checkpoint["agent_network"], strict=True)
    agent.eval()
    runner = MASIAEpisodeRunner(
        env=env,
        agent_network=agent,
        device=device,
        packet_loss=packet_loss,
    )
    return env, runner


def build_match_runner(
    checkpoint: dict[str, Any],
    training_arguments: dict[str, Any],
    device: torch.device,
    sight_range: float,
    packet_loss: float,
) -> tuple[CooperativeNavigation, MATCHEpisodeRunner]:
    env = CooperativeNavigation(
        n_agents=N_AGENTS,
        max_cycles=MAX_STEPS,
        occupancy_radius=OCCUPANCY_RADIUS,
    )
    env.reset(seed=0)
    first_agent = env.possible_agents[0]
    n_actions = int(env.action_space(first_agent).n)
    n_entities = N_AGENTS + N_LANDMARKS

    adapter = MPEEntityObservationAdapter(
        n_agents=N_AGENTS,
        n_landmarks=N_LANDMARKS,
        sight_range=sight_range,
    )
    hub = MATCHInformationHub(
        n_entities=n_entities,
        observation_dim=8,
        embedding_dim=32,
        state_dim=8,
        history_length=4,
    ).to(device)
    agent = MATCHAgent(
        n_agents=N_AGENTS,
        n_entities=n_entities,
        entity_observation_dim=8,
        hidden_dim=32,
        query_dim=32,
        reconstructed_entity_dim=32,
        n_actions=n_actions,
        use_local_belief=bool(training_arguments.get("use_local_belief", False)),
        belief_hidden_dim=int(training_arguments.get("belief_hidden_dim", 32)),
        belief_state_dim=8,
        fuse_local_belief=not bool(
            training_arguments.get("disable_belief_fusion", False)
        ),
        use_belief_history=not bool(
            training_arguments.get("disable_belief_history", False)
        ),
    ).to(device)
    agent.load_state_dict(checkpoint["match_agent"], strict=True)
    hub.load_state_dict(checkpoint["information_hub"], strict=True)
    agent.eval()
    hub.eval()
    runner = MATCHEpisodeRunner(
        env=env,
        match_agent=agent,
        information_hub=hub,
        entity_adapter=adapter,
        device=device,
        packet_loss=packet_loss,
    )
    return env, runner


def build_runner(
    method: str,
    checkpoint: dict[str, Any],
    training_arguments: dict[str, Any],
    device: torch.device,
    sight_range: float,
    packet_loss: float | None,
):
    if method == "qmix":
        return build_qmix_runner(checkpoint, device, sight_range)
    if method == "masia":
        assert packet_loss is not None
        return build_masia_runner(
            checkpoint, device, sight_range, packet_loss
        )
    assert packet_loss is not None
    return build_match_runner(
        checkpoint,
        training_arguments,
        device,
        sight_range,
        packet_loss,
    )


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "evaluation_summary.json"
    episode_path = output_dir / "evaluation_episodes.csv"
    progress_path = output_dir / "progress.json"

    device = choose_device(args.device)
    # Keep optimizer/mixer tensors on CPU.  Only the execution network is
    # copied to the requested device by the builder below.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    training_arguments = validate_checkpoint(
        args.method,
        checkpoint,
        checkpoint_path,
        args.expected_checkpoint_episode,
    )
    sight_range = resolve_sight_range(training_arguments, args.sight_range)
    packet_loss = resolve_packet_loss(
        args.method,
        training_arguments,
        args.packet_loss,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env, runner = build_runner(
        args.method,
        checkpoint,
        training_arguments,
        device,
        sight_range,
        packet_loss,
    )

    returns: list[float] = []
    pols: list[float] = []
    lengths: list[int] = []
    packet_rates: list[float] = []
    local_visible_rates: list[float] = []
    local_invisible_rates: list[float] = []
    received_packets = 0
    attempted_packets = 0

    fieldnames = [
        "evaluation_episode",
        "episode_seed",
        "episode_return",
        "pol",
        "full_occupancy",
        "episode_length",
        "packet_success_rate",
        "received_packets",
        "attempted_packets",
        "local_visible_entity_rate",
        "local_invisible_entity_rate",
    ]

    print("=" * 72)
    print("CN6 Unified Independent Evaluation")
    print("Method:", METHOD_LABELS[args.method])
    print("Checkpoint:", checkpoint_path)
    print("Checkpoint episode:", checkpoint["episode"])
    print("Training seed:", training_arguments.get("seed"))
    print("Evaluation episodes:", args.episodes)
    print("Evaluation base seed:", args.seed)
    print("Sight range:", sight_range)
    print("Packet loss:", "N/A" if packet_loss is None else packet_loss)
    print("Device:", device)
    print("=" * 72)

    try:
        with episode_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()

            for index in range(args.episodes):
                episode_seed = args.seed + index
                episode_data, statistics = runner.run(
                    epsilon=0.0,
                    seed=episode_seed,
                )
                episode_return = float(statistics["episode_return"])
                pol = float(statistics["pol"])
                episode_length = int(statistics["episode_length"])
                packet_rate_value = statistics.get(
                    "packet_success_rate",
                    statistics.get("packet_success"),
                )
                packet_rate = (
                    None
                    if packet_rate_value is None
                    else float(packet_rate_value)
                )
                received = statistics.get("received_packets")
                attempted = statistics.get("attempted_packets")
                local_visible_rate = None
                local_invisible_rate = None
                observation_masks = episode_data.get("observation_masks")
                if observation_masks is not None:
                    valid_masks = np.asarray(
                        observation_masks[:episode_length],
                        dtype=np.float64,
                    )
                    local_visible_rate = float(np.mean(valid_masks))
                    local_invisible_rate = 1.0 - local_visible_rate

                returns.append(episode_return)
                pols.append(pol)
                lengths.append(episode_length)
                if packet_rate is not None:
                    packet_rates.append(packet_rate)
                if received is not None:
                    received_packets += int(received)
                if attempted is not None:
                    attempted_packets += int(attempted)
                if local_visible_rate is not None:
                    local_visible_rates.append(local_visible_rate)
                    local_invisible_rates.append(local_invisible_rate)

                writer.writerow(
                    {
                        "evaluation_episode": index + 1,
                        "episode_seed": episode_seed,
                        "episode_return": episode_return,
                        "pol": pol,
                        "full_occupancy": int(pol >= 1.0 - 1e-6),
                        "episode_length": episode_length,
                        "packet_success_rate": (
                            "" if packet_rate is None else packet_rate
                        ),
                        "received_packets": (
                            "" if received is None else int(received)
                        ),
                        "attempted_packets": (
                            "" if attempted is None else int(attempted)
                        ),
                        "local_visible_entity_rate": (
                            "" if local_visible_rate is None else local_visible_rate
                        ),
                        "local_invisible_entity_rate": (
                            "" if local_invisible_rate is None else local_invisible_rate
                        ),
                    }
                )

                completed = index + 1
                if (
                    completed % args.progress_interval == 0
                    or completed == args.episodes
                ):
                    handle.flush()
                    progress = {
                        "method": args.method,
                        "method_label": METHOD_LABELS[args.method],
                        "training_seed": training_arguments.get("seed"),
                        "completed_episodes": completed,
                        "target_episodes": args.episodes,
                        "running_mean_pol": float(np.mean(pols)),
                        "running_mean_return": float(np.mean(returns)),
                    }
                    atomic_json_dump(progress_path, progress)
                    print(
                        f"episodes={completed}/{args.episodes} "
                        f"mean_POL={progress['running_mean_pol']:.4f} "
                        f"mean_return={progress['running_mean_return']:.4f}",
                        flush=True,
                    )
    finally:
        env.close()

    pol_distribution = dict(
        sorted(Counter(f"{value:.6f}" for value in pols).items())
    )
    mean_packet_success_rate = (
        None if not packet_rates else float(np.mean(packet_rates))
    )
    aggregate_packet_success_rate = (
        None
        if attempted_packets <= 0
        else float(received_packets / attempted_packets)
    )

    summary = {
        "status": "completed",
        "method": args.method,
        "method_label": METHOD_LABELS[args.method],
        "checkpoint": str(checkpoint_path),
        "checkpoint_episode": int(checkpoint["episode"]),
        "training_seed": training_arguments.get("seed"),
        "evaluation_episodes": args.episodes,
        "evaluation_seed": args.seed,
        "n_agents": N_AGENTS,
        "n_landmarks": N_LANDMARKS,
        "max_steps": MAX_STEPS,
        "sight_range": sight_range,
        "packet_loss": packet_loss,
        "mean_pol": float(np.mean(pols)),
        "std_pol": sample_std(pols),
        "pol_ci95_half_width_within_checkpoint": normal_ci95_half_width(pols),
        "mean_return": float(np.mean(returns)),
        "std_return": sample_std(returns),
        "return_ci95_half_width_within_checkpoint": normal_ci95_half_width(
            returns
        ),
        "full_occupancy_rate": float(
            np.mean(np.asarray(pols) >= 1.0 - 1e-6)
        ),
        "mean_episode_length": float(np.mean(lengths)),
        "mean_packet_success_rate": mean_packet_success_rate,
        "aggregate_packet_success_rate": aggregate_packet_success_rate,
        "received_packets": (
            None if attempted_packets <= 0 else received_packets
        ),
        "attempted_packets": (
            None if attempted_packets <= 0 else attempted_packets
        ),
        "mean_local_visible_entity_rate": (
            None
            if not local_visible_rates
            else float(np.mean(local_visible_rates))
        ),
        "mean_local_invisible_entity_rate": (
            None
            if not local_invisible_rates
            else float(np.mean(local_invisible_rates))
        ),
        "pol_distribution": pol_distribution,
        "device": str(device),
    }
    atomic_json_dump(summary_path, summary)

    print("=" * 72)
    print(f"Mean POL: {summary['mean_pol']:.4f}")
    print(f"Mean return: {summary['mean_return']:.4f}")
    print(f"Full occupancy: {summary['full_occupancy_rate']:.2%}")
    print("Saved:", summary_path)
    print("Saved:", episode_path)
    print("=" * 72)


if __name__ == "__main__":
    main()

