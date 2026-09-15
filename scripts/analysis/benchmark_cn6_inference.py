#!/usr/bin/env python3
"""Measure CN6 checkpoint parameters, memory, and end-to-end inference latency."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from src.envs.cooperative_navigation import CooperativeNavigation
from src.envs.entity_observation import MPEEntityObservationAdapter
from src.models.match_agent import MATCHAgent
from src.models.match_information_hub import MATCHInformationHub
from src.runners.match_episode_runner import MATCHEpisodeRunner


def state_dict_elements(state_dict: dict) -> int:
    return sum(value.numel() for value in state_dict.values() if torch.is_tensor(value))


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--warmup-episodes", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=40000)
    parser.add_argument("--packet-loss", type=float, default=0.3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if not 0.0 <= args.packet_loss <= 1.0:
        raise ValueError("packet loss must be between 0 and 1")

    device = torch.device("cuda:0")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    training_arguments = checkpoint.get("arguments", {})
    saved_sight_range = training_arguments.get("sight_range", -1.0)
    sight_range = None if saved_sight_range < 0.0 else saved_sight_range

    n_agents = 6
    n_landmarks = 6
    n_entities = 12
    max_steps = 25

    env = CooperativeNavigation(
        n_agents=n_agents,
        max_cycles=max_steps,
        occupancy_radius=0.3,
    )
    env.reset(seed=args.seed)
    n_actions = env.action_space(env.possible_agents[0]).n
    entity_adapter = MPEEntityObservationAdapter(
        n_agents=n_agents,
        n_landmarks=n_landmarks,
        sight_range=sight_range,
    )
    information_hub = MATCHInformationHub(
        n_entities=n_entities,
        observation_dim=8,
        embedding_dim=32,
        state_dim=8,
        history_length=4,
    ).to(device)
    match_agent = MATCHAgent(
        n_agents=n_agents,
        n_entities=n_entities,
        entity_observation_dim=8,
        hidden_dim=32,
        query_dim=32,
        reconstructed_entity_dim=32,
        n_actions=n_actions,
        use_local_belief=bool(training_arguments.get("use_local_belief", False)),
        belief_hidden_dim=int(training_arguments.get("belief_hidden_dim", 32)),
        belief_state_dim=8,
        fuse_local_belief=not bool(training_arguments.get("disable_belief_fusion", False)),
        use_belief_history=not bool(training_arguments.get("disable_belief_history", False)),
    ).to(device)
    match_agent.load_state_dict(checkpoint["match_agent"])
    information_hub.load_state_dict(checkpoint["information_hub"])
    match_agent.eval()
    information_hub.eval()

    runner = MATCHEpisodeRunner(
        env=env,
        match_agent=match_agent,
        information_hub=information_hub,
        entity_adapter=entity_adapter,
        device=device,
        packet_loss=args.packet_loss,
    )

    with torch.inference_mode():
        for episode in range(args.warmup_episodes):
            runner.run(epsilon=0.0, seed=args.seed + episode)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    episode_times = []
    pols = []
    returns = []

    with torch.inference_mode():
        for episode in range(args.episodes):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            _, statistics = runner.run(
                epsilon=0.0,
                seed=args.seed + args.warmup_episodes + episode,
            )
            torch.cuda.synchronize(device)
            episode_times.append(time.perf_counter() - start)
            pols.append(float(statistics["pol"]))
            returns.append(float(statistics["episode_return"]))

    elapsed = float(sum(episode_times))
    episode_times_ms = np.asarray(episode_times, dtype=float) * 1000.0
    method = "db_all" if match_agent.use_local_belief else "match"
    execution_parameters = count_parameters(match_agent) + count_parameters(information_hub)
    mixer_elements = state_dict_elements(checkpoint["mixer"])
    online_training_elements = execution_parameters + mixer_elements

    report = {
        "method": method,
        "checkpoint": args.checkpoint,
        "checkpoint_episode": int(checkpoint["episode"]),
        "training_seed": int(training_arguments.get("seed", -1)),
        "warmup_episodes": args.warmup_episodes,
        "timed_episodes": args.episodes,
        "steps_per_episode": max_steps,
        "packet_loss": args.packet_loss,
        "total_timed_seconds": elapsed,
        "mean_episode_latency_ms": float(episode_times_ms.mean()),
        "median_episode_latency_ms": float(np.median(episode_times_ms)),
        "p95_episode_latency_ms": float(np.percentile(episode_times_ms, 95)),
        "mean_end_to_end_step_latency_ms": float(episode_times_ms.mean() / max_steps),
        "episodes_per_second": args.episodes / elapsed,
        "environment_steps_per_second": args.episodes * max_steps / elapsed,
        "peak_inference_cuda_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_inference_cuda_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
        "execution_parameter_count": execution_parameters,
        "mixer_parameter_elements": mixer_elements,
        "online_training_parameter_elements": online_training_elements,
        "mean_pol": float(np.mean(pols)),
        "mean_return": float(np.mean(returns)),
        "gpu_name": torch.cuda.get_device_name(device),
    }
    for key, value in report.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"Non-finite result for {key}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    env.close()


if __name__ == "__main__":
    main()
