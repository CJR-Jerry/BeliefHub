#!/usr/bin/env python3
"""Measure five-method CN6 inference, model, memory, and communication cost."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from evaluate_cn6_checkpoint import (
    METHOD_LABELS,
    build_runner,
    resolve_packet_loss,
    resolve_sight_range,
    validate_checkpoint,
)


METHODS = tuple(METHOD_LABELS)
FLOAT_BYTES = 4


def state_dict_elements(state_dict: dict) -> int:
    return sum(value.numel() for value in state_dict.values() if torch.is_tensor(value))


def module_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def masia_active_parameters(module: torch.nn.Module) -> int:
    """Parameters reached by MASIAAgent.forward during decentralized execution."""
    inactive_prefixes = (
        "projection.",
        "predictor.",
        "target_encoder.",
        "target_projection.",
    )
    return sum(
        parameter.numel()
        for name, parameter in module.named_parameters()
        if not name.startswith(inactive_prefixes)
    )


def parameter_accounting(method: str, checkpoint: dict, runner) -> dict[str, int]:
    if method == "qmix":
        resident = module_parameters(runner.agent_network)
        active = resident
        training = (
            state_dict_elements(checkpoint["agent_network"])
            + state_dict_elements(checkpoint.get("mixer", {}))
        )
    elif method == "masia":
        resident = module_parameters(runner.agent_network)
        active = masia_active_parameters(runner.agent_network)
        training = (
            state_dict_elements(checkpoint["agent_network"])
            + state_dict_elements(checkpoint.get("mixer", {}))
            + state_dict_elements(checkpoint.get("latent_model", {}))
        )
    else:
        resident = (
            module_parameters(runner.match_agent)
            + module_parameters(runner.information_hub)
        )
        active = resident
        training = resident + state_dict_elements(checkpoint.get("mixer", {}))

    checkpoint_elements = sum(
        state_dict_elements(value)
        for value in checkpoint.values()
        if isinstance(value, dict)
    )
    return {
        "execution_parameter_count": resident,
        "active_execution_parameter_count": active,
        "online_training_parameter_elements": training,
        "checkpoint_parameter_elements": checkpoint_elements,
    }


def communication_for_episode(
    method: str,
    episode: dict,
    statistics: dict,
    runner,
) -> dict[str, float]:
    """Apply one documented logical-wire accounting model.

    MATCH-family: six agent-to-Hub uplinks and six personalized 32-float
    Hub-to-agent messages per step. A fixed uplink carries the 12 x 8 entity
    tensor. MASIA: expand every sender broadcast into five receiver deliveries,
    each carrying one local observation. QMIX: no execution-time messages.
    """
    steps = int(statistics["episode_length"])
    if method == "qmix":
        return {
            "steps": steps,
            "attempted_messages": 0.0,
            "delivered_messages": 0.0,
            "attempted_bytes": 0.0,
            "delivered_bytes": 0.0,
        }

    n_agents = int(runner.n_agents)
    if method == "masia":
        masks = np.asarray(episode["packet_loss_masks"][:steps], dtype=np.float64)
        receivers = n_agents - 1
        attempted_messages = float(steps * n_agents * receivers)
        delivered_messages = float(masks.sum() * receivers)
        payload_floats = int(runner.observation_dim)
        return {
            "steps": steps,
            "attempted_messages": attempted_messages,
            "delivered_messages": delivered_messages,
            "attempted_bytes": attempted_messages * payload_floats * FLOAT_BYTES,
            "delivered_bytes": delivered_messages * payload_floats * FLOAT_BYTES,
        }

    attempted_uplinks = int(statistics["attempted_packets"])
    delivered_uplinks = int(statistics["received_packets"])
    downlinks = steps * n_agents
    uplink_floats = int(runner.n_entities * runner.entity_observation_dim)
    downlink_floats = int(runner.match_agent.reconstructed_entity_dim)
    return {
        "steps": steps,
        "attempted_messages": float(attempted_uplinks + downlinks),
        "delivered_messages": float(delivered_uplinks + downlinks),
        "attempted_bytes": float(
            (attempted_uplinks * uplink_floats + downlinks * downlink_floats)
            * FLOAT_BYTES
        ),
        "delivered_bytes": float(
            (delivered_uplinks * uplink_floats + downlinks * downlink_floats)
            * FLOAT_BYTES
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--warmup-episodes", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=40000)
    parser.add_argument("--packet-loss", type=float, default=None)
    parser.add_argument("--sight-range", type=float, default=None)
    parser.add_argument("--expected-checkpoint-episode", type=int, default=100000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.episodes <= 0 or args.warmup_episodes < 0:
        raise ValueError("episode counts are invalid")

    device = torch.device("cuda:0")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    training_arguments = validate_checkpoint(
        args.method,
        checkpoint,
        checkpoint_path,
        args.expected_checkpoint_episode,
    )
    sight_range = resolve_sight_range(training_arguments, args.sight_range)
    packet_loss = resolve_packet_loss(args.method, training_arguments, args.packet_loss)
    env, runner = build_runner(
        args.method,
        checkpoint,
        training_arguments,
        device,
        sight_range,
        packet_loss,
    )
    parameters = parameter_accounting(args.method, checkpoint, runner)

    with torch.inference_mode():
        for episode_index in range(args.warmup_episodes):
            runner.run(epsilon=0.0, seed=args.seed + episode_index)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    episode_times: list[float] = []
    pols: list[float] = []
    returns: list[float] = []
    communication = {
        "steps": 0.0,
        "attempted_messages": 0.0,
        "delivered_messages": 0.0,
        "attempted_bytes": 0.0,
        "delivered_bytes": 0.0,
    }

    try:
        with torch.inference_mode():
            for episode_index in range(args.episodes):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                episode_data, statistics = runner.run(
                    epsilon=0.0,
                    seed=args.seed + args.warmup_episodes + episode_index,
                )
                torch.cuda.synchronize(device)
                episode_times.append(time.perf_counter() - start)
                pols.append(float(statistics["pol"]))
                returns.append(float(statistics["episode_return"]))
                episode_comm = communication_for_episode(
                    args.method, episode_data, statistics, runner
                )
                for key in communication:
                    communication[key] += float(episode_comm[key])
    finally:
        env.close()

    elapsed = float(sum(episode_times))
    episode_times_ms = np.asarray(episode_times, dtype=np.float64) * 1000.0
    total_steps = communication["steps"]
    if total_steps <= 0:
        raise RuntimeError("The benchmark executed no environment steps")

    report = {
        "status": "succeeded",
        "method": args.method,
        "method_label": METHOD_LABELS[args.method],
        "checkpoint": str(checkpoint_path),
        "checkpoint_episode": int(checkpoint["episode"]),
        "training_seed": int(training_arguments.get("seed", -1)),
        "warmup_episodes": args.warmup_episodes,
        "timed_episodes": args.episodes,
        "total_environment_steps": int(total_steps),
        "packet_loss": packet_loss,
        "sight_range": sight_range,
        "total_timed_seconds": elapsed,
        "mean_episode_latency_ms": float(episode_times_ms.mean()),
        "median_episode_latency_ms": float(np.median(episode_times_ms)),
        "p95_episode_latency_ms": float(np.percentile(episode_times_ms, 95)),
        "mean_end_to_end_step_latency_ms": elapsed / total_steps * 1000.0,
        "episodes_per_second": args.episodes / elapsed,
        "environment_steps_per_second": total_steps / elapsed,
        "peak_inference_cuda_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_inference_cuda_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
        **parameters,
        "attempted_messages_per_step": communication["attempted_messages"] / total_steps,
        "delivered_messages_per_step": communication["delivered_messages"] / total_steps,
        "attempted_communication_kib_per_episode": communication["attempted_bytes"] / args.episodes / 1024.0,
        "delivered_communication_kib_per_episode": communication["delivered_bytes"] / args.episodes / 1024.0,
        "mean_pol": float(np.mean(pols)),
        "mean_return": float(np.mean(returns)),
        "gpu_name": torch.cuda.get_device_name(device),
        "communication_accounting": (
            "QMIX: none; MASIA: point-to-point expansion of each sender "
            "broadcast, local observation float32 payload; MATCH family: "
            "fixed 12x8 float32 uplink plus personalized 32-float downlink."
        ),
    }
    for key, value in report.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"Non-finite result for {key}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

