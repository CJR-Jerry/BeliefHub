from typing import Dict

import numpy as np
import torch


class EpisodeReplayBuffer:
    """
    Replay buffer that stores complete MARL episodes.

    Observations and states contain T + 1 timesteps because
    the final item is used as the next observation/state.
    """

    def __init__(
        self,
        capacity: int,
        max_steps: int,
        n_agents: int,
        observation_dim: int,
        state_dim: int,
    ):
        self.capacity = capacity
        self.max_steps = max_steps
        self.n_agents = n_agents
        self.observation_dim = observation_dim
        self.state_dim = state_dim

        self.position = 0
        self.size = 0

        self.storage = {
            "observations": np.zeros(
                (
                    capacity,
                    max_steps + 1,
                    n_agents,
                    observation_dim,
                ),
                dtype=np.float32,
            ),
            "states": np.zeros(
                (capacity, max_steps + 1, state_dim),
                dtype=np.float32,
            ),
            "actions": np.zeros(
                (capacity, max_steps, n_agents, 1),
                dtype=np.int64,
            ),
            "rewards": np.zeros(
                (capacity, max_steps, 1),
                dtype=np.float32,
            ),
            "terminated": np.zeros(
                (capacity, max_steps, 1),
                dtype=np.float32,
            ),
            "filled": np.zeros(
                (capacity, max_steps, 1),
                dtype=np.float32,
            ),
            "pol": np.zeros(
                (capacity, 1),
                dtype=np.float32,
            ),
        }

    def __len__(self):
        return self.size

    def insert(self, episode: Dict[str, np.ndarray]):
        for key, storage_array in self.storage.items():
            if key not in episode:
                raise KeyError(f"Episode is missing key: {key}")

            value = np.asarray(
                episode[key],
                dtype=storage_array.dtype,
            )

            expected_shape = storage_array.shape[1:]

            if value.shape != expected_shape:
                raise ValueError(
                    f"{key} has shape {value.shape}, "
                    f"expected {expected_shape}"
                )

            storage_array[self.position] = value

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device):
        if batch_size > self.size:
            raise ValueError(
                f"Cannot sample {batch_size} episodes "
                f"from buffer containing {self.size}"
            )

        indices = np.random.choice(
            self.size,
            size=batch_size,
            replace=False,
        )

        batch = {
            key: torch.as_tensor(
                value[indices],
                device=device,
            )
            for key, value in self.storage.items()
        }

        return batch