import numpy as np

from src.data.episode_replay_buffer import EpisodeReplayBuffer


class MASIAEpisodeReplayBuffer(EpisodeReplayBuffer):
    def __init__(
        self,
        capacity,
        max_steps,
        n_agents,
        observation_dim,
        state_dim,
    ):
        super().__init__(
            capacity=capacity,
            max_steps=max_steps,
            n_agents=n_agents,
            observation_dim=observation_dim,
            state_dim=state_dim,
        )
        self.storage["packet_loss_masks"] = np.zeros(
            (capacity, max_steps + 1, n_agents),
            dtype=np.float32,
        )

