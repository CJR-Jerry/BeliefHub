import numpy as np

from src.data.episode_replay_buffer import EpisodeReplayBuffer
from src.data.match_episode_replay_buffer import MATCHEpisodeReplayBuffer


def _add_available_actions(storage, capacity, max_steps, n_agents, n_actions):
    if n_actions <= 0:
        raise ValueError("n_actions must be positive.")

    storage["avail_actions"] = np.zeros(
        (
            capacity,
            max_steps + 1,
            n_agents,
            n_actions,
        ),
        dtype=np.bool_,
    )


class SMACEpisodeReplayBuffer(EpisodeReplayBuffer):
    """Episode replay buffer for SMAC QMIX runs."""

    def __init__(
        self,
        capacity,
        max_steps,
        n_agents,
        observation_dim,
        state_dim,
        n_actions,
    ):
        super().__init__(
            capacity=capacity,
            max_steps=max_steps,
            n_agents=n_agents,
            observation_dim=observation_dim,
            state_dim=state_dim,
        )
        self.n_actions = n_actions
        _add_available_actions(
            self.storage,
            capacity,
            max_steps,
            n_agents,
            n_actions,
        )


class SMACMATCHEpisodeReplayBuffer(MATCHEpisodeReplayBuffer):
    """MATCH replay buffer with SMAC action-availability masks."""

    def __init__(
        self,
        capacity,
        max_steps,
        n_agents,
        observation_dim,
        state_dim,
        n_entities,
        entity_observation_dim,
        entity_state_dim,
        n_actions,
    ):
        super().__init__(
            capacity=capacity,
            max_steps=max_steps,
            n_agents=n_agents,
            observation_dim=observation_dim,
            state_dim=state_dim,
            n_entities=n_entities,
            entity_observation_dim=entity_observation_dim,
            entity_state_dim=entity_state_dim,
        )
        self.n_actions = n_actions
        _add_available_actions(
            self.storage,
            capacity,
            max_steps,
            n_agents,
            n_actions,
        )

