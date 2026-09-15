import numpy as np

from src.data.episode_replay_buffer import (
    EpisodeReplayBuffer,
)


class MATCHEpisodeReplayBuffer(EpisodeReplayBuffer):
    """
    Episode replay buffer for MATCH.

    In addition to the original QMIX fields, this buffer
    stores the entity-based information required by MATCH.

    T + 1 fields:
        observations
        states
        entity_observations
        observation_masks
        packet_loss_masks
        entity_states

    T fields:
        actions
        rewards
        terminated
        filled

    Reconstructed entity embeddings and Hub messages are
    not stored. They must be recomputed during training so
    that gradients can propagate through the Information Hub.
    """

    def __init__(
        self,
        capacity: int,
        max_steps: int,
        n_agents: int,
        observation_dim: int,
        state_dim: int,
        n_entities: int,
        entity_observation_dim: int,
        entity_state_dim: int,
    ):
        super().__init__(
            capacity=capacity,
            max_steps=max_steps,
            n_agents=n_agents,
            observation_dim=observation_dim,
            state_dim=state_dim,
        )

        if n_entities <= 0:
            raise ValueError(
                "n_entities must be positive."
            )

        if entity_observation_dim <= 0:
            raise ValueError(
                "entity_observation_dim must be positive."
            )

        if entity_state_dim <= 0:
            raise ValueError(
                "entity_state_dim must be positive."
            )

        self.n_entities = n_entities
        self.entity_observation_dim = (
            entity_observation_dim
        )
        self.entity_state_dim = entity_state_dim

        # Local entity observations of every agent.
        self.storage["entity_observations"] = np.zeros(
            (
                capacity,
                max_steps + 1,
                n_agents,
                n_entities,
                entity_observation_dim,
            ),
            dtype=np.float32,
        )

        # True when an entity is visible in the local
        # observation of an agent.
        self.storage["observation_masks"] = np.zeros(
            (
                capacity,
                max_steps + 1,
                n_agents,
                n_entities,
            ),
            dtype=np.bool_,
        )

        # True when the complete message sent by an agent
        # successfully reaches the Information Hub.
        self.storage["packet_loss_masks"] = np.zeros(
            (
                capacity,
                max_steps + 1,
                n_agents,
            ),
            dtype=np.bool_,
        )

        # Structured true global entity state used as the
        # target of the reconstruction loss.
        self.storage["entity_states"] = np.zeros(
            (
                capacity,
                max_steps + 1,
                n_entities,
                entity_state_dim,
            ),
            dtype=np.float32,
        )