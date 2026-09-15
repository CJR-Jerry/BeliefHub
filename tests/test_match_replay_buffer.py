import numpy as np
import torch

from src.data.match_episode_replay_buffer import (
    MATCHEpisodeReplayBuffer,
)


def create_random_match_episode(
    max_steps,
    n_agents,
    observation_dim,
    state_dim,
    n_actions,
    n_entities,
    entity_observation_dim,
    entity_state_dim,
):
    return {
        "observations": np.random.randn(
            max_steps + 1,
            n_agents,
            observation_dim,
        ).astype(np.float32),

        "states": np.random.randn(
            max_steps + 1,
            state_dim,
        ).astype(np.float32),

        "actions": np.random.randint(
            0,
            n_actions,
            size=(
                max_steps,
                n_agents,
                1,
            ),
            dtype=np.int64,
        ),

        "rewards": np.random.randn(
            max_steps,
            1,
        ).astype(np.float32),

        "terminated": np.concatenate(
            [
                np.zeros(
                    (max_steps - 1, 1),
                    dtype=np.float32,
                ),
                np.ones(
                    (1, 1),
                    dtype=np.float32,
                ),
            ],
            axis=0,
        ),

        "filled": np.ones(
            (max_steps, 1),
            dtype=np.float32,
        ),

        "pol": np.array(
            [np.random.random()],
            dtype=np.float32,
        ),

        "entity_observations": np.random.randn(
            max_steps + 1,
            n_agents,
            n_entities,
            entity_observation_dim,
        ).astype(np.float32),

        "observation_masks": (
            np.random.random(
                (
                    max_steps + 1,
                    n_agents,
                    n_entities,
                )
            )
            > 0.3
        ),

        "packet_loss_masks": (
            np.random.random(
                (
                    max_steps + 1,
                    n_agents,
                )
            )
            > 0.3
        ),

        "entity_states": np.random.randn(
            max_steps + 1,
            n_entities,
            entity_state_dim,
        ).astype(np.float32),
    }


def main():
    np.random.seed(0)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    capacity = 4
    max_steps = 25
    n_agents = 5
    observation_dim = 30
    state_dim = 150
    n_actions = 5
    n_entities = 10
    entity_observation_dim = 8
    entity_state_dim = 8

    buffer = MATCHEpisodeReplayBuffer(
        capacity=capacity,
        max_steps=max_steps,
        n_agents=n_agents,
        observation_dim=observation_dim,
        state_dim=state_dim,
        n_entities=n_entities,
        entity_observation_dim=(
            entity_observation_dim
        ),
        entity_state_dim=entity_state_dim,
    )

    # Insert six episodes into a capacity-four buffer.
    # This tests circular replacement.
    for _ in range(6):
        episode = create_random_match_episode(
            max_steps=max_steps,
            n_agents=n_agents,
            observation_dim=observation_dim,
            state_dim=state_dim,
            n_actions=n_actions,
            n_entities=n_entities,
            entity_observation_dim=(
                entity_observation_dim
            ),
            entity_state_dim=entity_state_dim,
        )

        buffer.insert(episode)

    assert len(buffer) == capacity

    batch = buffer.sample(
        batch_size=2,
        device=device,
    )

    print("Device:", device)
    print("Buffer size:", len(buffer))

    for key, value in batch.items():
        print(
            f"{key}: "
            f"{value.shape} "
            f"{value.dtype}"
        )

    assert batch["observations"].shape == (
        2,
        26,
        5,
        30,
    )

    assert batch["states"].shape == (
        2,
        26,
        150,
    )

    assert batch["actions"].shape == (
        2,
        25,
        5,
        1,
    )

    assert batch["rewards"].shape == (
        2,
        25,
        1,
    )

    assert batch[
        "entity_observations"
    ].shape == (
        2,
        26,
        5,
        10,
        8,
    )

    assert batch[
        "observation_masks"
    ].shape == (
        2,
        26,
        5,
        10,
    )

    assert batch[
        "packet_loss_masks"
    ].shape == (
        2,
        26,
        5,
    )

    assert batch["entity_states"].shape == (
        2,
        26,
        10,
        8,
    )

    assert (
        batch["entity_observations"].dtype
        == torch.float32
    )

    assert (
        batch["observation_masks"].dtype
        == torch.bool
    )

    assert (
        batch["packet_loss_masks"].dtype
        == torch.bool
    )

    assert (
        batch["entity_states"].dtype
        == torch.float32
    )

    # Verify packet success rate is reasonable.
    packet_success_rate = (
        batch["packet_loss_masks"]
        .float()
        .mean()
        .item()
    )

    print(
        "Packet success rate:",
        packet_success_rate,
    )

    assert 0.0 <= packet_success_rate <= 1.0

    print(
        "MATCH episode replay buffer test passed."
    )


if __name__ == "__main__":
    main()