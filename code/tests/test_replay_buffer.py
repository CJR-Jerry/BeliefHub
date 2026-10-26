import numpy as np
import torch

from src.data.episode_replay_buffer import EpisodeReplayBuffer


def create_random_episode(
    max_steps,
    n_agents,
    observation_dim,
    state_dim,
    n_actions,
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
            size=(max_steps, n_agents, 1),
            dtype=np.int64,
        ),

        "rewards": np.random.randn(
            max_steps,
            1,
        ).astype(np.float32),

        "terminated": np.concatenate(
            [
                np.zeros((max_steps - 1, 1), dtype=np.float32),
                np.ones((1, 1), dtype=np.float32),
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
    }


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    buffer = EpisodeReplayBuffer(
        capacity=4,
        max_steps=25,
        n_agents=5,
        observation_dim=30,
        state_dim=150,
    )

    # 插入6局，验证容量为4时能够循环覆盖
    for _ in range(6):
        episode = create_random_episode(
            max_steps=25,
            n_agents=5,
            observation_dim=30,
            state_dim=150,
            n_actions=5,
        )
        buffer.insert(episode)

    assert len(buffer) == 4

    batch = buffer.sample(
        batch_size=2,
        device=device,
    )

    print("Device:", device)
    print("Buffer size:", len(buffer))

    for key, value in batch.items():
        print(f"{key}: {value.shape} {value.dtype}")

    assert batch["observations"].shape == (2, 26, 5, 30)
    assert batch["states"].shape == (2, 26, 150)
    assert batch["actions"].shape == (2, 25, 5, 1)
    assert batch["rewards"].shape == (2, 25, 1)

    print("Episode replay buffer test passed.")


if __name__ == "__main__":
    main()
