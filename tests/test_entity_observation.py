import numpy as np
import torch

from src.envs.cooperative_navigation import CooperativeNavigation
from src.envs.entity_observation import (
    MPEEntityObservationAdapter,
)
from src.models.information_hub import (
    SelfAttentiveEntityPool,
)


def main():
    env = CooperativeNavigation(
        n_agents=5,
        max_cycles=25,
    )

    observations, _ = env.reset(seed=0)

    adapter = MPEEntityObservationAdapter(
        n_agents=5,
        n_landmarks=5,
        sight_range=None,
    )

    entity_observations, observation_mask = adapter.convert(
        observations=observations,
        agent_names=env.possible_agents,
        packet_loss=0.0,
        rng=np.random.default_rng(0),
    )

    print(
        "Entity observations:",
        entity_observations.shape,
    )

    print(
        "Observation mask:",
        observation_mask.shape,
    )

    assert entity_observations.shape == (5, 10, 8)
    assert observation_mask.shape == (5, 10)
    assert observation_mask.all()

    # 与环境真实相对位置对比
    world = env.env.unwrapped.world

    for observer_index, observer in enumerate(world.agents):
        for entity_index, agent in enumerate(world.agents):
            expected = (
                agent.state.p_pos
                - observer.state.p_pos
            )

            actual = entity_observations[
                observer_index,
                entity_index,
                0:2,
            ]

            assert np.allclose(
                actual,
                expected,
                atol=1e-5,
            )

        for landmark_index, landmark in enumerate(
            world.landmarks
        ):
            entity_index = 5 + landmark_index

            expected = (
                landmark.state.p_pos
                - observer.state.p_pos
            )

            actual = entity_observations[
                observer_index,
                entity_index,
                0:2,
            ]

            assert np.allclose(
                actual,
                expected,
                atol=1e-5,
            )

    # 测试100%消息丢失
    _, lost_mask = adapter.convert(
        observations=observations,
        agent_names=env.possible_agents,
        packet_loss=1.0,
        rng=np.random.default_rng(0),
    )

    assert not lost_mask.any()

    # 将真实实体观测送入Information Hub
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    pool = SelfAttentiveEntityPool(
        observation_dim=8,
        embedding_dim=32,
    ).to(device)

    entity_tensor = torch.as_tensor(
        entity_observations,
        device=device,
    ).unsqueeze(0)

    mask_tensor = torch.as_tensor(
        observation_mask,
        device=device,
    ).unsqueeze(0)

    aggregated, weights, observed = pool(
        entity_tensor,
        mask_tensor,
    )

    print("Aggregated entities:", aggregated.shape)
    print("Attention weights:", weights.shape)
    print("Observed entities:", observed.shape)

    assert aggregated.shape == (1, 10, 32)
    assert weights.shape == (1, 5, 10)
    assert observed.shape == (1, 10)

    print("Entity observation adapter test passed.")

    env.close()


if __name__ == "__main__":
    main()