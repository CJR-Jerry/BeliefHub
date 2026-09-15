import numpy as np
import torch

from src.data.match_episode_replay_buffer import (
    MATCHEpisodeReplayBuffer,
)
from src.envs.cooperative_navigation import (
    CooperativeNavigation,
)
from src.envs.entity_observation import (
    MPEEntityObservationAdapter,
)
from src.learners.match_learner import (
    MATCHLearner,
)
from src.models.match_agent import MATCHAgent
from src.models.match_information_hub import (
    MATCHInformationHub,
)
from src.models.qmix_mixer import QMixer
from src.runners.match_episode_runner import (
    MATCHEpisodeRunner,
)


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    n_agents = 5
    n_landmarks = 5
    n_entities = 10
    max_steps = 25
    packet_loss = 0.3

    env = CooperativeNavigation(
        n_agents=n_agents,
        max_cycles=max_steps,
        occupancy_radius=0.3,
    )

    first_agent = env.possible_agents[0]

    observation_dim = (
        env.observation_space(
            first_agent
        ).shape[0]
    )

    n_actions = env.action_space(
        first_agent
    ).n

    env.reset(seed=0)
    state_dim = env.env.state().size

    entity_adapter = (
        MPEEntityObservationAdapter(
            n_agents=n_agents,
            n_landmarks=n_landmarks,
            sight_range=0.8,
        )
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
    ).to(device)

    mixer = QMixer(
        n_agents=n_agents,
        state_dim=state_dim,
        mixing_hidden_dim=32,
        hypernet_hidden_dim=64,
    ).to(device)

    runner = MATCHEpisodeRunner(
        env=env,
        match_agent=match_agent,
        information_hub=information_hub,
        entity_adapter=entity_adapter,
        device=device,
        packet_loss=packet_loss,
    )

    buffer = MATCHEpisodeReplayBuffer(
        capacity=20,
        max_steps=max_steps,
        n_agents=n_agents,
        observation_dim=observation_dim,
        state_dim=state_dim,
        n_entities=n_entities,
        entity_observation_dim=8,
        entity_state_dim=8,
    )

    learner = MATCHLearner(
        match_agent=match_agent,
        information_hub=information_hub,
        mixer=mixer,
        learning_rate=5e-4,
        gamma=0.99,
        reconstruction_weight=0.1,
        grad_clip=10.0,
    )

    print("Device:", device)
    print("Collecting warm-up episodes...")

    warmup_episodes = 3

    for episode_index in range(
        warmup_episodes
    ):
        episode, statistics = runner.run(
            epsilon=1.0,
            seed=episode_index,
        )

        buffer.insert(episode)

        print(
            f"episode={episode_index + 1} "
            f"return="
            f"{statistics['episode_return']:.4f} "
            f"POL={statistics['pol']:.4f} "
            f"delivery="
            f"{statistics['packet_success_rate']:.3f}"
        )

    assert len(buffer) == warmup_episodes

    agent_parameter_before = (
        match_agent.q_network
        .value_layer
        .weight
        .detach()
        .clone()
    )

    hub_parameter_before = (
        information_hub
        .entity_pool
        .pool_projection
        .weight
        .detach()
        .clone()
    )

    mixer_parameter_before = (
        mixer.hyper_b1
        .weight
        .detach()
        .clone()
    )

    print("Running training updates...")

    training_updates = 2

    for update_index in range(
        training_updates
    ):
        batch = buffer.sample(
            batch_size=2,
            device=device,
        )

        train_statistics = (
            learner.train_batch(batch)
        )

        print(
            f"update={update_index + 1} "
            f"loss="
            f"{train_statistics['loss']:.4f} "
            f"td="
            f"{train_statistics['td_loss']:.4f} "
            f"state="
            f"{train_statistics['state_loss']:.4f} "
            f"grad="
            f"{train_statistics['gradient_norm']:.4f}"
        )

        for value in (
            train_statistics.values()
        ):
            assert np.isfinite(value)

    agent_parameter_after = (
        match_agent.q_network
        .value_layer
        .weight
        .detach()
    )

    hub_parameter_after = (
        information_hub
        .entity_pool
        .pool_projection
        .weight
        .detach()
    )

    mixer_parameter_after = (
        mixer.hyper_b1
        .weight
        .detach()
    )

    agent_parameter_change = (
        agent_parameter_after
        - agent_parameter_before
    ).abs().mean().item()

    hub_parameter_change = (
        hub_parameter_after
        - hub_parameter_before
    ).abs().mean().item()

    mixer_parameter_change = (
        mixer_parameter_after
        - mixer_parameter_before
    ).abs().mean().item()

    print(
        "Agent parameter change:",
        agent_parameter_change,
    )

    print(
        "Hub parameter change:",
        hub_parameter_change,
    )

    print(
        "Mixer parameter change:",
        mixer_parameter_change,
    )

    assert agent_parameter_change > 0.0
    assert hub_parameter_change > 0.0
    assert mixer_parameter_change > 0.0

    learner.update_targets()

    # Verify target-network synchronization.
    for online, target in zip(
        match_agent.parameters(),
        learner.target_match_agent.parameters(),
    ):
        assert torch.allclose(
            online,
            target,
        )

    for online, target in zip(
        information_hub.parameters(),
        learner
        .target_information_hub
        .parameters(),
    ):
        assert torch.allclose(
            online,
            target,
        )

    for online, target in zip(
        mixer.parameters(),
        learner.target_mixer.parameters(),
    ):
        assert torch.allclose(
            online,
            target,
        )

    print(
        "Running greedy pipeline evaluation..."
    )

    evaluation_episode, evaluation_statistics = (
        runner.run(
            epsilon=0.0,
            seed=100,
        )
    )

    print(
        "Evaluation return:",
        evaluation_statistics[
            "episode_return"
        ],
    )

    print(
        "Evaluation POL:",
        evaluation_statistics["pol"],
    )

    print(
        "Evaluation packet success rate:",
        evaluation_statistics[
            "packet_success_rate"
        ],
    )

    assert (
        evaluation_episode["filled"].sum()
        > 0
    )

    assert np.isfinite(
        evaluation_statistics[
            "episode_return"
        ]
    )

    assert 0.0 <= (
        evaluation_statistics["pol"]
    ) <= 1.0

    print(
        "MATCH training pipeline test passed."
    )

    env.close()


if __name__ == "__main__":
    main()