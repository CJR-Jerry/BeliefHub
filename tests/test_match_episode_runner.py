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
from src.models.match_agent import MATCHAgent
from src.models.match_information_hub import (
    MATCHInformationHub,
)
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

    runner = MATCHEpisodeRunner(
        env=env,
        match_agent=match_agent,
        information_hub=information_hub,
        entity_adapter=entity_adapter,
        device=device,
        packet_loss=0.3,
    )

    episode, statistics = runner.run(
        epsilon=1.0,
        seed=0,
    )

    buffer = MATCHEpisodeReplayBuffer(
        capacity=10,
        max_steps=max_steps,
        n_agents=n_agents,
        observation_dim=observation_dim,
        state_dim=state_dim,
        n_entities=n_entities,
        entity_observation_dim=8,
        entity_state_dim=8,
    )

    buffer.insert(episode)

    print("Device:", device)
    print(
        "Observation dimension:",
        observation_dim,
    )
    print(
        "State dimension:",
        state_dim,
    )
    print(
        "Episode return:",
        statistics["episode_return"],
    )
    print(
        "Episode length:",
        statistics["episode_length"],
    )
    print("POL:", statistics["pol"])
    print(
        "Packet success rate:",
        statistics["packet_success_rate"],
    )
    print(
        "Received packets:",
        statistics["received_packets"],
    )
    print(
        "Attempted packets:",
        statistics["attempted_packets"],
    )
    print(
        "Filled steps:",
        episode["filled"].sum(),
    )
    print("Buffer size:", len(buffer))

    assert episode["observations"].shape == (
        26,
        5,
        30,
    )


    assert episode[
        "entity_observations"
    ].shape == (
        26,
        5,
        10,
        8,
    )

    assert episode[
        "observation_masks"
    ].shape == (
        26,
        5,
        10,
    )

    assert episode[
        "packet_loss_masks"
    ].shape == (
        26,
        5,
    )

    assert episode["entity_states"].shape == (
        26,
        10,
        8,
    )

    assert episode["actions"].shape == (
        25,
        5,
        1,
    )

    assert statistics["episode_length"] == 25
    assert episode["filled"].sum() == 25
    assert len(buffer) == 1

    assert (
        statistics["attempted_packets"]
        == 25 * 5
    )

    assert (
        0.0
        <= statistics["packet_success_rate"]
        <= 1.0
    )

    # With 30% loss and 125 attempted packets,
    # both successful and lost packets should occur.
    assert statistics["received_packets"] > 0
    assert (
        statistics["received_packets"]
        < statistics["attempted_packets"]
    )

    batch = buffer.sample(
        batch_size=1,
        device=device,
    )

    assert batch[
        "entity_observations"
    ].shape == (
        1,
        26,
        5,
        10,
        8,
    )

    print(
        "MATCH episode runner test passed."
    )

    env.close()


if __name__ == "__main__":
    main()