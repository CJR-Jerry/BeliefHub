import torch

from src.data.episode_replay_buffer import EpisodeReplayBuffer
from src.envs.cooperative_navigation import CooperativeNavigation
from src.models.agent_rnn import AgentRNN
from src.runners.episode_runner import EpisodeRunner


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    env = CooperativeNavigation(
        n_agents=5,
        max_cycles=25,
        occupancy_radius=0.3,
    )

    first_agent = env.possible_agents[0]

    observation_dim = env.observation_space(
        first_agent
    ).shape[0]

    n_actions = env.action_space(first_agent).n

    env.reset(seed=0)
    state_dim = env.env.state().size

    agent_network = AgentRNN(
        observation_dim=observation_dim,
        n_actions=n_actions,
        n_agents=5,
        hidden_dim=64,
    ).to(device)

    runner = EpisodeRunner(
        env=env,
        agent_network=agent_network,
        device=device,
    )

    episode, statistics = runner.run(
        epsilon=1.0,
        seed=0,
    )

    buffer = EpisodeReplayBuffer(
        capacity=10,
        max_steps=25,
        n_agents=5,
        observation_dim=observation_dim,
        state_dim=state_dim,
    )

    buffer.insert(episode)

    print("Device:", device)
    print("Observation dimension:", observation_dim)
    print("State dimension:", state_dim)
    print("Episode return:", statistics["episode_return"])
    print("Episode length:", statistics["episode_length"])
    print("POL:", statistics["pol"])
    print("Filled steps:", episode["filled"].sum())
    print("Buffer size:", len(buffer))

    assert episode["observations"].shape == (26, 5, 30)
    assert episode["states"].shape == (26, 150)
    assert episode["actions"].shape == (25, 5, 1)
    assert statistics["episode_length"] == 25
    assert len(buffer) == 1

    print("Real episode runner test passed.")

    env.close()


if __name__ == "__main__":
    main()
