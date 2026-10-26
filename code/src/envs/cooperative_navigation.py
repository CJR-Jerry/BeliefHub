import numpy as np
from pettingzoo.mpe import simple_spread_v3


class CooperativeNavigation:
    """Cooperative Navigation environment used for MATCH reproduction."""

    def __init__(
        self,
        n_agents: int = 5,
        max_cycles: int = 25,
        occupancy_radius: float = 0.3,
        continuous_actions: bool = False,
    ):
        self.n_agents = n_agents
        self.max_cycles = max_cycles
        self.occupancy_radius = occupancy_radius

        self.env = simple_spread_v3.parallel_env(
            N=n_agents,
            local_ratio=0.5,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )

        self.possible_agents = list(self.env.possible_agents)

    def reset(self, seed=None):
        observations, infos = self.env.reset(seed=seed)
        return observations, infos

    def step(self, actions):
        return self.env.step(actions)

    @property
    def agents(self):
        return self.env.agents

    def action_space(self, agent):
        return self.env.action_space(agent)

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def calculate_pol(self) -> float:
        """
        Calculate Percentage of Landmarks Occupied (POL).

        A landmark is occupied if at least one agent is within
        occupancy_radius of that landmark.
        """
        world = self.env.unwrapped.world

        agent_positions = np.array(
            [agent.state.p_pos for agent in world.agents],
            dtype=np.float32,
        )

        landmark_positions = np.array(
            [landmark.state.p_pos for landmark in world.landmarks],
            dtype=np.float32,
        )

        differences = (
            agent_positions[:, None, :]
            - landmark_positions[None, :, :]
        )

        distances = np.linalg.norm(differences, axis=-1)

        occupied_landmarks = np.any(
            distances < self.occupancy_radius,
            axis=0,
        )

        return float(np.mean(occupied_landmarks))

    def close(self):
        self.env.close()


def run_random_episode(seed: int = 0) -> float:
    env = CooperativeNavigation(
        n_agents=5,
        max_cycles=25,
        occupancy_radius=0.3,
    )

    env.reset(seed=seed)

    while env.agents:
        actions = {
            agent: env.action_space(agent).sample()
            for agent in env.agents
        }

        env.step(actions)

    pol = env.calculate_pol()
    env.close()

    return pol


if __name__ == "__main__":
    episode_pols = []

    for episode in range(10):
        pol = run_random_episode(seed=episode)
        episode_pols.append(pol)

        print(
            f"Episode {episode + 1:02d}: "
            f"POL = {pol:.2f}"
        )

    print("-" * 40)
    print(f"Mean random-policy POL: {np.mean(episode_pols):.3f}")
