import numpy as np

from src.envs.cooperative_navigation import CooperativeNavigation


class LimitedSightCooperativeNavigation(CooperativeNavigation):
    """CN environment whose local observations obey a sight radius."""

    def __init__(
        self,
        n_agents=5,
        max_cycles=25,
        occupancy_radius=0.3,
        continuous_actions=False,
        sight_range=None,
    ):
        super().__init__(
            n_agents=n_agents,
            max_cycles=max_cycles,
            occupancy_radius=occupancy_radius,
            continuous_actions=continuous_actions,
        )

        self.sight_range = (
            None
            if sight_range is None or sight_range < 0.0
            else float(sight_range)
        )

    def _mask_observations(self, observations):
        if self.sight_range is None:
            return observations

        masked = {}

        landmark_start = 4
        landmark_end = landmark_start + 2 * self.n_agents

        other_position_start = landmark_end
        other_position_end = (
            other_position_start + 2 * (self.n_agents - 1)
        )

        communication_start = other_position_end

        for agent_name, observation in observations.items():
            observation = np.asarray(
                observation,
                dtype=np.float32,
            ).copy()

            # Mask landmarks outside the sight radius.
            for landmark_index in range(self.n_agents):
                start = landmark_start + 2 * landmark_index
                relative_position = observation[start:start + 2]

                if np.linalg.norm(relative_position) > self.sight_range:
                    observation[start:start + 2] = 0.0

            # Mask other agents and their communication features.
            for other_index in range(self.n_agents - 1):
                position_start = (
                    other_position_start + 2 * other_index
                )
                communication_index = (
                    communication_start + 2 * other_index
                )

                relative_position = observation[
                    position_start:position_start + 2
                ]

                if np.linalg.norm(relative_position) > self.sight_range:
                    observation[
                        position_start:position_start + 2
                    ] = 0.0
                    observation[
                        communication_index:communication_index + 2
                    ] = 0.0

            masked[agent_name] = observation

        return masked

    def reset(self, seed=None):
        observations, infos = super().reset(seed=seed)
        return self._mask_observations(observations), infos

    def step(self, actions):
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = super().step(actions)

        return (
            self._mask_observations(observations),
            rewards,
            terminations,
            truncations,
            infos,
        )
