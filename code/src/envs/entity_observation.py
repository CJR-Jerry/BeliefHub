import numpy as np


class MPEEntityObservationAdapter:
    """
    Convert PettingZoo Simple Spread observations into
    observer-entity representations.

    Entity feature layout:
        0:2  relative position
        2:4  observed velocity
        4:6  communication feature
        6:8  entity type: [agent, landmark]
    """

    def __init__(
        self,
        n_agents: int,
        n_landmarks: int,
        sight_range=None,
    ):
        self.n_agents = n_agents
        self.n_landmarks = n_landmarks
        self.n_entities = n_agents + n_landmarks
        self.entity_observation_dim = 8
        self.sight_range = sight_range

    def convert(
        self,
        observations,
        agent_names,
        packet_loss=0.0,
        rng=None,
    ):
        if rng is None:
            rng = np.random.default_rng()

        entity_observations = np.zeros(
            (
                self.n_agents,
                self.n_entities,
                self.entity_observation_dim,
            ),
            dtype=np.float32,
        )

        observation_mask = np.zeros(
            (
                self.n_agents,
                self.n_entities,
            ),
            dtype=bool,
        )

        landmark_start = 4
        landmark_end = (
            landmark_start + 2 * self.n_landmarks
        )

        other_position_start = landmark_end
        other_position_end = (
            other_position_start
            + 2 * (self.n_agents - 1)
        )

        communication_start = other_position_end
        communication_end = (
            communication_start
            + 2 * (self.n_agents - 1)
        )

        for observer_index, observer_name in enumerate(
            agent_names
        ):
            observation = np.asarray(
                observations[observer_name],
                dtype=np.float32,
            )

            self_velocity = observation[0:2]

            landmark_positions = observation[
                landmark_start:landmark_end
            ].reshape(self.n_landmarks, 2)

            other_agent_positions = observation[
                other_position_start:other_position_end
            ].reshape(self.n_agents - 1, 2)

            other_communications = observation[
                communication_start:communication_end
            ].reshape(self.n_agents - 1, 2)

            # 观察者自身
            entity_observations[
                observer_index,
                observer_index,
                0:2,
            ] = 0.0

            entity_observations[
                observer_index,
                observer_index,
                2:4,
            ] = self_velocity

            entity_observations[
                observer_index,
                observer_index,
                6:8,
            ] = np.array([1.0, 0.0])

            observation_mask[
                observer_index,
                observer_index,
            ] = True

            # 其他智能体按照全局智能体顺序排列
            other_indices = [
                index
                for index in range(self.n_agents)
                if index != observer_index
            ]

            for local_index, entity_index in enumerate(
                other_indices
            ):
                relative_position = (
                    other_agent_positions[local_index]
                )



                entity_observations[
                    observer_index,
                    entity_index,
                    0:2,
                ] = relative_position

                entity_observations[
                    observer_index,
                    entity_index,
                    4:6,
                ] = other_communications[local_index]

                entity_observations[
                    observer_index,
                    entity_index,
                    6:8,
                ] = np.array([1.0, 0.0])

                observation_mask[
                    observer_index,
                    entity_index,
                ] = self.is_visible(relative_position)

            # 地标
            for landmark_index in range(self.n_landmarks):
                entity_index = (
                    self.n_agents + landmark_index
                )

                relative_position = landmark_positions[
                    landmark_index
                ]

                entity_observations[
                    observer_index,
                    entity_index,
                    0:2,
                ] = relative_position

                entity_observations[
                    observer_index,
                    entity_index,
                    6:8,
                ] = np.array([0.0, 1.0])

                observation_mask[
                    observer_index,
                    entity_index,
                ] = self.is_visible(relative_position)

            # 模拟观察者到Information Hub的整条消息丢失
            message_received = rng.random() >= packet_loss

            if not message_received:
                observation_mask[observer_index, :] = False

        return entity_observations, observation_mask

    def is_visible(self, relative_position):
        if self.sight_range is None:
            return True

        distance = np.linalg.norm(relative_position)
        return bool(distance <= self.sight_range)
