import numpy as np


def build_global_entity_state(world):
    """
    Build the true global entity state from the MPE world.

    Feature layout:
        0:2  absolute position
        2:4  velocity
        4:6  communication
        6:8  entity type [agent, landmark]

    Output:
        [n_agents + n_landmarks, 8]
    """

    n_agents = len(world.agents)
    n_landmarks = len(world.landmarks)
    n_entities = n_agents + n_landmarks

    state = np.zeros(
        (n_entities, 8),
        dtype=np.float32,
    )

    for agent_index, agent in enumerate(world.agents):
        state[agent_index, 0:2] = agent.state.p_pos
        state[agent_index, 2:4] = agent.state.p_vel

        if agent.state.c is not None:
            communication = np.asarray(
                agent.state.c,
                dtype=np.float32,
            )

            state[
                agent_index,
                4:4 + min(2, communication.size),
            ] = communication[:2]

        state[agent_index, 6:8] = np.array(
            [1.0, 0.0],
            dtype=np.float32,
        )

    for landmark_index, landmark in enumerate(
        world.landmarks
    ):
        entity_index = n_agents + landmark_index

        state[entity_index, 0:2] = (
            landmark.state.p_pos
        )

        state[entity_index, 6:8] = np.array(
            [0.0, 1.0],
            dtype=np.float32,
        )

    return state
