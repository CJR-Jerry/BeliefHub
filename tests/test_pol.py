import numpy as np

from src.envs.cooperative_navigation import CooperativeNavigation


def main():
    env = CooperativeNavigation(
        n_agents=5,
        max_cycles=25,
        occupancy_radius=0.3,
    )

    env.reset(seed=0)
    world = env.env.unwrapped.world

    # 情况一：每个智能体正好位于一个不同地标上
    for agent, landmark in zip(world.agents, world.landmarks):
        agent.state.p_pos = landmark.state.p_pos.copy()

    full_pol = env.calculate_pol()
    print(f"All landmarks occupied: POL = {full_pol:.2f}")
    assert np.isclose(full_pol, 1.0)

    # 情况二：把所有智能体移动到远离地标的位置
    for agent in world.agents:
        agent.state.p_pos = np.array([10.0, 10.0], dtype=np.float32)

    empty_pol = env.calculate_pol()
    print(f"No landmarks occupied: POL = {empty_pol:.2f}")
    assert np.isclose(empty_pol, 0.0)

    env.close()
    print("POL test passed.")


if __name__ == "__main__":
    main()