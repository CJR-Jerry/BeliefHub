#!/usr/bin/env python3

import torch

from src.envs.cooperative_navigation import CooperativeNavigation
from src.models.match_agent import MATCHAgent


def main():
    assert torch.cuda.is_available(), "CUDA is not available"

    env = CooperativeNavigation(
        n_agents=6,
        max_cycles=25,
        occupancy_radius=0.3,
    )
    env.reset(seed=0)
    first_agent = env.possible_agents[0]
    observation_dim = env.observation_space(first_agent).shape[0]
    state_dim = int(env.env.state().size)
    n_actions = env.action_space(first_agent).n
    assert observation_dim == 36, observation_dim
    assert state_dim == 216, state_dim
    assert n_actions == 5, n_actions

    agent = MATCHAgent(
        n_agents=6,
        n_entities=12,
        entity_observation_dim=8,
        hidden_dim=32,
        query_dim=32,
        reconstructed_entity_dim=32,
        n_actions=5,
        use_local_belief=True,
        belief_hidden_dim=32,
        belief_state_dim=8,
        fuse_local_belief=False,
        use_belief_history=True,
    ).cuda()

    assert agent.use_local_belief
    assert not agent.fuse_local_belief
    assert agent.use_belief_history

    print("CN6 NoFusion-Invisible check: PASS")
    print("PyTorch:", torch.__version__)
    print("GPU:", torch.cuda.get_device_name(0))
    print("agents=6 obs=36 state=216 actions=5")
    print("belief_supervision=invisible is set by the launcher")
    print("belief_fusion=disabled")
    env.close()


if __name__ == "__main__":
    main()

