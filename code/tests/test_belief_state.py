import torch

from src.models.belief_state import (
    LocalBeliefState,
    ReliabilityGatedFusion,
    agent_centric_entity_targets,
    gaussian_belief_loss,
)
from src.models.match_agent import MATCHAgent


def test_local_belief_shapes_and_gradients():
    torch.manual_seed(0)
    batch_size, n_agents, n_entities = 2, 6, 12
    module = LocalBeliefState(
        n_agents=n_agents,
        n_entities=n_entities,
        entity_observation_dim=8,
        agent_hidden_dim=32,
        belief_hidden_dim=32,
        belief_state_dim=8,
        message_dim=32,
    )
    observations = torch.randn(batch_size, n_agents, n_entities, 8)
    mask = torch.rand(batch_size, n_agents, n_entities) > 0.3
    agent_hidden = torch.randn(batch_size, n_agents, 32)
    previous = module.initial_hidden(batch_size, observations.device)

    output = module(observations, mask, agent_hidden, previous)
    assert output["hidden"].shape == (batch_size, n_agents, n_entities, 32)
    assert output["mean"].shape == (batch_size, n_agents, n_entities, 8)
    assert output["logvar"].shape == (batch_size, n_agents, n_entities, 8)
    assert output["local_message"].shape == (batch_size, n_agents, 32)
    assert torch.isfinite(output["mean"]).all()

    (output["mean"].pow(2).mean() + output["logvar"].mean()).backward()
    assert module.mean_head.weight.grad is not None
    assert torch.isfinite(module.mean_head.weight.grad).all()


def test_no_communication_uses_local_message():
    torch.manual_seed(1)
    fusion = ReliabilityGatedFusion(message_dim=32)
    hub = torch.randn(3, 6, 32)
    local = torch.randn(3, 6, 32)
    variance = torch.ones(3, 6, 1)
    fused, gate = fusion(
        hub_message=hub,
        local_message=local,
        hub_reliability=torch.zeros(3, 1),
        mean_variance=variance,
    )
    assert torch.allclose(gate, torch.zeros_like(gate))
    assert torch.allclose(fused, local)


def test_agent_centric_targets_and_nll():
    batch_size, time_steps, n_agents, n_entities = 2, 4, 6, 12
    true_states = torch.randn(batch_size, time_steps, n_entities, 8)
    targets = agent_centric_entity_targets(true_states, n_agents=n_agents)
    for agent_index in range(n_agents):
        assert torch.allclose(
            targets[:, :, agent_index, agent_index, 0:2],
            torch.zeros_like(
                targets[:, :, agent_index, agent_index, 0:2]
            ),
            atol=1e-6,
        )

    logvar = torch.zeros_like(targets)
    observation_masks = torch.ones(
        batch_size,
        time_steps,
        n_agents,
        n_entities,
        dtype=torch.bool,
    )
    valid = torch.ones(batch_size, time_steps, dtype=torch.bool)
    statistics = gaussian_belief_loss(
        belief_mean=targets.clone().requires_grad_(True),
        belief_logvar=logvar,
        true_entity_states=true_states,
        observation_masks=observation_masks,
        state_valid_mask=valid,
        n_agents=n_agents,
        supervision="all",
    )
    assert torch.allclose(statistics["rmse"], torch.tensor(0.0))
    assert torch.allclose(statistics["loss"], torch.tensor(0.0))


def test_match_agent_local_belief_integration():
    torch.manual_seed(2)
    batch_size, n_agents, n_entities = 2, 6, 12
    model = MATCHAgent(
        n_agents=n_agents,
        n_entities=n_entities,
        entity_observation_dim=8,
        hidden_dim=32,
        query_dim=32,
        reconstructed_entity_dim=32,
        n_actions=5,
        use_local_belief=True,
        belief_hidden_dim=32,
        belief_state_dim=8,
    )
    observations = torch.randn(batch_size, n_agents, n_entities, 8)
    mask = torch.rand(batch_size, n_agents, n_entities) > 0.3
    hidden = model.initial_hidden(batch_size)
    belief_hidden = model.initial_belief_hidden(batch_size)
    result = model(
        entity_observations=observations,
        observation_mask=mask,
        previous_hidden=hidden,
        reconstructed_entities=torch.randn(batch_size, n_entities, 32),
        reconstructed_state=torch.randn(batch_size, n_entities, 8),
        previous_belief_hidden=belief_hidden,
        hub_reliability=torch.full((batch_size, 1), 0.7),
    )
    assert result["q_values"].shape == (batch_size, n_agents, 5)
    assert result["belief_mean"].shape == (
        batch_size,
        n_agents,
        n_entities,
        8,
    )
    assert result["belief_gate"].shape == (batch_size, n_agents, 1)
    assert torch.isfinite(result["q_values"]).all()

