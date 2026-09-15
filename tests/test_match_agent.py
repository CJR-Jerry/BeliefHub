import torch

from src.models.match_agent import MATCHAgent


def main():
    torch.manual_seed(0)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    batch_size = 4
    n_agents = 5
    n_entities = 10
    entity_observation_dim = 8
    hidden_dim = 32
    entity_embedding_dim = 32
    n_actions = 5

    model = MATCHAgent(
        n_agents=n_agents,
        n_entities=n_entities,
        entity_observation_dim=(
            entity_observation_dim
        ),
        hidden_dim=hidden_dim,
        query_dim=32,
        reconstructed_entity_dim=(
            entity_embedding_dim
        ),
        n_actions=n_actions,
    ).to(device)

    entity_observations = torch.randn(
        batch_size,
        n_agents,
        n_entities,
        entity_observation_dim,
        device=device,
        requires_grad=True,
    )

    observation_mask = (
        torch.rand(
            batch_size,
            n_agents,
            n_entities,
            device=device,
        )
        > 0.3
    )

    # Make sure that every agent has at least one
    # observable entity.
    observation_mask[:, :, 0] = True

    previous_hidden = model.initial_hidden(
        batch_size=batch_size,
        device=device,
    )

    reconstructed_entities = torch.randn(
        batch_size,
        n_entities,
        entity_embedding_dim,
        device=device,
        requires_grad=True,
    )

    reconstructed_state = torch.randn(
        batch_size,
        n_entities,
        8,
        device=device,
    )

    entity_mask = torch.ones(
        batch_size,
        n_entities,
        dtype=torch.bool,
        device=device,
    )

    result = model(
        entity_observations=entity_observations,
        observation_mask=observation_mask,
        previous_hidden=previous_hidden,
        reconstructed_entities=(
            reconstructed_entities
        ),
        reconstructed_state=reconstructed_state,
        entity_mask=entity_mask,
    )

    print("Device:", device)
    print(
        "Q-values:",
        result["q_values"].shape,
    )
    print(
        "Hidden states:",
        result["hidden_states"].shape,
    )
    print(
        "Queries:",
        result["queries"].shape,
    )
    print(
        "Hub messages:",
        result["hub_messages"].shape,
    )
    print(
        "Attention weights:",
        result["attention_weights"].shape,
    )

    assert result["q_values"].shape == (
        batch_size,
        n_agents,
        n_actions,
    )

    assert result["hidden_states"].shape == (
        batch_size,
        n_agents,
        hidden_dim,
    )

    assert result["queries"].shape == (
        batch_size,
        n_agents,
        32,
    )

    assert result["hub_messages"].shape == (
        batch_size,
        n_agents,
        entity_embedding_dim,
    )

    assert result["attention_weights"].shape == (
        batch_size,
        n_agents,
        n_entities,
    )

    attention_sums = result[
        "attention_weights"
    ].sum(dim=-1)

    assert torch.allclose(
        attention_sums,
        torch.ones_like(attention_sums),
        atol=1e-5,
    )

    assert torch.isfinite(
        result["q_values"]
    ).all()

    # Verify that the whole MATCH agent is
    # differentiable.
    loss = (
        result["q_values"].pow(2).mean()
        + result["hub_messages"].pow(2).mean()
    )

    loss.backward()

    assert entity_observations.grad is not None
    assert reconstructed_entities.grad is not None

    assert torch.isfinite(
        entity_observations.grad
    ).all()

    assert torch.isfinite(
        reconstructed_entities.grad
    ).all()

    # Verify temporal hidden-state propagation.
    next_result = model(
        entity_observations=(
            entity_observations.detach()
        ),
        observation_mask=observation_mask,
        previous_hidden=(
            result["hidden_states"].detach()
        ),
        reconstructed_entities=(
            reconstructed_entities.detach()
        ),
        reconstructed_state=reconstructed_state,
        entity_mask=entity_mask,
    )

    hidden_change = (
        next_result["hidden_states"]
        - result["hidden_states"].detach()
    ).abs().mean()

    print(
        "Mean temporal hidden change:",
        hidden_change.item(),
    )

    assert hidden_change.item() > 0.0

    print("Complete MATCH agent test passed.")


if __name__ == "__main__":
    main()