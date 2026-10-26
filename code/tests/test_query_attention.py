import torch

from src.models.query_attention import (
    AgentEntityHistoryEncoder,
    AgentQueryGenerator,
    DistanceAwareAttention,
)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    batch_size = 4
    n_agents = 5
    n_entities = 10

    history_encoder = AgentEntityHistoryEncoder(
        n_agents=n_agents,
        n_entities=n_entities,
        entity_observation_dim=8,
        hidden_dim=32,
    ).to(device)

    query_generator = AgentQueryGenerator(
        hidden_dim=32,
        query_dim=32,
    ).to(device)

    attention = DistanceAwareAttention(
        query_dim=32,
        entity_dim=32,
    ).to(device)

    entity_observations = torch.randn(
        batch_size,
        n_agents,
        n_entities,
        8,
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

    previous_hidden = history_encoder.initial_hidden(
        batch_size=batch_size,
        device=device,
    )

    new_hidden = history_encoder(
        entity_observations,
        observation_mask,
        previous_hidden,
    )

    queries = query_generator(new_hidden)

    reconstructed_entities = torch.randn(
        batch_size,
        n_entities,
        32,
        device=device,
        requires_grad=True,
    )

    reconstructed_state = torch.randn(
        batch_size,
        n_entities,
        8,
        device=device,
    )

    agent_positions = reconstructed_state[
        :, :n_agents, 0:2
    ]

    entity_positions = reconstructed_state[
        :, :, 0:2
    ]

    result = attention(
        queries=queries,
        entity_representations=reconstructed_entities,
        agent_positions=agent_positions,
        entity_positions=entity_positions,
    )

    print("Device:", device)
    print("Agent hidden:", new_hidden.shape)
    print("Queries:", queries.shape)
    print("Messages:", result["messages"].shape)
    print(
        "Attention weights:",
        result["attention_weights"].shape,
    )
    print(
        "Normalized distances:",
        result["normalized_distances"].shape,
    )

    assert new_hidden.shape == (4, 5, 32)
    assert queries.shape == (4, 5, 32)
    assert result["messages"].shape == (4, 5, 32)
    assert result["attention_weights"].shape == (
        4,
        5,
        10,
    )

    weight_sums = result[
        "attention_weights"
    ].sum(dim=-1)

    assert torch.allclose(
        weight_sums,
        torch.ones_like(weight_sums),
        atol=1e-5,
    )

    loss = (
        result["messages"].pow(2).mean()
        + new_hidden.pow(2).mean()
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

    print("Query attention test passed.")


if __name__ == "__main__":
    main()
