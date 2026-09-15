import torch

from src.models.information_hub import SelfAttentiveEntityPool


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    batch_size = 4
    n_agents = 5
    n_entities = 10
    observation_dim = 6
    embedding_dim = 32

    pool = SelfAttentiveEntityPool(
        observation_dim=observation_dim,
        embedding_dim=embedding_dim,
    ).to(device)

    entity_observations = torch.randn(
        batch_size,
        n_agents,
        n_entities,
        observation_dim,
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

    # 强制第0个实体完全不可见，模拟严重丢包
    observation_mask[:, :, 0] = False

    (
        aggregated_entities,
        attention_weights,
        observed_entities,
    ) = pool(
        entity_observations,
        observation_mask,
    )

    print("Device:", device)
    print(
        "Aggregated entity shape:",
        aggregated_entities.shape,
    )
    print(
        "Attention weight shape:",
        attention_weights.shape,
    )
    print(
        "Observed entity shape:",
        observed_entities.shape,
    )

    assert aggregated_entities.shape == (
        batch_size,
        n_entities,
        embedding_dim,
    )

    assert attention_weights.shape == (
        batch_size,
        n_agents,
        n_entities,
    )

    assert observed_entities.shape == (
        batch_size,
        n_entities,
    )

    # 完全不可见实体的权重和聚合表示应为0
    assert torch.allclose(
        attention_weights[:, :, 0],
        torch.zeros_like(
            attention_weights[:, :, 0]
        ),
    )

    assert torch.allclose(
        aggregated_entities[:, 0],
        torch.zeros_like(
            aggregated_entities[:, 0]
        ),
    )

    # 被观察实体的注意力权重之和应为1
    weight_sums = attention_weights.sum(dim=1)

    visible = observed_entities

    assert torch.allclose(
        weight_sums[visible],
        torch.ones_like(weight_sums[visible]),
        atol=1e-5,
    )

    # 检查梯度能否传播
    loss = aggregated_entities.pow(2).mean()
    loss.backward()

    assert entity_observations.grad is not None
    assert torch.isfinite(entity_observations.grad).all()

    print("Information Hub pooling test passed.")


if __name__ == "__main__":
    main()