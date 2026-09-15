import torch
import torch.nn as nn


class SelfAttentiveEntityPool(nn.Module):
    """
    Aggregate observations of the same entity from multiple agents.

    Inputs:
        entity_observations:
            [batch, observers, entities, observation_dim]

        observation_mask:
            [batch, observers, entities]
            True means the observation was successfully received.

    Outputs:
        aggregated_entities:
            [batch, entities, embedding_dim]

        attention_weights:
            [batch, observers, entities]

        observed_entities:
            [batch, entities]
    """

    def __init__(
        self,
        observation_dim: int,
        embedding_dim: int,
    ):
        super().__init__()

        self.observation_dim = observation_dim
        self.embedding_dim = embedding_dim

        self.pool_projection = nn.Linear(
            observation_dim,
            embedding_dim,
        )

        self.pool_query = nn.Parameter(
            torch.empty(embedding_dim)
        )

        nn.init.normal_(
            self.pool_query,
            mean=0.0,
            std=embedding_dim ** -0.5,
        )

    def forward(
        self,
        entity_observations,
        observation_mask,
    ):
        observation_mask = observation_mask.bool()

        projected_observations = self.pool_projection(
            entity_observations
        )

        attention_scores = torch.einsum(
            "bnej,j->bne",
            projected_observations,
            self.pool_query,
        )

        attention_scores = attention_scores.masked_fill(
            ~observation_mask,
            -1e9,
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=1,
        )

        # 删除丢失消息对应的权重
        attention_weights = (
            attention_weights
            * observation_mask.float()
        )

        # 当某个实体完全没有被观察到时，避免出现NaN
        normalizer = attention_weights.sum(
            dim=1,
            keepdim=True,
        )

        attention_weights = torch.where(
            normalizer > 0,
            attention_weights
            / normalizer.clamp(min=1e-8),
            torch.zeros_like(attention_weights),
        )

        aggregated_entities = torch.sum(
            attention_weights.unsqueeze(-1)
            * projected_observations,
            dim=1,
        )

        observed_entities = observation_mask.any(dim=1)

        return (
            aggregated_entities,
            attention_weights,
            observed_entities,
        )