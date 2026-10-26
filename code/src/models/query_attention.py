import math

import torch
import torch.nn as nn


class AgentEntityHistoryEncoder(nn.Module):
    """
    Encode each agent's entity observations together with
    its previous historical hidden state.
    """

    def __init__(
        self,
        n_agents=5,
        n_entities=10,
        entity_observation_dim=8,
        hidden_dim=32,
        n_heads=4,
        n_layers=2,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.n_entities = n_entities
        self.hidden_dim = hidden_dim

        self.entity_projection = nn.Linear(
            entity_observation_dim,
            hidden_dim,
        )

        self.agent_embedding = nn.Embedding(
            n_agents,
            hidden_dim,
        )

        self.entity_embedding = nn.Embedding(
            n_entities,
            hidden_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

    def initial_hidden(self, batch_size, device):
        return torch.zeros(
            batch_size,
            self.n_agents,
            self.hidden_dim,
            device=device,
        )

    def forward(
        self,
        entity_observations,
        observation_mask,
        previous_hidden,
    ):
        batch_size = entity_observations.shape[0]
        device = entity_observations.device

        projected_entities = self.entity_projection(
            entity_observations
        )

        entity_indices = torch.arange(
            self.n_entities,
            device=device,
        )

        entity_positions = self.entity_embedding(
            entity_indices
        ).view(
            1,
            1,
            self.n_entities,
            self.hidden_dim,
        )

        projected_entities = (
            projected_entities + entity_positions
        )

        agent_indices = torch.arange(
            self.n_agents,
            device=device,
        )

        agent_identity = self.agent_embedding(
            agent_indices
        ).view(
            1,
            self.n_agents,
            self.hidden_dim,
        )

        history_token = (
            previous_hidden + agent_identity
        ).unsqueeze(2)

        tokens = torch.cat(
            [
                history_token,
                projected_entities,
            ],
            dim=2,
        )

        tokens = tokens.reshape(
            batch_size * self.n_agents,
            self.n_entities + 1,
            self.hidden_dim,
        )

        history_valid = torch.ones(
            batch_size,
            self.n_agents,
            1,
            dtype=torch.bool,
            device=device,
        )

        valid_tokens = torch.cat(
            [
                history_valid,
                observation_mask.bool(),
            ],
            dim=2,
        )

        padding_mask = ~valid_tokens.reshape(
            batch_size * self.n_agents,
            self.n_entities + 1,
        )

        encoded_tokens = self.transformer(
            tokens,
            src_key_padding_mask=padding_mask,
        )

        new_hidden = encoded_tokens[:, 0].reshape(
            batch_size,
            self.n_agents,
            self.hidden_dim,
        )

        return new_hidden


class AgentQueryGenerator(nn.Module):
    def __init__(
        self,
        hidden_dim=32,
        query_dim=32,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, query_dim),
        )

    def forward(self, agent_hidden):
        return self.network(agent_hidden)


class DistanceAwareAttention(nn.Module):
    """
    Generate an agent-specific message from reconstructed
    global entity representations.

    score_ij =
        lambda * content_score_ij * sigmoid(-distance_ij)
    """

    def __init__(
        self,
        query_dim=32,
        entity_dim=32,
    ):
        super().__init__()

        self.query_dim = query_dim

        self.entity_projection = nn.Linear(
            entity_dim,
            query_dim,
            bias=False,
        )

        self.scale_parameter = nn.Parameter(
            torch.tensor(1.0)
        )

    def forward(
        self,
        queries,
        entity_representations,
        agent_positions,
        entity_positions,
        entity_mask=None,
    ):
        projected_entities = self.entity_projection(
            entity_representations
        )

        content_scores = torch.einsum(
            "bnq,beq->bne",
            queries,
            projected_entities,
        )

        content_scores = (
            content_scores / math.sqrt(self.query_dim)
        )

        distances = torch.cdist(
            agent_positions,
            entity_positions,
            p=2,
        )

        maximum_distance = distances.amax(
            dim=-1,
            keepdim=True,
        )

        normalized_distances = (
            distances
            / maximum_distance.clamp(min=1e-8)
        )

        locality_weight = torch.sigmoid(
            -normalized_distances
        )

        attention_scores = (
            self.scale_parameter
            * content_scores
            * locality_weight
        )

        if entity_mask is not None:
            expanded_mask = entity_mask.bool()

            if expanded_mask.ndim == 2:
                expanded_mask = expanded_mask.unsqueeze(1)

            attention_scores = attention_scores.masked_fill(
                ~expanded_mask,
                -1e9,
            )

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )

        messages = torch.einsum(
            "bne,bed->bnd",
            attention_weights,
            entity_representations,
        )

        return {
            "messages": messages,
            "attention_weights": attention_weights,
            "normalized_distances": normalized_distances,
            "locality_weights": locality_weight,
        }
