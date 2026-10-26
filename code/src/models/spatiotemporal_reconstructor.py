import torch
import torch.nn as nn


class SpatiotemporalReconstructor(nn.Module):
    """
    Transformer-based spatiotemporal state reconstruction.

    Current aggregated observations:
        [batch, entities, embedding_dim]

    Historical reconstructed states:
        [batch, history_length, entities, state_dim]

    Output reconstructed state:
        [batch, entities, state_dim]
    """

    def __init__(
        self,
        n_entities: int,
        entity_embedding_dim: int = 32,
        state_dim: int = 8,
        history_length: int = 4,
        n_heads: int = 4,
        n_layers: int = 2,
        feedforward_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.n_entities = n_entities
        self.entity_embedding_dim = entity_embedding_dim
        self.state_dim = state_dim
        self.history_length = history_length

        self.history_projection = nn.Linear(
            state_dim,
            entity_embedding_dim,
        )

        self.temporal_embedding = nn.Embedding(
            history_length + 1,
            entity_embedding_dim,
        )

        self.entity_embedding = nn.Embedding(
            n_entities,
            entity_embedding_dim,
        )

        self.missing_entity_token = nn.Parameter(
            torch.zeros(entity_embedding_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=entity_embedding_dim,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.decoder = nn.Sequential(
            nn.LayerNorm(entity_embedding_dim),
            nn.Linear(
                entity_embedding_dim,
                entity_embedding_dim,
            ),
            nn.GELU(),
            nn.Linear(
                entity_embedding_dim,
                state_dim,
            ),
        )

    def forward(
        self,
        current_entities,
        current_observed_mask,
        history_states,
        history_valid_mask=None,
    ):
        batch_size = current_entities.shape[0]
        device = current_entities.device

        if history_states.shape != (
            batch_size,
            self.history_length,
            self.n_entities,
            self.state_dim,
        ):
            raise ValueError(
                "Incorrect history shape: "
                f"{history_states.shape}"
            )

        if history_valid_mask is None:
            history_valid_mask = torch.ones(
                batch_size,
                self.history_length,
                dtype=torch.bool,
                device=device,
            )
        else:
            history_valid_mask = history_valid_mask.bool()

        projected_history = self.history_projection(
            history_states
        )

        current_observed_mask = (
            current_observed_mask.bool()
        )

        missing_mask = ~current_observed_mask

        current_tokens = current_entities + (
            missing_mask.unsqueeze(-1).float()
            * self.missing_entity_token
        )

        all_tokens = torch.cat(
            [
                projected_history,
                current_tokens.unsqueeze(1),
            ],
            dim=1,
        )


        temporal_indices = torch.arange(
            self.history_length + 1,
            device=device,
        )

        entity_indices = torch.arange(
            self.n_entities,
            device=device,
        )

        temporal_positions = self.temporal_embedding(
            temporal_indices
        ).view(
            1,
            self.history_length + 1,
            1,
            self.entity_embedding_dim,
        )

        entity_positions = self.entity_embedding(
            entity_indices
        ).view(
            1,
            1,
            self.n_entities,
            self.entity_embedding_dim,
        )

        all_tokens = (
            all_tokens
            + temporal_positions
            + entity_positions
        )

        flattened_tokens = all_tokens.reshape(
            batch_size,
            (self.history_length + 1)
            * self.n_entities,
            self.entity_embedding_dim,
        )

        # 无效历史位置不参与注意力
        valid_times = torch.cat(
            [
                history_valid_mask,
                torch.ones(
                    batch_size,
                    1,
                    dtype=torch.bool,
                    device=device,
                ),
            ],
            dim=1,
        )

        valid_tokens = valid_times.unsqueeze(-1).expand(
            -1,
            -1,
            self.n_entities,
        )

        padding_mask = ~valid_tokens.reshape(
            batch_size,
            -1,
        )

        transformed_tokens = self.transformer(
            flattened_tokens,
            src_key_padding_mask=padding_mask,
        )

        transformed_tokens = transformed_tokens.reshape(
            batch_size,
            self.history_length + 1,
            self.n_entities,
            self.entity_embedding_dim,
        )

        current_latent_state = transformed_tokens[:, -1]

        reconstructed_state = self.decoder(
            current_latent_state
        )

        return reconstructed_state, current_latent_state
