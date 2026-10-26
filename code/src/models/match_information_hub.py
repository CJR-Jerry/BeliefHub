import torch
import torch.nn as nn

from src.models.information_hub import (
    SelfAttentiveEntityPool,
)
from src.models.spatiotemporal_reconstructor import (
    SpatiotemporalReconstructor,
)


class MATCHInformationHub(nn.Module):
    def __init__(
        self,
        n_entities=10,
        observation_dim=8,
        embedding_dim=32,
        state_dim=8,
        history_length=4,
    ):
        super().__init__()

        self.n_entities = n_entities
        self.state_dim = state_dim
        self.history_length = history_length

        self.entity_pool = SelfAttentiveEntityPool(
            observation_dim=observation_dim,
            embedding_dim=embedding_dim,
        )

        self.reconstructor = SpatiotemporalReconstructor(
            n_entities=n_entities,
            entity_embedding_dim=embedding_dim,
            state_dim=state_dim,
            history_length=history_length,
            n_heads=4,
            n_layers=2,
            feedforward_dim=128,
            dropout=0.0,
        )

    def forward(
        self,
        entity_observations,
        observation_mask,
        history_states,
        history_valid_mask,
    ):
        (
            aggregated_entities,
            pooling_weights,
            observed_entities,
        ) = self.entity_pool(
            entity_observations,
            observation_mask,
        )

        (
            reconstructed_state,
            latent_state,
        ) = self.reconstructor(
            current_entities=aggregated_entities,
            current_observed_mask=observed_entities,
            history_states=history_states,
            history_valid_mask=history_valid_mask,
        )

        return {
            "reconstructed_state": reconstructed_state,
            "latent_state": latent_state,
            "aggregated_entities": aggregated_entities,
            "pooling_weights": pooling_weights,
            "observed_entities": observed_entities,
        }


class HubHistoryBuffer:
    """
    Online history buffer used during environment interaction.

    The newest reconstructed state is stored at the end.
    """

    def __init__(
        self,
        history_length,
        n_entities,
        state_dim,
    ):
        self.history_length = history_length
        self.n_entities = n_entities
        self.state_dim = state_dim

        self.states = None
        self.valid_mask = None

    def reset(self, batch_size, device):
        self.states = torch.zeros(
            batch_size,
            self.history_length,
            self.n_entities,
            self.state_dim,
            device=device,
        )

        self.valid_mask = torch.zeros(
            batch_size,
            self.history_length,
            dtype=torch.bool,
            device=device,
        )

    def get(self):
        if self.states is None:
            raise RuntimeError(
                "History buffer must be reset first."
            )

        return self.states, self.valid_mask

    def append(self, reconstructed_state):
        if self.states is None:
            raise RuntimeError(
                "History buffer must be reset first."
            )

        new_state = reconstructed_state.detach().unsqueeze(1)

        self.states = torch.cat(
            [
                self.states[:, 1:],
                new_state,
            ],
            dim=1,
        )

        new_valid = torch.ones(
            self.valid_mask.shape[0],
            1,
            dtype=torch.bool,
            device=self.valid_mask.device,
        )

        self.valid_mask = torch.cat(
            [
                self.valid_mask[:, 1:],
                new_valid,
            ],
            dim=1,
        )
