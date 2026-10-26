import torch
import torch.nn as nn


class CommunicationEnhancedAgent(nn.Module):
    """
    Communication-enhanced action-value network.

    MATCH computes:

        Q_i(tau_i, m_i, ·)
            = Psi(h_i, m_i; theta_value)

    where:

        h_i:
            Agent historical representation.

        m_i:
            Personalized message returned by the Hub.

    Layer normalization is applied before concatenation
    because the Transformer hidden state and Hub latent
    message may have different and changing scales.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        message_dim: int = 32,
        n_actions: int = 5,
        output_initialization_gain: float = 0.01,
    ):
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive."
            )

        if message_dim <= 0:
            raise ValueError(
                "message_dim must be positive."
            )

        if n_actions <= 0:
            raise ValueError(
                "n_actions must be positive."
            )

        if output_initialization_gain <= 0.0:
            raise ValueError(
                "output_initialization_gain must "
                "be positive."
            )

        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.n_actions = n_actions

        self.hidden_normalization = nn.LayerNorm(
            hidden_dim
        )

        self.message_normalization = nn.LayerNorm(
            message_dim
        )

        self.value_layer = nn.Linear(
            hidden_dim + message_dim,
            n_actions,
        )

        # Transformer representations can initially have
        # a much larger scale than GRU representations.
        # A small action-value head initialization prevents
        # large initial Q-values from being amplified by
        # the QMIX mixer and Bellman bootstrap.
        nn.init.orthogonal_(
            self.value_layer.weight,
            gain=output_initialization_gain,
        )

        nn.init.zeros_(
            self.value_layer.bias
        )

    def forward(
        self,
        agent_hidden,
        hub_messages,
    ):
        if agent_hidden.ndim != 3:
            raise ValueError(
                "agent_hidden must have shape "
                "[batch_size, n_agents, hidden_dim]."
            )

        if hub_messages.ndim != 3:
            raise ValueError(
                "hub_messages must have shape "
                "[batch_size, n_agents, message_dim]."
            )

        if (
            agent_hidden.shape[:2]
            != hub_messages.shape[:2]
        ):
            raise ValueError(
                "agent_hidden and hub_messages must have "
                "the same batch and agent dimensions."
            )

        if (
            agent_hidden.shape[-1]
            != self.hidden_dim
        ):
            raise ValueError(
                "The last dimension of agent_hidden must "
                f"be {self.hidden_dim}, but received "
                f"{agent_hidden.shape[-1]}."
            )

        if (
            hub_messages.shape[-1]
            != self.message_dim
        ):
            raise ValueError(
                "The last dimension of hub_messages must "
                f"be {self.message_dim}, but received "
                f"{hub_messages.shape[-1]}."
            )

        normalized_hidden = (
            self.hidden_normalization(
                agent_hidden
            )
        )

        normalized_messages = (
            self.message_normalization(
                hub_messages
            )
        )

        combined_features = torch.cat(
            [
                normalized_hidden,
                normalized_messages,
            ],
            dim=-1,
        )

        q_values = self.value_layer(
            combined_features
        )

        return q_values
