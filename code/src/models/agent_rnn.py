import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentRNN(nn.Module):
    """
    Shared recurrent Q-network used by all agents.

    Input:
        local observation
        previous action
        agent identity

    Output:
        Q-value of every available action
    """

    def __init__(
        self,
        observation_dim: int,
        n_actions: int,
        n_agents: int,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.observation_dim = observation_dim
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim

        input_dim = observation_dim + n_actions + n_agents

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, n_actions)

    def initial_hidden(self, batch_size: int, device=None):
        return torch.zeros(
            batch_size,
            self.n_agents,
            self.hidden_dim,
            device=device,
        )

    def forward(
        self,
        observations,
        previous_actions,
        hidden_states,
    ):
        batch_size, n_agents, _ = observations.shape

        agent_ids = torch.eye(
            self.n_agents,
            device=observations.device,
        )

        agent_ids = agent_ids.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )

        network_input = torch.cat(
            [
                observations,
                previous_actions,
                agent_ids,
            ],
            dim=-1,
        )

        network_input = network_input.reshape(
            batch_size * n_agents,
            -1,
        )

        hidden_states = hidden_states.reshape(
            batch_size * n_agents,
            self.hidden_dim,
        )

        features = F.relu(self.input_layer(network_input))
        next_hidden = self.gru(features, hidden_states)
        q_values = self.output_layer(next_hidden)

        q_values = q_values.reshape(
            batch_size,
            n_agents,
            self.n_actions,
        )

        next_hidden = next_hidden.reshape(
            batch_size,
            n_agents,
            self.hidden_dim,
        )

        return q_values, next_hidden
