import torch
import torch.nn as nn
import torch.nn.functional as F


class QMixer(nn.Module):
    """
    Mix individual agent Q-values into a joint team Q-value.

    Positive mixing weights guarantee:
        dQ_total / dQ_agent >= 0
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        mixing_hidden_dim: int = 32,
        hypernet_hidden_dim: int = 64,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.state_dim = state_dim
        self.mixing_hidden_dim = mixing_hidden_dim

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(
                hypernet_hidden_dim,
                n_agents * mixing_hidden_dim,
            ),
        )

        self.hyper_b1 = nn.Linear(
            state_dim,
            mixing_hidden_dim,
        )

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(
                hypernet_hidden_dim,
                mixing_hidden_dim,
            ),
        )

        self.state_value = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, 1),
        )

    def forward(self, agent_q_values, states):
        """
        agent_q_values:
            [batch, n_agents] or [batch, time, n_agents]

        states:
            [batch, state_dim] or [batch, time, state_dim]
        """
        leading_shape = agent_q_values.shape[:-1]

        flat_agent_q = agent_q_values.reshape(
            -1,
            1,
            self.n_agents,
        )

        flat_states = states.reshape(
            -1,
            self.state_dim,
        )

        weight_1 = torch.abs(self.hyper_w1(flat_states))
        weight_1 = weight_1.reshape(
            -1,
            self.n_agents,
            self.mixing_hidden_dim,
        )

        bias_1 = self.hyper_b1(flat_states)
        bias_1 = bias_1.reshape(
            -1,
            1,
            self.mixing_hidden_dim,
        )

        hidden = F.elu(
            torch.bmm(flat_agent_q, weight_1) + bias_1
        )

        weight_2 = torch.abs(self.hyper_w2(flat_states))
        weight_2 = weight_2.reshape(
            -1,
            self.mixing_hidden_dim,
            1,
        )

        state_value = self.state_value(flat_states)
        state_value = state_value.reshape(-1, 1, 1)

        total_q = (
            torch.bmm(hidden, weight_2)
            + state_value
        )

        return total_q.reshape(*leading_shape, 1)