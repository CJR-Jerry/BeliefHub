import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class MASIAStateEncoder(nn.Module):
    """Attention autoencoder used by MASIA to aggregate agent observations."""

    def __init__(
        self,
        input_dim,
        state_dim,
        n_agents,
        latent_dim=8,
        attention_dim=16,
        hidden_dim=32,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.latent_dim = latent_dim
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim

        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.encoder_rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.encoder = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Linear(n_agents * latent_dim, state_dim)

    def initial_hidden(self, batch_size, device):
        return torch.zeros(
            batch_size,
            self.n_agents,
            self.hidden_dim,
            device=device,
        )

    def encode(self, inputs, hidden, message_mask=None):
        batch_size = inputs.shape[0]
        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)

        logits = torch.matmul(query, key.transpose(1, 2))
        logits = logits / (self.attention_dim ** 0.5)

        if message_mask is not None:
            sender_mask = message_mask.bool().unsqueeze(1)
            logits = logits.masked_fill(~sender_mask, -1e9)

        scores = torch.softmax(logits, dim=-1)

        if message_mask is not None:
            sender_mask_float = message_mask.unsqueeze(1).to(scores.dtype)
            scores = scores * sender_mask_float
            scores = scores / scores.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        attended = torch.matmul(scores, value)
        next_hidden = self.encoder_rnn(
            attended.reshape(batch_size * self.n_agents, self.hidden_dim),
            hidden.reshape(batch_size * self.n_agents, self.hidden_dim),
        ).reshape(batch_size, self.n_agents, self.hidden_dim)

        latent = self.encoder(next_hidden).reshape(batch_size, -1)
        return latent, next_hidden

    def forward(self, inputs, hidden, message_mask=None):
        latent, next_hidden = self.encode(inputs, hidden, message_mask)
        reconstruction = self.decoder(latent)
        return reconstruction, latent, next_hidden


class MASIAAgent(nn.Module):
    """MASIA agent adapted to the repository's QMIX runner interface."""

    def __init__(
        self,
        observation_dim,
        state_dim,
        n_actions,
        n_agents,
        hidden_dim=64,
        encoder_hidden_dim=32,
        state_repre_dim=8,
        attention_dim=16,
        observation_embed_dim=32,
        spr_dim=32,
        momentum_tau=1.0,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.latent_dim = state_repre_dim * n_agents
        self.momentum_tau = momentum_tau

        encoder_input_dim = observation_dim + n_agents
        self.encoder = MASIAStateEncoder(
            input_dim=encoder_input_dim,
            state_dim=state_dim,
            n_agents=n_agents,
            latent_dim=state_repre_dim,
            attention_dim=attention_dim,
            hidden_dim=encoder_hidden_dim,
        )

        self.projection = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, spr_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(spr_dim, 64),
            nn.ReLU(),
            nn.Linear(64, spr_dim),
        )

        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_projection = copy.deepcopy(self.projection)
        for parameter in list(self.target_encoder.parameters()) + list(
            self.target_projection.parameters()
        ):
            parameter.requires_grad = False

        self.gate = nn.Linear(encoder_input_dim, self.latent_dim)
        self.observation_embedding = nn.Linear(
            observation_dim,
            observation_embed_dim,
        )
        self.input_layer = nn.Linear(
            observation_embed_dim + self.latent_dim + n_agents,
            hidden_dim,
        )
        self.policy_rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, n_actions)

    def _agent_ids(self, batch_size, device):
        return torch.eye(self.n_agents, device=device).unsqueeze(0).expand(
            batch_size, -1, -1
        )

    def _encoder_inputs(self, observations):
        return torch.cat(
            [
                observations,
                self._agent_ids(observations.shape[0], observations.device),
            ],
            dim=-1,
        )

    def initial_hidden(self, batch_size, device=None):
        policy_hidden = torch.zeros(
            batch_size,
            self.n_agents,
            self.hidden_dim,
            device=device,
        )
        encoder_hidden = self.encoder.initial_hidden(batch_size, device)
        return policy_hidden, encoder_hidden

    def _policy(self, observations, latent, policy_hidden):
        batch_size = observations.shape[0]
        ids = self._agent_ids(batch_size, observations.device)
        encoder_inputs = torch.cat([observations, ids], dim=-1)
        weighted_latent = torch.sigmoid(self.gate(encoder_inputs)) * latent.unsqueeze(1)
        observation_embedding = self.observation_embedding(observations)
        policy_input = torch.cat(
            [observation_embedding, weighted_latent, ids],
            dim=-1,
        )
        features = F.relu(self.input_layer(policy_input))
        next_policy_hidden = self.policy_rnn(
            features.reshape(batch_size * self.n_agents, self.hidden_dim),
            policy_hidden.reshape(batch_size * self.n_agents, self.hidden_dim),
        ).reshape(batch_size, self.n_agents, self.hidden_dim)
        q_values = self.output_layer(next_policy_hidden)
        return q_values, next_policy_hidden

    def forward_full(
        self,
        observations,
        previous_actions,
        hidden_states,
        message_mask=None,
    ):
        del previous_actions  # Official MASIA config uses obs_last_action=False.
        policy_hidden, encoder_hidden = hidden_states
        encoder_inputs = self._encoder_inputs(observations)
        reconstruction, latent, next_encoder_hidden = self.encoder(
            encoder_inputs,
            encoder_hidden,
            message_mask,
        )
        q_values, next_policy_hidden = self._policy(
            observations,
            latent,
            policy_hidden,
        )
        return (
            q_values,
            (next_policy_hidden, next_encoder_hidden),
            reconstruction,
            latent,
        )

    def forward(
        self,
        observations,
        previous_actions,
        hidden_states,
        message_mask=None,
    ):
        q_values, next_hidden, _, _ = self.forward_full(
            observations,
            previous_actions,
            hidden_states,
            message_mask,
        )
        return q_values, next_hidden

    def target_project_sequence(self, observations, message_masks):
        batch_size, sequence_length = observations.shape[:2]
        hidden = self.target_encoder.initial_hidden(
            batch_size,
            observations.device,
        )
        outputs = []
        for timestep in range(sequence_length):
            encoder_inputs = self._encoder_inputs(observations[:, timestep])
            latent, hidden = self.target_encoder.encode(
                encoder_inputs,
                hidden,
                message_masks[:, timestep],
            )
            outputs.append(self.target_projection(latent))
        return torch.stack(outputs, dim=1)

    @torch.no_grad()
    def momentum_update(self):
        tau = self.momentum_tau
        for target, online in zip(
            self.target_encoder.parameters(),
            self.encoder.parameters(),
        ):
            target.data.mul_(1.0 - tau).add_(online.data, alpha=tau)
        for target, online in zip(
            self.target_projection.parameters(),
            self.projection.parameters(),
        ):
            target.data.mul_(1.0 - tau).add_(online.data, alpha=tau)

