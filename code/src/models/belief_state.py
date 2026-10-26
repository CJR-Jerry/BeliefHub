import torch
import torch.nn as nn


class LocalBeliefState(nn.Module):
    """
    Per-agent, per-entity recurrent probabilistic belief.

    The module only consumes information available during decentralized
    execution: the agent's local entity observations, visibility mask, and
    historical representation. It predicts an agent-centric entity state
    distribution and a compact local message for action selection.
    """

    def __init__(
        self,
        n_agents,
        n_entities,
        entity_observation_dim,
        agent_hidden_dim,
        belief_hidden_dim=32,
        belief_state_dim=8,
        message_dim=32,
        logvar_min=-6.0,
        logvar_max=2.0,
    ):
        super().__init__()

        if logvar_min >= logvar_max:
            raise ValueError("logvar_min must be smaller than logvar_max.")

        self.n_agents = n_agents
        self.n_entities = n_entities
        self.belief_hidden_dim = belief_hidden_dim
        self.belief_state_dim = belief_state_dim
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        self.input_projection = nn.Sequential(
            nn.Linear(
                entity_observation_dim + agent_hidden_dim + 1,
                belief_hidden_dim,
            ),
            nn.GELU(),
        )
        self.agent_embedding = nn.Embedding(n_agents, belief_hidden_dim)
        self.entity_embedding = nn.Embedding(n_entities, belief_hidden_dim)
        self.recurrent_cell = nn.GRUCell(
            belief_hidden_dim,
            belief_hidden_dim,
        )
        self.mean_head = nn.Linear(belief_hidden_dim, belief_state_dim)
        self.logvar_head = nn.Linear(belief_hidden_dim, belief_state_dim)
        self.message_projection = nn.Sequential(
            nn.LayerNorm(belief_hidden_dim),
            nn.Linear(belief_hidden_dim, message_dim),
        )

    def initial_hidden(self, batch_size, device):
        return torch.zeros(
            batch_size,
            self.n_agents,
            self.n_entities,
            self.belief_hidden_dim,
            device=device,
        )

    def forward(
        self,
        entity_observations,
        observation_mask,
        agent_hidden,
        previous_belief_hidden,
    ):
        batch_size = entity_observations.shape[0]
        expected_prefix = (
            batch_size,
            self.n_agents,
            self.n_entities,
        )
        if entity_observations.shape[:3] != expected_prefix:
            raise ValueError("Unexpected entity_observations shape.")
        if observation_mask.shape != expected_prefix:
            raise ValueError("Unexpected observation_mask shape.")
        if previous_belief_hidden.shape != (
            *expected_prefix,
            self.belief_hidden_dim,
        ):
            raise ValueError("Unexpected previous_belief_hidden shape.")

        visibility = observation_mask.to(
            dtype=entity_observations.dtype
        ).unsqueeze(-1)
        masked_observations = entity_observations * visibility
        expanded_agent_hidden = agent_hidden.unsqueeze(2).expand(
            -1,
            -1,
            self.n_entities,
            -1,
        )

        recurrent_input = self.input_projection(
            torch.cat(
                [
                    masked_observations,
                    visibility,
                    expanded_agent_hidden,
                ],
                dim=-1,
            )
        )

        agent_ids = torch.arange(
            self.n_agents,
            device=entity_observations.device,
        )
        entity_ids = torch.arange(
            self.n_entities,
            device=entity_observations.device,
        )
        recurrent_input = (
            recurrent_input
            + self.agent_embedding(agent_ids).view(
                1, self.n_agents, 1, self.belief_hidden_dim
            )
            + self.entity_embedding(entity_ids).view(
                1, 1, self.n_entities, self.belief_hidden_dim
            )
        )

        new_hidden = self.recurrent_cell(
            recurrent_input.reshape(-1, self.belief_hidden_dim),
            previous_belief_hidden.reshape(-1, self.belief_hidden_dim),
        ).reshape(
            batch_size,
            self.n_agents,
            self.n_entities,
            self.belief_hidden_dim,
        )

        belief_mean = self.mean_head(new_hidden)
        belief_logvar = self.logvar_head(new_hidden).clamp(
            min=self.logvar_min,
            max=self.logvar_max,
        )

        entity_uncertainty = belief_logvar.exp().mean(dim=-1)
        confidence_weights = torch.softmax(
            -entity_uncertainty,
            dim=-1,
        )
        local_message = (
            self.message_projection(new_hidden)
            * confidence_weights.unsqueeze(-1)
        ).sum(dim=2)

        return {
            "hidden": new_hidden,
            "mean": belief_mean,
            "logvar": belief_logvar,
            "local_message": local_message,
            "mean_variance": entity_uncertainty.mean(dim=-1, keepdim=True),
        }


class ReliabilityGatedFusion(nn.Module):
    """Fuse Hub and local-belief messages using current Hub reliability."""

    def __init__(self, message_dim):
        super().__init__()
        self.message_dim = message_dim
        self.gate_network = nn.Sequential(
            nn.LayerNorm(message_dim * 2 + 2),
            nn.Linear(message_dim * 2 + 2, message_dim),
            nn.GELU(),
            nn.Linear(message_dim, 1),
        )

    def forward(
        self,
        hub_message,
        local_message,
        hub_reliability,
        mean_variance,
    ):
        if hub_reliability.ndim == 1:
            hub_reliability = hub_reliability[:, None, None]
        elif hub_reliability.ndim == 2:
            hub_reliability = hub_reliability.unsqueeze(-1)

        hub_reliability = hub_reliability.to(
            dtype=hub_message.dtype
        ).clamp(0.0, 1.0)
        hub_reliability = hub_reliability.expand(
            -1,
            hub_message.shape[1],
            -1,
        )

        learned_gate = torch.sigmoid(
            self.gate_network(
                torch.cat(
                    [
                        hub_message,
                        local_message,
                        hub_reliability,
                        mean_variance,
                    ],
                    dim=-1,
                )
            )
        )
        gate = hub_reliability * learned_gate
        fused_message = (
            gate * hub_message
            + (1.0 - gate) * local_message
        )
        return fused_message, gate


def agent_centric_entity_targets(true_entity_states, n_agents):
    """
    Convert global entity states to the local coordinate frame of each agent.

    Positions become relative to the observer. Velocity, communication, and
    type features retain the global entity-state representation.
    """

    targets = true_entity_states.unsqueeze(2).expand(
        -1,
        -1,
        n_agents,
        -1,
        -1,
    ).clone()
    observer_positions = true_entity_states[
        :, :, :n_agents, 0:2
    ].unsqueeze(3)
    targets[..., 0:2] = targets[..., 0:2] - observer_positions
    return targets


def gaussian_belief_loss(
    belief_mean,
    belief_logvar,
    true_entity_states,
    observation_masks,
    state_valid_mask,
    n_agents,
    supervision="all",
):
    if supervision not in {"all", "invisible"}:
        raise ValueError("supervision must be 'all' or 'invisible'.")

    targets = agent_centric_entity_targets(
        true_entity_states,
        n_agents=n_agents,
    )
    error = belief_mean - targets

    valid_mask = state_valid_mask[:, :, None, None]
    if supervision == "invisible":
        supervised_mask = valid_mask & ~observation_masks.bool()
    else:
        supervised_mask = valid_mask.expand_as(observation_masks.bool())

    element_mask = supervised_mask.unsqueeze(-1)
    active_elements = element_mask.sum() * belief_mean.shape[-1]

    if active_elements.item() == 0:
        zero = belief_mean.sum() * 0.0
        return {
            "loss": zero,
            "rmse": zero.detach(),
            "supervised_fraction": zero.detach(),
        }

    nll = 0.5 * (
        error.pow(2) * torch.exp(-belief_logvar)
        + belief_logvar
    )
    loss = (nll * element_mask).sum() / active_elements
    rmse = torch.sqrt(
        (error.pow(2) * element_mask).sum() / active_elements
    )
    supervised_fraction = supervised_mask.float().mean()

    return {
        "loss": loss,
        "rmse": rmse,
        "supervised_fraction": supervised_fraction,
    }

