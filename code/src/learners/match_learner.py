from copy import deepcopy

import torch

from src.losses.state_reconstruction import (
    state_reconstruction_loss,
)
from src.models.belief_state import gaussian_belief_loss


class MATCHLearner:
    """
    Centralized learner for MATCH.

    The optimized objective is:

        L = L_TD + eta * L_state

    The learner jointly updates:

        1. MATCH communication-enhanced agents
        2. MATCH Information Hub
        3. QMIX mixing network

    Separate target copies are maintained for all three
    components.
    """

    def __init__(
        self,
        match_agent,
        information_hub,
        mixer,
        learning_rate=5e-4,
        gamma=0.99,
        reconstruction_weight=0.1,
        belief_loss_weight=0.01,
        belief_supervision="all",
        grad_clip=10.0,
    ):
        if reconstruction_weight < 0.0:
            raise ValueError(
                "reconstruction_weight must be "
                "non-negative."
            )
        if belief_loss_weight < 0.0:
            raise ValueError(
                "belief_loss_weight must be non-negative."
            )
        if belief_supervision not in {"all", "invisible"}:
            raise ValueError(
                "belief_supervision must be 'all' or 'invisible'."
            )

        self.match_agent = match_agent
        self.information_hub = information_hub
        self.mixer = mixer

        self.target_match_agent = deepcopy(
            match_agent
        )

        self.target_information_hub = deepcopy(
            information_hub
        )

        self.target_mixer = deepcopy(
            mixer
        )

        self.gamma = gamma
        self.reconstruction_weight = (
            reconstruction_weight
        )
        self.grad_clip = grad_clip
        self.belief_loss_weight = belief_loss_weight
        self.belief_supervision = belief_supervision

        self.optimized_parameters = (
            list(
                self.match_agent.parameters()
            )
            + list(
                self.information_hub.parameters()
            )
            + list(
                self.mixer.parameters()
            )
        )

        self.optimizer = torch.optim.Adam(
            self.optimized_parameters,
            lr=learning_rate,
        )

    def build_state_valid_mask(
        self,
        filled,
    ):
        """
        Construct a validity mask for the T + 1 states.

        If transition t is filled, both state t and its
        next state t + 1 are valid.

        Input:
            filled:
                [batch, T, 1]

        Output:
            state_valid:
                [batch, T + 1]
        """

        transition_valid = filled[
            ...,
            0,
        ].bool()

        first_state_valid = (
            transition_valid[:, 0:1]
        )

        state_valid = torch.cat(
            [
                first_state_valid,
                transition_valid,
            ],
            dim=1,
        )

        return state_valid

    def unroll_match_network(
        self,
        match_agent,
        information_hub,
        entity_observations,
        observation_masks,
        packet_loss_masks,
        state_valid_mask,
    ):
        """
        Sequentially unroll MATCH over a complete episode.

        Packet loss affects observations received by the
        Information Hub, but does not erase an agent's own
        local observation.

        Returns:
            q_sequence:
                [batch, T + 1, agents, actions]

            reconstructed_sequence:
                [batch, T + 1, entities, state_dim]
        """

        batch_size = (
            entity_observations.shape[0]
        )

        sequence_length = (
            entity_observations.shape[1]
        )

        n_entities = (
            entity_observations.shape[3]
        )

        device = entity_observations.device

        agent_hidden = (
            match_agent.initial_hidden(
                batch_size=batch_size,
                device=device,
            )
        )
        belief_hidden = match_agent.initial_belief_hidden(
            batch_size=batch_size,
            device=device,
        )

        history_length = (
            information_hub.history_length
        )

        entity_state_dim = (
            information_hub.state_dim
        )

        history_states = torch.zeros(
            batch_size,
            history_length,
            n_entities,
            entity_state_dim,
            device=device,
        )

        history_valid_mask = torch.zeros(
            batch_size,
            history_length,
            dtype=torch.bool,
            device=device,
        )

        q_sequence = []
        reconstructed_sequence = []
        belief_mean_sequence = []
        belief_logvar_sequence = []
        belief_gate_sequence = []

        for timestep in range(
            sequence_length
        ):
            current_entity_observations = (
                entity_observations[
                    :,
                    timestep,
                ]
            )

            current_local_mask = (
                observation_masks[
                    :,
                    timestep,
                ].bool()
            )

            current_packet_mask = (
                packet_loss_masks[
                    :,
                    timestep,
                ].bool()
            )

            # True only when the entity is locally
            # observable and the observer's message
            # successfully arrives at the Hub.
            current_hub_mask = (
                current_local_mask
                & current_packet_mask.unsqueeze(-1)
            )
            hub_reliability = current_packet_mask.float().mean(
                dim=-1,
                keepdim=True,
            )

            hub_output = information_hub(
                entity_observations=(
                    current_entity_observations
                ),
                observation_mask=(
                    current_hub_mask
                ),
                history_states=history_states,
                history_valid_mask=(
                    history_valid_mask
                ),
            )

            agent_output = match_agent(
                entity_observations=(
                    current_entity_observations
                ),
                observation_mask=(
                    current_local_mask
                ),
                previous_hidden=agent_hidden,
                reconstructed_entities=(
                    hub_output["latent_state"]
                ),
                reconstructed_state=(
                    hub_output[
                        "reconstructed_state"
                    ]
                ),
                entity_mask=None,
                previous_belief_hidden=belief_hidden,
                hub_reliability=hub_reliability,
            )

            q_sequence.append(
                agent_output["q_values"]
            )

            reconstructed_sequence.append(
                hub_output[
                    "reconstructed_state"
                ]
            )

            agent_hidden = (
                agent_output["hidden_states"]
            )
            belief_hidden = agent_output["belief_hidden"]

            if match_agent.use_local_belief:
                belief_mean_sequence.append(
                    agent_output["belief_mean"]
                )
                belief_logvar_sequence.append(
                    agent_output["belief_logvar"]
                )
                belief_gate_sequence.append(
                    agent_output["belief_gate"]
                )

            current_state_valid = (
                state_valid_mask[
                    :,
                    timestep,
                ]
            )

            # Match HubHistoryBuffer behaviour:
            # historical reconstructions are treated as
            # buffered values rather than a recurrent
            # computational graph through the full episode.
            new_history_state = (
                hub_output[
                    "reconstructed_state"
                ]
                .detach()
                .unsqueeze(1)
            )

            history_states = torch.cat(
                [
                    history_states[:, 1:],
                    new_history_state,
                ],
                dim=1,
            )

            history_valid_mask = torch.cat(
                [
                    history_valid_mask[:, 1:],
                    current_state_valid.unsqueeze(1),
                ],
                dim=1,
            )

        output = {
            "q_values": torch.stack(
                q_sequence,
                dim=1,
            ),
            "reconstructed_states": torch.stack(
                reconstructed_sequence,
                dim=1,
            ),
        }
        if match_agent.use_local_belief:
            output.update(
                {
                    "belief_mean": torch.stack(
                        belief_mean_sequence,
                        dim=1,
                    ),
                    "belief_logvar": torch.stack(
                        belief_logvar_sequence,
                        dim=1,
                    ),
                    "belief_gate": torch.stack(
                        belief_gate_sequence,
                        dim=1,
                    ),
                }
            )
        return output

    def train_batch(self, batch):
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        terminated = batch["terminated"]
        filled = batch["filled"]

        entity_observations = batch[
            "entity_observations"
        ]

        observation_masks = batch[
            "observation_masks"
        ]

        packet_loss_masks = batch[
            "packet_loss_masks"
        ]

        true_entity_states = batch[
            "entity_states"
        ]

        state_valid_mask = (
            self.build_state_valid_mask(
                filled
            )
        )

        online_output = (
            self.unroll_match_network(
                match_agent=self.match_agent,
                information_hub=(
                    self.information_hub
                ),
                entity_observations=(
                    entity_observations
                ),
                observation_masks=(
                    observation_masks
                ),
                packet_loss_masks=(
                    packet_loss_masks
                ),
                state_valid_mask=(
                    state_valid_mask
                ),
            )
        )

        online_q_sequence = online_output[
            "q_values"
        ]

        reconstructed_states = online_output[
            "reconstructed_states"
        ]

        with torch.no_grad():
            target_output = (
                self.unroll_match_network(
                    match_agent=(
                        self.target_match_agent
                    ),
                    information_hub=(
                        self.target_information_hub
                    ),
                    entity_observations=(
                        entity_observations
                    ),
                    observation_masks=(
                        observation_masks
                    ),
                    packet_loss_masks=(
                        packet_loss_masks
                    ),
                    state_valid_mask=(
                        state_valid_mask
                    ),
                )
            )

            target_q_sequence = target_output[
                "q_values"
            ]

        # Q-values for the actions actually executed.
        chosen_agent_q = torch.gather(
            online_q_sequence[:, :-1],
            dim=-1,
            index=actions,
        ).squeeze(-1)

        # Double-Q:
        # online MATCH selects the next action.
        next_online_q = online_q_sequence[:, 1:].detach()

        # SMAC actions are state dependent.  Keep unavailable actions out of
        # the online-network argmax used by Double-Q.  The field is optional
        # so existing Cooperative Navigation runs are unchanged.
        if "avail_actions" in batch:
            next_avail_actions = batch["avail_actions"][:, 1:].bool()
            if next_avail_actions.shape != next_online_q.shape:
                raise ValueError(
                    "avail_actions must match the Q-value shape; "
                    f"got {next_avail_actions.shape} and "
                    f"{next_online_q.shape}."
                )
            next_online_q = next_online_q.masked_fill(
                ~next_avail_actions,
                -1e9,
            )

        next_actions = next_online_q.argmax(
            dim=-1,
            keepdim=True,
        )

        # Target MATCH evaluates that action.
        target_agent_q = torch.gather(
            target_q_sequence[:, 1:],
            dim=-1,
            index=next_actions,
        ).squeeze(-1)

        total_q = self.mixer(
            chosen_agent_q,
            states[:, :-1],
        )

        with torch.no_grad():
            target_total_q = (
                self.target_mixer(
                    target_agent_q,
                    states[:, 1:],
                )
            )

            td_target = (
                rewards
                + self.gamma
                * (1.0 - terminated)
                * target_total_q
            )

        td_error = total_q - td_target

        td_loss = (
            td_error.pow(2) * filled
        ).sum() / filled.sum().clamp(
            min=1.0
        )

        # Only valid T + 1 states participate in the
        # supervised reconstruction objective.
        valid_reconstructed_states = (
            reconstructed_states[
                state_valid_mask
            ]
        )

        valid_true_entity_states = (
            true_entity_states[
                state_valid_mask
            ]
        )

        reconstruction_losses = (
            state_reconstruction_loss(
                valid_reconstructed_states,
                valid_true_entity_states,
            )
        )

        reconstruction_loss = (
            reconstruction_losses["total"]
        )

        belief_loss = td_loss.new_zeros(())
        belief_rmse = td_loss.new_zeros(())
        belief_variance = td_loss.new_zeros(())
        belief_gate = td_loss.new_zeros(())
        belief_supervised_fraction = td_loss.new_zeros(())

        if self.match_agent.use_local_belief:
            belief_statistics = gaussian_belief_loss(
                belief_mean=online_output["belief_mean"],
                belief_logvar=online_output["belief_logvar"],
                true_entity_states=true_entity_states,
                observation_masks=observation_masks,
                state_valid_mask=state_valid_mask,
                n_agents=self.match_agent.n_agents,
                supervision=self.belief_supervision,
            )
            belief_loss = belief_statistics["loss"]
            belief_rmse = belief_statistics["rmse"]
            belief_supervised_fraction = belief_statistics[
                "supervised_fraction"
            ]

            state_weight = state_valid_mask[
                :, :, None, None, None
            ].to(dtype=online_output["belief_logvar"].dtype)
            belief_variance = (
                online_output["belief_logvar"].exp()
                * state_weight
            ).sum() / (
                state_weight.sum()
                * online_output["belief_logvar"].shape[2]
                * online_output["belief_logvar"].shape[3]
                * online_output["belief_logvar"].shape[4]
            ).clamp(min=1.0)

            gate_weight = state_valid_mask[
                :, :, None, None
            ].to(dtype=online_output["belief_gate"].dtype)
            belief_gate = (
                online_output["belief_gate"] * gate_weight
            ).sum() / (
                gate_weight.sum()
                * online_output["belief_gate"].shape[2]
            ).clamp(min=1.0)

        total_loss = (
            td_loss
            + self.reconstruction_weight
            * reconstruction_loss
            + self.belief_loss_weight
            * belief_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()

        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                self.optimized_parameters,
                self.grad_clip,
            )
        )

        self.optimizer.step()

        return {
            "loss": float(
                total_loss.item()
            ),
            "td_loss": float(
                td_loss.item()
            ),
            "state_loss": float(
                reconstruction_loss.item()
            ),
            "position_loss": float(
                reconstruction_losses[
                    "position"
                ].item()
            ),
            "attribute_loss": float(
                reconstruction_losses[
                    "attribute"
                ].item()
            ),
            "belief_loss": float(belief_loss.item()),
            "belief_rmse": float(belief_rmse.item()),
            "belief_variance": float(
                belief_variance.item()
            ),
            "belief_gate": float(belief_gate.item()),
            "belief_supervised_fraction": float(
                belief_supervised_fraction.item()
            ),
            "gradient_norm": float(
                gradient_norm.item()
                if torch.is_tensor(
                    gradient_norm
                )
                else gradient_norm
            ),
            "mean_total_q": float(
                total_q.mean().item()
            ),
            "mean_target_q": float(
                td_target.mean().item()
            ),
        }

    def update_targets(self):
        self.target_match_agent.load_state_dict(
            self.match_agent.state_dict()
        )

        self.target_information_hub.load_state_dict(
            self.information_hub.state_dict()
        )

        self.target_mixer.load_state_dict(
            self.mixer.state_dict()
        )

