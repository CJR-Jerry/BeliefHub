from copy import deepcopy

import torch
import torch.nn.functional as F


class MASIALearner:
    def __init__(
        self,
        agent_network,
        mixer,
        latent_model,
        learning_rate=5e-4,
        gamma=0.99,
        grad_clip=10.0,
        repr_coef=1.0,
        spr_coef=1.0,
        reward_prediction_coef=1.0,
        prediction_length=2,
    ):
        self.agent_network = agent_network
        self.mixer = mixer
        self.latent_model = latent_model
        self.target_agent_network = deepcopy(agent_network)
        self.target_mixer = deepcopy(mixer)
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.repr_coef = repr_coef
        self.spr_coef = spr_coef
        self.reward_prediction_coef = reward_prediction_coef
        self.prediction_length = prediction_length
        self.parameters = (
            list(agent_network.parameters())
            + list(mixer.parameters())
            + list(latent_model.parameters())
        )
        self.optimizer = torch.optim.Adam(
            self.parameters, lr=learning_rate
        )

    def _unroll(self, network, observations, message_masks):
        batch_size, sequence_length, n_agents = observations.shape[:3]
        hidden = network.initial_hidden(batch_size, observations.device)
        previous_actions = torch.zeros(
            batch_size,
            n_agents,
            network.n_actions,
            device=observations.device,
        )
        q_values = []
        reconstructions = []
        latents = []
        for timestep in range(sequence_length):
            q, hidden, reconstruction, latent = network.forward_full(
                observations[:, timestep],
                previous_actions,
                hidden,
                message_masks[:, timestep],
            )
            q_values.append(q)
            reconstructions.append(reconstruction)
            latents.append(latent)
        return (
            torch.stack(q_values, dim=1),
            torch.stack(reconstructions, dim=1),
            torch.stack(latents, dim=1),
        )

    def _representation_loss(
        self,
        observations,
        states,
        actions,
        rewards,
        filled,
        message_masks,
        reconstructions,
        latents,
    ):
        state_mask = torch.cat([filled, filled[:, -1:]], dim=1)
        reconstruction_error = (reconstructions - states).pow(2).mean(-1, keepdim=True)
        reconstruction_loss = (
            reconstruction_error * state_mask
        ).sum() / state_mask.sum().clamp(min=1.0)

        with torch.no_grad():
            target_projected = self.agent_network.target_project_sequence(
                observations, message_masks
            )

        predicted_current = self.agent_network.predictor(
            self.agent_network.projection(latents)
        )
        current_spr_error = (predicted_current - target_projected).pow(2).sum(-1, keepdim=True)
        spr_numerator = (current_spr_error * state_mask).sum()
        spr_denominator = state_mask.sum()

        actions_one_hot = F.one_hot(
            actions.squeeze(-1),
            num_classes=self.agent_network.n_actions,
        ).float()
        predicted_latent = latents[:, :-1]
        for horizon in range(1, self.prediction_length + 1):
            available_steps = actions_one_hot.shape[1] - horizon + 1
            if available_steps <= 0:
                break
            if horizon == 1:
                predicted_latent = self.latent_model(
                    predicted_latent,
                    actions_one_hot,
                )
            else:
                predicted_latent = self.latent_model(
                    predicted_latent[:, :-1],
                    actions_one_hot[:, horizon - 1 :],
                )
            predicted_projection = self.agent_network.predictor(
                self.agent_network.projection(predicted_latent)
            )
            target = target_projected[:, horizon:]
            horizon_mask = state_mask[:, horizon:]
            error = (predicted_projection - target).pow(2).sum(-1, keepdim=True)
            spr_numerator = spr_numerator + (error * horizon_mask).sum()
            spr_denominator = spr_denominator + horizon_mask.sum()
        spr_loss = spr_numerator / spr_denominator.clamp(min=1.0)

        predicted_rewards = self.latent_model.predict_reward(latents[:, :-1])
        reward_error = (predicted_rewards - rewards).pow(2)
        reward_loss = (
            reward_error * filled
        ).sum() / filled.sum().clamp(min=1.0)
        representation_loss = (
            reconstruction_loss
            + self.spr_coef * spr_loss
            + self.reward_prediction_coef * reward_loss
        )
        return representation_loss, reconstruction_loss, spr_loss, reward_loss

    def train_batch(self, batch):
        observations = batch["observations"]
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        terminated = batch["terminated"]
        filled = batch["filled"]
        message_masks = batch["packet_loss_masks"]

        online_q, reconstructions, latents = self._unroll(
            self.agent_network, observations, message_masks
        )
        with torch.no_grad():
            target_q, _, _ = self._unroll(
                self.target_agent_network,
                observations,
                message_masks,
            )

        chosen_q = torch.gather(
            online_q[:, :-1], dim=-1, index=actions
        ).squeeze(-1)
        next_actions = online_q[:, 1:].detach().argmax(
            dim=-1, keepdim=True
        )
        target_agent_q = torch.gather(
            target_q[:, 1:], dim=-1, index=next_actions
        ).squeeze(-1)
        total_q = self.mixer(chosen_q, states[:, :-1])
        with torch.no_grad():
            target_total_q = self.target_mixer(
                target_agent_q, states[:, 1:]
            )
            td_target = rewards + self.gamma * (1.0 - terminated) * target_total_q
        td_error = total_q - td_target
        td_loss = (td_error.pow(2) * filled).sum() / filled.sum().clamp(min=1.0)

        repr_loss, reconstruction_loss, spr_loss, reward_loss = (
            self._representation_loss(
                observations,
                states,
                actions,
                rewards,
                filled,
                message_masks,
                reconstructions,
                latents,
            )
        )
        total_loss = td_loss + self.repr_coef * repr_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters, self.grad_clip
        )
        self.optimizer.step()
        self.agent_network.momentum_update()

        return {
            "loss": float(total_loss.item()),
            "td_loss": float(td_loss.item()),
            "representation_loss": float(repr_loss.item()),
            "reconstruction_loss": float(reconstruction_loss.item()),
            "spr_loss": float(spr_loss.item()),
            "reward_prediction_loss": float(reward_loss.item()),
            "gradient_norm": float(gradient_norm),
        }

    def update_targets(self):
        self.target_agent_network.load_state_dict(
            self.agent_network.state_dict()
        )
        self.target_mixer.load_state_dict(self.mixer.state_dict())
