from copy import deepcopy

import torch
import torch.nn.functional as F


class QMIXLearner:
    def __init__(
        self,
        agent_network,
        mixer,
        learning_rate=5e-4,
        gamma=0.99,
        grad_clip=10.0,
    ):
        self.agent_network = agent_network
        self.mixer = mixer

        self.target_agent_network = deepcopy(agent_network)
        self.target_mixer = deepcopy(mixer)

        self.gamma = gamma
        self.grad_clip = grad_clip

        self.optimizer = torch.optim.Adam(
            list(agent_network.parameters())
            + list(mixer.parameters()),
            lr=learning_rate,
        )

    def unroll_agent_network(
        self,
        network,
        observations,
        actions,
    ):
        batch_size = observations.shape[0]
        sequence_length = observations.shape[1]
        n_agents = observations.shape[2]
        n_actions = network.n_actions
        device = observations.device

        hidden = network.initial_hidden(
            batch_size=batch_size,
            device=device,
        )

        zero_action = torch.zeros(
            batch_size,
            n_agents,
            n_actions,
            device=device,
        )

        q_sequence = []

        for timestep in range(sequence_length):
            if timestep == 0:
                previous_action = zero_action
            else:
                previous_action = F.one_hot(
                    actions[:, timestep - 1, :, 0],
                    num_classes=n_actions,
                ).float()

            q_values, hidden = network(
                observations[:, timestep],
                previous_action,
                hidden,
            )

            q_sequence.append(q_values)

        return torch.stack(q_sequence, dim=1)

    def train_batch(self, batch):
        observations = batch["observations"]
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        terminated = batch["terminated"]
        filled = batch["filled"]

        online_q_sequence = self.unroll_agent_network(
            self.agent_network,
            observations,
            actions,
        )

        with torch.no_grad():
            target_q_sequence = self.unroll_agent_network(
                self.target_agent_network,
                observations,
                actions,
            )

        # 当前实际执行动作对应的 Q 值
        chosen_agent_q = torch.gather(
            online_q_sequence[:, :-1],
            dim=-1,
            index=actions,
        ).squeeze(-1)

        # Double-Q：
        # 在线网络选择下一动作，目标网络评价该动作
        next_online_q = online_q_sequence[:, 1:].detach()

        # SMAC exposes a dynamic action-availability mask.  Selecting an
        # unavailable action in the Double-Q target silently overestimates
        # the bootstrap value, so mask it before the online argmax.  MPE
        # batches do not contain this optional field and keep the original
        # behaviour.
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
            target_total_q = self.target_mixer(
                target_agent_q,
                states[:, 1:],
            )

            td_target = (
                rewards
                + self.gamma
                * (1.0 - terminated)
                * target_total_q
            )

        td_error = total_q - td_target

        loss = (
            td_error.pow(2) * filled
        ).sum() / filled.sum().clamp(min=1.0)

        self.optimizer.zero_grad()
        loss.backward()

        parameters = (
            list(self.agent_network.parameters())
            + list(self.mixer.parameters())
        )

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            self.grad_clip,
        )

        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "gradient_norm": float(gradient_norm),
            "mean_total_q": float(total_q.mean().item()),
            "mean_target_q": float(td_target.mean().item()),
        }

    def update_targets(self):
        self.target_agent_network.load_state_dict(
            self.agent_network.state_dict()
        )

        self.target_mixer.load_state_dict(
            self.mixer.state_dict()
        )
