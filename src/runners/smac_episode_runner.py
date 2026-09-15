import numpy as np
import torch
import torch.nn.functional as F

from src.models.match_information_hub import HubHistoryBuffer


def _select_available_actions(q_values, avail_actions, epsilon, rng):
    actions = []
    q_array = q_values.detach().cpu().numpy()

    for agent_id, available in enumerate(avail_actions):
        valid = np.flatnonzero(available)
        if valid.size == 0:
            raise RuntimeError(f"Agent {agent_id} has no available action.")

        if rng.random() < epsilon:
            action = int(rng.choice(valid))
        else:
            masked_q = np.full(q_array.shape[-1], -np.inf, dtype=np.float32)
            masked_q[valid] = q_array[0, agent_id, valid]
            action = int(masked_q.argmax())
        actions.append(action)

    return actions


class SMACQMIXEpisodeRunner:
    """Collect one fixed-shape SMAC episode for QMIX."""

    def __init__(self, env, agent_network, device, rng_seed=0):
        self.env = env
        self.agent_network = agent_network
        self.device = device
        self.rng = np.random.default_rng(rng_seed)

        self.n_agents = env.n_agents
        self.n_actions = env.get_total_actions()
        self.max_steps = env.episode_limit
        self.observation_dim = env.get_obs_size()
        self.state_dim = env.get_state_size()

    def create_empty_episode(self):
        return {
            "observations": np.zeros(
                (self.max_steps + 1, self.n_agents, self.observation_dim),
                dtype=np.float32,
            ),
            "states": np.zeros(
                (self.max_steps + 1, self.state_dim),
                dtype=np.float32,
            ),
            "avail_actions": np.zeros(
                (self.max_steps + 1, self.n_agents, self.n_actions),
                dtype=np.bool_,
            ),
            "actions": np.zeros(
                (self.max_steps, self.n_agents, 1),
                dtype=np.int64,
            ),
            "rewards": np.zeros((self.max_steps, 1), dtype=np.float32),
            "terminated": np.zeros((self.max_steps, 1), dtype=np.float32),
            "filled": np.zeros((self.max_steps, 1), dtype=np.float32),
            # Kept only for compatibility with the shared replay buffer.
            "pol": np.zeros((1,), dtype=np.float32),
        }

    def _store_state(self, episode, timestep):
        episode["observations"][timestep] = np.asarray(
            self.env.get_obs(),
            dtype=np.float32,
        )
        episode["states"][timestep] = np.asarray(
            self.env.get_state(),
            dtype=np.float32,
        )
        episode["avail_actions"][timestep] = np.asarray(
            self.env.get_avail_actions(),
            dtype=np.bool_,
        )

    def run(self, epsilon=1.0):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1.")

        self.env.reset()
        episode = self.create_empty_episode()
        hidden = self.agent_network.initial_hidden(1, self.device)
        previous_actions = torch.zeros(
            1,
            self.n_agents,
            self.n_actions,
            device=self.device,
        )

        episode_return = 0.0
        battle_won = False
        executed_steps = 0

        for timestep in range(self.max_steps):
            self._store_state(episode, timestep)
            observations = torch.as_tensor(
                episode["observations"][timestep],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            with torch.no_grad():
                q_values, hidden = self.agent_network(
                    observations,
                    previous_actions,
                    hidden,
                )

            actions = _select_available_actions(
                q_values,
                episode["avail_actions"][timestep],
                epsilon,
                self.rng,
            )
            reward, terminated, info = self.env.step(actions)

            episode["actions"][timestep, :, 0] = actions
            episode["rewards"][timestep, 0] = float(reward)
            episode["terminated"][timestep, 0] = float(terminated)
            episode["filled"][timestep, 0] = 1.0

            previous_actions = F.one_hot(
                torch.as_tensor(
                    actions,
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0),
                num_classes=self.n_actions,
            ).float()

            episode_return += float(reward)
            executed_steps += 1
            battle_won = bool(info.get("battle_won", False))
            self._store_state(episode, timestep + 1)

            if terminated:
                break

        episode["pol"][0] = float(battle_won)
        return episode, {
            "episode_return": episode_return,
            "episode_length": executed_steps,
            "battle_won": battle_won,
            "epsilon": epsilon,
        }


class SMACMATCHEpisodeRunner:
    """Collect an SMAC episode with MATCH communication and packet loss."""

    def __init__(
        self,
        env,
        match_agent,
        information_hub,
        entity_adapter,
        device,
        packet_loss=0.3,
        rng_seed=0,
    ):
        if not 0.0 <= packet_loss <= 1.0:
            raise ValueError("packet_loss must be between 0 and 1.")

        self.env = env
        self.match_agent = match_agent
        self.information_hub = information_hub
        self.entity_adapter = entity_adapter
        self.device = device
        self.packet_loss = packet_loss
        self.rng = np.random.default_rng(rng_seed)

        self.n_agents = env.n_agents
        self.n_actions = env.get_total_actions()
        self.max_steps = env.episode_limit
        self.observation_dim = env.get_obs_size()
        self.state_dim = env.get_state_size()
        self.n_entities = entity_adapter.n_entities
        self.entity_dim = entity_adapter.entity_observation_dim

    def create_empty_episode(self):
        episode = SMACQMIXEpisodeRunner.create_empty_episode(self)
        episode.update(
            {
                "entity_observations": np.zeros(
                    (
                        self.max_steps + 1,
                        self.n_agents,
                        self.n_entities,
                        self.entity_dim,
                    ),
                    dtype=np.float32,
                ),
                "observation_masks": np.zeros(
                    (
                        self.max_steps + 1,
                        self.n_agents,
                        self.n_entities,
                    ),
                    dtype=np.bool_,
                ),
                "packet_loss_masks": np.zeros(
                    (self.max_steps + 1, self.n_agents),
                    dtype=np.bool_,
                ),
                "entity_states": np.zeros(
                    (
                        self.max_steps + 1,
                        self.n_entities,
                        self.entity_adapter.entity_state_dim,
                    ),
                    dtype=np.float32,
                ),
            }
        )
        return episode

    def _build_current_input(self):
        entity_observations, local_mask = self.entity_adapter.convert()
        living_agents = np.asarray(
            [self.env.agents[i].health > 0 for i in range(self.n_agents)],
            dtype=bool,
        )
        packet_received = (
            self.rng.random(self.n_agents) >= self.packet_loss
        ) & living_agents
        hub_mask = local_mask & packet_received[:, None]
        return entity_observations, local_mask, packet_received, hub_mask

    def _store_state(
        self,
        episode,
        timestep,
        entity_observations,
        local_mask,
        packet_received,
    ):
        episode["observations"][timestep] = np.asarray(
            self.env.get_obs(),
            dtype=np.float32,
        )
        episode["states"][timestep] = np.asarray(
            self.env.get_state(),
            dtype=np.float32,
        )
        episode["avail_actions"][timestep] = np.asarray(
            self.env.get_avail_actions(),
            dtype=np.bool_,
        )
        episode["entity_observations"][timestep] = entity_observations
        episode["observation_masks"][timestep] = local_mask
        episode["packet_loss_masks"][timestep] = packet_received
        episode["entity_states"][timestep] = (
            self.entity_adapter.build_global_state()
        )

    def run(self, epsilon=1.0):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1.")

        self.env.reset()
        episode = self.create_empty_episode()
        agent_hidden = self.match_agent.initial_hidden(1, self.device)
        belief_hidden = self.match_agent.initial_belief_hidden(1, self.device)
        history = HubHistoryBuffer(
            history_length=self.information_hub.history_length,
            n_entities=self.n_entities,
            state_dim=self.information_hub.state_dim,
        )
        history.reset(1, self.device)

        current = self._build_current_input()
        episode_return = 0.0
        battle_won = False
        executed_steps = 0
        received_packets = 0
        attempted_packets = 0

        for timestep in range(self.max_steps):
            entity_obs, local_mask, packet_received, hub_mask = current
            self._store_state(
                episode,
                timestep,
                entity_obs,
                local_mask,
                packet_received,
            )

            entity_tensor = torch.as_tensor(
                entity_obs,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            local_mask_tensor = torch.as_tensor(
                local_mask,
                dtype=torch.bool,
                device=self.device,
            ).unsqueeze(0)
            hub_mask_tensor = torch.as_tensor(
                hub_mask,
                dtype=torch.bool,
                device=self.device,
            ).unsqueeze(0)
            reliability = torch.as_tensor(
                packet_received,
                dtype=torch.float32,
                device=self.device,
            ).mean().reshape(1, 1)
            history_states, history_valid = history.get()

            with torch.no_grad():
                hub_output = self.information_hub(
                    entity_observations=entity_tensor,
                    observation_mask=hub_mask_tensor,
                    history_states=history_states,
                    history_valid_mask=history_valid,
                )
                agent_output = self.match_agent(
                    entity_observations=entity_tensor,
                    observation_mask=local_mask_tensor,
                    previous_hidden=agent_hidden,
                    reconstructed_entities=hub_output["latent_state"],
                    reconstructed_state=hub_output["reconstructed_state"],
                    entity_mask=None,
                    previous_belief_hidden=belief_hidden,
                    hub_reliability=reliability,
                )

            agent_hidden = agent_output["hidden_states"]
            belief_hidden = agent_output["belief_hidden"]
            history.append(hub_output["reconstructed_state"])

            actions = _select_available_actions(
                agent_output["q_values"],
                episode["avail_actions"][timestep],
                epsilon,
                self.rng,
            )
            reward, terminated, info = self.env.step(actions)

            episode["actions"][timestep, :, 0] = actions
            episode["rewards"][timestep, 0] = float(reward)
            episode["terminated"][timestep, 0] = float(terminated)
            episode["filled"][timestep, 0] = 1.0

            episode_return += float(reward)
            executed_steps += 1
            battle_won = bool(info.get("battle_won", False))
            received_packets += int(packet_received.sum())
            attempted_packets += int(
                np.diag(local_mask[:, : self.n_agents]).sum()
            )

            current = self._build_current_input()
            self._store_state(
                episode,
                timestep + 1,
                current[0],
                current[1],
                current[2],
            )
            if terminated:
                break

        episode["pol"][0] = float(battle_won)
        packet_success_rate = (
            received_packets / attempted_packets
            if attempted_packets > 0
            else 0.0
        )
        return episode, {
            "episode_return": episode_return,
            "episode_length": executed_steps,
            "battle_won": battle_won,
            "epsilon": epsilon,
            "packet_loss": self.packet_loss,
            "packet_success_rate": packet_success_rate,
        }
