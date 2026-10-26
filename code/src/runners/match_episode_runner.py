import numpy as np
import torch

from src.envs.global_entity_state import (
    build_global_entity_state,
)
from src.models.match_information_hub import (
    HubHistoryBuffer,
)


class MATCHEpisodeRunner:
    """
    Run one complete MATCH episode.

    At every timestep:

        1. Convert environment observations into
           entity-based observations.

        2. Sample agent-to-Hub packet loss.

        3. Reconstruct the global entity state through
           the Information Hub.

        4. Generate personalized Hub messages and
           per-agent Q-values.

        5. Select actions using epsilon-greedy.

        6. Store all fields required for MATCH training.
    """

    def __init__(
        self,
        env,
        match_agent,
        information_hub,
        entity_adapter,
        device,
        packet_loss: float = 0.3,
    ):
        if not 0.0 <= packet_loss <= 1.0:
            raise ValueError(
                "packet_loss must be between 0 and 1."
            )

        self.env = env
        self.match_agent = match_agent
        self.information_hub = information_hub
        self.entity_adapter = entity_adapter
        self.device = device
        self.packet_loss = packet_loss

        self.agent_names = list(
            env.possible_agents
        )

        self.n_agents = env.n_agents
        self.max_steps = env.max_cycles

        self.n_entities = (
            entity_adapter.n_entities
        )

        self.entity_observation_dim = (
            entity_adapter.entity_observation_dim
        )

        self.entity_state_dim = (
            information_hub.state_dim
        )

        first_agent = self.agent_names[0]

        self.observation_dim = (
            env.observation_space(
                first_agent
            ).shape[0]
        )

        self.n_actions = env.action_space(
            first_agent
        ).n

        env.reset(seed=0)

        self.state_dim = env.env.state().size

    def stack_observations(
        self,
        observations,
    ):
        """
        Convert the PettingZoo observation dictionary
        into a fixed agent-order array.
        """

        stacked = []

        for agent in self.agent_names:
            if agent in observations:
                stacked.append(
                    observations[agent]
                )
            else:
                stacked.append(
                    np.zeros(
                        self.observation_dim,
                        dtype=np.float32,
                    )
                )

        return np.asarray(
            stacked,
            dtype=np.float32,
        )

    def build_entity_input(
        self,
        observations,
        rng,
    ):
        """
        Build local entity observations and independently
        sample whether each agent's outgoing message reaches
        the Information Hub.

        Returns:
            entity_observations:
                [n_agents, n_entities, entity_dim]

            local_observation_mask:
                [n_agents, n_entities]

            packet_received_mask:
                [n_agents]

            hub_observation_mask:
                [n_agents, n_entities]
        """

        all_agents_present = all(
            agent in observations
            for agent in self.agent_names
        )

        if all_agents_present:
            (
                entity_observations,
                local_observation_mask,
            ) = self.entity_adapter.convert(
                observations=observations,
                agent_names=self.agent_names,
                packet_loss=0.0,
                rng=rng,
            )

            packet_received_mask = (
                rng.random(self.n_agents)
                >= self.packet_loss
            )
        else:
            # PettingZoo may return an empty observation
            # dictionary after a terminal transition.
            entity_observations = np.zeros(
                (
                    self.n_agents,
                    self.n_entities,
                    self.entity_observation_dim,
                ),
                dtype=np.float32,
            )

            local_observation_mask = np.zeros(
                (
                    self.n_agents,
                    self.n_entities,
                ),
                dtype=bool,
            )

            packet_received_mask = np.zeros(
                self.n_agents,
                dtype=bool,
            )

        hub_observation_mask = (
            local_observation_mask
            & packet_received_mask[:, None]
        )

        return (
            entity_observations,
            local_observation_mask,
            packet_received_mask,
            hub_observation_mask,
        )

    def create_empty_episode(self):
        return {
            "observations": np.zeros(
                (
                    self.max_steps + 1,
                    self.n_agents,
                    self.observation_dim,
                ),
                dtype=np.float32,
            ),

            "states": np.zeros(
                (
                    self.max_steps + 1,
                    self.state_dim,
                ),
                dtype=np.float32,
            ),

            "actions": np.zeros(
                (
                    self.max_steps,
                    self.n_agents,
                    1,
                ),
                dtype=np.int64,
            ),

            "rewards": np.zeros(
                (
                    self.max_steps,
                    1,
                ),
                dtype=np.float32,
            ),

            "terminated": np.zeros(
                (
                    self.max_steps,
                    1,
                ),
                dtype=np.float32,
            ),

            "filled": np.zeros(
                (
                    self.max_steps,
                    1,
                ),
                dtype=np.float32,
            ),

            "pol": np.zeros(
                (1,),
                dtype=np.float32,
            ),

            "entity_observations": np.zeros(
                (
                    self.max_steps + 1,
                    self.n_agents,
                    self.n_entities,
                    self.entity_observation_dim,
                ),
                dtype=np.float32,
            ),

            "observation_masks": np.zeros(
                (
                    self.max_steps + 1,
                    self.n_agents,
                    self.n_entities,
                ),
                dtype=bool,
            ),

            "packet_loss_masks": np.zeros(
                (
                    self.max_steps + 1,
                    self.n_agents,
                ),
                dtype=bool,
            ),

            "entity_states": np.zeros(
                (
                    self.max_steps + 1,
                    self.n_entities,
                    self.entity_state_dim,
                ),
                dtype=np.float32,
            ),
        }

    def store_state(
        self,
        episode,
        timestep,
        observations,
        entity_observations,
        local_observation_mask,
        packet_received_mask,
    ):
        """
        Store all state-like fields for one timestep.
        """

        episode["observations"][timestep] = (
            self.stack_observations(
                observations
            )
        )


        episode["states"][timestep] = (
            self.env.env.state()
        )

        episode[
            "entity_observations"
        ][timestep] = entity_observations

        episode[
            "observation_masks"
        ][timestep] = local_observation_mask

        episode[
            "packet_loss_masks"
        ][timestep] = packet_received_mask

        episode["entity_states"][timestep] = (
            build_global_entity_state(
                self.env.env.unwrapped.world
            )
        )

    def select_actions(
        self,
        q_values,
        epsilon,
    ):
        selected_actions = []

        for agent_index, agent in enumerate(
            self.agent_names
        ):
            if np.random.random() < epsilon:
                action = self.env.action_space(
                    agent
                ).sample()
            else:
                action = int(
                    q_values[
                        0,
                        agent_index,
                    ].argmax().item()
                )

            selected_actions.append(action)

        return selected_actions

    def run(
        self,
        epsilon: float = 1.0,
        seed=None,
    ):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(
                "epsilon must be between 0 and 1."
            )

        observations, _ = self.env.reset(
            seed=seed
        )

        rng = np.random.default_rng(seed)

        episode = self.create_empty_episode()

        agent_hidden = (
            self.match_agent.initial_hidden(
                batch_size=1,
                device=self.device,
            )
        )
        belief_hidden = (
            self.match_agent.initial_belief_hidden(
                batch_size=1,
                device=self.device,
            )
        )

        hub_history = HubHistoryBuffer(
            history_length=(
                self.information_hub.history_length
            ),
            n_entities=self.n_entities,
            state_dim=self.entity_state_dim,
        )

        hub_history.reset(
            batch_size=1,
            device=self.device,
        )

        (
            current_entity_observations,
            current_local_mask,
            current_packet_mask,
            current_hub_mask,
        ) = self.build_entity_input(
            observations=observations,
            rng=rng,
        )

        episode_return = 0.0
        executed_steps = 0
        received_packets = 0
        attempted_packets = 0

        for timestep in range(
            self.max_steps
        ):
            self.store_state(
                episode=episode,
                timestep=timestep,
                observations=observations,
                entity_observations=(
                    current_entity_observations
                ),
                local_observation_mask=(
                    current_local_mask
                ),
                packet_received_mask=(
                    current_packet_mask
                ),
            )

            entity_tensor = torch.as_tensor(
                current_entity_observations,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            local_mask_tensor = torch.as_tensor(
                current_local_mask,
                dtype=torch.bool,
                device=self.device,
            ).unsqueeze(0)

            hub_mask_tensor = torch.as_tensor(
                current_hub_mask,
                dtype=torch.bool,
                device=self.device,
            ).unsqueeze(0)

            packet_mask_tensor = torch.as_tensor(
                current_packet_mask,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            # The Hub returns the current fraction of successfully received
            # uplinks as a one-scalar reliability signal.
            hub_reliability = packet_mask_tensor.mean(
                dim=-1,
                keepdim=True,
            )

            (
                history_states,
                history_valid_mask,
            ) = hub_history.get()

            with torch.no_grad():
                hub_output = self.information_hub(
                    entity_observations=(
                        entity_tensor
                    ),
                    observation_mask=(
                        hub_mask_tensor
                    ),
                    history_states=history_states,
                    history_valid_mask=(
                        history_valid_mask
                    ),
                )

                agent_output = self.match_agent(
                    entity_observations=(
                        entity_tensor
                    ),
                    observation_mask=(
                        local_mask_tensor
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

            q_values = agent_output["q_values"]

            agent_hidden = (
                agent_output["hidden_states"]
            )
            belief_hidden = agent_output[
                "belief_hidden"
            ]

            hub_history.append(
                hub_output[
                    "reconstructed_state"
                ]
            )

            selected_actions = (
                self.select_actions(
                    q_values=q_values,
                    epsilon=epsilon,
                )
            )

            action_dictionary = {
                agent: selected_actions[index]
                for index, agent in enumerate(
                    self.agent_names
                )
                if agent in self.env.agents
            }

            (
                next_observations,
                rewards,
                terminations,
                truncations,
                _,
            ) = self.env.step(
                action_dictionary
            )

            team_reward = (
                float(
                    np.mean(
                        list(rewards.values())
                    )
                )
                if rewards
                else 0.0
            )

            done = all(
                terminations.get(
                    agent,
                    False,
                )
                or truncations.get(
                    agent,
                    False,
                )
                for agent in self.agent_names
            )

            episode[
                "actions"
            ][timestep, :, 0] = selected_actions

            episode[
                "rewards"
            ][timestep, 0] = team_reward

            episode[
                "terminated"
            ][timestep, 0] = float(done)

            episode[
                "filled"
            ][timestep, 0] = 1.0

            episode_return += team_reward
            executed_steps += 1

            received_packets += int(
                current_packet_mask.sum()
            )

            attempted_packets += (
                self.n_agents
            )

            observations = next_observations

            (
                current_entity_observations,
                current_local_mask,
                current_packet_mask,
                current_hub_mask,
            ) = self.build_entity_input(
                observations=observations,
                rng=rng,
            )

            # Store the next state at t + 1.
            self.store_state(
                episode=episode,
                timestep=timestep + 1,
                observations=observations,
                entity_observations=(
                    current_entity_observations
                ),
                local_observation_mask=(
                    current_local_mask
                ),
                packet_received_mask=(
                    current_packet_mask
                ),
            )

            if done:
                break

        episode["pol"][0] = (
            self.env.calculate_pol()
        )

        packet_success_rate = (
            received_packets
            / attempted_packets
            if attempted_packets > 0
            else 0.0
        )

        statistics = {
            "episode_return": episode_return,
            "episode_length": executed_steps,
            "pol": float(
                episode["pol"][0]
            ),
            "epsilon": epsilon,
            "packet_loss": self.packet_loss,
            "packet_success_rate": (
                packet_success_rate
            ),
            "received_packets": (
                received_packets
            ),
            "attempted_packets": (
                attempted_packets
            ),
        }

        return episode, statistics

