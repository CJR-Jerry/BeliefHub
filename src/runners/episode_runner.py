import numpy as np
import torch
import torch.nn.functional as F


class EpisodeRunner:
    def __init__(
        self,
        env,
        agent_network,
        device,
    ):
        self.env = env
        self.agent_network = agent_network
        self.device = device

        self.agent_names = list(env.possible_agents)
        self.n_agents = env.n_agents
        self.max_steps = env.max_cycles

        first_agent = self.agent_names[0]

        self.observation_dim = (
            env.observation_space(first_agent).shape[0]
        )

        self.n_actions = env.action_space(first_agent).n

        env.reset(seed=0)
        self.state_dim = env.env.state().size

    def stack_observations(self, observations):
        stacked = []

        for agent in self.agent_names:
            if agent in observations:
                stacked.append(observations[agent])
            else:
                stacked.append(
                    np.zeros(
                        self.observation_dim,
                        dtype=np.float32,
                    )
                )

        return np.asarray(stacked, dtype=np.float32)
    def run(self, epsilon=1.0, seed=None):
        observations, _ = self.env.reset(seed=seed)

        episode = {
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
                (self.max_steps, 1),
                dtype=np.float32,
            ),
            "terminated": np.zeros(
                (self.max_steps, 1),
                dtype=np.float32,
            ),
            "filled": np.zeros(
                (self.max_steps, 1),
                dtype=np.float32,
            ),
            "pol": np.zeros(
                (1,),
                dtype=np.float32,
            ),
        }

        hidden_states = self.agent_network.initial_hidden(
            batch_size=1,
            device=self.device,
        )

        previous_actions = torch.zeros(
            1,
            self.n_agents,
            self.n_actions,
            device=self.device,
        )

        episode_return = 0.0
        executed_steps = 0

        for timestep in range(self.max_steps):
            observation_array = self.stack_observations(
                observations
            )

            episode["observations"][timestep] = (
                observation_array
            )

            episode["states"][timestep] = (
                self.env.env.state()
            )

            observation_tensor = torch.as_tensor(
                observation_array,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            with torch.no_grad():
                q_values, hidden_states = self.agent_network(
                    observation_tensor,
                    previous_actions,
                    hidden_states,
                )

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
            ) = self.env.step(action_dictionary)

            team_reward = (
                float(np.mean(list(rewards.values())))
                if rewards
                else 0.0
            )

            done = all(
                terminations.get(agent, False)
                or truncations.get(agent, False)
                for agent in self.agent_names
            )

            episode["actions"][timestep, :, 0] = (
                selected_actions
            )

            episode["rewards"][timestep, 0] = team_reward
            episode["terminated"][timestep, 0] = float(done)
            episode["filled"][timestep, 0] = 1.0

            episode_return += team_reward
            executed_steps += 1

            previous_action_indices = torch.tensor(
                selected_actions,
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)

            previous_actions = F.one_hot(
                previous_action_indices,
                num_classes=self.n_actions,
            ).float()

            observations = next_observations

            episode["observations"][timestep + 1] = (
                self.stack_observations(observations)
            )

            episode["states"][timestep + 1] = (
                self.env.env.state()
            )

            if done:
                break

        episode["pol"][0] = self.env.calculate_pol()

        statistics = {
            "episode_return": episode_return,
            "episode_length": executed_steps,
            "pol": float(episode["pol"][0]),
            "epsilon": epsilon,
        }

        return episode, statistics