import torch

from src.learners.qmix_learner import QMIXLearner
from src.models.agent_rnn import AgentRNN
from src.models.qmix_mixer import QMixer


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    batch_size = 4
    time_steps = 25
    n_agents = 5
    observation_dim = 30
    state_dim = 150
    n_actions = 5

    agent_network = AgentRNN(
        observation_dim=observation_dim,
        n_actions=n_actions,
        n_agents=n_agents,
        hidden_dim=64,
    ).to(device)

    mixer = QMixer(
        n_agents=n_agents,
        state_dim=state_dim,
    ).to(device)

    learner = QMIXLearner(
        agent_network=agent_network,
        mixer=mixer,
    )

    actions = torch.randint(
        0,
        n_actions,
        (batch_size, time_steps, n_agents, 1),
        device=device,
    )

    batch = {
        "observations": torch.randn(
            batch_size,
            time_steps + 1,
            n_agents,
            observation_dim,
            device=device,
        ),
        "states": torch.randn(
            batch_size,
            time_steps + 1,
            state_dim,
            device=device,
        ),
        "actions": actions,
        "rewards": torch.randn(
            batch_size,
            time_steps,
            1,
            device=device,
        ),
        "terminated": torch.zeros(
            batch_size,
            time_steps,
            1,
            device=device,
        ),
        "filled": torch.ones(
            batch_size,
            time_steps,
            1,
            device=device,
        ),
    }

    batch["terminated"][:, -1] = 1.0

    statistics = learner.train_batch(batch)

    print("Device:", device)

    for name, value in statistics.items():
        print(f"{name}: {value:.6f}")

    assert torch.isfinite(
        torch.tensor(statistics["loss"])
    )

    learner.update_targets()

    print("QMIX learner test passed.")


if __name__ == "__main__":
    main()
