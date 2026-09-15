import torch

from src.models.agent_rnn import AgentRNN


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    batch_size = 8
    n_agents = 5
    observation_dim = 30
    n_actions = 5
    hidden_dim = 64

    model = AgentRNN(
        observation_dim=observation_dim,
        n_actions=n_actions,
        n_agents=n_agents,
        hidden_dim=hidden_dim,
    ).to(device)

    observations = torch.randn(
        batch_size,
        n_agents,
        observation_dim,
        device=device,
    )

    previous_actions = torch.zeros(
        batch_size,
        n_agents,
        n_actions,
        device=device,
    )

    hidden_states = model.initial_hidden(
        batch_size=batch_size,
        device=device,
    )

    q_values, next_hidden = model(
        observations,
        previous_actions,
        hidden_states,
    )

    print("Device:", device)
    print("Observation shape:", observations.shape)
    print("Q-value shape:", q_values.shape)
    print("Hidden-state shape:", next_hidden.shape)

    assert q_values.shape == (8, 5, 5)
    assert next_hidden.shape == (8, 5, 64)

    print("Agent RNN test passed.")


if __name__ == "__main__":
    main()