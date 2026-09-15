import torch

from src.models.qmix_mixer import QMixer


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    batch_size = 8
    time_steps = 25
    n_agents = 5
    state_dim = 150

    mixer = QMixer(
        n_agents=n_agents,
        state_dim=state_dim,
    ).to(device)

    agent_q_values = torch.randn(
        batch_size,
        time_steps,
        n_agents,
        device=device,
        requires_grad=True,
    )

    states = torch.randn(
        batch_size,
        time_steps,
        state_dim,
        device=device,
    )

    total_q = mixer(agent_q_values, states)

    print("Device:", device)
    print("Agent Q shape:", agent_q_values.shape)
    print("Global state shape:", states.shape)
    print("Total Q shape:", total_q.shape)

    assert total_q.shape == (
        batch_size,
        time_steps,
        1,
    )

    # 验证 QMIX 的单调性约束
    total_q.sum().backward()

    minimum_gradient = agent_q_values.grad.min().item()
    print("Minimum dQ_total/dQ_agent:", minimum_gradient)

    assert minimum_gradient >= -1e-7

    print("QMIX mixer test passed.")


if __name__ == "__main__":
    main()