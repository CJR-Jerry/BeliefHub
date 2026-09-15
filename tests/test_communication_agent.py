import torch

from src.models.communication_agent import (
    CommunicationEnhancedAgent,
)


def main():
    torch.manual_seed(0)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    batch_size = 4
    n_agents = 5
    hidden_dim = 32
    message_dim = 32
    n_actions = 5

    model = CommunicationEnhancedAgent(
        hidden_dim=hidden_dim,
        message_dim=message_dim,
        n_actions=n_actions,
    ).to(device)

    agent_hidden = torch.randn(
        batch_size,
        n_agents,
        hidden_dim,
        device=device,
        requires_grad=True,
    )

    hub_messages = torch.randn(
        batch_size,
        n_agents,
        message_dim,
        device=device,
        requires_grad=True,
    )

    q_values = model(
        agent_hidden=agent_hidden,
        hub_messages=hub_messages,
    )

    print("Device:", device)
    print("Agent hidden:", agent_hidden.shape)
    print("Hub messages:", hub_messages.shape)
    print("Q-values:", q_values.shape)

    assert q_values.shape == (
        batch_size,
        n_agents,
        n_actions,
    )

    assert torch.isfinite(q_values).all()

    # Test whether the output depends on Hub messages.
    zero_messages = torch.zeros_like(
        hub_messages
    )

    q_values_without_communication = model(
        agent_hidden=agent_hidden,
        hub_messages=zero_messages,
    )

    message_effect = (
        q_values
        - q_values_without_communication
    ).abs().mean()

    print(
        "Mean message effect:",
        message_effect.item(),
    )

    assert message_effect.item() > 0.0

    # Test gradient propagation.
    loss = q_values.pow(2).mean()
    loss.backward()

    assert agent_hidden.grad is not None
    assert hub_messages.grad is not None
    assert model.value_layer.weight.grad is not None
    assert model.value_layer.bias.grad is not None

    assert torch.isfinite(
        agent_hidden.grad
    ).all()

    assert torch.isfinite(
        hub_messages.grad
    ).all()

    assert torch.isfinite(
        model.value_layer.weight.grad
    ).all()

    # Test invalid batch/agent dimensions.
    invalid_messages = torch.randn(
        batch_size,
        n_agents + 1,
        message_dim,
        device=device,
    )

    try:
        model(
            agent_hidden=agent_hidden.detach(),
            hub_messages=invalid_messages,
        )
    except ValueError:
        invalid_shape_detected = True
    else:
        invalid_shape_detected = False

    assert invalid_shape_detected

    print(
        "Communication-enhanced agent test passed."
    )


if __name__ == "__main__":
    main()