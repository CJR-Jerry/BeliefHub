import torch

from src.learners.match_learner import (
    MATCHLearner,
)
from src.models.match_agent import MATCHAgent
from src.models.match_information_hub import (
    MATCHInformationHub,
)
from src.models.qmix_mixer import QMixer


def main():
    torch.manual_seed(0)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    batch_size = 2
    time_steps = 5
    n_agents = 5
    n_entities = 10
    n_actions = 5
    state_dim = 150

    match_agent = MATCHAgent(
        n_agents=n_agents,
        n_entities=n_entities,
        entity_observation_dim=8,
        hidden_dim=32,
        query_dim=32,
        reconstructed_entity_dim=32,
        n_actions=n_actions,
    ).to(device)

    information_hub = MATCHInformationHub(
        n_entities=n_entities,
        observation_dim=8,
        embedding_dim=32,
        state_dim=8,
        history_length=4,
    ).to(device)

    mixer = QMixer(
        n_agents=n_agents,
        state_dim=state_dim,
    ).to(device)

    reconstruction_weight = 0.1

    learner = MATCHLearner(
        match_agent=match_agent,
        information_hub=information_hub,
        mixer=mixer,
        learning_rate=5e-4,
        gamma=0.99,
        reconstruction_weight=(
            reconstruction_weight
        ),
        grad_clip=10.0,
    )

    actions = torch.randint(
        0,
        n_actions,
        (
            batch_size,
            time_steps,
            n_agents,
            1,
        ),
        device=device,
    )

    observation_masks = (
        torch.rand(
            batch_size,
            time_steps + 1,
            n_agents,
            n_entities,
            device=device,
        )
        > 0.3
    )

    # Every agent can observe itself locally.
    for agent_index in range(n_agents):
        observation_masks[
            :,
            :,
            agent_index,
            agent_index,
        ] = True

    packet_loss_masks = (
        torch.rand(
            batch_size,
            time_steps + 1,
            n_agents,
            device=device,
        )
        > 0.3
    )

    batch = {
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

        "entity_observations": torch.randn(
            batch_size,
            time_steps + 1,
            n_agents,
            n_entities,
            8,
            device=device,
        ),

        "observation_masks": (
            observation_masks
        ),

        "packet_loss_masks": (
            packet_loss_masks
        ),

        "entity_states": torch.randn(
            batch_size,
            time_steps + 1,
            n_entities,
            8,
            device=device,
        ),
    }

    batch["terminated"][:, -1] = 1.0

    parameter_before = (
        match_agent.q_network
        .value_layer
        .weight
        .detach()
        .clone()
    )

    statistics = learner.train_batch(
        batch
    )

    print("Device:", device)

    for name, value in statistics.items():
        print(f"{name}: {value:.6f}")

    for value in statistics.values():
        assert torch.isfinite(
            torch.tensor(value)
        )

    expected_total_loss = (
        statistics["td_loss"]
        + reconstruction_weight
        * statistics["state_loss"]
    )

    assert abs(
        statistics["loss"]
        - expected_total_loss
    ) < 1e-4

    parameter_after = (
        match_agent.q_network
        .value_layer
        .weight
        .detach()
    )

    assert not torch.allclose(
        parameter_before,
        parameter_after,
    )

    # Verify that the Information Hub receives gradients.
    hub_has_gradient = False

    for parameter in (
        information_hub.parameters()
    ):
        if parameter.grad is not None:
            assert torch.isfinite(
                parameter.grad
            ).all()

            hub_has_gradient = True

    assert hub_has_gradient

    learner.update_targets()

    for online_parameter, target_parameter in zip(
        match_agent.parameters(),
        learner.target_match_agent.parameters(),
    ):
        assert torch.allclose(
            online_parameter,
            target_parameter,
        )

    for online_parameter, target_parameter in zip(
        information_hub.parameters(),
        learner
        .target_information_hub
        .parameters(),
    ):
        assert torch.allclose(
            online_parameter,
            target_parameter,
        )

    for online_parameter, target_parameter in zip(
        mixer.parameters(),
        learner.target_mixer.parameters(),
    ):
        assert torch.allclose(
            online_parameter,
            target_parameter,
        )

    print("MATCH learner test passed.")


if __name__ == "__main__":
    main()