import numpy as np
import torch

from src.envs.cooperative_navigation import (
    CooperativeNavigation,
)
from src.envs.entity_observation import (
    MPEEntityObservationAdapter,
)
from src.envs.global_entity_state import (
    build_global_entity_state,
)
from src.losses.state_reconstruction import (
    state_reconstruction_loss,
)
from src.models.match_information_hub import (
    HubHistoryBuffer,
    MATCHInformationHub,
)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    env = CooperativeNavigation(
        n_agents=5,
        max_cycles=25,
    )

    observations, _ = env.reset(seed=0)

    adapter = MPEEntityObservationAdapter(
        n_agents=5,
        n_landmarks=5,
        sight_range=0.8,
    )

    hub = MATCHInformationHub(
        n_entities=10,
        observation_dim=8,
        embedding_dim=32,
        state_dim=8,
        history_length=4,
    ).to(device)

    history = HubHistoryBuffer(
        history_length=4,
        n_entities=10,
        state_dim=8,
    )

    history.reset(
        batch_size=1,
        device=device,
    )

    rng = np.random.default_rng(0)
    losses = []

    for timestep in range(6):
        (
            entity_observations,
            observation_mask,
        ) = adapter.convert(
            observations=observations,
            agent_names=env.possible_agents,
            packet_loss=0.3,
            rng=rng,
        )

        entity_tensor = torch.as_tensor(
            entity_observations,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        mask_tensor = torch.as_tensor(
            observation_mask,
            dtype=torch.bool,
            device=device,
        ).unsqueeze(0)

        history_states, history_valid = history.get()

        hub_output = hub(
            entity_observations=entity_tensor,
            observation_mask=mask_tensor,
            history_states=history_states,
            history_valid_mask=history_valid,
        )

        true_state = torch.as_tensor(
            build_global_entity_state(
                env.env.unwrapped.world
            ),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        reconstruction_losses = (
            state_reconstruction_loss(
                hub_output["reconstructed_state"],
                true_state,
            )
        )

        losses.append(reconstruction_losses["total"])

        print(
            f"timestep={timestep} "
            f"visible_entities="
            f"{hub_output['observed_entities'].sum().item()} "
            f"valid_history="
            f"{history_valid.sum().item()} "
            f"loss="
            f"{reconstruction_losses['total'].item():.4f}"
        )

        history.append(
            hub_output["reconstructed_state"]
        )

        actions = {
            agent: env.action_space(agent).sample()
            for agent in env.agents
        }

        (
            observations,
            _,
            _,
            _,
            _,
        ) = env.step(actions)

    total_loss = torch.stack(losses).mean()

    assert torch.isfinite(total_loss)

    total_loss.backward()

    has_finite_gradient = False

    for parameter in hub.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
            has_finite_gradient = True

    assert has_finite_gradient

    final_history, final_valid = history.get()

    assert final_history.shape == (1, 4, 10, 8)
    assert final_valid.all()

    print("Mean reconstruction loss:", total_loss.item())
    print("Complete Information Hub test passed.")

    env.close()


if __name__ == "__main__":
    main()