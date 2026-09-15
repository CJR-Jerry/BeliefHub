import torch

from src.models.spatiotemporal_reconstructor import (
    SpatiotemporalReconstructor,
)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    batch_size = 3
    n_entities = 10
    embedding_dim = 32
    state_dim = 8
    history_length = 4

    model = SpatiotemporalReconstructor(
        n_entities=n_entities,
        entity_embedding_dim=embedding_dim,
        state_dim=state_dim,
        history_length=history_length,
        n_heads=4,
        n_layers=2,
    ).to(device)

    current_entities = torch.randn(
        batch_size,
        n_entities,
        embedding_dim,
        device=device,
        requires_grad=True,
    )

    current_observed_mask = (
        torch.rand(
            batch_size,
            n_entities,
            device=device,
        )
        > 0.3
    )

    history_states = torch.randn(
        batch_size,
        history_length,
        n_entities,
        state_dim,
        device=device,
    )

    history_valid_mask = torch.tensor(
        [
            [True, True, True, True],
            [False, True, True, True],
            [False, False, True, True],
        ],
        device=device,
    )

    reconstructed, latent = model(
        current_entities=current_entities,
        current_observed_mask=current_observed_mask,
        history_states=history_states,
        history_valid_mask=history_valid_mask,
    )

    print("Device:", device)
    print(
        "Reconstructed state:",
        reconstructed.shape,
    )
    print("Latent state:", latent.shape)

    assert reconstructed.shape == (
        batch_size,
        n_entities,
        state_dim,
    )

    assert latent.shape == (
        batch_size,
        n_entities,
        embedding_dim,
    )

    target_state = torch.randn_like(reconstructed)

    loss = (
        reconstructed - target_state
    ).pow(2).mean()

    loss.backward()

    assert current_entities.grad is not None
    assert torch.isfinite(current_entities.grad).all()

    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()

    print("Reconstruction loss:", loss.item())
    print(
        "Spatiotemporal reconstructor test passed."
    )


if __name__ == "__main__":
    main()