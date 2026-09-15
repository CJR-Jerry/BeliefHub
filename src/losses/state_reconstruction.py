import torch


def state_reconstruction_loss(
    reconstructed_state,
    true_state,
):
    """
    Implements the MATCH reconstruction objective:

        L_state = L_pos + L_attr

    Both tensors:
        [..., entities, state_dim]

    Position dimensions:
        0:2

    Attribute dimensions:
        2:
    """

    reconstructed_positions = (
        reconstructed_state[..., 0:2]
    )

    true_positions = true_state[..., 0:2]

    reconstructed_center = (
        reconstructed_positions.mean(
            dim=-2,
            keepdim=True,
        )
    )

    true_center = true_positions.mean(
        dim=-2,
        keepdim=True,
    )

    centered_reconstructed_positions = (
        reconstructed_positions
        - reconstructed_center
    )

    centered_true_positions = (
        true_positions
        - true_center
    )

    position_error = (
        centered_true_positions
        - centered_reconstructed_positions
    )

    position_loss = (
        position_error.pow(2)
        .sum(dim=-1)
        .mean()
    )

    reconstructed_attributes = (
        reconstructed_state[..., 2:]
    )

    true_attributes = true_state[..., 2:]

    attribute_error = (
        true_attributes
        - reconstructed_attributes
    )

    attribute_loss = (
        attribute_error.pow(2)
        .sum(dim=-1)
        .mean()
    )

    total_loss = position_loss + attribute_loss

    return {
        "total": total_loss,
        "position": position_loss,
        "attribute": attribute_loss,
    }