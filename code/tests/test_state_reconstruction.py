import numpy as np
import torch

from src.envs.cooperative_navigation import (
    CooperativeNavigation,
)
from src.envs.global_entity_state import (
    build_global_entity_state,
)
from src.losses.state_reconstruction import (
    state_reconstruction_loss,
)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    env = CooperativeNavigation(
        n_agents=5,
        max_cycles=25,
    )

    env.reset(seed=0)

    world = env.env.unwrapped.world

    true_state_numpy = build_global_entity_state(
        world
    )

    print(
        "True entity state shape:",
        true_state_numpy.shape,
    )

    assert true_state_numpy.shape == (10, 8)

    # 前5个是智能体，后5个是地标
    assert np.allclose(
        true_state_numpy[:5, 6:8],
        np.array([1.0, 0.0]),
    )

    assert np.allclose(
        true_state_numpy[5:, 6:8],
        np.array([0.0, 1.0]),
    )

    true_state = torch.as_tensor(
        true_state_numpy,
        device=device,
    ).unsqueeze(0)

    # 完美重构：所有损失都应为0
    perfect_result = state_reconstruction_loss(
        true_state.clone(),
        true_state,
    )

    print(
        "Perfect reconstruction loss:",
        perfect_result["total"].item(),
    )

    assert torch.allclose(
        perfect_result["total"],
        torch.tensor(0.0, device=device),
    )

    # 整体平移不应产生位置损失
    translated_state = true_state.clone()
    translated_state[..., 0:2] += torch.tensor(
        [5.0, -3.0],
        device=device,
    )

    translated_result = state_reconstruction_loss(
        translated_state,
        true_state,
    )

    print(
        "Translation-invariant position loss:",
        translated_result["position"].item(),
    )

    assert torch.allclose(
        translated_result["position"],
        torch.tensor(0.0, device=device),
        atol=1e-6,
    )
    # 先完成所有修改，再启用梯度
    incorrect_state = true_state.clone().detach()

    # 改变第0个实体的位置
    incorrect_state[..., 0, 0] += 1.0

    # 改变第0个实体的速度属性
    incorrect_state[..., 0, 2] += 1.0

    incorrect_state.requires_grad_(True)

    incorrect_result = state_reconstruction_loss(
        incorrect_state,
        true_state,
    )

    print(
        "Incorrect position loss:",
        incorrect_result["position"].item(),
    )

    print(
        "Incorrect attribute loss:",
        incorrect_result["attribute"].item(),
    )

    assert incorrect_result["position"].item() > 0.0
    assert incorrect_result["attribute"].item() > 0.0

    incorrect_result["total"].backward()

    assert incorrect_state.grad is not None
    assert torch.isfinite(incorrect_state.grad).all()

    print("State reconstruction test passed.")

    env.close()

if __name__ == "__main__":
    main()
