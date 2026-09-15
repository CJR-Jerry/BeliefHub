import torch

from src.learners.match_learner import MATCHLearner
from src.models.match_agent import MATCHAgent
from src.models.match_information_hub import MATCHInformationHub
from src.models.qmix_mixer import QMixer


def test_local_belief_training_step():
    torch.manual_seed(3)
    batch_size, time_steps = 2, 3
    n_agents, n_entities, n_actions = 6, 12, 5
    state_dim = 216

    agent = MATCHAgent(
        n_agents=n_agents,
        n_entities=n_entities,
        entity_observation_dim=8,
        hidden_dim=32,
        query_dim=32,
        reconstructed_entity_dim=32,
        n_actions=n_actions,
        use_local_belief=True,
        belief_hidden_dim=32,
        belief_state_dim=8,
    )
    hub = MATCHInformationHub(
        n_entities=n_entities,
        observation_dim=8,
        embedding_dim=32,
        state_dim=8,
        history_length=4,
    )
    mixer = QMixer(n_agents=n_agents, state_dim=state_dim)
    learner = MATCHLearner(
        match_agent=agent,
        information_hub=hub,
        mixer=mixer,
        belief_loss_weight=0.01,
        belief_supervision="all",
    )

    observation_masks = torch.rand(
        batch_size,
        time_steps + 1,
        n_agents,
        n_entities,
    ) > 0.25
    batch = {
        "states": torch.randn(batch_size, time_steps + 1, state_dim),
        "actions": torch.randint(
            0,
            n_actions,
            (batch_size, time_steps, n_agents, 1),
        ),
        "rewards": torch.randn(batch_size, time_steps, 1),
        "terminated": torch.zeros(batch_size, time_steps, 1),
        "filled": torch.ones(batch_size, time_steps, 1),
        "entity_observations": torch.randn(
            batch_size,
            time_steps + 1,
            n_agents,
            n_entities,
            8,
        ),
        "observation_masks": observation_masks,
        "packet_loss_masks": torch.rand(
            batch_size,
            time_steps + 1,
            n_agents,
        ) > 0.3,
        "entity_states": torch.randn(
            batch_size,
            time_steps + 1,
            n_entities,
            8,
        ),
    }
    batch["terminated"][:, -1] = 1.0

    before = agent.local_belief.mean_head.weight.detach().clone()
    statistics = learner.train_batch(batch)
    after = agent.local_belief.mean_head.weight.detach()

    for value in statistics.values():
        assert torch.isfinite(torch.tensor(value))
    assert statistics["belief_supervised_fraction"] > 0.0
    assert statistics["belief_rmse"] > 0.0
    assert not torch.allclose(before, after)
