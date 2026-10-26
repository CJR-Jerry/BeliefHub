import torch
import torch.nn as nn
import torch.nn.functional as F


class MASIALatentModel(nn.Module):
    def __init__(
        self,
        latent_dim,
        n_actions,
        n_agents,
        action_embed_dim=8,
        hidden_dim=64,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.action_embedding = nn.Linear(n_actions, action_embed_dim)
        self.joint_action_embedding = nn.Sequential(
            nn.Linear(action_embed_dim * n_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.latent_embedding = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.transition = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.reward_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent, actions_one_hot):
        leading_shape = latent.shape[:-1]
        action_embedding = F.relu(self.action_embedding(actions_one_hot))
        action_embedding = action_embedding.reshape(
            *leading_shape,
            -1,
        )
        joint_action = self.joint_action_embedding(action_embedding)
        latent_embedding = self.latent_embedding(latent)
        delta = self.transition(
            torch.cat([latent_embedding, joint_action], dim=-1)
        )
        return latent + delta

    def predict_reward(self, latent):
        return self.reward_predictor(latent)

