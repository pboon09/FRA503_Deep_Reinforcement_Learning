from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from RL_Algorithm.RL_base_function import BaseAlgorithm


class MC_REINFORCE_network(nn.Module):
    def __init__(self, n_observations, hidden_size, n_actions, dropout, action_type="discrete"):
        super(MC_REINFORCE_network, self).__init__()
        assert action_type in ("discrete", "continuous")
        self.action_type = action_type

        self.network = nn.Sequential(
            nn.Linear(n_observations, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_actions),
        )

        if self.action_type == "continuous":
            self.log_std = nn.Parameter(torch.zeros(n_actions))

    def forward(self, x):
        return self.network(x)


class MC_REINFORCE(BaseAlgorithm):
    def __init__(
            self,
            device=None,
            num_of_action: int = None,
            action_range: list = [None, None],
            n_observations: int = None,
            hidden_dim: int = None,
            dropout: float = None,
            action_type: str = None,
            learning_rate: float = None,
            discount_factor: float = None,
    ) -> None:
        assert action_type in ("discrete", "continuous")

        self.action_type = action_type
        self.LR = learning_rate

        self.policy_net = MC_REINFORCE_network(
            n_observations, hidden_dim, num_of_action, dropout, action_type
        ).to(device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate)

        self.device = device
        self.steps_done = 0

        super(MC_REINFORCE, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )

    def _get_distribution(self, obs):
        output = self.policy_net(obs)
        if self.action_type == "discrete":
            return Categorical(logits=output)
        else:
            std = self.policy_net.log_std.exp()
            return Normal(output, std.expand_as(output))

    def _sample_action(self, dist):
        action = dist.sample()
        if self.action_type == "discrete":
            log_prob = dist.log_prob(action)
            action = action.unsqueeze(-1)
        else:
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def calculate_stepwise_returns(self, rewards):
        T = len(rewards)
        returns = torch.zeros(T, device=self.device)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + self.discount_factor * G
            returns[t] = G
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        return returns

    def generate_trajectory(self, env):
        obs, _ = env.reset()
        state = obs['policy'].to(self.device)

        rewards_list = []
        log_probs_list = []
        trajectory = []
        episode_return = 0.0
        done = False

        while not done:
            dist = self._get_distribution(state)
            action, log_prob = self._sample_action(dist)

            if self.action_type == "discrete":
                env_action = self.scale_action(action.item())
            else:
                env_action = action

            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            r = reward.mean().item()
            episode_return += r
            rewards_list.append(r)
            log_probs_list.append(log_prob.mean())
            trajectory.append((state.cpu(), action.cpu(), r))

            done = (terminated | truncated).any().item()
            state = next_obs['policy'].to(self.device)

        stepwise_returns = self.calculate_stepwise_returns(rewards_list)
        log_prob_tensor = torch.stack(log_probs_list)
        return episode_return, stepwise_returns, log_prob_tensor, trajectory

    def calculate_loss(self, stepwise_returns, log_prob_actions):
        returns = stepwise_returns.to(self.device)
        loss = -(log_prob_actions * returns).mean()
        return loss

    def update_policy(self, stepwise_returns, log_prob_actions):
        loss = self.calculate_loss(stepwise_returns, log_prob_actions)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

    def learn(self, env, num_agents: int = 1):
        self.policy_net.train()
        episode_return, stepwise_returns, log_prob_actions, trajectory = \
            self.generate_trajectory(env)
        loss = self.update_policy(stepwise_returns, log_prob_actions)
        return episode_return, loss, trajectory

    def save_model(self, path: str, filename: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy_net.state_dict(), os.path.join(path, filename))

    def load_model(self, path: str, filename: str) -> None:
        self.policy_net.load_state_dict(
            torch.load(os.path.join(path, filename), map_location=self.device)
        )
