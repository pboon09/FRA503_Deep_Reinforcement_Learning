from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from storage.on_policy import OnPolicyAlgorithm
from networks.mlp import MLP


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [None],
        activation: str = None,
        action_type: str = None,
        init_noise_std: float = None,
    ):
        super().__init__()
        assert action_type in ("continuous", "discrete")
        self.action_type = action_type
        self.action_dim  = action_dim

        self.actor  = MLP(state_dim, action_dim, hidden_dims, activation)
        self.critic = MLP(state_dim, 1,          hidden_dims, activation)

        if self.action_type == "continuous":
            self.std = nn.Parameter(init_noise_std * torch.ones(action_dim))

        self.distribution: Normal | Categorical | None = None

    @property
    def action_mean(self) -> torch.Tensor:
        if self.action_type == "continuous":
            return self.distribution.mean
        return self.distribution.probs

    @property
    def action_std(self) -> torch.Tensor:
        if self.action_type == "continuous":
            return self.distribution.stddev
        return torch.ones_like(self.distribution.probs)

    @property
    def entropy(self) -> torch.Tensor:
        if self.action_type == "continuous":
            return self.distribution.entropy().sum(dim=-1)
        return self.distribution.entropy()

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError("Use act() or evaluate().")

    def _update_distribution(self, obs: torch.Tensor) -> None:
        if self.action_type == "continuous":
            mean = self.actor(obs)
            self.distribution = Normal(mean, self.std.expand_as(mean))
        else:
            logits = self.actor(obs)
            self.distribution = Categorical(logits=logits)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        self._update_distribution(obs)
        actions = self.distribution.sample()
        if self.action_type == "discrete":
            actions = actions.unsqueeze(-1)
        return actions

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        if self.action_type == "continuous":
            return self.actor(obs)
        else:
            logits = self.actor(obs)
            return logits.argmax(dim=-1, keepdim=True)

    def evaluate(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_type == "continuous":
            return self.distribution.log_prob(actions).sum(dim=-1)
        else:
            return self.distribution.log_prob(actions.squeeze(-1))


class AC(OnPolicyAlgorithm):
    def __init__(
        self,
        device=None,
        num_of_action: int = None,
        action_range: list = [None, None],
        n_observations: int = None,
        hidden_dims: list[int] = [None],
        activation: str = None,
        action_type: str = None,
        init_noise_std: float = None,
        learning_rate: float = None,
        discount_factor: float = None,
        value_loss_coef: float = None,
        entropy_coef: float = None,
        max_grad_norm: float = None,
    ) -> None:

        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.policy = ActorCritic(
            state_dim=n_observations,
            action_dim=num_of_action,
            hidden_dims=hidden_dims,
            activation=activation,
            action_type=action_type,
            init_noise_std=init_noise_std,
        ).to(self.device)

        self.optimizer       = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.action_type     = action_type
        self.value_loss_coef = value_loss_coef
        self.entropy_coef    = entropy_coef
        self.max_grad_norm   = max_grad_norm

        super(AC, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )

    def generate_trajectory(self, env) -> tuple:
        log_probs = []
        values = []
        rewards = []
        entropies = []
        episode_return = 0.0

        obs, _ = env.reset()
        done = False
        timestep = 0

        while not done:
            state = obs['policy'].to(self.device)
            self.policy._update_distribution(state)
            action = self.policy.distribution.sample()
            if self.action_type == "discrete":
                action_for_log = action.unsqueeze(-1)
            else:
                action_for_log = action

            log_prob = self.policy.get_actions_log_prob(action_for_log)
            value = self.policy.evaluate(state)
            entropy = self.policy.entropy

            if self.action_type == "discrete":
                env_action = self.scale_action(action.item())
            else:
                env_action = action

            obs, reward, terminated, truncated, _ = env.step(env_action)
            done = (terminated | truncated).any().item()

            log_probs.append(log_prob.mean())
            values.append(value.mean())
            rewards.append(reward.mean().item())
            entropies.append(entropy.mean())
            episode_return += rewards[-1]
            timestep += 1

        log_probs_t = torch.stack(log_probs)
        values_t = torch.stack(values).squeeze(-1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        self._last_entropies = torch.stack(entropies)

        return episode_return, log_probs_t, values_t, rewards_t, timestep

    def compute_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        T = len(rewards)
        returns = torch.zeros(T, device=self.device)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + self.discount_factor * G
            returns[t] = G
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def calculate_loss(self, log_prob_actions, values, returns):
        advantage = (returns - values).detach()
        actor_loss = -(log_prob_actions * advantage).mean()
        critic_loss = nn.functional.mse_loss(values, returns)
        return actor_loss, critic_loss

    def update_policy(self, log_prob_actions, values, returns) -> float:
        actor_loss, critic_loss = self.calculate_loss(log_prob_actions, values, returns)
        entropy_loss = -self.entropy_coef * self._last_entropies.mean()
        total_loss = actor_loss + self.value_loss_coef * critic_loss + entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return total_loss.item()

    def learn(self, env, max_steps: int, num_agents: int) -> tuple:
        self.policy.train()

        if num_agents <= 1:
            episode_return, log_probs, values, rewards, timestep = \
                self.generate_trajectory(env)
            returns = self.compute_returns(rewards)
            loss = self.update_policy(log_probs, values, returns)
            return episode_return, loss, timestep

        T = 200
        obs, _ = env.reset()
        state = obs['policy'].to(self.device)

        log_probs_buf = []
        values_buf = []
        rewards_buf = []
        dones_buf = []
        entropies_buf = []

        for _ in range(T):
            self.policy._update_distribution(state)
            action = self.policy.distribution.sample()
            if self.action_type == "discrete":
                action_for_log = action.unsqueeze(-1)
            else:
                action_for_log = action

            log_prob = self.policy.get_actions_log_prob(action_for_log)
            value = self.policy.evaluate(state)
            entropy = self.policy.entropy

            if self.action_type == "discrete":
                action_vals = []
                for i in range(num_agents):
                    action_vals.append(
                        self.action_range[0] + (self.action_range[1] - self.action_range[0])
                        * action[i].item() / (self.num_of_action - 1)
                    )
                env_action = torch.tensor(action_vals, dtype=torch.float32).unsqueeze(-1).to(self.device)
            else:
                env_action = action

            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            done = (terminated | truncated).float().to(self.device)

            log_probs_buf.append(log_prob)
            values_buf.append(value.squeeze(-1))
            rewards_buf.append(reward.to(self.device))
            dones_buf.append(done)
            entropies_buf.append(entropy)

            state = next_obs['policy'].to(self.device)

        log_probs = torch.stack(log_probs_buf)
        values = torch.stack(values_buf)
        rewards = torch.stack(rewards_buf)
        dones = torch.stack(dones_buf)
        entropies = torch.stack(entropies_buf)

        G = torch.zeros(num_agents, device=self.device)
        returns = torch.zeros(T, num_agents, device=self.device)
        for t in reversed(range(T)):
            G = rewards[t] + self.discount_factor * G * (1.0 - dones[t])
            returns[t] = G

        returns_flat = returns.reshape(-1)
        returns_flat = (returns_flat - returns_flat.mean()) / (returns_flat.std() + 1e-8)
        returns = returns_flat.reshape(T, num_agents)

        advantage = (returns - values).detach()
        actor_loss = -(log_probs * advantage).mean()
        critic_loss = (values - returns).pow(2).mean()
        entropy_loss = -self.entropy_coef * entropies.mean()
        total_loss = actor_loss + self.value_loss_coef * critic_loss + entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return rewards.sum(0).mean().item(), total_loss.item(), T

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        return self.policy.act(obs)

    def process_env_step(self, rewards, dones) -> None:
        pass

    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        self.policy.eval()
        with torch.inference_mode():
            return self.policy.act_inference(obs)

    def save_model(self, path: str, filename: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(path, filename))

    def load_model(self, path: str, filename: str) -> None:
        self.policy.load_state_dict(
            torch.load(os.path.join(path, filename), map_location=self.device)
        )
