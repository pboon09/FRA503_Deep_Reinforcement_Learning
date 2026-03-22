from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.optim as optim
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
            self.log_std = nn.Parameter(torch.full((n_actions,), -0.5))

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
            entropy_coef: float = 0.01,
            num_epochs: int = 5,
    ) -> None:
        assert action_type in ("discrete", "continuous")
        self.action_type = action_type
        self.LR = learning_rate
        self.entropy_coef = entropy_coef
        self.num_epochs = num_epochs
        self.policy_net = MC_REINFORCE_network(
            n_observations, hidden_dim, num_of_action, dropout, action_type
        ).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.device = device

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

    def learn(self, env, num_agents: int = 1, n_episodes: int = 20000, rollout_steps: int = 512):
        self.policy_net.train()
        T = rollout_steps
        update_every = T

        obs, _ = env.reset()
        state = obs['policy'].to(self.device)

        total_episodes = 0
        total_return = 0.0
        sum_reward = 0.0
        last_log = 0
        global_step = 0
        ep_rewards = torch.zeros(num_agents, device=self.device)
        ep_steps = torch.zeros(num_agents, dtype=torch.int, device=self.device)

        while total_episodes < n_episodes:
            obs_buf = []
            actions_buf = []
            rewards_buf = []
            dones_buf = []

            with torch.no_grad():
                for _ in range(T):
                    obs_buf.append(state.clone())
                    dist = self._get_distribution(state)
                    action, log_prob = self._sample_action(dist)
                    actions_buf.append(action)

                    if self.action_type == "continuous":
                        env_action = torch.clamp(action, self.action_range[0], self.action_range[1])
                    else:
                        env_action = action.float()

                    next_obs, reward, terminated, truncated, _ = env.step(env_action)
                    done = (terminated | truncated).float().to(self.device)

                    rewards_buf.append(reward.to(self.device).squeeze())
                    dones_buf.append(done.squeeze())

                    ep_rewards += reward.to(self.device).squeeze()
                    ep_steps += 1
                    global_step += num_agents
                    for i in range(num_agents):
                        if done[i].item() > 0.5:
                            self.episode_log.append({
                                "episode": total_episodes,
                                "global_step": global_step,
                                "ep_return": ep_rewards[i].item(),
                                "ep_length": ep_steps[i].item(),
                            })
                            sum_reward += ep_rewards[i].item()
                            total_return += ep_rewards[i].item()
                            ep_rewards[i] = 0.0
                            ep_steps[i] = 0
                            total_episodes += 1

                    state = next_obs['policy'].to(self.device)

            obs_tensor = torch.stack(obs_buf)        # (T, N, 4)
            actions_tensor = torch.stack(actions_buf)  # (T, N, act_dim)
            rewards = torch.stack(rewards_buf)          # (T, N)
            dones = torch.stack(dones_buf)              # (T, N)

            # Compute MC returns (no grad needed)
            G = torch.zeros(num_agents, device=self.device)
            returns = torch.zeros(T, num_agents, device=self.device)
            for t in reversed(range(T)):
                G = rewards[t] + self.discount_factor * G * (1.0 - dones[t])
                returns[t] = G

            # Normalize PER-ENV (dim=0), not globally
            mean = returns.mean(dim=0, keepdim=True)
            std = returns.std(dim=0, keepdim=True)
            returns = (returns - mean) / (std + 1e-8)

            # Multi-epoch update (like PPO but without clipping)
            for _ in range(self.num_epochs):
                # Re-evaluate log_probs and entropy with current policy
                all_log_probs = []
                all_entropy = []
                for t in range(T):
                    dist = self._get_distribution(obs_tensor[t])
                    if self.action_type == "continuous":
                        lp = dist.log_prob(actions_tensor[t]).sum(dim=-1)
                    else:
                        lp = dist.log_prob(actions_tensor[t].squeeze(-1))
                    all_log_probs.append(lp)
                    all_entropy.append(dist.entropy().mean())

                log_probs = torch.stack(all_log_probs)   # (T, N)
                entropy = torch.stack(all_entropy).mean()

                loss = -(returns.detach() * log_probs).mean() - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
                self.optimizer.step()

            if total_episodes - last_log >= 100 and total_episodes > 0:
                n_new = total_episodes - last_log
                avg = sum_reward / n_new
                print(f"[MC_REINFORCE] ep {total_episodes} | avg_return={avg:.2f}")
                self.plot_durations(timestep=int(avg))
                sum_reward = 0.0
                last_log = total_episodes

        return total_return / max(total_episodes, 1), total_episodes

    def save_model(self, path: str, filename: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy_net.state_dict(), os.path.join(path, filename))

    def load_model(self, path: str, filename: str) -> None:
        self.policy_net.load_state_dict(
            torch.load(os.path.join(path, filename), map_location=self.device)
        )
