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
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
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
            value_loss_coef: float = 0.5,
    ) -> None:
        assert action_type in ("discrete", "continuous")
        self.action_type = action_type
        self.LR = learning_rate
        self.entropy_coef = entropy_coef
        self.num_epochs = num_epochs
        self.value_loss_coef = value_loss_coef
        self.policy_net = MC_REINFORCE_network(
            n_observations, hidden_dim, num_of_action, dropout, action_type
        ).to(device)
        # Value baseline network (REINFORCE with Baseline, Sutton & Barto Ch.13.4)
        self.value_net = nn.Sequential(
            nn.Linear(n_observations, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=learning_rate,
        )
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

        obs, _ = env.reset()
        state = obs['policy'].to(self.device)

        total_episodes = 0
        total_return = 0.0
        sum_reward = 0.0
        last_log = 0
        global_step = 0
        iteration = 0
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
                    action, _ = self._sample_action(dist)

                    actions_buf.append(action)

                    if self.action_type == "continuous":
                        env_action = torch.clamp(action, self.action_range[0], self.action_range[1])
                    else:
                        env_action = action.float()

                    next_obs, reward, terminated, truncated, _ = env.step(env_action)
                    done = (terminated | truncated).float().to(self.device)

                    rewards_buf.append(reward.to(self.device).view(-1))
                    dones_buf.append(done.view(-1))

                    ep_rewards += reward.to(self.device).view(-1)
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

            # # Compute MC returns -- bootstrap incomplete episodes from V(s_T)
            # with torch.no_grad():
            #     G = self.value_net(state).squeeze(-1)  # V(s_T) for incomplete episodes
            # returns = torch.zeros(T, num_agents, device=self.device)
            # for t in reversed(range(T)):
            #     G = rewards[t] + self.discount_factor * G * (1.0 - dones[t])
            #     returns[t] = G

            # Pure MC returns -- only use completed episodes, no bootstrapping
            returns = torch.zeros(T, num_agents, device=self.device)
            G = torch.zeros(num_agents, device=self.device)
            # Build a mask of timesteps that belong to completed episodes
            complete_mask = torch.zeros(T, num_agents, dtype=torch.bool, device=self.device)
            # Track which envs have at least one done in this rollout
            has_done = torch.zeros(num_agents, dtype=torch.bool, device=self.device)
            for t in range(T):
                if dones[t].sum() > 0:
                    has_done |= (dones[t] > 0.5)
            # Backward pass: accumulate returns, mark complete episode steps
            ep_complete = torch.zeros(num_agents, dtype=torch.bool, device=self.device)
            for t in reversed(range(T)):
                # At episode boundary, mark that everything from here backward (until previous done) is complete
                ep_complete = ep_complete | (dones[t] > 0.5)
                G = rewards[t] + self.discount_factor * G * (1.0 - dones[t])
                returns[t] = G
                complete_mask[t] = ep_complete

            # If no episodes completed in this rollout, skip the update
            if not complete_mask.any():
                iteration += 1
                continue

            # Multi-epoch update with value baseline (only on completed episode data)
            for _ in range(self.num_epochs):
                # Re-evaluate log_probs, entropy, and values with current networks
                all_log_probs = []
                all_entropy = []
                all_values = []
                for t in range(T):
                    dist = self._get_distribution(obs_tensor[t])
                    if self.action_type == "continuous":
                        lp = dist.log_prob(actions_tensor[t]).sum(dim=-1)
                    else:
                        lp = dist.log_prob(actions_tensor[t].squeeze(-1))
                    all_log_probs.append(lp)
                    all_entropy.append(dist.entropy().mean())
                    all_values.append(self.value_net(obs_tensor[t]).squeeze(-1))

                log_probs = torch.stack(all_log_probs)   # (T, N)
                entropy = torch.stack(all_entropy).mean()
                values = torch.stack(all_values)          # (T, N)

                # Advantage = MC return - value baseline (variance reduction)
                # Only use completed episode transitions
                masked_returns = returns[complete_mask].detach()
                masked_values = values[complete_mask]
                masked_log_probs = log_probs[complete_mask]

                advantage = masked_returns - masked_values.detach()
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                policy_loss = -(advantage * masked_log_probs).mean() - self.entropy_coef * entropy
                value_loss = (masked_values - masked_returns).pow(2).mean()
                loss = policy_loss + self.value_loss_coef * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(self.policy_net.parameters()) + list(self.value_net.parameters()),
                    max_norm=0.5,
                )
                self.optimizer.step()

            # Log training metrics
            with torch.no_grad():
                ev = 1.0 - (masked_returns - masked_values.detach()).var() / (masked_returns.var() + 1e-8)
            self.metrics_log.append({
                "iteration": iteration,
                "global_step": global_step,
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy.item(),
                "grad_norm": grad_norm.item(),
                "explained_variance": ev.item(),
            })
            iteration += 1

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
            torch.load(os.path.join(path, filename), map_location=self.device, weights_only=True)
        )
