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
    def __init__(self, state_dim, action_dim, hidden_dims=[None], activation=None,
                 action_type=None, init_noise_std=None):
        super().__init__()
        assert action_type in ("continuous", "discrete")
        self.action_type = action_type
        self.action_dim = action_dim
        self.actor = MLP(state_dim, action_dim, hidden_dims, activation)
        self.critic = MLP(state_dim, 1, hidden_dims, activation)
        # Orthogonal init (critical for PPO/AC stability)
        self.actor.init_weights(scales=1.0)
        self.critic.init_weights(scales=1.0)
        if self.action_type == "continuous":
            self.std = nn.Parameter(init_noise_std * torch.ones(action_dim))
        self.distribution = None

    @property
    def action_mean(self):
        return self.distribution.mean if self.action_type == "continuous" else self.distribution.probs

    @property
    def action_std(self):
        return self.distribution.stddev if self.action_type == "continuous" else torch.ones_like(self.distribution.probs)

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1) if self.action_type == "continuous" else self.distribution.entropy()

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def _update_distribution(self, obs):
        if self.action_type == "continuous":
            mean = self.actor(obs)
            self.distribution = Normal(mean, self.std.expand_as(mean))
        else:
            self.distribution = Categorical(logits=self.actor(obs))

    def act(self, obs):
        self._update_distribution(obs)
        actions = self.distribution.sample()
        if self.action_type == "discrete":
            actions = actions.unsqueeze(-1)
        return actions

    def act_inference(self, obs):
        if self.action_type == "continuous":
            return self.actor(obs)
        return self.actor(obs).argmax(dim=-1, keepdim=True)

    def evaluate(self, obs):
        return self.critic(obs)

    def get_actions_log_prob(self, actions):
        if self.action_type == "continuous":
            return self.distribution.log_prob(actions).sum(dim=-1)
        return self.distribution.log_prob(actions.squeeze(-1))


class AC(OnPolicyAlgorithm):
    def __init__(self, device=None, num_of_action=None, action_range=[None, None],
                 n_observations=None, hidden_dims=[None], activation=None,
                 action_type=None, init_noise_std=None, learning_rate=None,
                 discount_factor=None, value_loss_coef=None, entropy_coef=None,
                 max_grad_norm=None, num_epochs=5):

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.policy = ActorCritic(n_observations, num_of_action, hidden_dims, activation,
                                  action_type, init_noise_std).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.action_type = action_type
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        super(AC, self).__init__(
            num_of_action=num_of_action, action_range=action_range,
            learning_rate=learning_rate, discount_factor=discount_factor,
        )

    def learn(self, env, num_agents: int = 1, n_episodes: int = 20000, rollout_steps: int = 512):
        self.policy.train()
        T = rollout_steps

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

            # Collect rollout (no grad — we re-evaluate during update)
            with torch.no_grad():
                for _ in range(T):
                    obs_buf.append(state.clone())
                    self.policy._update_distribution(state)
                    action = self.policy.distribution.sample()
                    if self.action_type == "discrete":
                        action = action.unsqueeze(-1)
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

            obs_tensor = torch.stack(obs_buf)          # (T, N, 4)
            actions_tensor = torch.stack(actions_buf)  # (T, N, act_dim)
            rewards = torch.stack(rewards_buf)          # (T, N)
            dones = torch.stack(dones_buf)              # (T, N)

            # Bootstrap last value
            with torch.no_grad():
                last_value = self.policy.evaluate(state).squeeze(-1)
            G = last_value
            returns = torch.zeros(T, num_agents, device=self.device)
            for t in reversed(range(T)):
                G = rewards[t] + self.discount_factor * G * (1.0 - dones[t])
                returns[t] = G

            # Multi-epoch update (re-evaluate with current policy each epoch)
            for _ in range(self.num_epochs):
                all_log_probs = []
                all_values = []
                all_entropy = []
                for t in range(T):
                    self.policy._update_distribution(obs_tensor[t])
                    if self.action_type == "continuous":
                        lp = self.policy.get_actions_log_prob(actions_tensor[t])
                    else:
                        lp = self.policy.get_actions_log_prob(actions_tensor[t])
                    all_log_probs.append(lp)
                    all_values.append(self.policy.evaluate(obs_tensor[t]).squeeze(-1))
                    all_entropy.append(self.policy.entropy.mean())

                log_probs = torch.stack(all_log_probs)  # (T, N)
                values = torch.stack(all_values)          # (T, N)
                entropy = torch.stack(all_entropy).mean()

                # Normalize advantage PER-ENV (dim=0)
                advantage = (returns.detach() - values.detach())
                adv_mean = advantage.mean(dim=0, keepdim=True)
                adv_std = advantage.std(dim=0, keepdim=True)
                advantage = (advantage - adv_mean) / (adv_std + 1e-8)

                actor_loss = -(log_probs * advantage).mean()
                critic_loss = (values - returns.detach()).pow(2).mean()
                entropy_loss = -self.entropy_coef * entropy
                total_loss = actor_loss + self.value_loss_coef * critic_loss + entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if total_episodes - last_log >= 100 and total_episodes > 0:
                n_new = total_episodes - last_log
                avg = sum_reward / n_new
                print(f"[AC] ep {total_episodes} | avg_return={avg:.2f} | loss={total_loss.item():.4f}")
                self.plot_durations(timestep=int(avg))
                sum_reward = 0.0
                last_log = total_episodes

        return total_return / max(total_episodes, 1), total_episodes

    def act(self, obs):
        return self.policy.act(obs)

    def process_env_step(self, rewards, dones):
        pass

    def select_action(self, obs):
        self.policy.eval()
        with torch.inference_mode():
            return self.policy.act_inference(obs)

    def save_model(self, path, filename):
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(path, filename))

    def load_model(self, path, filename):
        self.policy.load_state_dict(torch.load(os.path.join(path, filename), map_location=self.device))
