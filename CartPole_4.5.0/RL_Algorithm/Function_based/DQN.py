from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
from storage.off_policy import OffPolicyAlgorithm


class DQN_network(nn.Module):
    def __init__(self, n_observations, hidden_size, n_actions, dropout):
        super(DQN_network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observations, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQN(OffPolicyAlgorithm):
    def __init__(
            self,
            device=None,
            num_of_action: int = None,
            action_range: list = [None, None],
            n_observations: int = None,
            hidden_dim: int = None,
            dropout: float = None,
            learning_rate: float = None,
            tau: float = None,
            initial_epsilon: float = None,
            epsilon_decay: float = None,
            final_epsilon: float = None,
            discount_factor: float = None,
            buffer_size: int = None,
            batch_size: int = None,
            learning_starts: int = 2000,
    ) -> None:
        self.learning_starts = learning_starts
        self.policy_net = DQN_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.target_net = DQN_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.device = device
        self.steps_done = 0
        self.num_of_action = num_of_action
        self.tau = tau
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50000, gamma=0.5)

        super(DQN, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

    def select_action(self, state):
        if torch.rand(1).item() < self.epsilon:
            action_idx = torch.randint(0, self.num_of_action, (1,)).item()
        else:
            self.policy_net.eval()
            with torch.no_grad():
                q_values = self.policy_net(state)
                action_idx = q_values.argmax(dim=-1).item()
            self.policy_net.train()
        scaled_action = self.scale_action(action_idx)
        return scaled_action, action_idx

    def calculate_loss(self, non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch):
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(state_batch.size(0), device=self.device)
        with torch.no_grad():
            if non_final_next_states.size(0) > 0:
                # # Double DQN: policy net selects action, target net evaluates
                # best_actions = self.policy_net(non_final_next_states).argmax(dim=1, keepdim=True)
                # next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, best_actions).squeeze(1)

                # Vanilla DQN: target net selects and evaluates
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(dim=1).values
        expected = (reward_batch + self.discount_factor * next_state_values).unsqueeze(1)
        return F.smooth_l1_loss(state_action_values, expected)

    def generate_sample(self, batch_size=None):
        batch = super().generate_sample()
        if batch is None:
            return None
        Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
        unpacked = Transition(*zip(*batch))
        state_batch = torch.cat(unpacked.state).to(self.device)
        action_batch = torch.tensor(unpacked.action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(unpacked.reward, dtype=torch.float32, device=self.device)
        non_final_mask = torch.tensor(
            [s is not None for s in unpacked.next_state], dtype=torch.bool, device=self.device
        )
        nf_states = [s for s in unpacked.next_state if s is not None]
        if nf_states:
            non_final_next_states = torch.cat(nf_states).to(self.device)
        else:
            non_final_next_states = torch.empty(0, state_batch.size(1), device=self.device)
        return non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch

    def update_policy(self):
        sample = self.generate_sample()
        if sample is None:
            return None
        loss = self.calculate_loss(*sample)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        # Mean Q-value for diagnostics
        with torch.no_grad():
            q_mean = self.policy_net(sample[2]).max(dim=1).values.mean().item()
        return {"td_loss": loss.item(), "grad_norm": grad_norm.item(), "q_mean": q_mean}

    def update_target_networks(self):
        for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.lerp_(pp.data, self.tau)

    def learn(self, env, num_agents: int = 1, n_episodes: int = 20000):
        obs, _ = env.reset()
        state = obs['policy'].to(self.device)

        episode_rewards = torch.zeros(num_agents, device=self.device)
        episode_steps = torch.zeros(num_agents, dtype=torch.int, device=self.device)
        total_episodes = 0
        total_return = 0.0
        sum_reward = 0.0
        last_log = 0
        global_step = 0

        action_min, action_max = self.action_range

        while total_episodes < n_episodes:
            self.policy_net.eval()
            with torch.no_grad():
                q_values = self.policy_net(state)
            self.policy_net.train()

            greedy_idx = q_values.argmax(dim=-1)
            random_mask = torch.rand(num_agents, device=self.device) < self.epsilon
            random_actions = torch.randint(0, self.num_of_action, (num_agents,), device=self.device)
            action_indices = torch.where(random_mask, random_actions, greedy_idx)

            scaled = action_min + (action_max - action_min) * action_indices.float() / (self.num_of_action - 1)
            action_tensor = scaled.unsqueeze(1)

            next_obs, reward, terminated, truncated, _ = env.step(action_tensor)
            next_state = next_obs['policy'].to(self.device)
            done_flags = (terminated | truncated)

            episode_rewards += reward.to(self.device).view(-1)
            episode_steps += 1
            global_step += num_agents

            for i in range(num_agents):
                term_i = bool(terminated[i].item())
                next_store = None if term_i else next_state[i:i+1].cpu()
                self.store_transition(
                    state[i:i+1].cpu(), action_indices[i].item(),
                    reward[i].item(), next_store, term_i,
                )
                if done_flags[i].item():
                    self.episode_log.append({
                        "episode": total_episodes,
                        "global_step": global_step,
                        "ep_return": episode_rewards[i].item(),
                        "ep_length": episode_steps[i].item(),
                        "epsilon": float(self.epsilon),
                    })
                    sum_reward += episode_rewards[i].item()
                    total_return += episode_rewards[i].item()
                    episode_rewards[i] = 0.0
                    episode_steps[i] = 0
                    total_episodes += 1

            # Epsilon decay per transition (not per vector step)
            for _ in range(num_agents):
                self.decay_epsilon()

            if global_step >= self.learning_starts:
                update_info = self.update_policy()
                self.update_target_networks()
                self.scheduler.step()
                if update_info is not None and total_episodes % 10 == 0:
                    self.metrics_log.append({
                        "iteration": len(self.metrics_log),
                        "global_step": global_step,
                        "td_loss": update_info["td_loss"],
                        "grad_norm": update_info["grad_norm"],
                        "q_mean": update_info["q_mean"],
                        "epsilon": float(self.epsilon),
                    })
            state = next_state

            if total_episodes - last_log >= 100 and total_episodes > 0:
                n_new = total_episodes - last_log
                avg = sum_reward / n_new
                print(f"[DQN] ep {total_episodes} | avg_return={avg:.2f} | eps={self.epsilon:.4f}")
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
        self.target_net.load_state_dict(self.policy_net.state_dict())
