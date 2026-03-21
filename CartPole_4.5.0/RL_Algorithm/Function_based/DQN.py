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
    ) -> None:

        self.policy_net = DQN_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.target_net = DQN_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.device        = device
        self.steps_done    = 0
        self.num_of_action = num_of_action
        self.tau           = tau

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)

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
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(dim=1).values
            )

        expected_state_action_values = (
            reward_batch + self.discount_factor * next_state_values
        ).unsqueeze(1)

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        return loss

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
            [s is not None for s in unpacked.next_state],
            dtype=torch.bool, device=self.device,
        )
        non_final_next_states = torch.cat(
            [s for s in unpacked.next_state if s is not None]
        ).to(self.device)

        return non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch

    def update_policy(self):
        sample = self.generate_sample()
        if sample is None:
            return
        non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch = sample
        loss = self.calculate_loss(non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

    def update_target_networks(self):
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.lerp_(policy_param.data, self.tau)

    def learn(self, env, num_agents: int = 1, max_steps: int = 1000):
        obs, _ = env.reset()
        state = obs['policy'].to(self.device)

        episode_return = 0.0
        timestep = 0

        for step in range(max_steps):
            scaled_action, action_idx = self.select_action(state)

            next_obs, reward, terminated, truncated, info = env.step(scaled_action)
            next_state = next_obs['policy'].to(self.device)

            reward_val = reward.item()
            term_val = terminated.item()
            trunc_val = truncated.item()
            episode_return += reward_val

            next_state_store = None if term_val else next_state.cpu()

            self.store_transition(
                state.cpu(), action_idx, reward_val,
                next_state_store, term_val,
            )

            self.update_policy()
            self.update_target_networks()
            self.decay_epsilon()

            timestep += 1
            state = next_state

            if term_val or trunc_val:
                break

        return episode_return, timestep

    def save_model(self, path: str, filename: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy_net.state_dict(), os.path.join(path, filename))

    def load_model(self, path: str, filename: str) -> None:
        self.policy_net.load_state_dict(
            torch.load(os.path.join(path, filename), map_location=self.device)
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
