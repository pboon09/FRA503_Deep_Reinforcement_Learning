from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from storage.off_policy import OffPolicyAlgorithm


class DQN_network(nn.Module):
    """
    Neural network model for the Deep Q-Network algorithm.

    Args:
        n_observations (int): Number of input features.
        hidden_size (int): Number of hidden neurons.
        n_actions (int): Number of possible actions.
        dropout (float): Dropout rate for regularization.
    """

    def __init__(self, n_observations, hidden_size, n_actions, dropout):
        super(DQN_network, self).__init__()
        # ========= put your code here ========= #
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.layer3 = nn.Linear(hidden_size, n_actions)
        # ====================================== #

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input state tensor.

        Returns:
            Tensor: Q-value estimates for each action.
        """
        # ========= put your code here ========= #
        x = self.dropout1(self.relu1(self.layer1(x)))
        x = self.dropout2(self.relu2(self.layer2(x)))
        x = self.layer3(x)
        return x
        # ====================================== #


class DQN(OffPolicyAlgorithm):
    """
    Deep Q-Network (DQN) — off-policy, value-based.

    Args:
        device: Torch device.
        num_of_action (int): Number of discrete actions.
        action_range (list): [min, max] for continuous action scaling.
        n_observations (int): Observation space dimension.
        hidden_dim (int): Hidden layer width.
        dropout (float): Dropout rate.
        learning_rate (float): Adam learning rate.
        tau (float): Polyak soft-update coefficient for target network.
        initial_epsilon (float): Starting exploration rate.
        epsilon_decay (float): Per-step epsilon decay.
        final_epsilon (float): Minimum exploration rate.
        discount_factor (float): Discount factor γ.
        buffer_size (int): Replay buffer capacity.
        batch_size (int): Mini-batch size per update.
    """

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

        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.policy_net = DQN_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.target_net = DQN_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.device        = device
        self.steps_done    = 0
        self.num_of_action = num_of_action
        self.tau           = tau

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
        pass
        # ====================================== #

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

    # ------------------------------------------------------------------ #
    # Core algorithm methods                                               #
    # ------------------------------------------------------------------ #

    def select_action(self, state):
        """
        Select an action using an epsilon-greedy policy.

        Args:
            state (Tensor): Current state.

        Returns:
            Tuple[Tensor, int]: Scaled action tensor and action index.
        """
        # ========= put your code here ========= #
        import random
        sample = random.random()
        if sample < self.epsilon:
            action_index = random.randrange(self.num_of_action)
        else:
            with torch.no_grad():
                # Ensure state has batch dimension
                if state.dim() == 1:
                    state = state.unsqueeze(0)
                q_values = self.policy_net(state)
                action_index = q_values.argmax(dim=1).item()
        self.decay_epsilon()
        scaled_action = self.scale_action(action_index)
        return scaled_action, action_index
        # ====================================== #

    def calculate_loss(self, non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch):
        """
        Compute the Bellman loss for a sampled mini-batch.

        Args:
            non_final_mask (Tensor): True where next state is not terminal.
            non_final_next_states (Tensor): Non-terminal next states.
            state_batch (Tensor): Batch of current states.
            action_batch (Tensor): Batch of action indices.
            reward_batch (Tensor): Batch of rewards.

        Returns:
            Tensor: Scalar Huber / MSE loss.
        """
        # ========= put your code here ========= #
        # Compute Q(s, a) - the policy net gives Q for all actions, we gather the ones taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s') for all next states: 0 for terminal, max_a Q_target(s', a) for non-terminal
        next_state_values = torch.zeros(state_batch.size(0), device=self.device)
        if non_final_next_states is not None and non_final_next_states.size(0) > 0:
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        # Compute expected Q values: r + gamma * V(s')
        expected_state_action_values = reward_batch + self.discount_factor * next_state_values

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        return loss
        # ====================================== #

    def generate_sample(self, batch_size=None):
        """
        Sample a mini-batch and unpack it into DQN-ready tensors.

        Returns:
            Tuple or None:
                - non_final_mask (Tensor)
                - non_final_next_states (Tensor)
                - state_batch (Tensor)
                - action_batch (Tensor)
                - reward_batch (Tensor)
            Returns None if the buffer is not ready.
        """
        # ========= put your code here ========= #
        batch = super().generate_sample()
        if batch is None:
            return None
        # ====================================== #

        # Unpack and prepare tensors from the Transition namedtuples
        # ========= put your code here ========= #
        # batch is a list of Transition namedtuples: (state, action, reward, next_state, done)
        state_batch = torch.cat([t.state if t.state.dim() >= 2 else t.state.unsqueeze(0) for t in batch]).to(self.device)
        action_batch = torch.tensor([[t.action] for t in batch], dtype=torch.long, device=self.device)
        reward_batch = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)

        # Build non-final mask and non-final next states
        non_final_mask = torch.tensor(
            [t.next_state is not None for t in batch], dtype=torch.bool, device=self.device
        )
        non_final_next_states_list = [
            t.next_state if t.next_state.dim() >= 2 else t.next_state.unsqueeze(0)
            for t in batch if t.next_state is not None
        ]
        if len(non_final_next_states_list) > 0:
            non_final_next_states = torch.cat(non_final_next_states_list).to(self.device)
        else:
            non_final_next_states = None

        return non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch
        # ====================================== #

    def update_policy(self):
        """Perform one gradient step on the policy network."""
        sample = self.generate_sample()
        if sample is None:
            return
        non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch = sample
        loss = self.calculate_loss(non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch)

        # ========= put your code here ========= #
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        # ====================================== #

    def update_target_networks(self):
        # ========= put your code here ========= #
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        # ====================================== #

    def learn(self, env, num_agents: int = 256, max_steps: int = 1000):
        """
        Train the agent across parallel vectorized environments.

        Isaac Lab auto-resets envs when done. The returned next_obs is
        already the reset observation, so we never break on done -- we
        simply keep stepping for max_steps and track per-env returns.

        Args:
            env: Vectorized Isaac Lab environment with ``num_agents`` sub-envs.
            num_agents (int): Number of parallel environments (default 256).
            max_steps (int): Total number of environment steps to take.

        Returns:
            Tuple[float, float]: (mean_episode_return, mean_episode_length)
                over all episodes that completed during the run.
        """
        # ========= put your code here ========= #
        import random as _random

        action_min, action_max = self.action_range

        # --- Reset env and get initial obs (num_agents, obs_dim) ---
        obs, _ = env.reset()
        if isinstance(obs, dict):
            obs = obs["policy"]
        # obs shape: (num_agents, obs_dim) tensor on GPU

        # Per-env tracking for episode returns and lengths
        env_returns = torch.zeros(num_agents, device=self.device)
        env_lengths = torch.zeros(num_agents, dtype=torch.long, device=self.device)
        completed_returns = []
        completed_lengths = []

        for timestep in range(1, max_steps + 1):
            # --- Batch epsilon-greedy action selection ---
            with torch.no_grad():
                q_values = self.policy_net(obs)  # (num_agents, num_of_action)
                greedy_actions = q_values.argmax(dim=1)  # (num_agents,)

            # Random actions for exploration
            random_actions = torch.randint(
                0, self.num_of_action, (num_agents,), device=self.device
            )

            # Per-env epsilon-greedy mask
            explore_mask = torch.rand(num_agents, device=self.device) < self.epsilon
            action_indices = torch.where(explore_mask, random_actions, greedy_actions)  # (num_agents,)

            # --- Build (num_agents,) scaled action tensor ---
            scaled_actions = (
                action_min
                + (action_indices.float() / (self.num_of_action - 1))
                * (action_max - action_min)
            )  # (num_agents,)

            # --- Step all envs simultaneously ---
            # Isaac Lab expects (num_envs, action_dim) shape
            next_obs, reward, terminated, truncated, _ = env.step(scaled_actions.unsqueeze(-1))
            if isinstance(next_obs, dict):
                next_obs = next_obs["policy"]

            done = terminated | truncated  # (num_agents,) bool tensor

            # --- Single batch CPU transfer (avoid per-env .item() GPU syncs) ---
            act_cpu = action_indices.cpu().numpy()
            rew_cpu = reward.cpu().numpy()
            term_cpu = terminated.cpu().numpy()
            done_cpu = done.cpu().numpy()

            # --- Store transitions for ALL envs in replay buffer ---
            for i in range(num_agents):
                if done_cpu[i]:
                    self.store_transition(obs[i], int(act_cpu[i]), float(rew_cpu[i]), None, bool(term_cpu[i]))
                else:
                    self.store_transition(obs[i], int(act_cpu[i]), float(rew_cpu[i]), next_obs[i], bool(term_cpu[i]))

            # --- Update policy and target networks ---
            self.update_policy()
            self.update_target_networks()

            # --- Track per-env returns (GPU) ---
            env_returns += reward
            env_lengths += 1

            # --- Log completed episodes (vectorized) ---
            done_mask = done.bool()
            if done_mask.any():
                done_idx = done_mask.nonzero(as_tuple=True)[0]
                completed_returns.extend(env_returns[done_idx].cpu().tolist())
                completed_lengths.extend(env_lengths[done_idx].cpu().tolist())
                env_returns[done_mask] = 0.0
                env_lengths[done_mask] = 0

            # Isaac Lab auto-resets; next_obs is already the new obs
            obs = next_obs

            self.decay_epsilon()

        # Compute means over completed episodes (fallback to running totals)
        if len(completed_returns) > 0:
            mean_return = sum(completed_returns) / len(completed_returns)
            mean_length = sum(completed_lengths) / len(completed_lengths)
        else:
            mean_return = env_returns.mean().item()
            mean_length = env_lengths.float().mean().item()

        return mean_return, mean_length
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save_model(self, path: str, filename: str) -> None:
        """
        Save policy network weights.

        Args:
            path (str): Directory to save.
            filename (str): File name (e.g., 'dqn_cartpole.pth').
        """
        # ========= put your code here ========= #
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy_net.state_dict(), os.path.join(path, filename))
        # ====================================== #

    def load_model(self, path: str, filename: str) -> None:
        """
        Load policy network weights and sync to target network.

        Args:
            path (str): Directory of saved model.
            filename (str): File name (e.g., 'dqn_cartpole.pth').
        """
        # ========= put your code here ========= #
        state_dict = torch.load(os.path.join(path, filename), map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # ====================================== #