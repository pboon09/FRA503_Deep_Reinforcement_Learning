from __future__ import annotations
import os
import numpy as np
import torch
from RL_Algorithm.RL_base_function import BaseAlgorithm


class Linear_QN(BaseAlgorithm):
    """
    Linear Q-Learning with function approximation.

    Args:
        num_of_action (int): Number of discrete actions.
        action_range (list): [min, max] continuous action range.
        learning_rate (float): TD weight-update step size.
        initial_epsilon (float): Starting exploration rate.
        epsilon_decay (float): Per-step epsilon decay.
        final_epsilon (float): Minimum exploration rate.
        discount_factor (float): Discount factor γ.
    """

    def __init__(
            self,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            learning_rate: float = 0.01,
            initial_epsilon: float = 1.0,
            epsilon_decay: float = 1e-3,
            final_epsilon: float = 0.001,
            discount_factor: float = 0.95,
    ) -> None:

        super().__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )

        # ===== Linear weight matrix ===== #
        # Shape: (obs_feature_dim, num_of_action)
        self.w = np.zeros((4, num_of_action))

        # Observation scale for normalization: [cart_pos, pole_angle, cart_vel, pole_ang_vel]
        # CartPole obs ranges: pos~±3, angle~±0.42rad, vel~±5, ang_vel~±5
        self.obs_scale = np.array([3.0, 0.419, 5.0, 5.0], dtype=np.float32)

        # GPU version of weights (created lazily in learn())
        self._w_gpu = None
        self._device = None

    # ------------------------------------------------------------------ #
    # Linear Q-value estimation                                           #
    # ------------------------------------------------------------------ #

    def q(self, obs, a=None):
        """
        Return the linearly-estimated Q-value(s) for a given observation.

        Args:
            obs: State feature vector φ(s), shape (obs_dim,).
            a (int | None): Action index. If None, returns Q for all actions
                            as a 1-D array of shape (num_of_action,).

        Returns:
            float | np.ndarray: Q(s, a) scalar, or Q(s, :) array.
        """
        # ========= put your code here ========= #
        q_all = obs @ self.w  # shape (num_of_action,)
        if a is None:
            return q_all
        return q_all[a]
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Core algorithm methods                                               #
    # ------------------------------------------------------------------ #

    def update(
        self,
        obs,
        action: int,
        reward: float,
        next_obs,
        next_action: int,
        terminated: bool,
    ):
        """
        Update the weight vector using the TD error.

        Args:
            obs: Current state feature vector φ(s).
            action (int): Action index taken in state s.
            reward (float): Reward received.
            next_obs: Next state feature vector φ(s').
            next_action (int): Next action taken (for SARSA-style update).
            terminated (bool): True if the episode ended.
        """
        # ========= put your code here ========= #
        td_target = reward + self.discount_factor * np.max(self.q(next_obs)) * (1 - terminated)
        td_error = td_target - self.q(obs, action)
        td_error = np.clip(td_error, -5.0, 5.0)  # clip to prevent overflow
        self.w[:, action] += self.lr * td_error * obs
        # ====================================== #

    def select_action(self, state):
        """
        Select an action using an epsilon-greedy policy over Q(s, :).

        Args:
            state: Current state feature vector φ(s).

        Returns:
            Tuple[Tensor, int]: Scaled continuous action tensor and action index.
        """
        # ========= put your code here ========= #
        state_norm = state / self.obs_scale
        if np.random.random() < self.epsilon:
            action_index = np.random.randint(self.num_of_action)
        else:
            action_index = int(np.argmax(self.q(state_norm)))
        scaled_action = self.scale_action(action_index)
        return scaled_action, action_index
        # ====================================== #

    def learn(self, env, max_steps: int, num_agents: int = 256):
        """
        Train the agent across parallel vectorized environments.
        Uses pure GPU torch operations to avoid CPU transfers.

        Args:
            env: Vectorized Isaac Lab environment.
            max_steps (int): Steps per iteration.
            num_agents (int): Number of parallel environments.

        Returns:
            Tuple[float, float]: (mean_episode_return, mean_episode_length)
        """
        # ========= put your code here ========= #
        action_min, action_max = self.action_range

        obs, _ = env.reset()
        if isinstance(obs, dict):
            obs = obs["policy"]
        device = obs.device

        # Sync numpy weights to GPU tensor
        if self._w_gpu is None or self._device != device:
            self._device = device
        self._w_gpu = torch.tensor(self.w, dtype=torch.float32, device=device)
        obs_scale_gpu = torch.tensor(self.obs_scale, dtype=torch.float32, device=device)

        # Normalize initial observations
        obs = obs / obs_scale_gpu

        # Per-env tracking (all on GPU)
        env_returns = torch.zeros(num_agents, device=device)
        env_lengths = torch.zeros(num_agents, device=device)
        completed_returns = []
        completed_lengths = []

        for step in range(max_steps):
            # --- Vectorized Q-values on GPU: (N, 4) @ (4, A) -> (N, A) ---
            all_q = obs @ self._w_gpu  # (num_agents, num_of_action)
            greedy_actions = all_q.argmax(dim=1)  # (num_agents,)

            # Epsilon-greedy
            random_actions = torch.randint(0, self.num_of_action, (num_agents,), device=device)
            explore_mask = torch.rand(num_agents, device=device) < self.epsilon
            action_indices = torch.where(explore_mask, random_actions, greedy_actions)

            # Scale actions: (num_agents,)
            scaled = action_min + (action_indices.float() / (self.num_of_action - 1)) * (action_max - action_min)

            # Step env: (num_agents, 1) for Isaac Lab
            next_obs, reward, terminated, truncated, _ = env.step(scaled.unsqueeze(-1))
            if isinstance(next_obs, dict):
                next_obs = next_obs["policy"]
            next_obs = next_obs / obs_scale_gpu
            done = (terminated | truncated).bool()

            # --- Vectorized TD update on GPU ---
            # Q(s, a_taken) for all envs
            next_q = next_obs @ self._w_gpu  # (N, A)
            next_max_q = next_q.max(dim=1).values  # (N,)
            current_q = all_q.gather(1, action_indices.unsqueeze(1)).squeeze(1)  # (N,)
            td_target = reward + self.discount_factor * next_max_q * (~terminated).float()
            td_error = (td_target - current_q).clamp(-5.0, 5.0)  # (N,)

            # Batch weight update: for each action, accumulate gradients
            for a in range(self.num_of_action):
                mask = (action_indices == a)
                if mask.any():
                    # grad = lr * td_error * obs for envs that took action a
                    grads = self.lr * td_error[mask].unsqueeze(1) * obs[mask]  # (count, 4)
                    self._w_gpu[:, a] += grads.mean(dim=0)  # average over envs

            # Track returns
            env_returns += reward
            env_lengths += 1
            if done.any():
                done_idx = done.nonzero(as_tuple=True)[0]
                completed_returns.extend(env_returns[done_idx].cpu().tolist())
                completed_lengths.extend(env_lengths[done_idx].cpu().tolist())
                env_returns[done] = 0.0
                env_lengths[done] = 0.0

            obs = next_obs
            self.decay_epsilon()

        # Sync GPU weights back to numpy
        self.w = self._w_gpu.cpu().numpy()

        if completed_returns:
            return float(np.mean(completed_returns)), float(np.mean(completed_lengths))
        return float(env_returns.mean().item()), float(env_lengths.mean().item())
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Persistence — linear weights only                                    #
    # ------------------------------------------------------------------ #

    def save_model(self, path: str, filename: str) -> None:
        """
        Save the weight matrix self.w to disk as a .npy file.

        Args:
            path (str): Directory to save the file.
            filename (str): File name (e.g., 'linear_q_cartpole.npy').
        """
        # ========= put your code here ========= #
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, filename), self.w)
        # ====================================== #

    def load_model(self, path: str, filename: str) -> None:
        """
        Load the weight matrix self.w from a .npy file.

        Args:
            path (str): Directory containing the file.
            filename (str): File name (e.g., 'linear_q_cartpole.npy').
        """
        # ========= put your code here ========= #
        self.w = np.load(os.path.join(path, filename))
        # ====================================== #
