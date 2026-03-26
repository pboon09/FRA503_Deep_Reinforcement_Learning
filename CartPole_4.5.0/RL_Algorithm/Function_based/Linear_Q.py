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
        if np.random.random() < self.epsilon:
            action_index = np.random.randint(self.num_of_action)
        else:
            action_index = int(np.argmax(self.q(state)))
        scaled_action = self.scale_action(action_index)
        return scaled_action, action_index
        # ====================================== #

    def learn(self, env, max_steps: int, num_agents: int = 256):
        """
        Train the agent across parallel vectorized environments.

        Isaac Lab auto-resets envs when done. The returned next_obs is
        already the reset observation, so we never break on done -- we
        simply keep stepping for max_steps and track per-env returns.

        Args:
            env: Vectorized Isaac Lab environment with ``num_agents`` sub-envs.
            max_steps (int): Total number of environment steps to take.
            num_agents (int): Number of parallel environments (default 256).

        Returns:
            Tuple[float, float]: (mean_episode_return, mean_episode_length)
                over all episodes that completed during the run.
        """
        # ========= put your code here ========= #
        action_min, action_max = self.action_range

        # --- Reset env and get initial obs (num_agents, obs_dim) ---
        obs, _ = env.reset()
        if isinstance(obs, dict):
            obs = obs["policy"]
        obs_np = obs.cpu().numpy()  # (num_agents, 4)

        # Per-env tracking
        env_returns = np.zeros(num_agents)
        env_lengths = np.zeros(num_agents, dtype=int)
        completed_returns = []
        completed_lengths = []

        # Number of envs to learn from each step (subset for speed)
        learn_batch = min(16, num_agents)

        for timestep in range(1, max_steps + 1):
            # --- Vectorized action selection (epsilon-greedy) ---
            # Q-values for all envs: obs_np @ self.w -> (num_agents, num_actions)
            all_q = obs_np @ self.w  # (num_agents, num_of_action)
            greedy_actions = np.argmax(all_q, axis=1)  # (num_agents,)
            random_actions = np.random.randint(0, self.num_of_action, size=num_agents)
            explore_mask = np.random.random(num_agents) < self.epsilon
            action_indices = np.where(explore_mask, random_actions, greedy_actions)

            # --- Build scaled action tensor ---
            scaled_np = action_min + (action_indices / (self.num_of_action - 1)) * (action_max - action_min)
            scaled_actions = torch.tensor(scaled_np, dtype=torch.float32, device=obs.device)

            # --- Step all envs (Isaac Lab expects (num_envs, action_dim)) ---
            next_obs, reward, terminated, truncated, _ = env.step(scaled_actions.unsqueeze(-1))
            if isinstance(next_obs, dict):
                next_obs = next_obs["policy"]

            next_obs_np = next_obs.cpu().numpy()
            reward_np = reward.cpu().numpy().flatten()
            terminated_np = terminated.cpu().numpy().flatten()
            truncated_np = truncated.cpu().numpy().flatten()
            done_np = terminated_np | truncated_np

            # --- Update weights from a random subset of envs (for speed) ---
            batch_idx = np.random.choice(num_agents, learn_batch, replace=False)
            for i in batch_idx:
                next_action = int(np.argmax(self.q(next_obs_np[i])))
                self.update(
                    obs_np[i], action_indices[i], reward_np[i],
                    next_obs_np[i], next_action, bool(terminated_np[i]),
                )

            # --- Track per-env returns ---
            env_returns += reward_np
            env_lengths += 1
            done_mask = done_np.astype(bool)
            if done_mask.any():
                completed_returns.extend(env_returns[done_mask].tolist())
                completed_lengths.extend(env_lengths[done_mask].tolist())
                env_returns[done_mask] = 0.0
                env_lengths[done_mask] = 0

            obs_np = next_obs_np
            obs = next_obs
            self.decay_epsilon()

        if completed_returns:
            return float(np.mean(completed_returns)), float(np.mean(completed_lengths))
        return float(np.mean(env_returns)), float(np.mean(env_lengths))
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