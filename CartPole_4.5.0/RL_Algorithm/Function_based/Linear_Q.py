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
        # obs is a (num_agents, 4) GPU tensor; convert to numpy
        obs_np = obs.cpu().numpy()  # (num_agents, 4)

        # Per-env tracking for episode returns and lengths
        env_returns = np.zeros(num_agents)
        env_lengths = np.zeros(num_agents, dtype=int)
        completed_returns = []
        completed_lengths = []

        for timestep in range(1, max_steps + 1):
            # --- Select actions for ALL envs (epsilon-greedy per env) ---
            action_indices = np.empty(num_agents, dtype=int)
            for i in range(num_agents):
                if np.random.random() < self.epsilon:
                    action_indices[i] = np.random.randint(self.num_of_action)
                else:
                    action_indices[i] = int(np.argmax(self.q(obs_np[i])))

            # --- Build (num_agents,) scaled action tensor on GPU ---
            scaled_actions = torch.tensor(
                action_min + (action_indices / (self.num_of_action - 1)) * (action_max - action_min),
                dtype=torch.float32,
                device=obs.device,
            )  # (num_agents,)

            # --- Step all envs simultaneously ---
            # Isaac Lab expects (num_envs, action_dim) shape
            next_obs, reward, terminated, truncated, _ = env.step(scaled_actions.unsqueeze(-1))
            if isinstance(next_obs, dict):
                next_obs = next_obs["policy"]

            next_obs_np = next_obs.cpu().numpy()       # (num_agents, 4)
            reward_np = reward.cpu().numpy()            # (num_agents,)
            terminated_np = terminated.cpu().numpy()    # (num_agents,)
            truncated_np = truncated.cpu().numpy()      # (num_agents,)
            done_np = np.logical_or(terminated_np, truncated_np)

            # --- Update weights from ALL envs' transitions ---
            for i in range(num_agents):
                # Pick next action for SARSA-style next_action arg
                next_action_index = int(np.argmax(self.q(next_obs_np[i])))
                self.update(
                    obs_np[i],
                    action_indices[i],
                    reward_np[i],
                    next_obs_np[i],
                    next_action_index,
                    bool(terminated_np[i]),
                )

            # --- Accumulate per-env returns and lengths ---
            env_returns += reward_np
            env_lengths += 1

            # --- Log completed episodes and reset trackers ---
            for i in range(num_agents):
                if done_np[i]:
                    completed_returns.append(env_returns[i])
                    completed_lengths.append(env_lengths[i])
                    env_returns[i] = 0.0
                    env_lengths[i] = 0

            # Isaac Lab auto-resets; next_obs is already the new obs
            obs_np = next_obs_np
            obs = next_obs

            self.decay_epsilon()

        # Compute means over completed episodes (fallback to running totals)
        if len(completed_returns) > 0:
            mean_return = float(np.mean(completed_returns))
            mean_length = float(np.mean(completed_lengths))
        else:
            mean_return = float(np.mean(env_returns))
            mean_length = float(np.mean(env_lengths))

        return mean_return, mean_length
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