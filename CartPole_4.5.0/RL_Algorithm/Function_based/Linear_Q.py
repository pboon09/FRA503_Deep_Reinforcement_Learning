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

    def learn(self, env, max_steps: int):
        """
        Train the agent for one episode.

        Args:
            env: The environment.
            max_steps (int): Maximum steps per episode.

        Returns:
            Tuple[float, int]: (episode_return, timestep)
        """
        # ========= put your code here ========= #
        obs, _ = env.reset()
        # Convert observation to numpy
        if isinstance(obs, dict):
            obs = obs["policy"]
        if isinstance(obs, torch.Tensor):
            obs = obs.squeeze().cpu().numpy()

        action, action_index = self.select_action(obs)
        episode_return = 0.0

        for timestep in range(1, max_steps + 1):
            next_obs, reward, terminated, truncated, _ = env.step(action)

            # Convert next_obs to numpy
            if isinstance(next_obs, dict):
                next_obs = next_obs["policy"]
            if isinstance(next_obs, torch.Tensor):
                next_obs = next_obs.squeeze().cpu().numpy()

            reward_value = reward.item() if isinstance(reward, torch.Tensor) else reward
            terminated_value = terminated.item() if isinstance(terminated, torch.Tensor) else terminated
            truncated_value = truncated.item() if isinstance(truncated, torch.Tensor) else truncated

            next_action_tensor, next_action_index = self.select_action(next_obs)

            self.update(obs, action_index, reward_value, next_obs, next_action_index, terminated_value)

            episode_return += reward_value

            done = terminated_value or truncated_value
            if done:
                break

            obs = next_obs
            action = next_action_tensor
            action_index = next_action_index

        self.decay_epsilon()
        return episode_return, timestep
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