from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType

class SARSA(BaseAlgorithm):
    def __init__(
            self,
            num_of_action: int,
            action_range: list,
            discretize_state_weight: list,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float,
    ) -> None:
        """
        Initialize the SARSA algorithm.

        Args:
            num_of_action (int): Number of possible actions.
            action_range (list): Scaling factor for actions.
            discretize_state_weight (list): Scaling factor for discretizing states.
            learning_rate (float): Learning rate for Q-value updates.
            initial_epsilon (float): Initial value for epsilon in epsilon-greedy policy.
            epsilon_decay (float): Rate at which epsilon decays.
            final_epsilon (float): Minimum value for epsilon.
            discount_factor (float): Discount factor for future rewards.
        """
        super().__init__(
            control_type=ControlType.TEMPORAL_DIFFERENCE,
            num_of_action=num_of_action,
            action_range=action_range,
            discretize_state_weight=discretize_state_weight,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )
        
    def update(
        self,
        obs_dis,
        action,
        reward,
        next_obs_dis,
        next_action,
        done,
    ):
        """
        Update Q-values using SARSA (on-policy TD).

        Q(s,a) <- Q(s,a) + lr * [r + gamma * Q(s',a') - Q(s,a)]
        next_action must be chosen by the same epsilon-greedy policy before calling update.

        Args:
            obs_dis (tuple): Discretized current state.
            action (int): Action taken at current state.
            reward (float): Reward received.
            next_obs_dis (tuple): Discretized next state.
            next_action (int): Action selected at next state (epsilon-greedy).
            done (bool): Whether the episode has ended.
        """
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * self.q_values[next_obs_dis][next_action]

        td_error = target - self.q_values[obs_dis][action]
        self.q_values[obs_dis][action] += self.lr * td_error
        self.training_error.append(abs(td_error))