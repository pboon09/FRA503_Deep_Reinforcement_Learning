from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType

class MC(BaseAlgorithm):
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
            q_init: float = 0.0,
    ) -> None:
        """
        Initialize the Monte Carlo algorithm.

        Args:
            num_of_action (int): Number of possible actions.
            action_range (list): Scaling factor for actions.
            discretize_state_weight (list): Scaling factor for discretizing states.
            learning_rate (float): Learning rate for Q-value updates.
            initial_epsilon (float): Initial value for epsilon in epsilon-greedy policy.
            epsilon_decay (float): Rate at which epsilon decays.
            final_epsilon (float): Minimum value for epsilon.
            discount_factor (float): Discount factor for future rewards.
            q_init (float): Initial Q-value for all state-action pairs.
        """
        super().__init__(
            control_type=ControlType.MONTE_CARLO,
            num_of_action=num_of_action,
            action_range=action_range,
            discretize_state_weight=discretize_state_weight,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
            q_init=q_init,
        )
        
    def update(
        self,
        obs_dis,
        action,
        reward,
        done,
    ):
        """
        Update Q-values using Monte Carlo.

        Accumulates (obs_dis, action, reward) each step. When done=True, computes
        discounted returns backwards and updates Q-values via constant-alpha.

        Args:
            obs_dis (tuple): Discretized state at current step.
            action (int): Discrete action taken.
            reward (float): Reward received.
            done (bool): Whether the episode has ended.
        """
        self.obs_hist.append(obs_dis)
        self.action_hist.append(action)
        self.reward_hist.append(reward)

        if done:
            G = 0
            for t in reversed(range(len(self.reward_hist))):
                G = self.discount_factor * G + self.reward_hist[t]
                s = self.obs_hist[t]
                a = self.action_hist[t]
                error = G - self.q_values[s][a]
                # Constant-alpha update for non-stationary policy improvement
                self.q_values[s][a] += self.lr * error
                self.training_error.append(abs(error))
            self.obs_hist.clear()
            self.action_hist.clear()
            self.reward_hist.clear()