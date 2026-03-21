from __future__ import annotations
import os
import numpy as np
import torch
from RL_Algorithm.RL_base_function import BaseAlgorithm


class Linear_QN(BaseAlgorithm):
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
        self.w = np.zeros((4, num_of_action))

    def q(self, obs, a=None):
        obs = np.asarray(obs, dtype=np.float64).flatten()[:4]
        if a is None:
            return obs @ self.w
        return obs @ self.w[:, a]

    def update(self, obs, action, reward, next_obs, next_action, terminated):
        obs = np.asarray(obs, dtype=np.float64).flatten()[:4]
        next_obs = np.asarray(next_obs, dtype=np.float64).flatten()[:4]

        if terminated:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(next_obs @ self.w)

        delta = target - (obs @ self.w[:, action])
        self.w[:, action] += self.lr * delta * obs

    def select_action(self, state):
        if isinstance(state, dict):
            state = state['policy']
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy().flatten()[:4]

        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(0, self.num_of_action)
        else:
            q_vals = self.q(state)
            action_idx = int(np.argmax(q_vals))

        scaled_action = self.scale_action(action_idx)
        return scaled_action, action_idx

    def learn(self, env, max_steps: int):
        obs, _ = env.reset()
        state = obs['policy'].cpu().numpy().flatten()[:4]
        total_reward = 0.0
        timestep = 0

        for step in range(max_steps):
            _, action_idx = self.select_action(state)
            scaled_action = self.scale_action(action_idx)

            next_obs, reward, terminated, truncated, _ = env.step(scaled_action)
            next_state = next_obs['policy'].cpu().numpy().flatten()[:4]

            r = reward.item()
            term = terminated.item()
            trunc = truncated.item()
            total_reward += r

            self.update(state, action_idx, r, next_state, None, term)
            self.decay_epsilon()

            timestep += 1
            state = next_state

            if term or trunc:
                break

        return total_reward, timestep

    def save_model(self, path: str, filename: str) -> None:
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, filename), self.w)

    def load_model(self, path: str, filename: str) -> None:
        self.w = np.load(os.path.join(path, filename))
