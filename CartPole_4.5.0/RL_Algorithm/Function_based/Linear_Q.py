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
            lr_decay: float = 1.0,
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
        self.lr_decay = lr_decay
        self.lr_min = 0.001
        self.w = np.zeros((4, num_of_action))
        self.obs_scale = np.array([3.0, 0.419, 5.0, 5.0], dtype=np.float64)

    def q(self, obs, a=None):
        obs = np.asarray(obs, dtype=np.float64).reshape(-1, 4)
        if a is None:
            return obs @ self.w
        return np.einsum('ij,j->i', obs, self.w[:, a])

    def update_batch(self, states, actions, rewards, next_states, terminateds):
        N = len(actions)
        q_next_all = next_states @ self.w
        max_q_next = np.max(q_next_all, axis=1)
        targets = rewards + self.discount_factor * max_q_next * (1.0 - terminateds)
        q_current = np.array([states[i] @ self.w[:, actions[i]] for i in range(N)])
        deltas = np.clip(targets - q_current, -1.0, 1.0)  # clip TD error for stability
        # Average gradient across all envs to prevent lr scaling with num_envs
        grad = np.zeros_like(self.w)
        for i in range(N):
            grad[:, actions[i]] += deltas[i] * states[i]
        grad /= N
        self.w += self.lr * grad
        # Decay learning rate
        self.lr = max(self.lr_min, self.lr * self.lr_decay)

    def select_action(self, state):
        if isinstance(state, dict):
            state = state['policy']
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy().flatten()[:4]
        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(0, self.num_of_action)
        else:
            state_norm = np.clip(state[:4] / self.obs_scale, -1.0, 1.0)
            q_vals = state_norm @ self.w
            action_idx = int(np.argmax(q_vals))
        scaled_action = self.scale_action(action_idx)
        return scaled_action, action_idx

    def learn(self, env, num_agents: int = 1, n_episodes: int = 20000):
        obs, _ = env.reset()
        states = obs['policy'].cpu().numpy()

        episode_rewards = np.zeros(num_agents)
        episode_steps = np.zeros(num_agents, dtype=int)
        total_episodes = 0
        total_return = 0.0
        sum_reward = 0.0
        last_log = 0
        global_step = 0

        action_min, action_max = self.action_range

        while total_episodes < n_episodes:
            # Normalize observations
            states_norm = np.clip(states / self.obs_scale, -1.0, 1.0)
            q_all = states_norm @ self.w
            greedy = np.argmax(q_all, axis=1)
            random_mask = np.random.random(num_agents) < self.epsilon
            random_actions = np.random.randint(0, self.num_of_action, size=num_agents)
            action_indices = np.where(random_mask, random_actions, greedy)

            scaled = action_min + (action_max - action_min) * action_indices.astype(np.float32) / (self.num_of_action - 1)
            action_tensor = torch.tensor(scaled.reshape(-1, 1), dtype=torch.float32)

            next_obs, reward, terminated, truncated, _ = env.step(action_tensor)
            next_states = next_obs['policy'].cpu().numpy()
            reward_np = reward.cpu().numpy().flatten()
            term_np = terminated.cpu().numpy().flatten().astype(np.float64)
            done_np = (terminated | truncated).cpu().numpy().flatten()

            episode_rewards += reward_np
            episode_steps += 1
            global_step += num_agents
            next_states_norm = np.clip(next_states / self.obs_scale, -1.0, 1.0)
            self.update_batch(states_norm, action_indices, reward_np, next_states_norm, term_np)

            for i in range(num_agents):
                if done_np[i]:
                    self.episode_log.append({
                        "episode": total_episodes,
                        "global_step": global_step,
                        "ep_return": float(episode_rewards[i]),
                        "ep_length": int(episode_steps[i]),
                        "epsilon": float(self.epsilon),
                    })
                    sum_reward += episode_rewards[i]
                    total_return += episode_rewards[i]
                    episode_rewards[i] = 0.0
                    episode_steps[i] = 0
                    total_episodes += 1
                    self.decay_epsilon()  # per episode, not per step

            if total_episodes - last_log >= 100 and total_episodes > 0:
                n_new = total_episodes - last_log
                print(f"[Linear_Q] ep {total_episodes} | avg_return={sum_reward/n_new:.2f} | eps={self.epsilon:.4f}")
                self.plot_durations(timestep=int(sum_reward / n_new))
                sum_reward = 0.0
                last_log = total_episodes

            states = next_states

        return total_return / max(total_episodes, 1), total_episodes

    def save_model(self, path: str, filename: str) -> None:
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, filename), self.w)

    def load_model(self, path: str, filename: str) -> None:
        self.w = np.load(os.path.join(path, filename))
