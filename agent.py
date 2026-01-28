import numpy as np
from typing import Literal


class Agent:
    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.1,
        c: float = 2.0,
        strategy: Literal["epsilon_greedy", "ucb"] = "epsilon_greedy",
        initial_q: float = 0.0,
        alpha: float = None,
    ):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.c = c
        self.strategy = strategy
        self.initial_q = initial_q
        self.alpha = alpha

        self.q_estimates = np.full(n_arms, initial_q, dtype=float)
        self.action_counts = np.zeros(n_arms, dtype=int)
        self.total_steps = 0

    def select_action(self) -> int:
        if self.strategy == "epsilon_greedy":
            return self._epsilon_greedy_action()
        elif self.strategy == "ucb":
            return self._ucb_action()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _epsilon_greedy_action(self) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            max_q = np.max(self.q_estimates)
            best_actions = np.where(self.q_estimates == max_q)[0]
            return np.random.choice(best_actions)

    def _ucb_action(self) -> int:
        unvisited = np.where(self.action_counts == 0)[0]
        if len(unvisited) > 0:
            return np.random.choice(unvisited)

        exploration_bonus = self.c * np.sqrt(
            np.log(self.total_steps) / self.action_counts
        )
        ucb_values = self.q_estimates + exploration_bonus

        max_ucb = np.max(ucb_values)
        best_actions = np.where(ucb_values == max_ucb)[0]
        return np.random.choice(best_actions)

    def update(self, action: int, reward: float):
        self.action_counts[action] += 1
        self.total_steps += 1

        if self.alpha is not None:
            self.q_estimates[action] += self.alpha * (reward - self.q_estimates[action])
        else:
            n = self.action_counts[action]
            self.q_estimates[action] += (1 / n) * (reward - self.q_estimates[action])

    def reset(self):
        self.q_estimates = np.full(self.n_arms, self.initial_q, dtype=float)
        self.action_counts = np.zeros(self.n_arms, dtype=int)
        self.total_steps = 0

    def get_greedy_action(self) -> int:
        return int(np.argmax(self.q_estimates))
